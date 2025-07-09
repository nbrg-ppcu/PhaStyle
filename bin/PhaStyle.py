from os.path import join
import os
import pandas as pd
import torch
import numpy as np
from transformers import TrainingArguments, Trainer
from prokbert.sequtils import *
from prokbert.config_utils import ProkBERTConfig, get_user_provided_args
from prokbert.training_utils import *
from prokbert.models import ProkBertForSequenceClassification
from prokbert.tokenizer import LCATokenizer
import multiprocessing
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset



default_segmentation_length = 512


def preprocess_function(sample, tokenizer, max_length=512):
    """
    Tokenizes the sample's 'segment' field. Adjust max_length and padding strategy as needed.
    """
    tokenized = tokenizer(
        sample["segment"],
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    attention_masks = tokenized['attention_mask']
    for attention_mask in attention_masks:
        attention_mask[0] = 0
        attention_mask[-1] = 0
    results = {}
    results['input_ids'] = tokenized['input_ids']
    results['attention_mask'] = attention_masks
    #results["labels"] = sample["y"]
    return results




def prepare_input_arguments():
    """
    Define and parse only the essential CLI arguments for PhaBERT inference.

    Returns:
        parameters (dict): Nested dict for 'finetuning' and 'inference'.
        args (Namespace): Raw argparse results.
    """
    parser = argparse.ArgumentParser(description="ProkBERT PhaSTYLE inference")
    parser.add_argument(
        "--fastain", required=True,
        help="Path to input FASTA file for segmentation."
    )
    parser.add_argument(
        "--out", required=True,
        help="Path where the TSV results will be written."
    )
    parser.add_argument(
        "--ftmodel", default="neuralbioinfo/PhaStyle-mini",
        help="Pretrained model identifier or local path."
    )
    parser.add_argument(
        "-c", "--num-cores", dest="num_cores", type=int,
        help="Number of CPU cores to use for preprocessing. Defaults to all available."
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=32,
        help="Per-device evaluation batch size."
    )

    args = parser.parse_args()

    # Determine CPU cores
    if args.num_cores and args.num_cores > 0:
        cpu_cores = args.num_cores
    else:
        cpu_cores = multiprocessing.cpu_count()
    print(f"Using {cpu_cores} CPU core(s) for preprocessing.")

    parameters = {
        "finetuning": {
            "ftmodel": args.ftmodel,
            "num_cores": cpu_cores
        },
        "inference": {
            "per_device_eval_batch_size": args.batch_size
        }
    }
    return parameters, args

def prepare_model(model_path):
    """
    Load ProkBertForSequenceClassification and LCATokenizer from disk (or remote),
    with trust_remote_code=True, but fall back to local_files_only if network fails.
    Move model to CUDA if available.
    
    Returns:
        model (torch.nn.Module)
        tokenizer (PreTrainedTokenizer)
    """
    try:
        # Attempt a normal load (may hit HF network)
        model = ProkBertForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer = LCATokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception as e:
        print(
            f"Warning: network/proxy error while loading from '{model_path}'.\n"
            f"  Error: {e}\n"
            f"Falling back to local cache..."
        )
        model = ProkBertForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        tokenizer = LCATokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )


    return model, tokenizer

def prepare_dataset(fasta_path, tokenizer, num_cores, max_length=512):
    """
    1) Load contigs from the given FASTA file.
    2) Segment sequences into chunks of length `max_length` (contiguous).
    3) Turn the resulting pandas DataFrame into an in-memory Dataset.
    4) Tokenize segments with `num_cores` processes, keep in memory.

    Returns:
        tokenized_ds (datasets.Dataset): ready for Trainer.predict()
        raw_segment_df (pandas.DataFrame): original segment-level DataFrame
    """
    print(f"[prepare_dataset] Loading sequences from: {fasta_path}")
    sequences = load_contigs(
        [fasta_path],
        IsAddHeader=True,
        adding_reverse_complement=False,
        AsDataFrame=True,
        to_uppercase=True,
        is_add_sequence_id=True,
    )
    print(f"[prepare_dataset] Number of raw sequences: {len(sequences)}")

    print("[prepare_dataset] Running segmentation")
    segmentation_params = {
        "max_length": max_length,
        "min_length": int(max_length * 0.5),
        "type": "contiguous",
    }
    raw_segment_df = segment_sequences(
        sequences, segmentation_params, AsDataFrame=True
    )
    print(f"[prepare_dataset] Number of segments: {len(raw_segment_df)}")

    # Wrap into HF Dataset (in memory)
    hf_dataset = Dataset.from_pandas(raw_segment_df)

    # Tokenization function (same as before, except no labels)
    def _tokenize_fn(batch):
        tokenized = tokenizer(
            batch["segment"],
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        # Zero out first/last attention token
        masks = tokenized["attention_mask"]
        for m in masks:
            m[0] = 0
            m[-1] = 0
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": masks
        }

    print(f"[prepare_dataset] Tokenizing with {num_cores} CPU core(s)")
    tokenized_ds = hf_dataset.map(
        _tokenize_fn,
        batched=True,
        num_proc=num_cores,
        remove_columns=hf_dataset.column_names,
        keep_in_memory=True,
    )

    return sequences, tokenized_ds, hf_dataset


def post_processing_predictions(predictions, hf_dataset, sequences):

    final_columns = ['sequence_id', 'fasta_id', 'predicted_label', 'score_temperate', 'score_virulent']
    final_columns_rename = ['sequence_id', 'predicted_label', 'score_temperate', 'score_virulent', 'fasta_id']

    final_table = inference_binary_sequence_predictions(predictions, hf_dataset)
    final_table['predicted_label'] = final_table.apply(lambda x:  'virulent' if x['predicted_label']=='class_1' else 'temperate', axis=1)

    final_table = final_table.merge(sequences[['sequence_id', 'fasta_id']], how='left',
                                     left_on='sequence_id', right_on='sequence_id')
    final_table.columns = final_columns_rename
    final_table = final_table[final_columns]
    return final_table


def main(parameters, args):
    """
    Main function to perform inference using ProkBERT.

    Args:
        prokbert_config (ProkBERTConfig): Configuration object for ProkBERT inference.
        args (Namespace): Parsed command-line arguments.
    """
    print('ProkBERT PhaSTYLE prediction!')

    model_path = parameters["finetuning"]["ftmodel"]
    model, tokenizer = prepare_model(model_path)


    fasta_in = args.fastain
    num_cores = parameters["finetuning"]["num_cores"]

    sequences, tokenized_ds, hf_dataset = prepare_dataset(
        fasta_in, tokenizer, num_cores, max_length=default_segmentation_length
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tmp_output = "./prokbert_inference_output"
    os.makedirs(tmp_output, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=tmp_output,
        do_train=False,
        do_eval=False,
        per_device_eval_batch_size=parameters.get("inference", {}).get(
            "per_device_eval_batch_size", 32
        ),
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[main] Running prediction on segments...")
    predictions = trainer.predict(tokenized_ds)
    final_table = post_processing_predictions(predictions, hf_dataset, sequences)


    print(f'Writing the results into : {args.out}')
    final_table.to_csv(args.out, sep='\t', index=False)
    #print(final_table)
    



if __name__ == "__main__":
    print('Parsing input arguments!')
    parameters, args = prepare_input_arguments()
    main(parameters, args)
