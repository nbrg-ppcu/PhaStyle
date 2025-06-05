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
    Prepare and validate input arguments for ProkBERT inference.

    Parses command-line arguments and sets the configuration for the inference process.

    Returns:
        prokbert_config (ProkBERTConfig): Configuration object for ProkBERT inference.
        args (Namespace): Parsed command-line arguments.
    """
    prokbert_config = ProkBERTConfig()
    keyset = ['finetuning'] 
    # Get the argument parser and mappings
    parser, cmd_argument2group_param, group2param2cmdarg = prokbert_config.get_cmd_arg_parser(keyset)

    fastain_found = False
    for action in parser._actions:
        if '--fastain' in action.option_strings:
            action.required = True
            # Append "(required)" to the help text so users see it in --help
            action.help = (action.help or "") + "  (required)"
            fastain_found = True
            break
    if not fastain_found:
        raise RuntimeError("Parser did not define --fastain; cannot mark it required.")  # Sanity check


    out_found = False
    for action in parser._actions:
        if '--out' in action.option_strings:
            action.required = True
            action.help = (action.help or "") + "  (required)"
            out_found = True
            break
    if not out_found:
        raise RuntimeError("Parser did not define --out; cannot mark it required.")

    parser.set_defaults(ftmodel="neuralbioinfo/PhaStyle-mini") 

    # Parse the arguments
    args = parser.parse_args()
    user_provided_args = get_user_provided_args(args, parser)
    input_args2check = list( (set(user_provided_args.keys()) - {'help'}) | {'ftmodel'})
    parameter_group_names = list(prokbert_config.parameters.keys()) + ['inference']
    # Initialize the input parameter set
    parameters = {k: {} for k in parameter_group_names}
    for provided_input_argument in input_args2check:
        print(f'Setting: {provided_input_argument}')
        param_group, param_name = cmd_argument2group_param[provided_input_argument]
        act_value = getattr(args, provided_input_argument)
        parameters[param_group][param_name] = act_value

    requested_cores = getattr(args, "num_proc", None)
    if requested_cores is not None:
        cpu_cores = int(requested_cores)
    else:
        cpu_cores = multiprocessing.cpu_count()
    print(f"Using {cpu_cores} CPU core(s) for preprocessing.")
    parameters['finetuning']['num_cores']=cpu_cores

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
