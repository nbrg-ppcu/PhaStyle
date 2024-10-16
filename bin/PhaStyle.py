import yaml
import pathlib
from os.path import join
import os
import sys
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import TrainingArguments, Trainer

# Import custom modules from the prokbert package
from prokbert.training_utils import (
    get_default_pretrained_model_parameters,
    get_torch_data_from_segmentdb_classification,
    compute_metrics_eval_prediction,
    evaluate_binary_classification_bert_build_pred_results,
    evaluate_binary_classification_bert,
    weighted_voting
)
from prokbert.models import BertForBinaryClassificationWithPooling
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.config_utils import ProkBERTConfig, get_user_provided_args
from prokbert.sequtils import load_contigs, segment_sequences
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT

# Get the local rank for distributed training (if any)
local_rank = int(os.getenv('LOCAL_RANK', '0'))
print('Local rank:', local_rank)

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
    # Parse the arguments
    args = parser.parse_args()
    # Get user-provided arguments (excluding defaults)
    user_provided_args = get_user_provided_args(args, parser)
    input_args2check = list(set(user_provided_args.keys()) - {'help'})
    parameter_group_names = list(prokbert_config.parameters.keys()) + ['inference']
    # Initialize the input parameter set
    parameters = {k: {} for k in parameter_group_names}
    for provided_input_argument in input_args2check:
        print(f'Setting: {provided_input_argument}')
        param_group, param_name = cmd_argument2group_param[provided_input_argument]
        act_value = getattr(args, provided_input_argument)
        parameters[param_group][param_name] = act_value

    # Set configuration parameters
    _ = prokbert_config.get_and_set_pretraining_parameters(parameters['pretraining'])
    _ = prokbert_config.get_and_set_tokenization_parameters(parameters['tokenization'])
    _ = prokbert_config.get_and_set_segmentation_parameters(parameters['segmentation'])
    _ = prokbert_config.get_and_set_computation_params(parameters['computation'])
    _ = prokbert_config.get_and_set_datacollator_parameters(parameters['data_collator'])
    _ = prokbert_config.get_and_set_finetuning_parameters(parameters['finetuning'])
    prokbert_config.default_torchtype = torch.long
    return prokbert_config, args

def process_prediction(args, predictions, torchdb_annot, sequences):
    """
    Process the raw model predictions to generate final results.

    Args:
        args (Namespace): Parsed command-line arguments.
        predictions (PredictionOutput): Output from the Trainer's predict method.
        torchdb_annot (DataFrame): Annotated DataFrame mapping torch IDs to segment IDs.
        sequences (DataFrame): Original sequences loaded from the FASTA files.
    """
    final_cols = ['fasta_id', 'sequence_id', 'predicted_label', 'p_temperate', 'p_virulent']
    logits = predictions.predictions
    labels = predictions.label_ids
    print('Building predictions')
    # Evaluate predictions and compute probabilities
    pred_results = evaluate_binary_classification_bert_build_pred_results(logits, labels)
    logits = pred_results[:, 2:]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    combined_results = np.concatenate([pred_results, probabilities], axis=1)
    combined_results = pd.DataFrame(
        combined_results,
        columns=['y_true', 'y_pred', 'logit_y0', 'logit_y1', 'p_class_0', 'p_class1']
    )
    combined_results['torch_id'] = range(len(combined_results))
    # Merge with annotations
    combined_results_annot = combined_results.merge(
        torchdb_annot, how='left', left_on='torch_id', right_on='torch_id'
    )
    # Aggregate predictions using weighted voting
    sequence_predictions = weighted_voting(combined_results_annot)
    # Prepare the final output
    seq_db = sequences[['sequence_id', 'fasta_id']]
    sequence_predictions.rename(
        {'score_class_0': 'p_temperate', 'score_class_1': 'p_virulent'},
        axis=1, inplace=True
    )
    sequence_predictions = sequence_predictions.merge(seq_db)
    sequence_predictions['predicted_label'] = np.where(
        sequence_predictions['y_pred'] == 1, 'virulent', 'temperate'
    )
    final_table = sequence_predictions[final_cols]
    print(f'Writing the final prediction into: {args.out}')
    final_table.to_csv(args.out, sep='\t', index=False)

def get_prokbert_dataset(tokenizer, segment_db):
    """
    Prepare the ProkBERT dataset for inference.

    Args:
        tokenizer (ProkBERTTokenizer): Tokenizer for tokenizing sequences.
        segment_db (DataFrame): DataFrame containing segmented sequences.

    Returns:
        ds (Dataset): PyTorch dataset for inference.
        torchdb_annot (DataFrame): Annotated DataFrame mapping torch IDs to segment IDs.
        torchdb (DataFrame): DataFrame containing torch IDs and segment IDs.
    """
   
    [X, y, torchdb] = get_torch_data_from_segmentdb_classification(tokenizer, segment_db, randomize=False)
    torchdb_annot = torchdb.merge(segment_db, how='left', left_on='segment_id', right_on='segment_id')
    ds = ProkBERTTrainingDatasetPT(X, y, AddAttentionMask=True)    
    return ds, torchdb_annot, torchdb

def main(prokbert_config, args):
    """
    Main function to perform inference using ProkBERT.

    Args:
        prokbert_config (ProkBERTConfig): Configuration object for ProkBERT inference.
        args (Namespace): Parsed command-line arguments.
    """
    print('ProkBERT PhaSTYLE prediction!')
    # Configuration for tokenizer, segmentation, and computation
    tokenizer_config = prokbert_config.tokenization_params
    segmentation_config = prokbert_config.segmentation_params
    computation_config = prokbert_config.computation_params

    #print(args)
    print(f'Loading the model: {args.ftmodel}')
    act_model_name = args.ftmodel
    genomic_lm_type = 'prokbert'
    # Initialize the tokenizer
    tokenizer = ProkBERTTokenizer(tokenization_params=tokenizer_config)

    print(f'Estimated batch size: {args.per_device_eval_batch_size}')
    input_fasta_files = [args.fastain]

    print('Loading the sequence data:')
    # Load sequences from FASTA files
    sequences = load_contigs(
        input_fasta_files,
        IsAddHeader=True,
        adding_reverse_complement=False,
        AsDataFrame=True,
        to_uppercase=True,
        is_add_sequence_id=True
    )
    print('Number of sequences in db:', len(sequences))
    # Assign dummy labels (not used in inference)
    sequences['y'] = np.random.randint(0, 2, size=len(sequences))
    sequences['label'] = np.where(sequences['y'] == 1, 'virulent', 'temperate')

    print('Segmenting the database')
    start_time = time.time()
    segment_db = segment_sequences(sequences, segmentation_config, AsDataFrame=True)
    segment_db = segment_db.merge(
        sequences[['sequence_id', 'y', 'label']],
        how='left', left_on='sequence_id', right_on='sequence_id'
    )
    print(f'Segment db size: {len(segment_db)}')
    segment_time = time.time() - start_time
    print(f"Segmentation took {segment_time:.2f} seconds.")

    start_time = time.time()
    # Prepare the dataset
    ds, torchdb_annot, torchdb = get_prokbert_dataset(tokenizer, segment_db)
    tokenization_time = time.time() - start_time
    print(f"Tokenization and data preparation took {tokenization_time:.2f} seconds.")

    print('Loading the model')
    # Load the model
    if args.modelclass == 'BertForBinaryClassificationWithPooling':
        print(f'Loading the {args.ftmodel} into a {args.modelclass} class')
        model = BertForBinaryClassificationWithPooling.from_pretrained(args.ftmodel)
    elif args.modelclass == 'AutoModelForSequenceClassification':
        model = AutoModelForSequenceClassification.from_pretrained(
            args.ftmodel, trust_remote_code=True
        )
    else:
        print('Trying to load a BertForBinaryClassificationWithPooling model')
        model = BertForBinaryClassificationWithPooling.from_pretrained(args.ftmodel)

    print('Building the input arguments')
    # Set up training arguments for inference
    training_args = TrainingArguments(
        output_dir="/tmp",  # Directory to store results
        local_rank=local_rank,
        do_predict=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # Adjust as needed
        ddp_backend=args.ddp_backend,
        dataloader_drop_last=False,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode
        # Additional arguments as needed
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=None  # No metrics needed during inference
    )
    print('Running predictions')
    # Perform prediction
    predictions = trainer.predict(ds)
    # Process predictions (only on rank 0 if distributed training is used)
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            process_prediction(args, predictions, torchdb_annot, sequences)
    else:
        process_prediction(args, predictions, torchdb_annot, sequences)

if __name__ == "__main__":
    print('Parsing input arguments!')
    prokbert_config, args = prepare_input_arguments()
    main(prokbert_config, args)
