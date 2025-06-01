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
from prokbert.sequtils import *
from prokbert.config_utils import ProkBERTConfig, get_user_provided_args
from prokbert.training_utils import *
from prokbert.models import ProkBertForSequenceClassification
from prokbert.tokenizer import LCATokenizer

from transformers import AutoModelForSequenceClassification

# Get the local rank for distributed training (if any)
local_rank = int(os.getenv('LOCAL_RANK', '0'))
print('Local rank:', local_rank)

default_segmentation_length = 512


def preprocess_function(sample, tokenizer, max_length=200):
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
    results["labels"] = sample["y"]
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

    return parameters, args

def main(prokbert_config, args):
    """
    Main function to perform inference using ProkBERT.

    Args:
        prokbert_config (ProkBERTConfig): Configuration object for ProkBERT inference.
        args (Namespace): Parsed command-line arguments.
    """
    print('ProkBERT PhaSTYLE prediction!')

    # Loading the model


    model_path = parameters['finetuning']['ftmodel']
    print(model_path)
    model=ProkBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = LCATokenizer.from_pretrained(model_path)

    # Loading the data
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
    #print(sequences)

    print('Running segmentation')
    segmentation_params = {
        "max_length": default_segmentation_length,
        "min_length": int(default_segmentation_length*0.5),
        "type": "contiguous",
    }

    segmentdb = segment_sequences(sequences, segmentation_params, AsDataFrame=True)
    print('Finished')

    dataset = dataset_original.map(
            lambda sample: preprocess_function(sample, tokenizer, max_token_length),
            batched=True,
            num_proc=cpu_cores,
            remove_columns=cols_to_remove,
        )
    


    print(segmentdb)    






    print(parameters)





if __name__ == "__main__":
    print('Parsing input arguments!')
    parameters, args = prepare_input_arguments()
    main(parameters, args)
