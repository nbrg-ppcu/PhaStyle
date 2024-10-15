# ProkBERT PhaStyle


# ProkBERT PhaStyle
** UNDER DEVELOPMENT **
RELEASE after 12th June



## Description

ProkBERT PhaStyle is a tool designed for phage lifestyle prediction, classifying phages as either **virulent** or **temperate**. This repository contains the code and models for ProkBERT PhaStyle, utilizing genomic language models (GLMs) for efficient analysis directly from nucleotide sequences without the need for complex preprocessing pipelines or manually curated databases.

## Table of Contents

1. [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
2. [Usage](#usage)
    - [Quick Start](#quick-start)
    - [Command-Line Arguments](#command-line-arguments)
    - [Example](#example)
3. [Models](#models)
4. [Benchmarking](#benchmarking)
5. [Evaluation](#evaluation)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)
9. [References](#references)

## Installation

### Prerequisites
- **ProkBERT** package

### Installation Steps

1. **Clone the repository**

    ```bash
    git clone https://github.com/nbrg-ppcu/prokbert-phastyle.git
    cd prokbert-phastyle
    ```

2. **Set up a virtual environment (optional but recommended)**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the ProkBERT package**

    If ProkBERT is not a standalone package, include the `prokbert` directory in your `PYTHONPATH` or install it directly:

    ```bash
    pip install -e .
    ```

    or

    ```bash
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/prokbert"
    ```

## Usage

### Quick Start

To perform phage lifestyle prediction on a FASTA file using a fine-tuned ProkBERT model, run the following command:

```bash
python bin/PhaStyle.py \
    --fastain path/to/input_sequences.fasta \
    --out path/to/output_predictions.tsv \
    --ftmodel path/to/finetuned_model \
    --modelclass BertForBinaryClassificationWithPooling \
    --per_device_eval_batch_size 8
```
