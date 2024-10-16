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
4. [Acknowledgements](#acknowledgements)
5. [References](#references)

## Installation
The only prerequestics is the ProkBERT. We reccomend to use python 3.10 for fully get advantage of the features of pytorch (model compilation).

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
    Followed the instruction of installing ProkBERT.     

    ```bash
    pip install prokbert
    ```


## Usage

### Quick Start

To perform phage lifestyle prediction on a FASTA file using a fine-tuned ProkBERT model, run the following command:

```bash
python bin/PhaStyle.py \
    --fastain data/EXTREMOPHILE/extremophiles.fasta \
    --out output_predictions.tsv \
    --ftmodel neuralbioinfo/PhaStyle-mini \
    --modelclass BertForBinaryClassificationWithPooling \
    --per_device_eval_batch_size 196
```
The script supports distributed GPU inference (tested with the NVCC framework). For an example command, refer to the `bin/run_PhaStyle.sh` script. For large-scale inference tasks, consider using the `torch.compile` option for performance optimization.

The recommended fine-tuned model is `neuralbioinfo/PhaStyle-mini`. For detailed arguments related to tokenization and segmentation, please consult the ProkBERT documentation or the following example notebooks:
- [Tokenization Notebook](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Tokenization.ipynb)
- [Segmentation Notebook](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Segmentation.ipynb)

Both notebooks provide illustrative examples with nice figures and tables. Additionally, common parameters for Hugging Face's `TrainingArguments` can be customized and passed as necessary. For more details, see the [Hugging Face documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).


## Available models and datasets
### Finetuned models for phage life style prediction

| Model Name | k-mer | Shift | Hugging Face URL |
| --- | --- | --- | --- |
| `neuralbioinfo/prokbert-mini-phage` | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-phage) |
| `neuralbioinfo/prokbert-mini-long-phage` | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long-phage) |
| `neuralbioinfo/prokbert-mini-c-phage` | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c-phage) |



### Datasets

| Dataset Name | Hugging Face URL |
| --- | --- |
| `neuralbioinfo/ESKAPE-genomic-features` | [Link](https://huggingface.co/datasets/neuralbioinfo/ESKAPE-genomic-features) |
| `neuralbioinfo/phage-test-10k` | [Link](https://huggingface.co/datasets/neuralbioinfo/phage-test-10k) |
| `neuralbioinfo/bacterial_promoters` | [Link](https://huggingface.co/datasets/neuralbioinfo/bacterial_promoters) |
| `neuralbioinfo/ESKAPE-masking` | [Link](https://huggingface.co/datasets/neuralbioinfo/ESKAPE-masking) |

## Project Structure

The repository is organized as follows:
```
./
├── assets/                     # Contains figures and other resources used in documentation and presentations
│   ├── figure_01.jpg
│   ├── figure_02_method.png
│   ├── figure_03.jpg
├── bin/                        # Contains executable scripts for running the PhaStyle predictions
│   ├── PhaStyle.py
│   ├── PhaStyle_example.sh
├── containers/                 # Container-related files, including Docker and Singularity definitions
├── data/                       # Datasets used for model training and testing, organized by phage type
│   ├── ESCHERICHIA/
│   ├── EXTREMOPHILE/
├── examples/                   # Example scripts and usage notebooks
├── LICENSE                     # License information for the repository
├── README.md                   # The main README file
```

- **`bin/`**: Includes the executable scripts for phage lifestyle prediction, such as `PhaStyle.py` and example shell scripts.
- **`containers/`**: Stores container files for easy setup using Docker or Singularity.
- **`data/`**: Contains various datasets used for phage lifestyle predictions, organized by type (e.g., Escherichia, Extremophile).
- **`examples/`**: Contains example code or usage notebooks to help users get started with the project.


# Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{ProkBERT2024,
  author  = {Ligeti, Balázs and Szepesi-Nagy, István and Bodnár, Babett and Ligeti-Nagy, Noémi and Juhász, János},
  journal = {Frontiers in Microbiology},
  title   = {{ProkBERT} family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14},
  URL={https://www.frontiersin.org/articles/10.3389/fmicb.2023.1331233},       
	DOI={10.3389/fmicb.2023.1331233},      
	ISSN={1664-302X}
}
```



