## Description

ProkBERT PhaStyle is a BERT-based genomic language model fine-tuned for phage lifestyle prediction. It classifies phages as either **virulent** or **temperate** directly from nucleotide sequences, providing a fast, efficient, and accurate alternative to traditional database-based approaches.

This model is particularly useful in scenarios involving fragmented sequences from metagenomic and metavirome studies, eliminating the need for complex preprocessing pipelines or manually curated databases. 

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

Before installing ProkBERT PhaStyle, ensure that the [ProkBERT package](https://github.com/nbrg-ppcu/prokbert) is installed. We highly recommend setting up a virtual environment to isolate dependencies.

### Prerequisites

- Python 3.10 (recommended for optimal PyTorch performance)
- **ProkBERT** package (install instructions below)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/nbrg-ppcu/prokbert-phastyle.git
    cd prokbert-phastyle
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the ProkBERT package:

    Follow the instructions to install [ProkBERT](https://github.com/nbrg-ppcu/prokbert). After that, install the required dependencies:

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

## Model Description

### How ProkBERT PhaStyle Works

![ProkBERT PhaStyle Workflow](https://github.com/nbrg-ppcu/PhaStyle/blob/main/assets/figure_02.jpg)

ProkBERT PhaStyle is a genomic language model fine-tuned to predict phage lifestyles—specifically, whether a phage is **virulent** or **temperate**—directly from nucleotide sequences.

Here's a quick rundown of how it works:

1. **Segmentation**: We start with phage genomic sequences (contigs). Since these sequences can be quite long, we break each contig into smaller pieces called segments (e.g., S₁ becomes S₁₁, S₁₂, S₁₃). This makes processing more manageable and helps in handling fragmented sequences from metagenomic data.

2. **Tokenization**: Each segment is then converted into a series of k-mers using Local Context Attention (LCA) tokenization. Think of k-mers as overlapping chunks of k nucleotides that help the model grasp the sequence patterns.

3. **Encoding with ProkBERT**: The tokenized segments are fed into the ProkBERT encoder. ProkBERT is a transformer-based model pretrained on a vast collection of prokaryotic genomes. It generates contextual embeddings for each token, capturing intricate patterns in the genomic data.

4. **Classification**: A classification head (a simple neural layer added on top) processes the embeddings to predict the probability of each segment being virulent (P_vir) or temperate (P_tem).

5. **Aggregation**: To determine the lifestyle of the entire contig, we aggregate the predictions from all its segments. This is usually done by averaging the probabilities or using a weighted voting scheme.

6. **Final Prediction**: The aggregated probabilities give us a final verdict on whether the phage is virulent or temperate.

### Why It Matters

ProkBERT PhaStyle can efficiently and accurately predict phage lifestyles without the need for complex bioinformatics pipelines or extensive manual annotations. This is especially handy when dealing with fragmented sequences from metagenomic studies, where traditional methods might falter.


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



