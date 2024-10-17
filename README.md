# ProkBERT PhaStyle


## Description

ProkBERT PhaStyle is a BERT-based genomic language model fine-tuned for phage lifestyle prediction. It classifies phages as either **virulent** or **temperate** directly from nucleotide sequences, providing a fast, efficient, and accurate alternative to traditional database-based approaches.

This model is particularly useful in scenarios involving fragmented sequences from metagenomic and metavirome studies, eliminating the need for complex preprocessing pipelines or manually curated databases.

## Table of Contents

1. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
2. [Usage](#usage)
   - [Quick Start](#quick-start)
3. [Model Description](#model-description)
   - [How ProkBERT PhaStyle Works](#how-prokbert-phastyle-works)
   - [Why It Matters](#why-it-matters)
4. [Results](#results)
   - [Performance Comparison](#performance-comparison)
     - [Evaluation Metrics](#evaluation-metrics)
     - [Results on *Escherichia* Test Set (512 bp segments)](#results-on-escherichia-test-set-512-bp-segments)
     - [Results on *Escherichia* Test Set (1022 bp segments)](#results-on-escherichia-test-set-1022-bp-segments)
     - [Results on EXTREMOPHILE Test Set (512 bp segments)](#results-on-extremophile-test-set-512-bp-segments)
     - [Results on EXTREMOPHILE Test Set (1022 bp segments)](#results-on-extremophile-test-set-1022-bp-segments)
     - [Summary](#summary)
   - [Inference Speed and Running Times](#inference-speed-and-running-times)
5. [Available Models and Datasets](#available-models-and-datasets)
   - [Fine-tuned Models for Phage Lifestyle Prediction](#fine-tuned-models-for-phage-lifestyle-prediction)
6. [Datasets](#datasets)
   - [Available Datasets](#available-datasets)
   - [Summary Table](#summary-table)
   - [Dataset Details](#dataset-details)
   - [How to Access the Datasets](#how-to-access-the-datasets)
7. [Project Structure](#project-structure)
8. [License](#license)
9. [Citing this Work](#citing-this-work)

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

### Explanation of Parameters:
- `--fastain`: Specifies the path to the input FASTA file containing the sequences you want to classify. In this example, it is set to `data/EXTREMOPHILE/extremophiles.fasta`, which is the input dataset.

- `--out`: Defines the output file where the inference results will be saved. Here, it is set to `output_predictions.tsv`, meaning the results will be written in a tab-separated values format.

- `--ftmodel`: Defines the fine-tuned model to be used for inference. In this case, the model `neuralbioinfo/PhaStyle-mini` is used, which is a pre-trained version of PhaStyle.

- `--modelclass`: Specifies the model class that implements the neural network structure for the analysis. Here, `BertForBinaryClassificationWithPooling` is used, which is suited for binary classification tasks with an added pooling layer for feature aggregation.

- `--per_device_eval_batch_size`: Sets the number of samples processed per device (GPU/CPU) during evaluation. A batch size of `196` is used in this example for efficient processing.



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

## Results

### Performance Comparison

We evaluated **ProkBERT PhaStyle** against other state-of-the-art phage lifestyle prediction methods, including DNABERT-2, Nucleotide Transformer (NT), DeePhage, and PhaTYP. The evaluation was performed on two datasets:

- **Escherichia Phages**: A collection of 96 taxonomically diverse *Escherichia* bacteriophages.
- **EXTREMOPHILE Phages**: Phages isolated from extreme environments such as deep-sea trenches, acidic, and arsenic-rich habitats.

#### Evaluation Metrics

We used standard binary classification metrics:

- **Balanced Accuracy (Bal. Acc.)**
- **Matthews Correlation Coefficient (MCC)**
- **Sensitivity (Sens.)**
- **Specificity (Spec.)**

### Results on *Escherichia* Test Set (512 bp segments)

| Method                     | Bal. Acc. |   MCC  | Sens. | Spec. |
|----------------------------|-----------|--------|-------|-------|
| **ProkBERT-mini**          | **0.91**  | **0.83** | 0.94  | **0.89** |
| ProkBERT-mini-long         | 0.90      | 0.82   | **0.96** | 0.85  |
| ProkBERT-mini-c            | 0.89      | 0.80   | 0.95  | 0.84  |
| DNABERT-2-117M             | 0.84      | 0.72   | 0.95  | 0.74  |
| Nucleotide Transformer-50M | 0.85      | 0.72   | 0.92  | 0.78  |
| Nucleotide Transformer-100M| 0.87      | 0.75   | 0.93  | 0.82  |
| Nucleotide Transformer-500M| 0.88      | 0.78   | **0.96** | 0.80  |
| DeePhage                   | 0.86      | 0.71   | 0.84  | 0.88  |
| PhaTYP                     | **0.91**  | **0.83** | 0.94  | 0.88  |

### Results on *Escherichia* Test Set (1022 bp segments)

| Method                     | Bal. Acc. |   MCC  | Sens. | Spec. |
|----------------------------|-----------|--------|-------|-------|
| **ProkBERT-mini**          | **0.94**  | 0.88   | **0.97** | **0.91** |
| **ProkBERT-mini-long**     | **0.94**  | **0.89** | **0.97** | **0.91** |
| ProkBERT-mini-c            | 0.93      | 0.87   | **0.97** | 0.89  |
| DNABERT-2-117M             | 0.90      | 0.80   | 0.95  | 0.85  |
| Nucleotide Transformer-50M | 0.90      | 0.80   | 0.94  | 0.85  |
| Nucleotide Transformer-100M| 0.92      | 0.83   | 0.94  | 0.89  |
| Nucleotide Transformer-500M| 0.91      | 0.84   | 0.96  | 0.87  |
| DeePhage                   | 0.91      | 0.82   | 0.94  | 0.88  |
| PhaTYP                     | 0.92      | 0.84   | 0.96  | 0.87  |

### Results on EXTREMOPHILE Test Set (512 bp segments)

| Method                     | Bal. Acc. |   MCC  | Sens. | Spec. |
|----------------------------|-----------|--------|-------|-------|
| **ProkBERT-mini**          | **0.93**  | **0.83** | 0.99  | 0.87  |
| **ProkBERT-mini-long**     | **0.93**  | 0.82   | **1.00** | 0.86  |
| ProkBERT-mini-c            | 0.92      | 0.80   | 0.99  | 0.84  |
| DNABERT-2-117M             | 0.89      | 0.74   | 0.99  | 0.79  |
| Nucleotide Transformer-50M | 0.91      | 0.79   | 0.98  | 0.84  |
| Nucleotide Transformer-100M| 0.90      | 0.76   | 0.97  | 0.82  |
| Nucleotide Transformer-500M| 0.91      | 0.78   | 0.99  | 0.82  |
| DeePhage                   | 0.87      | 0.75   | 0.84  | **0.91** |
| PhaTYP                     | 0.76      | 0.52   | 0.74  | 0.79  |

### Results on EXTREMOPHILE Test Set (1022 bp segments)

| Method                     | Bal. Acc. |   MCC  | Sens. | Spec. |
|----------------------------|-----------|--------|-------|-------|
| **ProkBERT-mini**          | **0.96**  | **0.91** | **1.00** | **0.93** |
| ProkBERT-mini-long         | **0.96**  | 0.90   | **1.00** | 0.92  |
| ProkBERT-mini-c            | 0.94      | 0.86   | **1.00** | 0.89  |
| DNABERT-2-117M             | 0.94      | 0.85   | 0.98  | 0.90  |
| Nucleotide Transformer-50M | 0.93      | 0.83   | 0.99  | 0.87  |
| Nucleotide Transformer-100M| 0.95      | 0.88   | 0.98  | 0.91  |
| Nucleotide Transformer-500M| 0.96      | 0.89   | **1.00** | 0.91  |
| DeePhage                   | 0.92      | 0.80   | 0.96  | 0.87  |
| PhaTYP                     | 0.80      | 0.58   | 0.84  | 0.76  |

### Summary

- **ProkBERT PhaStyle** consistently outperforms other models, especially on shorter sequence fragments (512 bp), which are common in metagenomic datasets.
- ProkBERT models demonstrate excellent generalization capabilities, performing well even on phages from extreme environments not represented in the training data.
- Despite having fewer parameters (~25 million) compared to larger models like DNABERT-2 and Nucleotide Transformer, ProkBERT achieves superior performance.
- The model is efficient and suitable for large-scale applications, offering faster inference times compared to other methods.

### Inference Speed and Running Times

We also evaluated the computational efficiency of ProkBERT PhaStyle compared to other models. The evaluation was performed on a consistent hardware setup using NVIDIA Tesla A100 GPUs.

| Model                      | Execution Time (seconds) | Inference Speed (MB/sec) |
|----------------------------|--------------------------|--------------------------|
| **ProkBERT-mini-long**     | **132**                  | **0.52**                 |
| ProkBERT-mini              | 141                      | 0.49                     |
| ProkBERT-mini-c            | 146                      | 0.47                     |
| DNABERT-2-117M             | 284                      | 0.23                     |
| Nucleotide Transformer-50M | 292                      | 0.21                     |
| Nucleotide Transformer-100M| 313                      | 0.20                     |
| Nucleotide Transformer-500M| 500                      | 0.15                     |
| DeePhage                   | 159                      | 0.43                     |
| PhaTYP                     | 2,718                    | 0.10                     |
| BACPHLIP                   | 7,125                    | 0.04                     |

**Key Takeaways:**

- **ProkBERT-mini-long** is the fastest model, making it ideal for large-scale analyses.
- ProkBERT models are significantly faster than database search-based methods like PhaTYP and BACPHLIP.
- Even with GPU support, larger models like DNABERT-2 and Nucleotide Transformer are slower due to their size.

---

By leveraging ProkBERT PhaStyle, you can achieve high accuracy in phage lifestyle prediction with improved computational efficiency, making it a valuable tool for both research and clinical applications.


## Available models and datasets
### Finetuned models for phage life style prediction

| Model Name | k-mer | Shift | Hugging Face URL |
| --- | --- | --- | --- |
| `neuralbioinfo/prokbert-mini-phage` | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-phage) |
| `neuralbioinfo/prokbert-mini-long-phage` | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long-phage) |
| `neuralbioinfo/prokbert-mini-c-phage` | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c-phage) |

## Datasets

The ProkBERT PhaStyle model was trained and evaluated using several carefully curated datasets. These datasets consist of phage sequences labeled with their lifestyles (virulent or temperate) and are segmented to simulate real-world scenarios where sequences may be fragmented. Below is a summary of the available datasets, including descriptions and links to their corresponding Hugging Face repositories.
The structure of the dataset is explained visually in the following figure:

![ProkBERT PhaStyle Workflow](https://github.com/nbrg-ppcu/PhaStyle/blob/main/assets/figure_01.jpg)
*Figure 2: The dataset used in the ProkBERT PhaStyle study. Phage sequences from multiple independent data sources were segmented into 512bp and 1022bp fragments for training and testing models on phage lifestyle prediction. The dataset consists of the BACPHLIP training and validation sets, Escherichia phages (from the Guelin collection), and phages from extreme environments.*




### Available Datasets

1. **PhaStyle-SequenceDB**
   - **Description**: A comprehensive dataset containing phage sequences from multiple sources, including the BACPHLIP training and validation sets, *Escherichia* phages from the Guelin collection, and phages from extreme environments.
   - **Usage**: Used for training and evaluating the ProkBERT PhaStyle model on phage lifestyle prediction tasks.
   - **Hugging Face Link**: [PhaStyle-SequenceDB](https://huggingface.co/datasets/neuralbioinfo/PhaStyle-SequenceDB)

2. **PhageStyle-BACPHLIP**
   - **Description**: Consists of phage sequences from the BACPHLIP dataset, segmented into 512bp and 1022bp fragments. The training set excludes *Escherichia coli* sequences to test the model's generalization capabilities.
   - **Usage**: Used for training the ProkBERT PhaStyle model, ensuring it can generalize to phages not seen during training.
   - **Hugging Face Link**: [PhageStyle-BACPHLIP](https://huggingface.co/datasets/neuralbioinfo/PhageStyle-BACPHLIP)

3. **PhaStyle-EXTREMOPHILE**
   - **Description**: A test dataset containing phage sequences isolated from extreme environments such as deep-sea trenches, acidic, and arsenic-rich habitats. These sequences are segmented into 512bp and 1022bp fragments.
   - **Usage**: Used to evaluate the model's performance and generalization on phages from underrepresented environments.
   - **Hugging Face Link**: [PhaStyle-EXTREMOPHILE](https://huggingface.co/datasets/neuralbioinfo/PhaStyle-EXTREMOPHILE)

4. **PhaStyle-ESCHERICHIA**
   - **Description**: Contains *Escherichia* phage sequences, including the Guelin collection and additional high-quality temperate phages. The sequences are segmented to simulate fragmented assemblies.
   - **Usage**: Used to test the model's ability to generalize to *Escherichia* phages, which were excluded from the training set.
   - **Hugging Face Link**: [PhaStyle-ESCHERICHIA](https://huggingface.co/datasets/neuralbioinfo/PhaStyle-ESCHERICHIA)

### Summary Table


| Dataset Name                | Description                                                                                                          | Hugging Face Link                                                                                       |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **PhaStyle-SequenceDB**     | Comprehensive dataset of phage sequences from multiple sources, segmented for training and evaluation.               | [PhaStyle-SequenceDB](https://huggingface.co/datasets/neuralbioinfo/PhaStyle-SequenceDB)                |
| **PhageStyle-BACPHLIP**     | BACPHLIP training and validation sets segmented into 512bp and 1022bp fragments, excluding *E. coli* sequences.      | [PhageStyle-BACPHLIP](https://huggingface.co/datasets/neuralbioinfo/PhageStyle-BACPHLIP)                |
| **PhaStyle-EXTREMOPHILE**   | Phage sequences from extreme environments, segmented to test model generalization on underrepresented phages.        | [PhaStyle-EXTREMOPHILE](https://huggingface.co/datasets/neuralbioinfo/PhaStyle-EXTREMOPHILE)            |
| **PhaStyle-ESCHERICHIA**    | *Escherichia* phage sequences segmented to evaluate model performance on this genus, not included in training data. | [PhaStyle-ESCHERICHIA](https://huggingface.co/datasets/neuralbioinfo/PhaStyle-ESCHERICHIA)              |

### Dataset Details

#### 1. PhaStyle-SequenceDB

- **Structure**: Includes sequences from the BACPHLIP dataset, *Escherichia* phages, and extremophile phages.
- **Segment Lengths**: Sequences are segmented into 512bp and 1022bp fragments.
- **Purpose**: Provides a diverse dataset for training and evaluating phage lifestyle prediction models.

#### 2. PhageStyle-BACPHLIP

- **Training Set**: Contains 1,868 phage sequences (excluding *E. coli*), segmented into 512bp and 1022bp fragments.
- **Validation Set**: Includes 246 *E. coli* phage sequences, used to validate model generalization.

#### 3. PhaStyle-EXTREMOPHILE

- **Phage Sources**: Phages from deep-sea environments, acidic habitats, and arsenic-rich microbial mats.
- **Segment Lengths**: 512bp and 1022bp segments.
- **Challenge**: Tests model performance on phages not represented in standard training datasets.

#### 4. PhaStyle-ESCHERICHIA

- **Composition**: 394 *Escherichia* phages, including the Guelin collection and high-quality temperate phages.
- **Segmentation**: Sequences are broken into fragments to simulate assembly artifacts.
- **Evaluation**: Assesses the model's ability to accurately predict lifestyles of *Escherichia* phages.

### How to Access the Datasets

The datasets are available on Hugging Face and can be accessed using the `datasets` library:

```python
from datasets import load_dataset

# Load PhaStyle-SequenceDB
sequence_db = load_dataset("neuralbioinfo/PhaStyle-SequenceDB")

# Load PhageStyle-BACPHLIP
bacphlip_db = load_dataset("neuralbioinfo/PhageStyle-BACPHLIP")

# Load PhaStyle-EXTREMOPHILE
extremophile_db = load_dataset("neuralbioinfo/PhaStyle-EXTREMOPHILE")

# Load PhaStyle-ESCHERICHIA
escherichia_db = load_dataset("neuralbioinfo/PhaStyle-ESCHERICHIA")
```

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

## Contact

For questions, feedback, or collaboration opportunities, please contact:

- **Balázs Ligeti** (Corresponding Author)
  - Email: [obalasz@gmail.com](mailto:obalasz@gmail.com)
  - ORCID: [0000-0003-0301-0434](https://orcid.org/0000-0003-0301-0434)
  - 

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



