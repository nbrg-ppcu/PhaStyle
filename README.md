# ProkBERT PhaStyle


## Description

ProkBERT PhaStyle is a genomic language model based solution for phage lifestyle prediction. It classifies phages as either **virulent** or **temperate** directly from nucleotide sequences, providing a fast, efficient, and accurate alternative to traditional database-based approaches.

## TLDR example
For start you can try the PhaStyle in google colab notebook: 
 - PhaStyle notebook: [colab link](https://colab.research.google.com/github/nbrg-ppcu/PhaStyle/blob/main/bin/PhaStyleExample.ipynb) 


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


### Prerequisites
We highly recommend setting up a virtual environment to isolate dependencies.

- Python 3.12 (recommended)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/nbrg-ppcu/PhaStyle/
    cd PhaStyle
    ```

2. Install the ProkBERT package:

    ```bash
    pip install git+https://github.com/nbrg-ppcu/prokbert.git
    pip install transformers datasets
    ```
  
### Containers
We provide docker containerized version of the PhaStyle:
```bash
docker pull obalasz/phastyle:latest
```


Building singularity container:
```bash
singularity pull phastyle.sif docker://obalasz/phastyle:latest
```





## Usage


### Colab notebook 

No instalation required. For start you can try the PhaStyle in google colab notebook: 
 - PhaStyle notebook: [colab link](https://colab.research.google.com/github/nbrg-ppcu/PhaStyle/blob/main/bin/PhaStyleExample.ipynb) 



### Quick Start

To perform phage lifestyle prediction on a FASTA file using a fine-tuned ProkBERT model, run the following command:

```bash
python bin/PhaStyle.py \
    --fastain data/EXTREMOPHILE/extremophiles.fasta \
    --out output_predictions.tsv \
    --ftmodel neuralbioinfo/PhaStyle-mini \
    --per_device_eval_batch_size 196
```

### Explanation of Parameters:
- `--fastain`: Specifies the path to the input FASTA file containing the sequences you want to classify. In this example, it is set to `data/EXTREMOPHILE/extremophiles.fasta`, which is the input dataset.

- `--out`: Defines the output file where the inference results will be saved. Here, it is set to `output_predictions.tsv`, meaning the results will be written in a tab-separated values format.

- `--ftmodel`: Defines the fine-tuned model to be used for inference. In this case, the model `neuralbioinfo/PhaStyle-mini` is used, which is a pre-trained version of PhaStyle.

- `--per_device_eval_batch_size`: Sets the number of samples processed per device (GPU/CPU) during evaluation. A batch size of `196` is used in this example for efficient processing.

For large-scale inference tasks, consider using the `torch.compile` option as well as using nvcc or accelerate for performance optimization.


### Using containers

Docker example with GPU support. Assuming a folder that contains the fasta file named 'input.fasta':


```bash
docker run --rm --gpus 1 \
  -v "$(pwd)/phastyle_data":/workspace \
  obalasz/phastyle:latest \
  python bin/PhaStyle.py \
    --fastain /workspace/input.fasta \
    --out /workspace/output_predictions.tsv \
    --ftmodel neuralbioinfo/PhaStyle-mini \
    --per_device_eval_batch_size 196
```

#### Singularity or apptainer:
To use ProkBERT PhaStyle in an Apptainer (formerly Singularity) container, follow these steps. This example assumes you have a local folder (`./phastyle_data`) containing your `input.fasta` file and that you want GPU support.

1. **Pull (or build) the Apptainer image**  
   You can pull the image directly from Docker Hub:
   ```bash
   apptainer pull phastyle.sif docker://obalasz/phastyle:latest
    ```

2. **Run the container with GPU support and bind your data folder**

Assuming you have a local folder (e.g., ./phastyle_data) with input.fasta,
run the container, bind the folder, and enable GPU support if available:


```bash
apptainer exec --nv \
  --bind "$(pwd)/phastyle_data":/workspace \
  phastyle.sif \
  python bin/PhaStyle.py \
    --fastain /workspace/input.fasta \
    --out /workspace/output_predictions.tsv \
    --ftmodel neuralbioinfo/PhaStyle-mini \
    --per_device_eval_batch_size 196
```

  


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
- **Specificity**
  

**Key Takeaways:**

- **ProkBERT-mini-long** is the fastest model, making it ideal for large-scale analyses.
- ProkBERT models are significantly faster than database search-based methods like PhaTYP and BACPHLIP.
- Even with GPU support, larger models like DNABERT-2 and Nucleotide Transformer are slower due to their size.

---


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



