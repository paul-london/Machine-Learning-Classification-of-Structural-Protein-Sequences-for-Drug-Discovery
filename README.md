🧬 Machine Learning Classification of Structural Protein Sequences for Drug Discovery

Authors:
Paul London, Ernest Bonat, Ph.D.

📘 Overview

This repository accompanies the project “Machine Learning Classification of Structural Protein Sequences for Drug Discovery.”
The project investigates how modern machine learning (ML) and language-modeling techniques can classify proteins based solely on their amino acid sequences.

By treating protein sequences as biological text, we apply natural language processing (NLP), recurrent neural networks (LSTMs), and pre-trained large language models (LLMs, e.g., ESM-2) to analyze patterns within primary structure and predict each protein’s functional class.

The project emphasizes:

Lightweight, GPU-friendly workflows

Methods that balance computational efficiency with biological relevance

Practical comparisons between NLP, LSTM, and LLM-based approaches

Exploratory data analysis (EDA) of protein sequences from the Protein Data Bank (PDB)

📂 Repository Structure
repo/
│
├── notebooks/
│   └── protein_classification.ipynb
│
├── data/                # (not included in repo; see instructions below)
│   ├── metadata.csv
│   └── sequences.csv
│
├── figures/             # Auto-generated EDA and model evaluation plots
│
├── models/              # Saved model weights / Optuna results (optional)
│
├── README.md
└── requirements.txt

📦 Dataset

Data originate from:

Kaggle – Protein Data Set
(originally sourced from the RCSB Protein Data Bank)

Two files are used:

metadata — experimental and structural metadata

sequences — amino acid sequences for all chains

🔍 Key preprocessing steps

Merge metadata and sequences on structureId

Filter for proteins only (exclude DNA/RNA)

Drop missing sequences or labels

Clean sequences to the 20 standard amino acids

Filter by length 20–1024 residues

Concatenate multi-chain sequences

Select the top 20 protein classes by representation

Split into train/validation/test sets (70/15/15)

🚀 Project Pipelines

Three modeling pipelines were developed to compare ML approaches:

🧵 Pipeline 1 — NLP + Tree-Based Models

Treats protein sequences as text:

3-mer tokenization

CountVectorizer

SMOTE for class imbalance

Dimensionality reduction (SVD)

Baseline model search with LazyClassifier

Hyperparameter tuning using Optuna

Best model: Tuned Random Forest
Accuracy: ~69.7%

Strengths: extremely lightweight, fast to train
Limitations: weak for long-range dependencies

🔁 Pipeline 2 — LSTM Sequence Model

A BiLSTM built in Keras/TensorFlow:

Tokenized and padded sequences

Bidirectional LSTM

Class weighting for imbalanced classes

Trained for 50–100 epochs

Best validation accuracy: ~78.2%

Strengths: captures long-range sequence patterns
Limitations: training speed bottleneck on CPU hardware

🧠 Pipeline 3 — LLM Embeddings (ESM-2)

Uses pre-trained transformer embeddings:

Model: esm2_t6_8M_UR50D (Meta AI)

Extracted high-dimensional embeddings per sequence

Trained tree-based and shallow neural classifiers on embeddings

UMAP visualization demonstrated biologically meaningful clusters

Strengths: highest biological fidelity with minimal training
Limitations: generating full embeddings is computationally expensive

📊 Exploratory Data Analysis

The notebook includes:

Sequence length distributions

Protein class frequency

Amino acid composition and biochemical grouping

PCA and UMAP visualizations

Secondary structure propensity logo plots (Logomaker)

These analyses highlight the complexity of functional classification and motivate the use of advanced sequence models.

🛠️ Installation

Python version: 3.10+ recommended

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

📁 Data Download

Due to size restrictions, data files are not stored in the repository.

Download from Kaggle:

https://www.kaggle.com/datasets/shahir/protein-data-set

Place the two CSVs in:

data/metadata.csv
data/sequences.csv

▶️ Usage

Run the Jupyter notebook:

jupyter notebook notebooks/protein_classification.ipynb


The notebook includes:

Data preprocessing

All three modeling pipelines

EDA visualizations

Hyperparameter tuning

Performance comparisons

🔬 Results Summary
Pipeline	Method	Validation Accuracy	Notes
1	NLP + Random Forest	~69.7%	Lightweight baseline
2	BiLSTM	~78.2%	Captures sequence relationships
3	ESM-2 Embeddings	TBD (in progress)	Best biological clustering
📘 Discussion

This project shows how treating protein sequences as language unlocks powerful machine-learning workflows. While classic NLP and LSTMs provide solid baselines, pre-trained LLM embeddings demonstrate the most promise, especially for complex biological classification tasks.

Future work will expand embedding generation, investigate fine-tuning transformer models, and explore generative approaches for intelligent drug design.

📝 Citation

If you use this repository in your work, please cite:

Machine Learning Classification of Structural Protein Sequences for Drug Discovery
Paul London & Ernest Bonat, Ph.D. (2025)

📧 Contact

For questions, collaboration, or dataset access issues:

Paul London
💼 Bioinformatics & Data Scientist
📫 [Your email or GitHub profile link]
