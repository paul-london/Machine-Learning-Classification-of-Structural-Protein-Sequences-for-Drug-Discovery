# 🧬 Machine Learning Classification of Structural Protein Sequences for Drug Discovery

*Authors:*  
**Paul London, M.S., Ernest Bonat, Ph.D.**

---

## 📘 Overview

This repository accompanies the project **“Machine Learning Classification of Structural Protein Sequences for Drug Discovery.”**  
The project investigates how modern machine learning (ML) and language-modeling techniques can classify proteins based solely on their amino acid sequences.

By treating protein sequences as *biological text*, we apply natural language processing (NLP), recurrent neural networks (LSTMs), and pre-trained large language models (LLMs, e.g., ESM-2) to analyze patterns within primary structure and predict each protein’s functional class.

The project emphasizes:

- Lightweight, GPU-friendly workflows  
- Methods balancing computational efficiency with biological relevance  
- Comparisons between NLP, LSTM, and LLM-based approaches  
- Exploratory data analysis (EDA) of protein sequences from the Protein Data Bank (PDB)

---

## 📂 Repository Structure

- Notebook.ipynb
- .gitignore
- README.md
- requirements.txt

---

## 📦 Dataset

Data originate from:

- **Kaggle – Protein Data Set**  
  (originally sourced from the RCSB Protein Data Bank)

Two files are required:

- `metadata.csv` — (originally `pdb_data_no_dups.csv`) experimental and structural metadata  
- `sequences.csv` (originally `pdb_data_seq.csv`) — amino acid sequences for all chains  

### 🔍 Key preprocessing steps

1. Merge metadata and sequences on `structureId`  
2. Filter for **proteins only**  
3. Remove missing sequences/labels  
4. Clean sequences to the **20 standard amino acids**  
5. Keep sequence lengths **20–1024 residues**  
6. Concatenate multi-chain sequences  
7. Select the **top 20 protein classes**  
8. Train/validation/test split (70/15/15)

---

## 🚀 Project Pipelines

### **🧵 Pipeline 1 — NLP + Tree-Based Models**

- 3-mer tokenization  
- CountVectorizer  
- SMOTE for imbalance  
- Dimensionality reduction (SVD)  
- LazyClassifier baseline  
- Optuna hyperparameter tuning  

**Best model:** Tuned Random Forest  
**Accuracy:** ~69.7%

---

### **🔁 Pipeline 2 — LSTM Sequence Model (Keras)**

- Tokenized + padded sequences  
- Bidirectional LSTM  
- Class weighting  
- Trained 50–100 epochs  

**Accuracy:** ~78.2%

---

### **🧠 Pipeline 3 — LLM Embeddings (ESM-2)**

- Pre-trained model: `esm2_t6_8M_UR50D`  
- Extract embeddings per sequence  
- Train shallow neural nets & tree models  
- UMAP visualization of protein structure space  

**Status:** Embedding-based model comparison in progress

---

## 📊 Exploratory Data Analysis

Includes:

- Sequence length distributions  
- Protein class counts  
- Amino acid composition & grouping (WordCloud) 
- PCA and UMAP  
- Secondary-structure motif logos (Logomaker)

## 📊 Example Visualization

![](umap.jpg)

*UMAP projection of ESM-2 embeddings showing strong overall clustering, but limited separation between individual protein classes.*

---

## 🛠️ Installation

```bash
git clone https://github.com/paul-london/Machine-Learning-Based-Classification-of-Structural-Protein-Sequences-for-Drug-Discovery.git
cd Machine-Learning-Based-Classification-of-Structural-Protein-Sequences-for-Drug-Discovery
pip install -r requirements.txt
```

---

## 📁 Data Download

Download the dataset from Kaggle:

[Protein Data Set](https://www.kaggle.com/datasets/shahir/protein-data-set)

After downloading, place the files in the main project directory.

---

## ▶️ Usage

Run the main notebook:

```bash
jupyter notebook notebooks/protein_classification.ipynb
```

---

## 🔬 Results Summary
| Pipeline | Method               | Test Accuracy (%) | Notes                                    |
| :--------: | :--------------------: | :-------------------: | ---------------------------------------- |
| **1**    | NLP + Random Forest  | 69.7              | Lightweight baseline using 3-mers & SVD  |
| **2**    | Bidirectional LSTM   | 78.2              | Best sequence-based model so far         |
| **3**    | ESM-2 LLM Embeddings | In progress         | Strongest clustering; evaluation ongoing |

---

## 📘 Discussion

This project highlights how protein sequences can be modeled using modern machine learning techniques, including NLP tokenization, deep learning architectures, and protein language model embeddings.

- **NLP + tree-based models** provide fast, lightweight baselines and capture local sequence patterns.
- **Bidirectional LSTMs** leverage long-range dependencies within primary structure and deliver strong performance with modest compute.
- **ESM-2 protein LLM embeddings** offer the richest biological representations and show the strongest potential for accurate, structure-aware classification.

---

## 🔮 Future Work

- Full evaluation and benchmarking of ESM-based classifiers  
- Fine-tuning transformer-based protein models  
- Integrating secondary-structure or 3D-derived features  
- Applying generative modeling approaches for drug discovery  

---

## 📝 Article Citation

**Machine Learning Classification of Structural Protein Sequences for Drug Discovery**  
Paul London, M.S. & Ernest Bonat, Ph.D. (2025)

---

## 📧 Contact

**Paul London**  
Bioinformatics & Data Scientist  
[Email](palondon@hotmail.com)
[LinkedIn](https://www.linkedin.com/in/palondon)

