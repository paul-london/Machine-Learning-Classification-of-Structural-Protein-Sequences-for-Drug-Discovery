# [markdown]
#  Machine Learning-Based Classification of Structural Protein Sequences for Drug Discovery
#
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>1. Introduction</strong>
# </div>
#
# ## 1.1. Background
#
# Proteins are fundamental biomolecules that carry out essential functions within living organisms. Understanding the structure of a protein is critical for drug discovery, as the structural conformation often determines the protein’s function, interaction with other molecules, and its role in disease mechanisms. Traditional experimental methods to determine protein structure, such as X-ray crystallography and NMR spectroscopy, are time-consuming and expensive.
#
# Advances in computational biology and machine learning offer an efficient alternative. By leveraging sequence-based features, deep learning models such as Long Short-Term Memory (LSTM) networks and transformer-based large language models (LLMs) can predict protein structural classes directly from amino acid sequences. These approaches accelerate the identification of potential drug targets and facilitate the design of novel therapeutics.
#
# ## 1.2. Purpose
#
# ---
#
# **Classify protein function category based solely on amino acid sequence using supervised machine learning.**
#
# ---
#
# The purpose of this project is to develop a machine learning pipeline for classifying structural protein sequences into their respective structural classes, using sequence data alone. Specifically, this project aims to:
#
# 1. Preprocess protein sequences and convert them into machine-readable formats suitable for deep learning models.
#
# 2. Apply Natural Language Processing (NLP) techniques, including embeddings and sequence modeling, to capture meaningful patterns in protein sequences.
#
# 3. Compare the performance of different models using NLP, LSTM networks, and LLM, in structural protein classification.
#
# 4. Provide insights into how computational sequence analysis can accelerate drug discovery and facilitate the identification of novel therapeutic targets.
#
# ## 1.3. Dataset
#
# The protein sequence data used is publicly available at [Kaggle](https://www.kaggle.com/code/davidhjek/protein-sequence-classification). It was retrieved from Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB).
#
# - `pdb_data_no_dups.csv` contains protein metadata which includes details on protein classification, extraction methods, etc.
#
# | Column                     | Description                                                                                 |
# | -------------------------- | ------------------------------------------------------------------------------------------- |
# | `structureId`              | Unique identifier for each protein structure in the PDB (Protein Data Bank).                |
# | `classification`           | Structural class or category of the protein (e.g., enzyme, transporter).                    |
# | `experimentalTechnique`    | Method used to determine the protein structure (e.g., X-ray crystallography, NMR, cryo-EM). |
# | `macromoleculeType`        | Type of macromolecule (e.g., protein, DNA, RNA).                                            |
# | `residueCount`             | Number of amino acid residues in the protein chain.                                         |
# | `resolution`               | Resolution of the protein structure (Ångströms), relevant for X-ray crystallography.        |
# | `structureMolecularWeight` | Molecular weight of the protein structure (Daltons).                                        |
# | `crystallizationMethod`    | Method used for crystallizing the protein (if applicable).                                  |
# | `crystallizationTemp`      | Temperature used for protein crystallization (Kelvin or Celsius).                           |
# | `densityMatthews`          | Matthews coefficient, a measure of crystal packing density.                                 |
# | `densityPercentSol`        | Estimated solvent content (%) in the crystal.                                               |
# | `pdbxDetails`              | Additional details about the structure (text description).                                  |
# | `phValue`                  | pH at which the protein structure was determined.                                           |
# | `publicationYear`          | Year the protein structure was published in the PDB.                                        |
#
#     
# - `pdb_data_seq.csv` contains >400,000 protein structure sequences.
#
# | Column              | Description                                                                   |
# | ------------------- | ----------------------------------------------------------------------------- |
# | `structureId`       | Unique identifier for the protein structure (matches `pdb_data_no_dups.csv`). |
# | `chainId`           | Identifier for the protein chain within the structure (A, B, C, etc.).        |
# | `sequence`          | Amino acid sequence of the chain (one-letter codes).                          |
# | `residueCount`      | Number of residues in this chain.                                             |
# | `macromoleculeType` | Type of macromolecule (e.g., protein, DNA, RNA).                              |
#
# ## 1.3. Amino Acid Code
#
# Protein sequences are expressed as a series of one letter abbreviations for each amino acid, given below:
#
# ### 1.3.1. Alphabetical
#
# | One-Letter Code | Amino Acid    |
# | :---------------: | ------------- |
# | A               | Alanine       |
# | R               | Arginine      |
# | N               | Asparagine    |
# | D               | Aspartic Acid |
# | C               | Cysteine      |
# | E               | Glutamic Acid |
# | Q               | Glutamine     |
# | G               | Glycine       |
# | H               | Histidine     |
# | I               | Isoleucine    |
# | L               | Leucine       |
# | K               | Lysine        |
# | M               | Methionine    |
# | F               | Phenylalanine |
# | P               | Proline       |
# | S               | Serine        |
# | T               | Threonine     |
# | W               | Tryptophan    |
# | Y               | Tyrosine      |
# | V               | Valine        |
#
# ### 1.3.2. By Properties
#
# #### Nonpolar, Aliphatic
# | One-letter Code| Amino Acid       | Notes                        |
# |:------------:|-----------------|-------------------------------|
# | A          | Alanine          | Small, hydrophobic            |
# | G          | Glycine          | Smallest residue, flexible    |
# | I          | Isoleucine       | Hydrophobic, aliphatic        |
# | L          | Leucine          | Hydrophobic, aliphatic        |
# | M          | Methionine       | Contains sulfur               |
# | P          | Proline          | Cyclic, rigid structure       |
# | V          | Valine           | Hydrophobic, aliphatic        |
#
# #### Aromatic
# | One-letter Code| Amino Acid       | Notes                        |
# |:------------:|-----------------|-------------------------------|
# | F          | Phenylalanine    | Nonpolar, aromatic            |
# | W          | Tryptophan       | Aromatic, slightly polar      |
# | Y          | Tyrosine         | Polar, aromatic               |
#
# #### Polar, Uncharged
# | One-letter Code| Amino Acid       | Notes                        |
# |:------------:|-----------------|-------------------------------|
# | C          | Cysteine         | Can form disulfide bonds      |
# | N          | Asparagine       | Polar, uncharged              |
# | Q          | Glutamine        | Polar, uncharged              |
# | S          | Serine           | Polar, uncharged              |
# | T          | Threonine        | Polar, uncharged              |
#
# #### Acidic (Negative)
# | One-letter Code| Amino Acid       | Notes                        |
# |:------------:|-----------------|-------------------------------|
# | D          | Aspartic Acid    | Acidic, negatively charged    |
# | E          | Glutamic Acid    | Acidic, negatively charged    |
#
# #### Basic (Positive)
# | One-letter Code| Amino Acid       | Notes                        |
# |:------------:|-----------------|-------------------------------|
# | K          | Lysine           | Basic, positively charged     |
# | R          | Arginine         | Basic, positively charged     |
# | H          | Histidine        | Basic, partially charged at physiological pH |
#
# #### Ambiguous / Wildcards
# | One-letter Code| Amino Acid / Meaning              | Notes                        |
# |:------------:|---------------------------------|-------------------------------|
# | B          | Aspartic Acid (D) / Asparagine (N) | Ambiguous                     |
# | Z          | Glutamic Acid (E) / Glutamine (Q)  | Ambiguous                     |
# | X          | Unknown amino acid                 | Wildcard                       |
# | J          | Leucine (L) / Isoleucine (I)      | Ambiguous                     |
# | U          | Selenocysteine                     | Rare non-standard amino acid  |
# | O          | Pyrrolysine                        | Rare non-standard amino acid  |
#

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>2. Import Libraries and Data</strong>
# </div>
#
# ## 2.1. Import Libraries

# Core
import os
import platform
import numpy as np
import pandas as pd
import pickle

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker

# Data preprocessing
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from lazypredict.Supervised import LazyClassifier

# Modeling
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD, PCA

# Tuning
import optuna
from tqdm import tqdm

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLP / Deep Learning (LSTM, embeddings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all TF logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Concatenate, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow_hub as hub
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# LLM / Transformers (protein language models)
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import sentencepiece
import esm
import torch
import h5py

# Bioinformatics
from Bio import SeqIO

# Notebook Utilities
from tqdm.notebook import tqdm
import ipywidgets as widgets


print("=== GENERAL ===")
print("TensorFlow version:", tf.__version__)
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

print("\n=== SYSTEM INFO ===")
print("Python:", platform.python_version(), platform.architecture())
print("OS:", platform.system(), platform.release())
print("TensorFlow version:", tf.__version__)

print("\n=== PHYSICAL DEVICES ===")
devices = tf.config.list_physical_devices()
print(devices)

print("\n=== CUDA PATH CHECK ===")
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
dlls = os.listdir(cuda_bin) if os.path.exists(cuda_bin) else []
print("cudart64 present:", any("cudart64" in f for f in dlls))
print("cudnn present:", any("cudnn" in f for f in dlls))
print("All DLLs found:", "Yes" if any("cudart64" in f for f in dlls) and any("cudnn" in f for f in dlls) else "No")

print("\n=== ENVIRONMENT VARIABLES ===")
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
print("PATH includes CUDA bin:", any(cuda_bin.lower() in p.lower() for p in os.environ.get("PATH", "").split(";")))
exit()

# [markdown]
# ## 2.2. Import Datasets

# Metadata
metadata = pd.read_csv("pdb_data_no_dups.csv")

# Sequences
sequences = pd.read_csv("pdb_data_seq.csv")

print("Metadata:")
display(metadata.head())
display(metadata.info())
display(metadata.shape)
print()
print("Sequences:")
display(sequences.head())
display(sequences.info())
display(sequences.shape)

# [markdown]
# ## 2.3. Analysis of Features
#
# ### 2.3.1. Metadata
#
# | Column                     | Use for modeling? | Notes                                                         |
# | -------------------------- | ----------------- | ------------------------------------------------------------- |
# | `structureId`              | ✅ Yes             | Key for merging sequences with labels                         |
# | `classification`           | ✅ Yes             | **Target variable** (structural class)                            |
# | `experimentalTechnique`    | ❌ No              | Optional metadata, not used for sequence-based ML             |
# | `macromoleculeType`        | ❌ No              | Could filter out non-proteins, but not a feature for LSTM/LLM |
# | `residueCount`             | ❌ No              | Sequence length is captured from sequences themselves         |
# | `resolution`               | ❌ No              | Metadata, not used in current model                           |
# | `structureMolecularWeight` | ❌ No              | Metadata                                                      |
# | `crystallizationMethod`    | ❌ No              | Metadata                                                      |
# | `crystallizationTempK`     | ❌ No              | Metadata                                                      |
# | `densityMatthews`          | ❌ No              | Metadata                                                      |
# | `densityPercentSol`        | ❌ No              | Metadata                                                      |
# | `pdbxDetails`              | ❌ No              | Metadata                                                      |
# | `phValue`                  | ❌ No              | Metadata                                                      |
# | `publicationYear`          | ❌ No              | Metadata                                                      |
#
# ### 2.3.2. Sequences
#
# | Column              | Use for modeling? | Notes                                                                   |
# | ------------------- | ----------------- | ----------------------------------------------------------------------- |
# | `structureId`       | ✅ Yes             | Merge key                                                               |
# | `chainId`           | ❌ Optional        | Could treat different chains separately; usually just keeps unique rows |
# | `sequence`          | ✅ Yes             | **Main feature** for NLP/LLM/LSTM                                       |
# | `residueCount`      | ❌ Optional        | Length can be derived from `sequence`                                   |
# | `macromoleculeType` | ❌ Optional        | Could filter out non-proteins, usually redundant                        |
#

# [markdown]
# ## 2.4. Merge, Filter, and Clean Up Datasets
#
# ### 2.4.1. Merge and Filter

# Merge sequences with metadata for complete dataset
data_all = pd.merge(sequences[['chainId', 'sequence', 'structureId']], metadata, on='structureId', how='inner', suffixes=('', '')) # Avoids redundant columns duplicating

# Keep only protein data as a new df
data = data_all[data_all['macromoleculeType'] == 'Protein']

# Drop missing label and sequences since these will be the feature (X) and target (y) of models later
data = data[data['classification'].notnull()]
data = data[data['sequence'].notnull()]

# Reset index
data.reset_index()

# Check
print(data.shape)
data.head()

# Save CSV at this step
data.to_csv("data.csv", index=False)

# [markdown]
# ### 2.4.2. Clean Up
#
# #### 2.4.2.1. Rename Columns

# Rename columns for readability and consistency
# All lowercase and separate words with underscores
data = data.rename(columns={
    'structureId': 'structure_id',
    'chainId': 'chain_id',
    'sequence': 'sequence',
    'residueCount': 'residue_count',
    'macromoleculeType': 'macromolecule_type',
    'classification': 'classification'
})

# Change all classifications to lowercase (target variable)
#data['classification'] = data['classification'].lower()

data.info()
data.head()

# [markdown]
# #### 2.4.2.2. Check for Missing Values
#
# We have already accounted for missing `sequence` (`X`) and `classification` (`y`) but we will explore others.

"""
# Heatmap of missing values
plt.figure(figsize=(12, len(data)/10))
sns.heatmap(data.isnull(), cbar=False, cmap="viridis", linewidths=0)  # True = missing, False = not missing
plt.title("Heatmap of Missing Values")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
"""

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>3. Exploratory Data Analysis - General</strong>
# </div>
#
# In this section, we will explore the feature (`sequence`) and target (`classification`) variables in the raw data.
#
# ## 3.1. Sequence
#
# ### 3.1.1. Length

# Add new column
data['sequence_length'] = data['sequence'].str.len()

# Plot
plt.figure(figsize=(8, 5))
sns.histplot(data['sequence_length'], bins=50, kde=True)
plt.title('Distribution of Protein Sequence Lengths')
plt.xlabel('Protein Sequence Length (Amino Acid Residues)')
plt.xticks(ticks=range(0, 1500, 100))
plt.ylabel('Count')
plt.show()

# Describe distribution
data['sequence_length'].describe()

# Create function
def sequence_lengths(df):
    df['sequence_length'] = df['sequence'].apply(len)
    df['sequence_length'].hist(bins=50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Sequences')
    plt.show()

# [markdown]
# ### 3.1.2. Properties
#
# #### 3.1.2.1. Amino Acid Composition

aa_counts = Counter("".join(data['sequence']))
aa_freq_pct = {aa: (count / sum(aa_counts.values())) * 100 for aa, count in aa_counts.items()}

# Convert to DataFrame for plotting
aa_df = pd.DataFrame(list(aa_freq_pct.items()), columns=['Amino Acid', 'Frequency']).sort_values('Frequency', ascending=False)

aa_df.plot(kind='bar', x='Amino Acid', y='Frequency', figsize=(12,5))
plt.xticks(rotation=0)
plt.ylabel('Frequency (%)')
plt.legend().set_visible(False)
plt.show()

# Create function
def aa_composition(df):
    aa_counts = Counter("".join(df['sequence']))
    aa_freq = {aa: count / sum(aa_counts.values()) * 100 for aa, count in aa_counts.items()}
    aa_df = pd.DataFrame(list(aa_freq.items()), columns=['Amino Acid', 'Frequency']).sort_values('Frequency', ascending=False)
    aa_df.plot(kind='bar', x='AminoAcid', y='Frequency', legend=False)
    plt.ylabel('Frequency (%)')
    plt.xticks(rotation=45)
    plt.show()

# [markdown]
# #### 3.1.2.2. Biochemical Properties

# Hydrophobic (nonpolar) residues
hydrophobic = set('AILMFWV')

# Polar uncharged residues
polar_uncharged = set('CSTNQ')

# Acidic (negatively charged)
acidic = set('DE')

# Basic (positively charged)
basic = set('KRH')

# Aromatic residues
aromatic = set('FYW')

# Compute fraction for each property
def aa(seq, aa_set):
    return sum(aa in aa_set for aa in seq) / len(seq)

# Apply to data
data['hydrophobic'] = data['sequence'].apply(lambda s: aa(s, hydrophobic))
data['polar'] = data['sequence'].apply(lambda s: aa(s, polar_uncharged))
data['acidic'] = data['sequence'].apply(lambda s: aa(s, acidic))
data['basic'] = data['sequence'].apply(lambda s: aa(s, basic))
data['aromatic'] = data['sequence'].apply(lambda s: aa(s, aromatic))

# Plot
properties = ['hydrophobic', 'polar', 'acidic', 'basic', 'aromatic']

plt.figure(figsize=(12,6))
for prop in properties:
    data[prop].hist(alpha=0.75, bins=30, label=prop)

plt.xlabel('Fraction')
plt.ylabel('Number of sequences')
plt.title('Distribution of Biochemical Properties Across Sequences')
plt.legend()
plt.show()

# Average property fraction per class
class_props = data.groupby('classification')[properties].mean()
class_props

# Create function
def biochemical_properties(df):
    hydrophobic = set('AILMFWV')
    polar_uncharged = set('CSTNQ')
    acidic = set('DE')
    basic = set('KRH')
    aromatic = set('FYW')

    df['hydrophobic'] = df['sequence'].apply(lambda s: sum(aa in hydrophobic for aa in s)/len(s))
    df['polar'] = df['sequence'].apply(lambda s: sum(aa in polar_uncharged for aa in s)/len(s))
    df['acidic'] = df['sequence'].apply(lambda s: sum(aa in acidic for aa in s)/len(s))
    df['basic'] = df['sequence'].apply(lambda s: sum(aa in basic for aa in s)/len(s))
    df['aromatic'] = df['sequence'].apply(lambda s: sum(aa in aromatic for aa in s)/len(s))
    
    properties = ['hydrophobic', 'polar', 'acidic', 'basic', 'aromatic']

    plt.figure(figsize=(12,6))
    for prop in properties:
        data[prop].hist(alpha=0.75, bins=30, label=prop)

    plt.xlabel('Fraction')
    plt.ylabel('Number of sequences')
    plt.title('Distribution of Biochemical Properties Across Sequences')
    plt.legend()
    plt.show()

    return df

# [markdown]
# #### 3.1.2.3. K-mer Counts

def get_kmers(seq, k=2):
    """Return list of all k-mers in a sequence."""
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

# Example: count dipeptides (k=2) across all sequences
k = 2
all_kmers = Counter()

for seq in data['sequence']:
    all_kmers.update(get_kmers(seq, k=k))

# Convert to DataFrame for easy plotting
kmer_df = pd.DataFrame(all_kmers.items(), columns=['k-mer', 'Count']).sort_values('Count', ascending=False)

# Plot
top_n = 20
plt.figure(figsize=(12,5))
plt.bar(kmer_df['k-mer'].head(top_n), kmer_df['Count'].head(top_n))
plt.xticks(rotation=45)
plt.xlabel(f'Top {k}-mers')
plt.ylabel('Count')
plt.title(f'Top {top_n} {k}-mers Across All Sequences')
plt.show()

# Frequency
# Create a dictionary of Counter objects per class
class_kmers = {cls: Counter() for cls in data['classification'].unique()}

for cls, group in data.groupby('classification'):
    for seq in group['sequence']:
        class_kmers[cls].update(get_kmers(seq, k=k))

# Convert to DataFrame (classes x k-mers)
class_kmer_df = pd.DataFrame.from_dict(class_kmers, orient='index').fillna(0)

# Heatmap
plt.figure(figsize=(15,10))
sns.heatmap(class_kmer_df.iloc[:, :50], cmap='viridis')  # first 50 k-mers for readability
plt.title(f'{k}-mer Frequency Heatmap per Class')
plt.show()


# [markdown]
# ## 3.2. Classification
#
# ### 3.2.1. Values

print(Counter(data['classification']))
print(len(Counter(data['classification'])))

# [markdown]
# There are too many classes here to reliably model (4468). I will:
# 1. Check for redundancy and combine equivalent classes, and
# 2. Group rare classes into an "Other" category.

# Normalize classification text by making all uppercase and stripping whitespace and punctuation
data['classification'] = (
    data['classification']
    .str.upper()
    .str.strip()
    .str.replace(r'[^A-Z0-9 /]', '', regex=True)
)

# Extract top 20 classes
#top_20_classes = data['classification'].value_counts().head(20).index.tolist()
top_50_classes = data['classification'].value_counts().head(50).index.tolist()

# Check classes 21-30 for safety
#classes_21_30 = data['classification'][20:30]
#print(f"21st - 30th classes: {classes_21_30}")

# Check classes 31-50 for safety
#classes_31_50 = data['classification'][30:50]
#print(f"31st - 50th classes: {classes_31_50}")

# Use Fuzzywuzzy to check for synonyms
synonyms = {}

for c in top_50_classes:
    matches = process.extract(c, top_50_classes, limit=10)
    # Keep matches with similarity >= 90 (adjust threshold if needed)
    synonyms[c] = [m[0] for m in matches if m[1] >= 90 and m[0] != c]

synonyms

# [markdown]
# Based on the top 50 classes, I will make a final set of no more than 50 (best for language learning models):
# 1. The top 50 classes, with any synonyms included.
# 3. An "OTHER" class to capture the remainder of any biologically relevant classes.
# 4. Classes must be represented by at least 200 sequences.

# [markdown]
# ### 3.2.2. Frequency

# Add 5, 10, 15




# Compute counts for top 20
counts = data['classification'].value_counts().sort_values(ascending=False).head(20)

# Plot descending barplot
plt.figure(figsize=(8, 12))
sns.barplot(x=counts.values, y=counts.index)
plt.xlabel("Count")
plt.ylabel("Classification")
plt.title("Counts of Protein Classes (Top 20)")
plt.show()

# Compute counts for top 50
counts = data['classification'].value_counts().sort_values(ascending=False).head(50)

# Plot descending barplot
plt.figure(figsize=(8, 15))
sns.barplot(x=counts.values, y=counts.index)
plt.xlabel("Count")
plt.ylabel("Classification")
plt.title("Counts of Protein Classes (Top 50)")
plt.yticks(fontsize=10)
plt.show()

# [markdown]
# We can see a large imbalance in the top 50 classes, which will need to be accounted for later.
#
# ### 3.2.3. Sequences vs. Classification
#
# The types of models being explored here do best with fewer than 50 classes and/or more than 200 sets of data (sequences in this case). Below, we can rank the classes by their nmber of representative sequences and decide on breakpoints.

# Count sequences per class
class_counts = data['classification'].value_counts().sort_values(ascending=False)

# Create rank index
ranks = range(1, len(class_counts)+1)

# Plot
plt.figure(figsize=(14,7))
plt.plot(ranks, class_counts.values, marker='o', markersize=4, linewidth=1)
plt.xlabel('Class rank (most to least sequences)', fontsize=12)
plt.ylabel('Number of sequences', fontsize=12)
plt.title('Class rank vs number of sequences (high resolution)', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Cutoff lines
# Sequences (> 200)
y_cutoff = 200
plt.axhline(y=y_cutoff, color='red', linestyle='--', label=f'Cutoff = > {y_cutoff}')
# Number of classes (< 50)
x_cutoff = 50
plt.axvline(x=x_cutoff, color='green', linestyle='--', label=f'Cutoff = < {x_cutoff}')
plt.legend()

# Optional: log scale to emphasize lower counts
plt.yscale('log')
plt.show()

# Filter classes (start with top 20)
#top_classes = class_counts.head(50).index.tolist()
top_classes = class_counts.head(20).index.tolist()
top_classes

# When finished with final classes
# top_classes = [...]  # list of your 50 chosen classes
data_filtered = data[data['classification'].isin(top_classes)].reset_index(drop=True) # Do EDA on these (move above sequences section)

# Save filtered csv
data_filtered.to_csv("data_filtered.csv")

# Check
print(data_filtered.shape)
data_filtered.head(10)

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>4. Data Preprocessing I - General</strong>
# </div>
#
# In this section, we will determine the final classes and make sure all sequences are appropriate for modeling.
#
# ## 4.1. Classification

# [markdown]
# ## 4.2. Sequences
#
# Here we will ensure all sequences are ready for modeling.

# Ensure sequences contain only the standard 20 amino acids
valid_aas = set('ACDEFGHIKLMNPQRSTVWY')  # standard 20 amino acids

def clean_sequence(seq):
    seq = seq.upper().strip()
    return ''.join([aa for aa in seq if aa in valid_aas])

# Maintain positions of invalid AAs using X placeholder - removing them would confuse model
def clean_sequence_keep_placeholder(seq):
    seq = seq.upper().strip()
    return ''.join([aa if aa in valid_aas else 'X' for aa in seq])

data_filtered['sequence'] = data_filtered['sequence'].apply(clean_sequence)

# We will also exclude very short sequences (< 20)
min_len = 20
data_filtered = data_filtered[data_filtered['sequence'].str.len() >= min_len].reset_index(drop=True)

# We will exclude very long sequences (> 700) at this step and match it to the tokenizer later
max_len = 1024
data_filtered = data_filtered[data_filtered['sequence'].str.len() <= max_len].reset_index(drop=True)

print(data.shape)

# We also need to combine sequences for proteins with multiple chains
df_combined_sequences = (
    data_filtered.groupby('structure_id')['sequence']
      .apply(lambda seqs: ''.join(seqs))
      .reset_index(name='combined_sequence')   # Put combined sequence into a new column
)

# Merge new column back into original data
data_filtered = data_filtered.merge(df_combined_sequences, on='structure_id', how='left')

# Remove duplicate structure IDs to account for combined sequences
data_filtered = data_filtered.drop_duplicates(subset='structure_id')

print(data_filtered.head())
print(data_filtered.shape)

# [markdown]
# ## 4.3. Define Variables for Modeling

X = data_filtered['combined_sequence']
y = data_filtered['classification']

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>5. Exploratory Data Analysis II - Specific</strong>
# </div>
#
# Here, the data analysis above will be revisited on the final selected classes only

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>6. Data Preprocessing II - Preparation for Modeling</strong>
# </div>

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>7. Model Training and Evaluation</strong>
# </div>
#
# ## 7.1. Overview of Modeling Approaches
#
# - NLP (k-mer) bag-of-words features with top model candidates from LazyClassifer
#
# - LSTM: Sequence-based and tuned neural network model
#
# - LLM: Using embeddings from a protein language model, then feeding to top NLP-fed model plus neural network model
#
# ## 7.2. Split Dataset (Training, Validation, Test)

def train_test_split_custom (X, y, size, state):
    """
    Generates training, validation, and test datasets given parameters:
    X: feature(s)
    y: target
    size: size (%) of training set
    state: random state value
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=1-size/100,
    stratify=y,
    random_state=state
    )

    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

    return X_train, X_val, X_test, y_train, y_val, y_test

size = 70   # Size (%) of training set
state = 42

# Run function
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_custom(X, y, size, state)

# Check shapes
print("X:")
print(X_train.shape, X_val.shape, X_test.shape)

print()
print("y:")
print(y_train.shape, y_val.shape, y_test.shape)

# [markdown]
# ## 7.3. Preprocessing `y`
#
# This processed `y` will be used for all models.

# Use LabelEncoder to encode y
le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

print(y_train_enc.shape, y_val_enc.shape, y_test_enc.shape)

# [markdown]
# ## 7.3. NLP
#
# ### 7.3.1. Preprocessing `X`
#
# The NLP approach with tree/gradient models requires creating *k*-mers for processing sequences (`X`).

# Split sequences into overlapping k-mers (k=3)
def kmer_seq(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

# Use list comprehension instead of .apply()
X_train_nlp = [' '.join(kmer_seq(seq, k=3)) for seq in X_train]
X_val_nlp   = [' '.join(kmer_seq(seq, k=3)) for seq in X_val]
X_test_nlp  = [' '.join(kmer_seq(seq, k=3)) for seq in X_test]

# Check first few examples
print(X_train_nlp[:5])

# Initialize vectorizer
vectorizer = CountVectorizer(max_features=5000)  # or TfidfVectorizer() ?

# Fit on training data and transform
X_train_nlp = vectorizer.fit_transform(X_train_nlp)

# Transform validation and test sets
X_val_nlp   = vectorizer.transform(X_val_nlp)
X_test_nlp  = vectorizer.transform(X_test_nlp)

# Check shapes
print(X_train_nlp.shape, X_val_nlp.shape, X_test_nlp.shape)

# [markdown]
# ### 7.3.2. Address Feature Imbalance
#
# SMOTE will be used to address feature imbalance for the NLP approach only.

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# NLP
print(X_train_nlp.shape)
print(Counter(y_train_enc))
X_train_nlp_res, y_train_enc_res = apply_smote(X_train_nlp, y_train_enc)
print(X_train_nlp_res.shape)
print(Counter(y_train_enc_res))

# [markdown]
# ### 7.3.3. Baseline Models with LazyClassifier
#
# LazyClassifier will be used to see which models perform the best at baseline with no tuning.

# Dimensionality reduction directly on sparse data (no toarray)
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_svd = svd.fit_transform(X_train_nlp_res)
X_test_svd  = svd.transform(X_test_nlp)

print(X_train_svd.shape)
print(X_test_svd.shape)

""" (Already run and results saved)
# Run LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_svd, X_test_svd, y_train_enc_res, y_test_enc)

print(models)
"""

# [markdown]
# ### 7.3.4. RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier 
#
# These were the top 3 models from LazyClassifer and will be investigated and tuned further.

# RandomForestClassifier
def objective_randomforest(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train_svd, y_train_enc_res, cv=cv, scoring='accuracy').mean()
    return score

# ExtraTreesClassifier
def objective_extratrees(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train_svd, y_train_enc_res, cv=cv, scoring='accuracy').mean()
    return score

# BaggingClassifier
def objective_bagging(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_samples = trial.suggest_float('max_samples', 0.3, 0.7)
    max_features = trial.suggest_float('max_features', 0.3, 0.7)
    
    model = BaggingClassifier(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train_svd, y_train_enc_res, cv=cv, scoring='accuracy').mean()
    return score

# Run Optimizations
print("\nOptimizing RandomForestClassifier...")
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_randomforest, n_trials=25, show_progress_bar=True)

print("Optimizing ExtraTreesClassifier...")
study_et = optuna.create_study(direction="maximize")
study_et.optimize(objective_extratrees, n_trials=25, show_progress_bar=True)

print("\nOptimizing BaggingClassifier...")
study_bg = optuna.create_study(direction="maximize")
study_bg.optimize(objective_bagging, n_trials=25, show_progress_bar=True)

# Compare Results
print("\nBest RandomForest params:", study_rf.best_params)
print("Best RandomForest accuracy:", study_rf.best_value)

print("\nBest ExtraTrees params:", study_et.best_params)
print("Best ExtraTrees accuracy:", study_et.best_value)

print("\nBest Bagging params:", study_bg.best_params)
print("Best Bagging accuracy:", study_bg.best_value)

# [markdown]
# ### 7.3.5. Model Evaluation

# Dictionary of best models from each study
best_models = {
    "RandomForest": RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1),
    "ExtraTreesClassifier": ExtraTreesClassifier(**study_et.best_params, random_state=42, n_jobs=-1),
    "BaggingClassifier": BaggingClassifier(**study_bg.best_params, random_state=42, n_jobs=-1)
}

# Train and evaluate each optimized model
for name, model in best_models.items():
    print(f"\n===== {name} (Best Optuna Params) =====")
    model.fit(X_train_svd, y_train_enc_res)
    preds = model.predict(X_test_svd)
    
    acc = accuracy_score(y_test_enc, preds)
    print(f"Accuracy: {acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test_enc, preds))

# Save each optimized model with Pickle
# Create folder for saved models
os.makedirs("saved_models", exist_ok=True)

# Save each model
for name, model in best_models.items():
    filename = f"saved_models/{name}_optuna.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")

# [markdown]
# ## 7.4. LSTM
#
# ### 7.4.1. Preprocessing `X`
#
# The neural network approach with LSTM requires a Tokenizer for processing sequences (`X`).

"""
# Create k-mer function
def create_kmers(seq, k=3):
    seq = seq.upper().strip()
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

# Run k-mer function
k = 3
X_train_kmers = [create_kmers(seq, k) for seq in X_train]
X_val_kmers   = [create_kmers(seq, k) for seq in X_val]
X_test_kmers  = [create_kmers(seq, k) for seq in X_test]
"""

# Character level/k-mer-level tokenization (1 or 3 AA's per token)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)

# Tokenize each X dataset - still enforcing max length of 1024 even though it was included earlier
max_len = 1024

X_train_lstm = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
X_val_lstm   = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding='post')
X_test_lstm  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')

# Check
print(X_train_lstm.shape, X_val_lstm.shape, X_test_lstm.shape)

# [markdown]
# ### 7.4.2. Create Neural Network

# SETUP
# Class imbalance
classes = np.unique(y_train_enc)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=classes,
                                     y=y_train_enc)
# Make a dict for Keras
class_weight_dict = dict(zip(classes, class_weights))

# Define remaining embedding variables
vocab_size = len(tokenizer.word_index) + 1  # plus 1 for padding token

# Check input variables
try:
    print("X_train_lstm shape:", X_train_lstm.shape)
    print("y_train_enc shape:", y_train_enc.shape)
    print("X_val_lstm shape:", X_val_lstm.shape)
    print("y_val_enc shape:", y_val_enc.shape)
    print("X_test_lstm shape:", X_test_lstm.shape)
    print("y_test_enc shape:", y_test_enc.shape)
    print("max_len:", max_len)
    print("vocab_size:", vocab_size)
    print("Number of classes:", len(le.classes_))
except NameError as e:
    print(f"❌ Missing variable: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

# MODEL ARCHITECTURE
inputs = Input(shape=(max_len,))

# Embedding
x = Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(inputs)

# BiLSTM
x = Bidirectional(LSTM(256, dropout=0.3, return_sequences=False))(x)

# Dense
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

# Output
outputs = Dense(len(le.classes_), activation='softmax')(x)

# Build & compile
model = Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# CALLBACKS
# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',       # what to watch
    patience=2,               # wait 2 epochs
    restore_best_weights=True # revert to best model
)

# Reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,       # reduce LR
    patience=2,       # wait 2 epochs
    min_lr=1e-6
)

# Save best epoch to file
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    verbose=1
)

model.summary()

import numpy as np
print(data_filtered['combined_sequence'].head())
lengths = [len(seq) for seq in data_filtered['combined_sequence']]
print("Number of sequences:", len(lengths))
print("Median:", np.median(lengths))
print("95th percentile:", np.percentile(lengths, 95))
print("Max:", max(lengths))

# [markdown]
# ### 7.4.3. Train Model

history_lstm = model.fit(
    X_train_lstm, y_train_enc,
    validation_data=(X_val_lstm, y_val_enc),
    batch_size=256,
    epochs=50,                     # Max epochs
    class_weight=class_weight_dict, # Class imbalance dict
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1 
)

test_loss, test_acc = model.evaluate(X_test_lstm, y_test_enc)
print(f"Test Accuracy: {test_acc:.4f}")

# [markdown]
# ### 7.4.4. Plot Training History

# history is returned from model.fit()
history_lstm.history.keys()

# Plot
plt.figure(figsize=(8,5))
plt.plot(history_lstm.history['accuracy'], label='Train Accuracy', color='royalblue', linewidth=2)
plt.plot(history_lstm.history['val_accuracy'], label='Validation Accuracy', color='darkorange', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot function

# [markdown]
# ## 7.5. LLM
#
# ### 7.5.1. Preprocess `X`
#
# Sequences need to be converted to embeddings from a pre-trained language model. We will use ESM-2 as it is the most suited to protein sequences.

# Subset to test
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Choose a tiny subset so it definitely runs
tiny_sequences = data_filtered['combined_sequence'][:100]  # limited # of sequences

def save_esm_embeddings(sequences, out_file, batch_size=1):
    with h5py.File(out_file, 'w') as f:
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]

            # Format for the batch converter
            batch_data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(batch_data)

            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[6])
                # Change repr layer index to match your model (t6 → layer 6)
                reps = results["representations"][6].cpu().numpy()

            # Save each embedding individually
            for k, seq in enumerate(batch_seqs):
                f.create_dataset(f"seq_{i+k}", data=reps[k])

    print(f"Saved {len(sequences)} embeddings to {out_file}")

# Run test
save_esm_embeddings(tiny_sequences, "esm_test_embeddings.h5", batch_size=1)


# Inspect embeddings file
with h5py.File("esm_test_embeddings.h5", "r") as f:
    print("Datasets:", list(f.keys()))
    for key in f.keys():
        print(key, f[key].shape)   

# ---------------------------
# 1. Load mean-pooled embeddings
# ---------------------------
def load_mean_pooled_embeddings(h5_file):
    """
    Load embeddings from h5 file and mean-pool over sequence length.
    Returns a dict: {seq_key: embedding_vector}
    """
    pooled = {}
    with h5py.File(h5_file, "r") as f:
        for key in f.keys():
            emb = f[key][:]
            pooled[key] = emb.mean(axis=0)  # mean pooling
    return pooled

pooled = load_mean_pooled_embeddings("esm_test_embeddings.h5")

# ---------------------------
# 2. Prepare embedding matrix and labels
# ---------------------------
labels = list(pooled.keys())                 # ['seq_0', 'seq_1', ...]
X = np.vstack([pooled[k] for k in labels])  # shape: (n_sequences, embedding_dim)

# ---------------------------
# 3. Map classification
# ---------------------------
# Assumes the order of pooled.keys() matches data_filtered rows
class_list = data_filtered['classification'].tolist()[:len(labels)]

# ---------------------------
# 4. PCA
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ---------------------------
# 5. Map classes to colors
# ---------------------------
unique_classes = list(set(class_list))
colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
class_to_color = {cls: c for cls, c in zip(unique_classes, colors_map)}

# ---------------------------
# 6. Plot PCA
# ---------------------------
plt.figure(figsize=(10,6))
for i, label in enumerate(labels):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], color=class_to_color[class_list[i]], s=60)

# Add legend
for cls, color in class_to_color.items():
    plt.scatter([], [], color=color, label=cls)
plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of ESM-2 Mean-Pooled Embeddings (Colored by Classification)")
plt.tight_layout()
plt.show()


model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def save_esm_embeddings(sequences, out_file, batch_size=1):
    with h5py.File(out_file, 'w') as f:
        # Get embedding dim
        batch_data = [(str(0), sequences[0])]
        _, _, batch_tokens = batch_converter(batch_data)
        with torch.no_grad():
            token_embeddings = model(batch_tokens, repr_layers=[5], return_contacts=False)["representations"][5]
        embedding_dim = token_embeddings.size(-1)
        dset = f.create_dataset('embeddings', (len(sequences), embedding_dim), dtype='float16')

        idx = 0
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            batch_data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                token_embeddings = model(batch_tokens, repr_layers=[5], return_contacts=False)["representations"][5]
                token_embeddings = token_embeddings.half()  # reduce memory

                for j, (_, seq) in enumerate(batch_data):
                    seq_len = len(seq)
                    if seq_len == 0:
                        continue
                    emb = token_embeddings[j, 1:seq_len+1].mean(0)
                    dset[idx] = emb.cpu().numpy()
                    idx += 1

            del batch_tokens, token_embeddings
            torch.cuda.empty_cache()

# Example usage:
save_esm_embeddings(X_train, "X_train_emb.h5")
save_esm_embeddings(X_val, "X_val_emb.h5")
save_esm_embeddings(X_test, "X_test_emb.h5")


# [markdown]
# ### 7.5.2. Tree Model
#
# We will first train our best model from the NLP section using the embeddings from ESM-2.

# Tree approach
# Train best classifiers from previous section
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
clf.fit(X_train_emb, y_train_enc)

# Evaluate
y_val_pred  = clf.predict(X_val_emb)
y_test_pred = clf.predict(X_test_emb)

print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test_enc, y_test_pred))

# [markdown]
# ### 7.5.3. Neural Network
#
# Next, we will train a neural network (similar to LSTM but simplified) with the ESM-2 embeddings.

# Neural network approach
input_dim = X_train_emb.shape[1]  # embedding size

# Model architecture
model_llm = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

# Build model
model_llm.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Run model
history_llm = model_llm.fit(
    X_train_llm, y_train_llm,
    validation_data=(X_val_llm, y_val_llm),
    batch_size=256,
    epochs=20,
    class_weight=class_weight_dict,  # if classes are imbalanced
)

# [markdown]
# ## 7.6. Model Comparison

# [markdown]
# <div style="width: 100%; background-color: #1565C0; text-align: left; font-size: 30px; padding: 10px; border-radius: 5px;">
#   <strong>8. Conclusion</strong>
# </div>

# [markdown]
# - NLP 90 maybe 93-95
# - LSTM 78
# - LLM need full embeddings
# 
# (Paper)
#  Machine Learning-Based Classification of Structural Protein Sequences for Drug Discovery
# 
# 1. Overview
#     - How ML can improve drug discovery
#     - In this paper we will compare 3 different ML methods that could be used in drug discovery and protein targetting
#         - NLP
#         - LSTM
#         - LLM
# 2. Data Source/Explanation/Definitions
#     - How many protein types, max, min, EDA
#     - Biochemical (AA) plots, etc.
# 3. Application of Methods: NLP (Tree Models)
#     - Methods
#     - Results
#     - Limitations/Improvements
# 4. Application of Methods: LSTM (Keras/TensorFlow)
# 5. Application of Methods: LLM (ESM-2)
# 6. Conclusion/Discussion
#     - Difficulties (computation power)
#     - Recommendations

# [markdown]
#

