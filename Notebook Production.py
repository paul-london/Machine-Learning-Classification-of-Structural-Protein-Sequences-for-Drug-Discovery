"""
Machine Learning-Based Classification of Structural Protein Sequences for Drug Discovery
Object-Oriented Programming Refactored Version
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import optuna
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Embedding, 
                                      Bidirectional, Input, BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class ProteinDataLoader:
    """Handles loading and initial processing of protein data."""
    
    def __init__(self, metadata_path: str, sequences_path: str):
        self.metadata_path = metadata_path
        self.sequences_path = sequences_path
        self.metadata = None
        self.sequences = None
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and merge protein metadata and sequences."""
        self.metadata = pd.read_csv(self.metadata_path)
        self.sequences = pd.read_csv(self.sequences_path)
        
        # Merge datasets
        self.data = pd.merge(
            self.sequences[['chainId', 'sequence', 'structureId']], 
            self.metadata, 
            on='structureId', 
            how='inner'
        )
        
        # Filter for proteins only
        self.data = self.data[self.data['macromoleculeType'] == 'Protein']
        self.data = self.data[self.data['classification'].notnull()]
        self.data = self.data[self.data['sequence'].notnull()]
        
        return self.data
    
    def get_info(self):
        """Display dataset information."""
        print("Metadata shape:", self.metadata.shape)
        print("Sequences shape:", self.sequences.shape)
        print("Merged data shape:", self.data.shape)
        return self.data.head()


class ProteinDataCleaner:
    """Cleans and preprocesses protein data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        
    def rename_columns(self):
        """Standardize column names."""
        self.data = self.data.rename(columns={
            'structureId': 'structure_id',
            'chainId': 'chain_id',
            'residueCount': 'residue_count',
            'macromoleculeType': 'macromolecule_type'
        })
        return self
    
    def normalize_classification(self):
        """Normalize classification labels."""
        self.data['classification'] = (
            self.data['classification']
            .str.upper()
            .str.strip()
            .str.replace(r'[^A-Z0-9 /]', '', regex=True)
        )
        return self
    
    def clean_sequences(self, min_len: int = 20, max_len: int = 1024):
        """Clean and filter sequences."""
        # Clean sequences
        self.data['sequence'] = self.data['sequence'].apply(
            lambda seq: ''.join([aa for aa in seq.upper().strip() if aa in self.valid_aas])
        )
        
        # Filter by length
        self.data = self.data[
            (self.data['sequence'].str.len() >= min_len) & 
            (self.data['sequence'].str.len() <= max_len)
        ].reset_index(drop=True)
        
        return self
    
    def combine_chains(self):
        """Combine sequences for multi-chain proteins."""
        df_combined = (
            self.data.groupby('structure_id')['sequence']
            .apply(lambda seqs: ''.join(seqs))
            .reset_index(name='combined_sequence')
        )
        
        self.data = self.data.merge(df_combined, on='structure_id', how='left')
        self.data = self.data.drop_duplicates(subset='structure_id')
        
        return self
    
    def filter_top_classes(self, n_classes: int = 20):
        """Keep only top N most frequent classes."""
        top_classes = self.data['classification'].value_counts().head(n_classes).index
        self.data = self.data[self.data['classification'].isin(top_classes)].reset_index(drop=True)
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return cleaned data."""
        return self.data


class ProteinEDA:
    """Exploratory Data Analysis for protein sequences."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def plot_sequence_lengths(self):
        """Plot distribution of sequence lengths."""
        self.data['sequence_length'] = self.data['sequence'].str.len()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['sequence_length'], bins=50, kde=True)
        plt.title('Distribution of Protein Sequence Lengths')
        plt.xlabel('Sequence Length (AA Residues)')
        plt.ylabel('Count')
        plt.show()
        
    def plot_aa_composition(self):
        """Plot amino acid composition."""
        aa_counts = Counter("".join(self.data['sequence']))
        aa_freq_pct = {aa: (count / sum(aa_counts.values())) * 100 
                       for aa, count in aa_counts.items()}
        
        aa_df = pd.DataFrame(
            list(aa_freq_pct.items()), 
            columns=['Amino Acid', 'Frequency']
        ).sort_values('Frequency', ascending=False)
        
        plt.figure(figsize=(12, 5))
        aa_df.plot(kind='bar', x='Amino Acid', y='Frequency', legend=False)
        plt.xticks(rotation=0)
        plt.ylabel('Frequency (%)')
        plt.title('Amino Acid Composition')
        plt.show()
        
    def plot_class_distribution(self, top_n: int = 20):
        """Plot distribution of protein classes."""
        counts = self.data['classification'].value_counts().head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=counts.values, y=counts.index)
        plt.xlabel("Count")
        plt.ylabel("Classification")
        plt.title(f"Top {top_n} Protein Classes")
        plt.tight_layout()
        plt.show()
        
    def compute_biochemical_properties(self):
        """Compute biochemical properties of sequences."""
        hydrophobic = set('AILMFWV')
        polar_uncharged = set('CSTNQ')
        acidic = set('DE')
        basic = set('KRH')
        aromatic = set('FYW')
        
        def compute_fraction(seq, aa_set):
            return sum(aa in aa_set for aa in seq) / len(seq) if len(seq) > 0 else 0
        
        self.data['hydrophobic'] = self.data['sequence'].apply(lambda s: compute_fraction(s, hydrophobic))
        self.data['polar'] = self.data['sequence'].apply(lambda s: compute_fraction(s, polar_uncharged))
        self.data['acidic'] = self.data['sequence'].apply(lambda s: compute_fraction(s, acidic))
        self.data['basic'] = self.data['sequence'].apply(lambda s: compute_fraction(s, basic))
        self.data['aromatic'] = self.data['sequence'].apply(lambda s: compute_fraction(s, aromatic))
        
        return self.data


class ProteinDataSplitter:
    """Splits data into train/val/test sets."""
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, X, y) -> Tuple:
        """Split data into train, validation, and test sets."""
        # First split: train vs temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class NLPPreprocessor:
    """Preprocesses sequences for NLP-based models."""
    
    def __init__(self, k: int = 3, max_features: int = 5000):
        self.k = k
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        
    @staticmethod
    def create_kmers(seq: str, k: int = 3) -> str:
        """Convert sequence to k-mers."""
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        return ' '.join(kmers)
    
    def fit_transform(self, X_train):
        """Fit vectorizer and transform training data."""
        X_train_kmers = [self.create_kmers(seq, self.k) for seq in X_train]
        return self.vectorizer.fit_transform(X_train_kmers)
    
    def transform(self, X):
        """Transform new data."""
        X_kmers = [self.create_kmers(seq, self.k) for seq in X]
        return self.vectorizer.transform(X_kmers)
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for class imbalance."""
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
    
    def apply_svd(self, X_train, X_test):
        """Apply dimensionality reduction."""
        X_train_svd = self.svd.fit_transform(X_train)
        X_test_svd = self.svd.transform(X_test)
        return X_train_svd, X_test_svd


class LSTMPreprocessor:
    """Preprocesses sequences for LSTM models."""
    
    def __init__(self, max_len: int = 1024):
        self.max_len = max_len
        self.tokenizer = Tokenizer(char_level=True)
        
    def fit_transform(self, X_train):
        """Fit tokenizer and transform training data."""
        self.tokenizer.fit_on_texts(X_train)
        return pad_sequences(
            self.tokenizer.texts_to_sequences(X_train),
            maxlen=self.max_len,
            padding='post'
        )
    
    def transform(self, X):
        """Transform new data."""
        return pad_sequences(
            self.tokenizer.texts_to_sequences(X),
            maxlen=self.max_len,
            padding='post'
        )
    
    @property
    def vocab_size(self):
        """Get vocabulary size."""
        return len(self.tokenizer.word_index) + 1


class TreeModelTrainer:
    """Trains and optimizes tree-based models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.best_params = None
        self.model = None
        
    def optimize(self, X_train, y_train, n_trials: int = 25):
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            if self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif self.model_type == 'extra_trees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = ExtraTreesClassifier(**params)
                
            elif self.model_type == 'bagging':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                    'max_samples': trial.suggest_float('max_samples', 0.3, 0.7),
                    'max_features': trial.suggest_float('max_features', 0.3, 0.7),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = BaggingClassifier(**params)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
            return score
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        return study.best_value
    
    def train(self, X_train, y_train):
        """Train model with best parameters."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.best_params, random_state=42, n_jobs=-1)
        elif self.model_type == 'extra_trees':
            self.model = ExtraTreesClassifier(**self.best_params, random_state=42, n_jobs=-1)
        elif self.model_type == 'bagging':
            self.model = BaggingClassifier(**self.best_params, random_state=42, n_jobs=-1)
        
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model."""
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, report
    
    def save_model(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)


class LSTMModelTrainer:
    """Builds and trains LSTM models."""
    
    def __init__(self, vocab_size: int, max_len: int, n_classes: int):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture."""
        inputs = Input(shape=(self.max_len,))
        
        x = Embedding(input_dim=self.vocab_size, output_dim=128, mask_zero=True)(inputs)
        x = Bidirectional(LSTM(256, dropout=0.3, return_sequences=False))(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self
    
    def train(self, X_train, y_train, X_val, y_val, 
              class_weights: Optional[Dict] = None,
              epochs: int = 50, batch_size: int = 256):
        """Train LSTM model."""
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
            ModelCheckpoint('best_lstm_model.keras', monitor='val_accuracy', 
                          save_best_only=True, mode='max')
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model."""
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        return test_acc
    
    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


# ========== EXAMPLE USAGE ==========

def main():
    """Main pipeline execution."""
    
    # 1. Load data
    print("Loading data...")
    loader = ProteinDataLoader("pdb_data_no_dups.csv", "pdb_data_seq.csv")
    data = loader.load_data()
    
    # 2. Clean data
    print("Cleaning data...")
    cleaner = ProteinDataCleaner(data)
    cleaned_data = (cleaner
                    .rename_columns()
                    .normalize_classification()
                    .clean_sequences(min_len=20, max_len=1024)
                    .combine_chains()
                    .filter_top_classes(n_classes=20)
                    .get_cleaned_data())
    
    # 3. EDA
    print("Performing EDA...")
    eda = ProteinEDA(cleaned_data)
    eda.plot_sequence_lengths()
    eda.plot_aa_composition()
    eda.plot_class_distribution()
    
    # 4. Prepare data
    X = cleaned_data['combined_sequence']
    y = cleaned_data['classification']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    splitter = ProteinDataSplitter(test_size=0.3)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y_encoded)
    
    # 5. NLP Approach
    print("\n=== NLP Approach ===")
    nlp_prep = NLPPreprocessor(k=3, max_features=5000)
    
    X_train_nlp = nlp_prep.fit_transform(X_train)
    X_val_nlp = nlp_prep.transform(X_val)
    X_test_nlp = nlp_prep.transform(X_test)
    
    # Apply SMOTE
    X_train_nlp_res, y_train_res = nlp_prep.apply_smote(X_train_nlp, y_train)
    
    # Apply SVD
    X_train_svd, X_test_svd = nlp_prep.apply_svd(X_train_nlp_res, X_test_nlp)
    
    # Train ExtraTreesClassifier
    tree_trainer = TreeModelTrainer(model_type='extra_trees')
    tree_trainer.optimize(X_train_svd, y_train_res, n_trials=25)
    tree_trainer.train(X_train_svd, y_train_res)
    acc, report = tree_trainer.evaluate(X_test_svd, y_test)
    print(f"ExtraTrees Accuracy: {acc:.4f}")
    tree_trainer.save_model("extra_trees_model.pkl")
    
    # 6. LSTM Approach
    print("\n=== LSTM Approach ===")
    lstm_prep = LSTMPreprocessor(max_len=1024)
    
    X_train_lstm = lstm_prep.fit_transform(X_train)
    X_val_lstm = lstm_prep.transform(X_val)
    X_test_lstm = lstm_prep.transform(X_test)
    
    # Compute class weights
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights_array))
    
    # Train LSTM
    lstm_trainer = LSTMModelTrainer(
        vocab_size=lstm_prep.vocab_size,
        max_len=1024,
        n_classes=len(le.classes_)
    )
    
    lstm_trainer.build_model()
    lstm_trainer.train(X_train_lstm, y_train, X_val_lstm, y_val, 
                      class_weights=class_weight_dict, epochs=50)
    
    lstm_acc = lstm_trainer.evaluate(X_test_lstm, y_test)
    print(f"LSTM Test Accuracy: {lstm_acc:.4f}")
    lstm_trainer.plot_history()
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()