"""
Improved Deep Learning Architecture for Protein Sequence Classification
Multiple architecture options with best practices
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, TensorBoard)
from sklearn.utils.class_weight import compute_class_weight


class ProteinClassificationModel:
    """Enhanced protein classification with multiple architecture options."""
    
    def __init__(self, vocab_size, max_len, num_classes, architecture='hybrid'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        
    def build_improved_bilstm(self):
        """
        Improved BiLSTM with:
        - Deeper architecture
        - Residual connections
        - Better regularization
        """
        inputs = Input(shape=(self.max_len,), name='sequence_input')
        
        # Embedding with larger dimension
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=256,
            mask_zero=True,
            name='embedding'
        )(inputs)
        
        # Spatial Dropout (better for sequences than regular dropout)
        x = layers.SpatialDropout1D(0.2)(x)
        
        # First BiLSTM layer with return_sequences=True
        lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_1'
        )(x)
        lstm1 = layers.LayerNormalization()(lstm1)
        
        # Second BiLSTM layer
        lstm2 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_2'
        )(lstm1)
        lstm2 = layers.LayerNormalization()(lstm2)
        
        # Global pooling (capture both max and average patterns)
        avg_pool = layers.GlobalAveragePooling1D()(lstm2)
        max_pool = layers.GlobalMaxPooling1D()(lstm2)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers with BatchNorm
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='ImprovedBiLSTM')
        return self
    
    def build_cnn_lstm_hybrid(self):
        """
        CNN-LSTM Hybrid:
        - CNN extracts local patterns (motifs)
        - LSTM captures long-range dependencies
        """
        inputs = Input(shape=(self.max_len,), name='sequence_input')
        
        # Embedding
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=256,
            mask_zero=True,
            name='embedding'
        )(inputs)
        x = layers.SpatialDropout1D(0.2)(x)
        
        # Multi-scale CNN (different kernel sizes capture different motif lengths)
        conv_blocks = []
        for kernel_size in [3, 5, 7]:
            conv = layers.Conv1D(
                filters=128,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                name=f'conv_{kernel_size}'
            )(x)
            conv = layers.BatchNormalization()(conv)
            conv_blocks.append(conv)
        
        # Concatenate multi-scale features
        x = layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        # BiLSTM to capture sequential dependencies
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3),
            name='bilstm'
        )(x)
        x = layers.LayerNormalization()(x)
        
        # Attention mechanism (focus on important regions)
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(512)(attention)  # 512 = 256*2 (BiLSTM output)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Hybrid')
        return self
    
    def build_transformer_encoder(self):
        """
        Transformer Encoder:
        - Self-attention captures relationships between all positions
        - Better for long-range dependencies
        """
        inputs = Input(shape=(self.max_len,), name='sequence_input')
        
        # Embedding + Positional encoding
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=256,
            mask_zero=True,
            name='embedding'
        )(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.max_len,
            output_dim=256
        )(positions)
        x = x + position_embedding
        x = layers.Dropout(0.2)(x)
        
        # Transformer encoder blocks
        for i in range(2):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=32,
                dropout=0.1,
                name=f'attention_{i}'
            )(x, x)
            
            # Residual connection + LayerNorm
            x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = layers.Dense(512, activation='relu')(x)
            ffn = layers.Dropout(0.1)(ffn)
            ffn = layers.Dense(256)(ffn)
            
            # Residual connection + LayerNorm
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global pooling
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='TransformerEncoder')
        return self
    
    def build_residual_lstm(self):
        """
        Residual LSTM:
        - Skip connections help gradient flow
        - Prevents vanishing gradients in deep networks
        """
        inputs = Input(shape=(self.max_len,), name='sequence_input')
        
        # Embedding
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=256,
            mask_zero=True,
            name='embedding'
        )(inputs)
        x = layers.SpatialDropout1D(0.2)(embedding)
        
        # First LSTM block
        lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3)
        )(x)
        lstm1 = layers.LayerNormalization()(lstm1)
        
        # Second LSTM block with residual
        lstm2 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3)
        )(lstm1)
        lstm2 = layers.LayerNormalization()(lstm2)
        lstm2 = layers.Add()([lstm1, lstm2])  # Residual connection
        
        # Third LSTM block with residual
        lstm3 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3)
        )(lstm2)
        lstm3 = layers.LayerNormalization()(lstm3)
        lstm3 = layers.Add()([lstm2, lstm3])  # Residual connection
        
        # Attention pooling
        attention = layers.Dense(1, activation='tanh')(lstm3)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(512)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.Multiply()([lstm3, attention])
        x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='ResidualLSTM')
        return self
    
    def build(self):
        """Build model based on selected architecture."""
        if self.architecture == 'improved_bilstm':
            return self.build_improved_bilstm()
        elif self.architecture == 'hybrid':
            return self.build_cnn_lstm_hybrid()
        elif self.architecture == 'transformer':
            return self.build_transformer_encoder()
        elif self.architecture == 'residual':
            return self.build_residual_lstm()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def compile_model(self, learning_rate=1e-3, label_smoothing=0.1):
        """
        Compile with advanced options:
        - Label smoothing helps with overconfidence
        - Cosine decay for learning rate
        """
        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            alpha=0.1
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        # Label smoothing reduces overconfidence
        loss = keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=label_smoothing
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        return self
    
    def get_callbacks(self, checkpoint_path='best_protein_model.keras', 
                      log_dir='./logs'):
        """Enhanced callbacks with TensorBoard."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive reduction
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build() first.")


# ========== USAGE EXAMPLE ==========

def train_protein_classifier(X_train, y_train, X_val, y_val, 
                             tokenizer, label_encoder,
                             architecture='hybrid'):
    """
    Complete training pipeline with best practices.
    """
    
    # Setup
    vocab_size = len(tokenizer.word_index) + 1
    max_len = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    # Class weights for imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Vocab size: {vocab_size}")
    print(f"Max length: {max_len}")
    print(f"Number of classes: {num_classes}")
    print(f"Architecture: {architecture}")
    
    # Build model
    model_builder = ProteinClassificationModel(
        vocab_size=vocab_size,
        max_len=max_len,
        num_classes=num_classes,
        architecture=architecture
    )
    
    model_builder.build().compile_model(learning_rate=1e-3)
    model_builder.summary()
    
    # Train
    history = model_builder.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=64,  # Smaller batch for better generalization
        epochs=100,
        class_weight=class_weight_dict,
        callbacks=model_builder.get_callbacks(),
        verbose=1
    )
    
    return model_builder.model, history


# Example usage with all architectures
def compare_architectures(X_train, y_train, X_val, y_val, 
                         tokenizer, label_encoder):
    """Train and compare all architectures."""
    
    architectures = ['improved_bilstm', 'hybrid', 'transformer', 'residual']
    results = {}
    
    for arch in architectures:
        print(f"\n{'='*50}")
        print(f"Training {arch}")
        print('='*50)
        
        model, history = train_protein_classifier(
            X_train, y_train, X_val, y_val,
            tokenizer, label_encoder,
            architecture=arch
        )
        
        # Get best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        results[arch] = best_val_acc
        
        print(f"\n{arch} best validation accuracy: {best_val_acc:.4f}")
    
    # Print comparison
    print("\n" + "="*50)
    print("ARCHITECTURE COMPARISON")
    print("="*50)
    for arch, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{arch:20s}: {acc:.4f}")
    
    return results


# Quick start with recommended architecture
if __name__ == "__main__":
    # Assuming you have these variables from your preprocessing:
    # X_train_lstm, y_train_enc, X_val_lstm, y_val_enc, tokenizer, le
    
    """
    model, history = train_protein_classifier(
        X_train_lstm, y_train_enc,
        X_val_lstm, y_val_enc,
        tokenizer, le,
        architecture='hybrid'  # Recommended: CNN-LSTM hybrid
    )
    
    # Evaluate on test set
    test_loss, test_acc, test_top3 = model.evaluate(X_test_lstm, y_test_enc)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Top-3 Accuracy: {test_top3:.4f}")
    """
    
    pass