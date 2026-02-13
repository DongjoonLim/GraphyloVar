"""
Train GraphyloVar base model.

This script provides a template for training the base GraphyloVar model architecture
with multi-task learning (allele classification + polymorphism prediction).
"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, NUM_SPECIES
from utils import create_output_directory, load_numpy_array

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_graphylo_model(input_shape, A):
    """
    Build Graphylo model (CNN + GCN + etc.).

    This is a template implementation. Customize based on your specific architecture needs.

    Args:
        input_shape: Shape of input data (species, sequence_length, features)
        A: Adjacency matrix for phylogenetic graph

    Returns:
        Compiled Keras model with two outputs: allele_softmax and binary_sigmoid

    Note:
        This is a simplified placeholder. Extend with:
        - Proper GCN layers using spektral or custom implementation
        - Transformer layers for sequence attention
        - Additional regularization (dropout, batch norm)
    """
    inputs = tf.keras.Input(shape=input_shape, name="sequence_input")

    # Example CNN feature extraction
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same")
    )(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    # TODO: Add GCN layers here using the adjacency matrix A
    # For now, using a simple Dense layer as placeholder
    x = tf.keras.layers.Dense(128, activation="relu", name="gcn_placeholder")(x)

    # Global pooling across species dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Shared dense layers
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # Multi-task outputs
    allele_output = tf.keras.layers.Dense(5, activation="softmax", name="allele_softmax")(x)
    poly_output = tf.keras.layers.Dense(1, activation="sigmoid", name="binary_sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[allele_output, poly_output])
    return model


def get_phylogenetic_adjacency():
    """
    Generate phylogenetic adjacency matrix.

    Returns:
        Adjacency matrix of shape (NUM_SPECIES, NUM_SPECIES)

    Note:
        This currently returns an identity matrix as placeholder.
        In production, load actual phylogenetic tree structure from:
        - Newick format tree file
        - Pre-computed adjacency matrix
        - Graph database
    """
    logger.warning("Using identity matrix for adjacency. Load real phylogenetic structure in production.")
    return np.eye(NUM_SPECIES)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GraphyloVar model.")
    parser.add_argument("--x_train", type=str, required=True, help="Path to training data X")
    parser.add_argument("--y_train", type=str, required=True, help="Path to training data Y")
    parser.add_argument("--x_val", type=str, required=True, help="Path to validation data X")
    parser.add_argument("--y_val", type=str, required=True, help="Path to validation data Y")
    parser.add_argument("--output_dir", type=str, default="models/saved_model", help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    return parser.parse_args()


def prepare_labels(Y_train, Y_val):
    """
    Prepare labels for multi-task learning.

    Args:
        Y_train: Training labels
        Y_val: Validation labels

    Returns:
        Tuple of (train_allele, train_poly, val_allele, val_poly)
    """
    Y_train_allele = Y_train[:, :5].astype("float32")
    Y_train_poly = Y_train[:, 5].astype("float32")
    Y_val_allele = Y_val[:, :5].astype("float32")
    Y_val_poly = Y_val[:, 5].astype("float32")

    logger.info(f"Label shapes - Train allele: {Y_train_allele.shape}, Train poly: {Y_train_poly.shape}")
    logger.info(f"Label shapes - Val allele: {Y_val_allele.shape}, Val poly: {Y_val_poly.shape}")

    return Y_train_allele, Y_train_poly, Y_val_allele, Y_val_poly


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Load data
    logger.info("Loading training data...")
    X_train = load_numpy_array(args.x_train, mmap_mode="r")
    Y_train = load_numpy_array(args.y_train)
    logger.info(f"Training data loaded: X shape {X_train.shape}, Y shape {Y_train.shape}")

    logger.info("Loading validation data...")
    X_val = load_numpy_array(args.x_val, mmap_mode="r")
    Y_val = load_numpy_array(args.y_val)
    logger.info(f"Validation data loaded: X shape {X_val.shape}, Y shape {Y_val.shape}")

    # Prepare labels
    Y_train_allele, Y_train_poly, Y_val_allele, Y_val_poly = prepare_labels(Y_train, Y_val)

    # Get adjacency matrix
    A = get_phylogenetic_adjacency()

    # Build model
    input_shape = X_train.shape[1:]
    logger.info(f"Building model with input shape: {input_shape}")
    model = build_graphylo_model(input_shape, A)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss={"allele_softmax": "categorical_crossentropy", "binary_sigmoid": "binary_crossentropy"},
        metrics={"allele_softmax": "accuracy", "binary_sigmoid": "accuracy"},
    )

    logger.info("Model summary:")
    model.summary()

    # Setup callbacks
    create_output_directory(args.output_dir)
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1))

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        x=X_train,
        y=[Y_train_allele, Y_train_poly],
        validation_data=(X_val, [Y_val_allele, Y_val_poly]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    history_path = os.path.join(args.output_dir, "history.npy")
    np.save(history_path, history.history)
    logger.info(f"Training history saved to {history_path}")

    # Print final metrics
    logger.info("Training complete!")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation allele accuracy: {history.history['val_allele_softmax_accuracy'][-1]:.4f}")
    logger.info(f"Final validation poly accuracy: {history.history['val_binary_sigmoid_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()
