"""
Train GraphyloVar Siamese model for variant impact prediction.

This script trains a Siamese neural network that combines CNNs for sequence feature extraction,
GCNs for phylogenetic relationship modeling, and LSTMs for temporal dependencies.
The model uses focal loss to handle class imbalance in variant classification.
"""

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from focal_loss import BinaryFocalLoss
from tensorflow.keras import Input, Model, layers

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CNN_FILTERS,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL_LOSS_GAMMA,
    DEFAULT_GCN_UNITS,
    DEFAULT_LSTM_UNITS,
    NUM_SPECIES,
)
from utils import create_output_directory, load_numpy_array

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GraphyloVar Siamese model.")
    parser.add_argument("data_x_path", type=str, help="Path to X data.")
    parser.add_argument("output_dir", type=str, help="Output directory.")
    parser.add_argument("data_y_path", type=str, help="Path to y data.")
    parser.add_argument("gpu_id", type=str, help="GPU ID.")
    parser.add_argument("filters", type=int, help="CNN filters.")
    parser.add_argument("fcnn", type=int, help="FCNN units.")
    parser.add_argument("gcn", type=int, help="GCN units.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument(
        "--focal_gamma", type=float, default=DEFAULT_FOCAL_LOSS_GAMMA, help="Focal loss gamma parameter."
    )
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio.")
    return parser.parse_args()

def build_siamese_graphylo(num_species: int, seq_len: int, filters: int, fcnn: int, gcn: int) -> Model:
    """
    Build GraphyloVar Siamese model.

    Architecture:
        1. TimeDistributed CNN for sequence feature extraction per species
        2. GCN layer for phylogenetic relationship modeling
        3. LSTM for capturing temporal dependencies
        4. Dense layer for binary classification

    Args:
        num_species: Number of species in phylogenetic graph
        seq_len: Length of input sequences
        filters: Number of CNN filters
        fcnn: Number of LSTM units
        gcn: Number of GCN units

    Returns:
        Compiled Keras Model

    Note:
        Current GCN implementation is a placeholder using simple matrix multiplication.
        In production, use proper GCN layers from spektral or similar libraries.
    """
    # Input layers
    input_seq = Input(shape=(num_species, seq_len, 4), name="seq_input")
    input_adj = Input(shape=(num_species, num_species), name="adj_input")

    # CNN feature extraction per species
    cnn = layers.TimeDistributed(layers.Conv1D(filters=filters, kernel_size=5, activation="relu", padding="same"))(
        input_seq
    )
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(cnn)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # GCN layer (placeholder - use proper GCN in production)
    x_gcn = layers.Dense(gcn, activation="relu")(x)

    def graph_conv(inputs):
        """Simple graph convolution using adjacency matrix multiplication."""
        features, adj = inputs
        return tf.matmul(adj, features)

    x_gcn = layers.Lambda(graph_conv, name="graph_conv")([x_gcn, input_adj])

    # LSTM for temporal modeling
    x_lstm = layers.LSTM(fcnn, name="lstm_layer")(x_gcn)

    # Binary classification output
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x_lstm)

    model = Model(inputs=[input_seq, input_adj], outputs=outputs)
    return model


def create_adjacency_matrices(num_samples: int, num_species: int) -> np.ndarray:
    """
    Create adjacency matrices for the phylogenetic graph.

    Args:
        num_samples: Number of samples in the dataset
        num_species: Number of species in the phylogenetic graph

    Returns:
        Adjacency matrix array of shape (num_samples, num_species, num_species)

    Note:
        Currently returns identity matrices as placeholder.
        In production, load actual phylogenetic adjacency from tree structure.
    """
    logger.warning("Using identity matrix as placeholder for adjacency. Load real phylogenetic structure in production.")
    return np.tile(np.eye(num_species), (num_samples, 1, 1))


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    logger.info(f"Using GPU: {args.gpu_id}")

    # Load data
    logger.info("Loading data...")
    X = load_numpy_array(args.data_x_path)  # [Batch, Species, Len, 4]
    y = load_numpy_array(args.data_y_path)
    logger.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")

    # Validate data shapes
    num_samples, num_species, seq_len, num_features = X.shape
    if num_features != 4:
        raise ValueError(f"Expected 4 features (one-hot DNA), got {num_features}")
    if num_species != NUM_SPECIES:
        logger.warning(f"Expected {NUM_SPECIES} species, got {num_species}")

    # Create adjacency matrices
    A = create_adjacency_matrices(num_samples, num_species)

    # Build model
    logger.info("Building model...")
    model = build_siamese_graphylo(num_species, seq_len, args.filters, args.fcnn, args.gcn)

    # Compile model
    model.compile(optimizer="adam", loss=BinaryFocalLoss(gamma=args.focal_gamma), metrics=["accuracy"])

    logger.info("Model summary:")
    model.summary()

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        [X, A],
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=1,
    )

    # Save model
    create_output_directory(args.output_dir)
    model.save(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.npy")
    np.save(history_path, history.history)
    logger.info(f"Training history saved to {history_path}")

    # Print final metrics
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    logger.info(f"Training complete!")
    logger.info(f"Final training loss: {final_loss:.4f}, accuracy: {final_acc:.4f}")
    logger.info(f"Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_acc:.4f}")

if __name__ == "__main__":
    main()
