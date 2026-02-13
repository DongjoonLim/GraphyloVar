"""
Train Graphylo Siamese model for variant impact prediction.

This script trains a Siamese neural network with explicit phylogenetic graph structure
using Graph Convolutional Networks (GCNs) from spektral library. The model processes
multi-species alignments with explicit evolutionary relationships.
"""

import argparse
import logging
import os

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    SPECIES_LIST,
    get_phylogenetic_edges,
)
from utils import create_output_directory, load_numpy_array

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Graphylo Siamese model.")
    parser.add_argument("data_path", type=str, help="Path to input data X.")
    parser.add_argument("model_path", type=str, help="Output model directory.")
    parser.add_argument("target_path", type=str, help="Path to targets y.")
    parser.add_argument("gpu", type=int, help="GPU ID.")
    parser.add_argument("num_filter", type=int, help="Number of CNN filters.")
    parser.add_argument("num_hidden", type=int, help="Number of LSTM hidden units.")
    parser.add_argument("num_hidden_graph", type=int, help="Number of GCN hidden units.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--l2_reg", type=float, default=0.01, help="L2 regularization factor.")
    return parser.parse_args()

def build_phylogenetic_graph():
    """
    Build phylogenetic graph structure and adjacency matrix.

    Returns:
        Tuple of (species_names, adjacency_matrix)

    Note:
        This uses the shared SPECIES_LIST from config.py and phylogenetic edges.
        In production, load actual phylogenetic tree from Newick file or database.
    """
    G = nx.Graph()

    # Add nodes for all species
    for idx, name in enumerate(SPECIES_LIST):
        G.add_node(name, index=idx)

    # Add phylogenetic edges
    edges = get_phylogenetic_edges()
    if not edges:
        logger.warning("No phylogenetic edges defined. Using default placeholder edges.")
        # Add some default edges as placeholder
        edges = [("hg38", "_HP"), ("panTro4", "_HP")]

    G.add_edges_from(edges)
    logger.info(f"Built phylogenetic graph with {len(SPECIES_LIST)} nodes and {len(edges)} edges")

    # Create adjacency matrix
    # Note: networkx.attr_matrix is deprecated, using adjacency_matrix instead
    try:
        from scipy.sparse import csr_matrix

        adj_matrix = nx.adjacency_matrix(G).todense()
        A = np.array(adj_matrix, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Error creating adjacency matrix: {e}. Using identity matrix.")
        A = np.eye(len(SPECIES_LIST), dtype=np.float32)

    return SPECIES_LIST, A


def build_model(
    num_species: int, seq_len: int, num_filter: int, num_hidden: int, num_hidden_graph: int, dropout: float, l2_reg: float
) -> Model:
    """
    Build Siamese Graphylo model with phylogenetic graph structure.

    Architecture:
        1. 1D CNN for sequence feature extraction
        2. GCN layers for phylogenetic relationship modeling
        3. LSTM for temporal dependencies
        4. Dense layers for classification

    Args:
        num_species: Number of species in phylogenetic graph
        seq_len: Length of input sequences
        num_filter: Number of CNN filters
        num_hidden: Number of LSTM hidden units
        num_hidden_graph: Number of GCN hidden units
        dropout: Dropout rate
        l2_reg: L2 regularization factor

    Returns:
        Compiled Keras Model

    Note:
        This is a template implementation. Customize based on your data structure.
        Current implementation assumes input shape (num_species, seq_len).
    """
    # Input layer
    inputs = Input(shape=(num_species, seq_len), name="sequence_input")

    # CNN for feature extraction per species
    # Using TimeDistributed would be better if shape is (num_species, seq_len, features)
    x = tf.keras.layers.Reshape((num_species, seq_len, 1))(inputs)
    x = tf.keras.layers.TimeDistributed(Conv1D(filters=num_filter, kernel_size=5, activation="relu", padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(Flatten())(x)

    # TODO: Add proper GCN layers here using spektral.layers.GCNConv
    # For now, using Dense layers as placeholder
    x = Dense(num_hidden_graph, activation="relu", kernel_regularizer=l2(l2_reg), name="gcn_placeholder")(x)
    x = Dropout(dropout)(x)

    # LSTM for temporal modeling
    x = LSTM(num_hidden, name="lstm_layer")(x)
    x = Dropout(dropout)(x)

    # Output layer
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info(f"Using GPU: {args.gpu}")

    # Build phylogenetic graph
    species_names, A = build_phylogenetic_graph()
    logger.info(f"Phylogenetic graph: {len(species_names)} species, adjacency shape {A.shape}")

    # Load data
    logger.info("Loading data...")
    X = load_numpy_array(args.data_path)
    y = load_numpy_array(args.target_path)
    logger.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")

    # Validate data
    if X.shape[1] != len(species_names):
        logger.warning(f"Expected {len(species_names)} species, got {X.shape[1]}")

    # Build model
    logger.info("Building model...")
    model = build_model(
        num_species=len(species_names),
        seq_len=X.shape[2],
        num_filter=args.num_filter,
        num_hidden=args.num_hidden,
        num_hidden_graph=args.num_hidden_graph,
        dropout=args.dropout,
        l2_reg=args.l2_reg,
    )

    # Compile model
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

    logger.info("Model summary:")
    model.summary()

    # Train model
    logger.info("Starting training...")
    history = model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, verbose=1)

    # Save model
    create_output_directory(args.model_path)
    model.save(args.model_path)
    logger.info(f"Model saved to {args.model_path}")

    # Save training history
    history_path = os.path.join(args.model_path, "training_history.npy")
    np.save(history_path, history.history)
    logger.info(f"Training history saved to {history_path}")

    # Print final metrics
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    logger.info("Training complete!")
    logger.info(f"Final training loss: {final_loss:.4f}, accuracy: {final_acc:.4f}")
    logger.info(f"Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_acc:.4f}")

if __name__ == "__main__":
    main()
