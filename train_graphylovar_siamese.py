import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from spektral.layers import GCNConv
from focal_loss import BinaryFocalLoss

def build_siamese_graphylo(num_species, seq_len, filters, fcnn_units, gcn_units):
    """
    Constructs the GraphyloVar Architecture:
    CNN (Seq constraints) -> GCN (Phylo constraints) -> LSTM -> Dense
    """
    
    # --- Input Definition ---
    # Shape: [Num_Species, Seq_Len, 4 (One-Hot)]
    # Note: For GCN, we usually treat Species as Nodes.
    
    # 1. Sequence Feature Extraction (Shared CNN across species)
    # We treat each species sequence as a channel or handle via TimeDistributed
    input_seq = Input(shape=(num_species, seq_len, 4), name="seq_input")
    input_adj = Input(shape=(num_species, num_species), name="adj_input")
    
    # Shared CNN Encoder
    # We apply the same CNN to every species row
    cnn = layers.Conv1D(filters=filters, kernel_size=5, activation='relu', padding='same')
    
    # Apply CNN to each species independently (TimeDistributed)
    x = layers.TimeDistributed(cnn)(input_seq)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x) 
    
    # Now x shape is [Batch, Num_Species, Features]
    
    # 2. Graph Convolution (Phylogenetic Mixing)
    # Spektral layers usually expect (Nodes, Features) + Adjacency
    # We implement a custom lambda or Spektral loop here for the batch
    
    # Simplified GCN for dense adjacency implementation
    # A * X * W
    def graph_conv(args):
        features, adj = args
        # features: [Batch, N, F]
        # adj: [Batch, N, N] or [N, N] broadcasted
        return tf.matmul(adj, features)

    x_gcn = layers.Dense(gcn_units, activation='relu')(x)
    x_gcn = layers.Lambda(graph_conv)([x_gcn, input_adj])
    
    # 3. LSTM (Capturing dependencies if treating species as sequence, 
    # OR capturing dependencies along the genomic window if architected differently)
    # Based on your README, LSTM is likely for the sequence part, but here we put it after GCN
    # to aggregate the species info.
    
    x_lstm = layers.LSTM(fcnn_units)(x_gcn)
    
    # 4. Classification
    outputs = layers.Dense(1, activation='sigmoid')(x_lstm)
    
    model = Model(inputs=[input_seq, input_adj], outputs=outputs)
    return model

if __name__ == "__main__":
    # Args: [DATA_X] [OUTPUT_DIR] [DATA_Y] [GPU_ID] [FILTERS] [FCNN] [GCN]
    if len(sys.argv) < 8:
        print("Invalid arguments. See README.")
        sys.exit(1)

    data_x_path = sys.argv[1]
    output_dir = sys.argv[2]
    data_y_path = sys.argv[3]
    gpu_id = sys.argv[4]
    filters = int(sys.argv[5])
    fcnn = int(sys.argv[6])
    gcn = int(sys.argv[7])

    # GPU Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Load Data
    print("Loading data...")
    X = np.load(data_x_path) # Expected: [Batch, Species, Len, 4]
    y = np.load(data_y_path)
    
    # Mock Adjacency (Since it's not in the NPY usually)
    # In real usage, load this from a file
    num_samples = X.shape[0]
    num_species = X.shape[1]
    seq_len = X.shape[2]
    
    # Create dummy identity adjacency for runnable demo
    A = np.tile(np.eye(num_species), (num_samples, 1, 1))

    # Build Model
    model = build_siamese_graphylo(num_species, seq_len, filters, fcnn, gcn)
    
    model.compile(optimizer='adam',
                  loss=BinaryFocalLoss(gamma=2),
                  metrics=['accuracy'])
    
    model.summary()
    
    # Train
    print("Starting training...")
    model.fit([X, A], y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(output_dir)
    print(f"Model saved to {output_dir}")
