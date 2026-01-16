import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from spektral.layers import GCNConv
from focal_loss import BinaryFocalLoss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train GraphyloVar Siamese model.")
    parser.add_argument("data_x_path", type=str, help="Path to X data.")
    parser.add_argument("output_dir", type=str, help="Output directory.")
    parser.add_argument("data_y_path", type=str, help="Path to y data.")
    parser.add_argument("gpu_id", type=str, help="GPU ID.")
    parser.add_argument("filters", type=int, help="CNN filters.")
    parser.add_argument("fcnn", type=int, help="FCNN units.")
    parser.add_argument("gcn", type=int, help="GCN units.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    return parser.parse_args()

def build_siamese_graphylo(num_species: int, seq_len: int, filters: int, fcnn: int, gcn: int) -> Model:
    """Build GraphyloVar Siamese model."""
    input_seq = Input(shape=(num_species, seq_len, 4), name="seq_input")
    input_adj = Input(shape=(num_species, num_species), name="adj_input")
    
    cnn = layers.TimeDistributed(layers.Conv1D(filters=filters, kernel_size=5, activation='relu', padding='same'))(input_seq)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(cnn)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    x_gcn = layers.Dense(gcn, activation='relu')(x)
    def graph_conv(inputs):
        features, adj = inputs
        return tf.matmul(adj, features)
    x_gcn = layers.Lambda(graph_conv)([x_gcn, input_adj])
    
    x_lstm = layers.LSTM(fcnn)(x_gcn)
    outputs = layers.Dense(1, activation='sigmoid')(x_lstm)
    
    model = Model(inputs=[input_seq, input_adj], outputs=outputs)
    return model

def main():
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    print("Loading data...")
    X = np.load(args.data_x_path)  # [Batch, Species, Len, 4]
    y = np.load(args.data_y_path)
    
    num_samples, num_species, seq_len, _ = X.shape
    A = np.tile(np.eye(num_species), (num_samples, 1, 1))  # Dummy adj; load real if available
    
    model = build_siamese_graphylo(num_species, seq_len, args.filters, args.fcnn, args.gcn)
    model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy'])
    model.summary()
    
    print("Training...")
    model.fit([X, A], y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)
    
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
