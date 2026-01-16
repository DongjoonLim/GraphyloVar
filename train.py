import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from models import build_graphylo_model
from utils import get_phylogenetic_adjacency

def main():
    parser = argparse.ArgumentParser(description="Train GraphyloVar")
    parser.add_argument('--x_train', type=str, required=True)
    parser.add_argument('--y_train', type=str, required=True)
    parser.add_argument('--x_val', type=str, required=True)
    parser.add_argument('--y_val', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models/saved_model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data...")
    X_train = np.load(args.x_train, mmap_mode='r')
    Y_train = np.load(args.y_train, allow_pickle=True)
    X_val = np.load(args.x_val, mmap_mode='r')
    Y_val = np.load(args.y_val, allow_pickle=True)

    print(f"Training Data Shape: {X_train.shape}") # Expected: (N, 115, 402)

    # 2. Split Targets (Multi-task)
    # Columns 0-4: Allele Identity (One-hot)
    # Column 5: Polymorphism Status (Binary)
    Y_train_allele = Y_train[:, :5].astype('float32')
    Y_train_poly = Y_train[:, 5].astype('float32')
    Y_val_allele = Y_val[:, :5].astype('float32')
    Y_val_poly = Y_val[:, 5].astype('float32')

    # 3. Build Model
    A = get_phylogenetic_adjacency() # From utils.py
    
    # Input shape: (115 species, 402 sequence length)
    # Note: 402 comes from (100+1+100) * 2
    input_shape = X_train.shape[1:] 
    
    model = build_graphylo_model(input_shape, A)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'allele_softmax': 'categorical_crossentropy',
            'binary_sigmoid': 'binary_crossentropy'
        },
        metrics={
            'allele_softmax': 'accuracy',
            'binary_sigmoid': 'accuracy'
        }
    )
    
    model.summary()

    # 4. Train
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, 'best_model.h5'), 
        save_best_only=True,
        monitor='val_loss'
    )

    history = model.fit(
        x=X_train,
        y=[Y_train_allele, Y_train_poly],
        validation_data=(X_val, [Y_val_allele, Y_val_poly]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint]
    )
    
    np.save(os.path.join(args.output_dir, 'history.npy'), history.history)
    print("Training Finished.")

if __name__ == "__main__":
    main()
