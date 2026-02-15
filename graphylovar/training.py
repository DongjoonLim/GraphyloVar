"""
Training utilities for GraphyloVar models.

Provides:
    - train_model : full training loop with callbacks, plotting, and saving
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for servers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: str,
    epochs: int = 9999,
    batch_size: int = 64,
    patience: int = 7,
    plot: bool = True,
    plot_dir: Optional[str] = None,
) -> tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    Train a compiled model with early stopping and checkpointing.

    Parameters
    ----------
    model      : compiled Keras model
    X_train    : training features
    y_train    : training labels (one-hot, shape (N,2))
    X_val      : validation features
    y_val      : validation labels
    save_path  : where to save the best checkpoint (no extension needed)
    epochs     : maximum number of epochs
    batch_size : mini-batch size
    patience   : early-stopping patience
    plot       : whether to save a loss curve
    plot_dir   : directory for loss plot; defaults to same dir as save_path

    Returns
    -------
    (history, model) — the History object and the best-weights model
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_loss", verbose=1, save_best_only=True
        ),
    ]

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
    )

    # ── Loss curve ──────────────────────────────────────────────────
    if plot:
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"], label="train")
        ax.plot(history.history["val_loss"], label="val")
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper left")

        if plot_dir is None:
            plot_dir = os.path.dirname(save_path) or "."
        os.makedirs(plot_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(save_path))[0]
        fig.savefig(os.path.join(plot_dir, f"{base}_loss.png"), dpi=150)
        plt.close(fig)

    return history, model
