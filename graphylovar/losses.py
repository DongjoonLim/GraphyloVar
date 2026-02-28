# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Loss functions for GraphyloVar training.

Provides:
    - binary_focal_loss : focal loss for class-imbalanced binary classification
    - get_loss          : factory to get loss by name
"""

from __future__ import annotations

import tensorflow as tf


class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for binary / two-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the contribution of easy (well-classified) examples so the
    model concentrates on hard negatives, which is critical for the
    imbalanced conserved-vs-mutated labels in CADD-style data.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default 2.0). Higher = more focus on hard examples.
    alpha : float or None
        Balancing factor. If None, defaults to 0.25.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1.0 - tf.keras.backend.epsilon())
        # For two-class softmax output, take the positive-class column
        if y_pred.shape[-1] == 2:
            p_t = y_true[:, 1] * y_pred[:, 1] + y_true[:, 0] * y_pred[:, 0]
        else:
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        alpha_t = y_true[:, 1] * self.alpha + y_true[:, 0] * (1 - self.alpha) \
            if y_pred.shape[-1] == 2 else \
            y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        loss = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


def get_loss(name: str = "focal", **kwargs):
    """
    Return a loss function by name.

    Parameters
    ----------
    name : "focal", "bce", or "cce"
    """
    if name == "focal":
        gamma = kwargs.get("gamma", 2.0)
        alpha = kwargs.get("alpha", 0.25)
        return BinaryFocalLoss(gamma=gamma, alpha=alpha)
    if name == "bce":
        return "binary_crossentropy"
    if name == "cce":
        return "categorical_crossentropy"
    raise ValueError(f"Unknown loss '{name}'. Choose from: focal, bce, cce")
