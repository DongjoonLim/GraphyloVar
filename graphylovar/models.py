# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
GraphyloVar model architectures with multitask capabilities.

Three model families, all sharing the same phylogenetic GCN tail:
    1. CNN-GCN      (model1D_siamese_se)      – siamese 1-D convolutions
    2. LSTM-GCN     (model_lstm_siamese_se)    – bidirectional LSTM with attention
    3. Transformer-GCN (transformer_gcn)       – embedding + multi-head self-attention

Each model:
    * accepts uint8 input of shape (batch, 115, seq_len)
    * one-hot encodes to 6 channels [A C G T N -]
    * splits into forward/reverse-complement halves (siamese)
    * applies species-attention (SE-style) and/or channel-attention
    * feeds through 2 GCN layers on the phylogenetic adjacency matrix
    * outputs 2-class softmax (conserved vs. mutated)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    AdditiveAttention,
    Bidirectional,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    LSTM,
    LayerNormalization,
    MaxPooling2D,
    MaxPooling3D,
    MultiHeadAttention,
    Permute,
    Reshape,
    TimeDistributed,
    multiply,
)
from tensorflow.keras.regularizers import l2
import numpy as np

from graphylovar.phylogeny import NUM_NODES


try:
    from keras import ops as Kops  # Keras 3 path
except Exception:  # pragma: no cover - TensorFlow-only fallback
    class _TensorflowOps:
        """Minimal keras.ops compatibility shim for TensorFlow-backed environments."""

        @staticmethod
        def arange(start=0, stop=None, step=1):
            if stop is None:
                return tf.range(start)
            return tf.range(start=start, limit=stop, delta=step)

        @staticmethod
        def reshape(x, shape):
            return tf.reshape(x, shape)

        @staticmethod
        def one_hot(x, num_classes):
            return tf.one_hot(x, num_classes)

        @staticmethod
        def flip(x, axis):
            return tf.reverse(x, axis=[axis] if isinstance(axis, int) else axis)

        @staticmethod
        def concatenate(values, axis=-1):
            return tf.concat(values, axis=axis)

        @staticmethod
        def cast(x, dtype):
            return tf.cast(x, dtype)

        @staticmethod
        def maximum(x, y):
            return tf.maximum(x, y)

        @staticmethod
        def exp(x):
            return tf.exp(x)

        @staticmethod
        def square(x):
            return tf.square(x)

        @staticmethod
        def sum(x, axis=None):
            return tf.reduce_sum(x, axis=axis)

        @staticmethod
        def multiply(x, y):
            return tf.multiply(x, y)

        @staticmethod
        def ones_like(x):
            return tf.ones_like(x)

        @staticmethod
        def shape(x):
            return tf.shape(x)

        @staticmethod
        def expand_dims(x, axis):
            return tf.expand_dims(x, axis=axis)

        @staticmethod
        def squeeze(x, axis=None):
            return tf.squeeze(x, axis=axis)

        @staticmethod
        def mean(x, axis=None, keepdims=False):
            return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

    Kops = _TensorflowOps()


# =====================================================================
# Attention modules
# =====================================================================


def _normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Return D^(-1/2) (A + I) D^(-1/2) for a fixed graph."""
    A = np.asarray(A, dtype=np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    degree = np.sum(A, axis=1)
    inv_sqrt_degree = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt_degree[degree <= 0] = 0.0
    return (A * inv_sqrt_degree[:, None]) * inv_sqrt_degree[None, :]


@tf.keras.utils.register_keras_serializable(package="graphylovar")
class FixedAdjacencyGraphConv(tf.keras.layers.Layer):
    """Simple graph convolution with a fixed normalized adjacency matrix."""

    def __init__(
        self,
        units: int,
        adjacency,
        activation=None,
        kernel_regularizer=None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self._adjacency_array = _normalize_adjacency(adjacency)
        self._adjacency = tf.constant(self._adjacency_array, dtype=tf.float32)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        x = tf.sparse.to_dense(inputs) if isinstance(inputs, tf.SparseTensor) else inputs
        compute_dtype = x.dtype
        adjacency = tf.cast(self._adjacency, compute_dtype)
        kernel = tf.cast(self.kernel, compute_dtype)
        x = tf.linalg.matmul(adjacency, x)
        x = tf.linalg.matmul(x, kernel)
        if self.bias is not None:
            x = x + tf.cast(self.bias, compute_dtype)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "adjacency": self._adjacency_array.tolist(),
                "activation": tf.keras.activations.serialize(self.activation),
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
                "use_bias": self.use_bias,
            }
        )
        return config

def species_attention(x, ratio: int = 2):
    """
    Squeeze-and-Excitation along the *species* axis (axis=1, channels-first).
    Learns which species channels are most informative.
    """
    residual = x
    try:
        se = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")(residual)
    except Exception:
        se = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(residual)
    se = Reshape((1, 1, residual.shape[1]))(se)
    se = Dense(residual.shape[1] // ratio, activation="relu", use_bias=False)(se)
    se = Dense(residual.shape[1] // ratio, activation="relu", use_bias=False)(se)
    se = Dense(residual.shape[1], activation="sigmoid", use_bias=False)(se)
    se = Permute((3, 1, 2))(se)
    return multiply([residual, se])


def channel_attention(x, ratio: int = 2):
    """
    Squeeze-and-Excitation along the *channel / feature* axis (last dim).
    """
    residual = x
    try:
        se = tf.keras.layers.GlobalAveragePooling2D()(residual)
    except Exception:
        se = tf.keras.layers.GlobalAveragePooling1D()(residual)
    se = Dense(residual.shape[-1] // ratio, activation="relu", use_bias=False)(se)
    se = Dense(residual.shape[-1] // ratio, activation="relu", use_bias=False)(se)
    se = Dense(residual.shape[-1], activation="sigmoid", use_bias=False)(se)
    return multiply([residual, se])


# =====================================================================
# Shared GCN tail
# =====================================================================

def _gcn_classifier(x, A: np.ndarray, gcn_units: int = 32,
                    dense_units: int = 64, num_classes: int = 2,
                    dropout: float = 0.3):
    """
    GCN → GCN → Flatten → Dense → output.
    Shared across all three model families.
    """
    x = Reshape((NUM_NODES, -1))(x)
    x = FixedAdjacencyGraphConv(gcn_units, A, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.5)(x)
    logits = Dense(num_classes, name="logits")(x)
    outputs = Activation("softmax", name="softmax")(logits)
    return outputs


# =====================================================================
# Transformer block (row-wise self-attention over species)
# =====================================================================

def _transformer_block(inputs, num_heads: int = 4, dff: int = 16,
                       num_layers: int = 6, max_seq_len: int = 402,
                       dropout_rate: float = 0.4, token_dim: int = 8,
                       vocab_size: int = 6):
    """
    Applies embedding + positional encoding + stacked transformer layers.
    """
    embedding = Embedding(input_dim=vocab_size, output_dim=token_dim, name="token_embedding")(inputs)
    embed_dim = max_seq_len * token_dim

    # Positional embeddings
    positions = Kops.arange(start=0, stop=max_seq_len, step=1)
    pos_emb = Embedding(input_dim=max_seq_len, output_dim=token_dim, name="position_embedding")(positions)
    pos_emb = Kops.reshape(pos_emb, [1, 1, max_seq_len, token_dim])

    embedding = embedding + pos_emb
    flattened = Kops.reshape(embedding, [-1, embedding.shape[1], embed_dim])

    for _ in range(num_layers):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=64)(flattened, flattened)
        attn = Dropout(rate=dropout_rate)(attn)
        attn = LayerNormalization(epsilon=1e-6)(flattened + attn)

        ff = Dense(units=dff, activation="relu")(attn)
        ff = Dense(units=embed_dim)(ff)
        ff = Dropout(rate=dropout_rate)(ff)
        flattened = LayerNormalization(epsilon=1e-6)(attn + ff)

    output = Kops.reshape(
        flattened,
        [-1, inputs.shape[1], inputs.shape[2], embedding.shape[-1]],
    )
    return output


# =====================================================================
# Model 1: CNN-GCN  (model1D_siamese_se)
# =====================================================================

def build_cnn_gcn(
    input_shape: tuple,
    A: np.ndarray,
    loss="binary_crossentropy",
    num_classes: int = 2,
) -> Model:
    """
    Siamese 1-D CNN with species-attention → GCN classifier.

    Parameters
    ----------
    input_shape : (115, seq_len)  –  e.g. (115, 42) for context21
    A           : adjacency matrix from phylogeny.build_graph()
    loss        : loss function or string
    """
    seq_len = input_shape[-1]
    half = seq_len // 2

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    conv1_l = TimeDistributed(Conv1D(32, 11, padding="same", activation="relu"))
    conv1_r = TimeDistributed(Conv1D(32, 11, padding="same", activation="relu"))
    x_left = conv1_l(x_left)
    x_left = channel_attention(x_left)
    x_right = conv1_r(x_right)
    x_right = Kops.flip(x_right, axis=2)

    x = Kops.concatenate([x_left, x_right], axis=-1)
    x = TimeDistributed(Conv1D(32, 11, padding="same", activation="relu"))(x)
    x = species_attention(x)
    x = Dropout(0.3)(x)

    outputs = _gcn_classifier(x, A, num_classes=num_classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model


# =====================================================================
# Model 1b: CNN-GCN v2 (variant-center-aware)
# =====================================================================

def _build_center_mask(half_len: int):
    """
    Build a binary mask (1.0 at the variant center, 0.0 elsewhere)
    that will be broadcast-concatenated as a 7th one-hot channel.

    The forward segment has 2*context_flank+1 positions with the variant
    at position context_flank (the center). We create a (1, 1, half, 1)
    mask tensor that TF will broadcast over (batch, species, half, 1).
    """
    mask = np.zeros((1, 1, half_len, 1), dtype=np.float32)
    center = half_len // 2
    mask[0, 0, center, 0] = 1.0
    return tf.constant(mask)


def _center_weighted_pool(x, half_len: int, sigma: float = 0.25):
    """
    Weighted average pooling along the sequence axis, with Gaussian
    weights centered on the variant position. This ensures that the
    model always gives the highest weight to the central variant while
    still extracting information from flanking positions.

    sigma is in fraction-of-half-len units (0.25 means the Gaussian
    std-dev is 1/4 of the half-length).
    """
    seq_len = Kops.cast(half_len, "int32")
    center = Kops.cast(seq_len // 2, "float32")
    positions = Kops.cast(Kops.arange(seq_len), "float32")
    sigma_abs = Kops.maximum(Kops.cast(sigma, "float32") * Kops.cast(seq_len, "float32"), 1.0)
    weights = Kops.exp(-0.5 * Kops.square((positions - center) / sigma_abs))
    weights = weights / Kops.sum(weights)
    weights = Kops.reshape(weights, (1, 1, -1, 1))
    weights = Kops.cast(weights, x.dtype)
    # x: (batch, species, seq, channels) -> weighted sum over seq
    return Kops.sum(Kops.multiply(x, weights), axis=2)  # (batch, species, channels)


def build_cnn_gcn_v2(
    input_shape: tuple,
    A: np.ndarray,
    loss="binary_crossentropy",
    num_classes: int = 2,
) -> Model:
    """
    Variant-center-aware CNN-GCN.

    Architectural improvements over v1 for better flank utilisation:
    1. A binary center-mask channel (7th) marks the variant position.
    2. Dual-branch encoder:
       - LOCAL: small-kernel Conv1D focused on the variant center
       - CONTEXT: dilated Conv1D stack to capture flanking conservation
    3. Center-weighted Gaussian pooling before the GCN tail.

    Parameters
    ----------
    input_shape : (115, seq_len)
    A           : adjacency matrix from phylogeny.build_graph()
    loss        : loss function or string
    """
    seq_len = input_shape[-1]
    half = seq_len // 2

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)  # (batch, 115, seq_len, 6)

    # Split into forward and reverse-complement halves
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    # -- Add variant-center mask channel -------------------------
    center_mask = _build_center_mask(half)
    # Broadcast: (1,1,half,1) over (batch, 115, half, 6)
    cm_left = Kops.ones_like(x_left[:, :, :, :1]) * center_mask
    cm_right = Kops.ones_like(x_right[:, :, :, :1]) * center_mask
    x_left = Kops.concatenate([x_left, Kops.cast(cm_left, "float32")], axis=-1)   # 7 channels
    x_right = Kops.concatenate([x_right, Kops.cast(cm_right, "float32")], axis=-1)

    # -- LOCAL BRANCH: focus on variant center -------------------
    local_conv = TimeDistributed(Conv1D(32, 3, padding="same", activation="relu"))
    local_l = local_conv(x_left)
    local_l = channel_attention(local_l)
    local_r = local_conv(x_right)
    local_r = Kops.flip(local_r, axis=2)
    local_out = Kops.concatenate([local_l, local_r], axis=-1)  # (batch, 115, half, 64)

    # -- CONTEXT BRANCH: dilated convolutions for flanking -------
    ctx_conv1 = TimeDistributed(Conv1D(16, 5, dilation_rate=1, padding="same", activation="relu"))
    ctx_conv2 = TimeDistributed(Conv1D(16, 5, dilation_rate=2, padding="same", activation="relu"))
    ctx_conv3 = TimeDistributed(Conv1D(16, 5, dilation_rate=4, padding="same", activation="relu"))

    ctx_l = ctx_conv3(ctx_conv2(ctx_conv1(x_left)))
    ctx_r = ctx_conv3(ctx_conv2(ctx_conv1(x_right)))
    ctx_r = Kops.flip(ctx_r, axis=2)
    ctx_out = Kops.concatenate([ctx_l, ctx_r], axis=-1)  # (batch, 115, half, 32)

    # -- MERGE local + context -----------------------------------
    merged = Kops.concatenate([local_out, ctx_out], axis=-1)  # (batch, 115, half, 96)
    merged = TimeDistributed(Conv1D(32, 1, activation="relu"))(merged)    # reduce channels

    # Species attention across the (species, seq, channels) tensor
    merged = species_attention(merged)
    merged = Dropout(0.3)(merged)

    # -- Center-weighted pooling => (batch, 115, 32) -------------
    pooled = _center_weighted_pool(merged, half)

    # -- GCN classifier tail -------------------------------------
    outputs = _gcn_classifier(pooled, A, num_classes=num_classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model


# =====================================================================
# Model 2: LSTM-GCN  (model_lstm_siamese_se)
# =====================================================================

def build_lstm_gcn(
    input_shape: tuple,
    A: np.ndarray,
    loss="binary_crossentropy",
    lstm_units: int = 4,
    num_classes: int = 2,
) -> Model:
    """
    Siamese Bidirectional-LSTM with multi-head attention → GCN classifier.

    Parameters
    ----------
    input_shape : (115, seq_len)
    A           : adjacency matrix
    loss        : loss function or string
    lstm_units  : units in each LSTM layer
    """
    seq_len = input_shape[-1]
    half = seq_len // 2
    n_species = input_shape[0]  # 115

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    # Flatten species × position for LSTM
    x_left = Reshape((n_species * half, 6))(x_left)
    x_right = Reshape((n_species * half, 6))(x_right)

    x_left = Bidirectional(LSTM(lstm_units, return_sequences=True))(x_left)
    x_left = Reshape((n_species, half, -1))(x_left)
    attn_l = MultiHeadAttention(num_heads=4, key_dim=4, attention_axes=(2,))
    x_left = attn_l(x_left, x_left)

    x_right = Bidirectional(LSTM(lstm_units, return_sequences=True))(x_right)
    x_right = Reshape((n_species, half, -1))(x_right)
    attn_r = MultiHeadAttention(num_heads=4, key_dim=4, attention_axes=(2,))
    x_right = attn_r(x_right, x_right)

    x = Kops.concatenate([x_left, x_right], axis=-1)
    x = Reshape((n_species * half, -1))(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Reshape((n_species, half, -1))(x)
    x = species_attention(x)
    x = Dropout(0.3)(x)

    outputs = _gcn_classifier(x, A, num_classes=num_classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model


# =====================================================================
# Model 3: Transformer-GCN
# =====================================================================

def build_transformer_gcn(
    input_shape: tuple,
    A: np.ndarray,
    loss="binary_crossentropy",
    num_heads: int = 4,
    dff: int = 16,
    num_layers: int = 6,
    num_classes: int = 2,
) -> Model:
    """
    Transformer encoder + species-attention → GCN classifier.

    Parameters
    ----------
    input_shape : (115, seq_len)
    A           : adjacency matrix
    loss        : loss function or string
    """
    seq_len = input_shape[-1]

    inputs = Input(shape=input_shape, dtype="uint8")
    x = _transformer_block(inputs, num_heads=num_heads, dff=dff,
                           num_layers=num_layers, max_seq_len=seq_len)
    x = species_attention(x)
    x = Dropout(0.3)(x)

    outputs = _gcn_classifier(x, A, num_classes=num_classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model


def build_multitask_transformer_gcn(
    input_shape: tuple,
    A: np.ndarray,
    nucleotide_loss="categorical_crossentropy",
    binary_loss="binary_crossentropy",
    binary_loss_weight: float = 1.0,
    num_heads: int = 4,
    dff: int = 16,
    num_layers: int = 6,
    gcn_units: int = 32,
    dense_units: int = 128,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Transformer-GCN with dual TOPMed pretraining heads.

    Outputs:
        * `nucleotide`: 5-way softmax over A/C/G/T/gap
        * `binary`:     sigmoid polymorphism probability
    """
    seq_len = input_shape[-1]

    inputs = Input(shape=input_shape, dtype="uint8")
    x = _transformer_block(
        inputs,
        num_heads=num_heads,
        dff=dff,
        num_layers=num_layers,
        max_seq_len=seq_len,
    )
    x = species_attention(x)
    x = Dropout(dropout)(x)

    x = Reshape((NUM_NODES, -1))(x)
    x = FixedAdjacencyGraphConv(gcn_units, A, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.4)(x)

    nucleotide_head = Dense(dense_units // 2, activation="relu")(x)
    nucleotide_head = Dropout(dropout)(nucleotide_head)
    nucleotide_output = Dense(5, activation="softmax", name="nucleotide")(nucleotide_head)

    binary_head = Dense(dense_units // 2, activation="relu")(x)
    binary_head = Dropout(dropout)(binary_head)
    binary_output = Dense(1, activation="sigmoid", name="binary")(binary_head)

    model = Model(inputs=inputs, outputs=[nucleotide_output, binary_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"nucleotide": nucleotide_loss, "binary": binary_loss},
        loss_weights={"nucleotide": 1.0, "binary": binary_loss_weight},
        metrics={"nucleotide": ["accuracy"], "binary": ["accuracy"]},
    )
    return model


def build_multitask_cnn_gcn_v2(
    input_shape: tuple,
    A: np.ndarray,
    nucleotide_loss="categorical_crossentropy",
    binary_loss="binary_crossentropy",
    binary_loss_weight: float = 1.0,
    gcn_units: int = 32,
    dense_units: int = 128,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Variant-center-aware CNN-GCN with dual TOPMed pretraining heads.

    This architecture explicitly separates local variant context from broader
    flanking context, making it a better fit for flank ablation than the
    original multitask transformer path.
    """
    seq_len = input_shape[-1]
    half = seq_len // 2

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    center_mask = _build_center_mask(half)
    cm_left = Kops.ones_like(x_left[:, :, :, :1]) * center_mask
    cm_right = Kops.ones_like(x_right[:, :, :, :1]) * center_mask
    x_left = Kops.concatenate([x_left, Kops.cast(cm_left, "float32")], axis=-1)
    x_right = Kops.concatenate([x_right, Kops.cast(cm_right, "float32")], axis=-1)

    local_conv = TimeDistributed(Conv1D(32, 3, padding="same", activation="relu"))
    local_l = channel_attention(local_conv(x_left))
    local_r = Kops.flip(local_conv(x_right), axis=2)
    local_out = Kops.concatenate([local_l, local_r], axis=-1)

    ctx_conv1 = TimeDistributed(Conv1D(16, 5, dilation_rate=1, padding="same", activation="relu"))
    ctx_conv2 = TimeDistributed(Conv1D(16, 5, dilation_rate=2, padding="same", activation="relu"))
    ctx_conv3 = TimeDistributed(Conv1D(16, 5, dilation_rate=4, padding="same", activation="relu"))
    ctx_l = ctx_conv3(ctx_conv2(ctx_conv1(x_left)))
    ctx_r = Kops.flip(ctx_conv3(ctx_conv2(ctx_conv1(x_right))), axis=2)
    ctx_out = Kops.concatenate([ctx_l, ctx_r], axis=-1)

    merged = Kops.concatenate([local_out, ctx_out], axis=-1)
    merged = TimeDistributed(Conv1D(32, 1, activation="relu"))(merged)
    merged = species_attention(merged)
    merged = Dropout(dropout)(merged)

    x = _center_weighted_pool(merged, half)
    x = FixedAdjacencyGraphConv(gcn_units, A, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.4)(x)

    nucleotide_head = Dense(dense_units // 2, activation="relu")(x)
    nucleotide_head = Dropout(dropout)(nucleotide_head)
    nucleotide_output = Dense(5, activation="softmax", name="nucleotide")(nucleotide_head)

    binary_head = Dense(dense_units // 2, activation="relu")(x)
    binary_head = Dropout(dropout)(binary_head)
    binary_output = Dense(1, activation="sigmoid", name="binary")(binary_head)

    model = Model(inputs=inputs, outputs=[nucleotide_output, binary_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"nucleotide": nucleotide_loss, "binary": binary_loss},
        loss_weights={"nucleotide": 1.0, "binary": binary_loss_weight},
        metrics={"nucleotide": ["accuracy"], "binary": ["accuracy"]},
    )
    return model


# =====================================================================
# Model 4b: Hybrid v3 — self-attention + center extraction + GCN
# =====================================================================

def _sinusoidal_positional_encoding(seq_len: int, d_model: int):
    """Create sinusoidal positional encoding matrix (1, seq_len, d_model)."""
    positions = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    dims = np.arange(d_model, dtype=np.float32)[np.newaxis, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, seq, d)


def build_multitask_hybrid_v3(
    input_shape: tuple,
    A: np.ndarray,
    nucleotide_loss="categorical_crossentropy",
    binary_loss="binary_crossentropy",
    binary_loss_weight: float = 1.0,
    gcn_units: int = 32,
    dense_units: int = 128,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    num_attn_heads: int = 4,
    attn_key_dim: int = 8,
    num_attn_layers: int = 2,
    embed_dim: int = 32,
) -> Model:
    """
    Hybrid v3: Self-attention over positions + direct center extraction + GCN.

    Fixes v2's flank-scaling problem by replacing limited-receptive-field
    CNNs with self-attention over sequence positions. Every flanking
    position can attend to every other position, so longer flanks provide
    richer context instead of adding noise.

    Key design choices:
    1. Conv1D(embed_dim, 3) embeds local nucleotide patterns
    2. Sinusoidal positional encoding preserves position information
    3. N layers of MultiHeadAttention let distant positions interact
    4. Center position is EXTRACTED directly (not averaged)
       → flanking info flows to center via attention, then we grab center
    5. Forward + reverse center features concatenated → species attention → GCN
    """
    seq_len = input_shape[-1]
    half = seq_len // 2
    center_idx = half // 2  # center position in each half

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)  # (batch, 115, seq_len, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    # --- Shared embedding: Conv1D projects one-hot to embed_dim ---
    shared_embed = Conv1D(embed_dim, 3, padding="same", activation="gelu")
    shared_norm = LayerNormalization(epsilon=1e-6)

    # --- Positional encoding (precomputed, shared) ---
    pos_enc = _sinusoidal_positional_encoding(half, embed_dim)

    def encode_half(x_half):
        """Embed + positional encode + self-attention + extract center."""
        # x_half: (batch, 115, half, 6)
        # Process per-species: reshape to merge batch and species
        batch_shape = Kops.shape(x_half)
        n_batch = batch_shape[0]
        n_species = NUM_NODES  # 115

        # (batch * 115, half, 6)
        x_flat = Kops.reshape(x_half, [-1, half, 6])

        # Embed: (batch*115, half, embed_dim)
        x_emb = shared_embed(x_flat)
        x_emb = shared_norm(x_emb)

        # Add positional encoding (cast to match dtype for mixed-precision)
        x_emb = x_emb + tf.cast(pos_enc, x_emb.dtype)

        # Self-attention layers over positions
        for _ in range(num_attn_layers):
            # Multi-head self-attention
            attn_out = MultiHeadAttention(
                num_heads=num_attn_heads,
                key_dim=attn_key_dim,
                dropout=dropout,
            )(x_emb, x_emb)
            x_emb = LayerNormalization(epsilon=1e-6)(x_emb + attn_out)

            # Feed-forward
            ff = Dense(embed_dim * 2, activation="gelu")(x_emb)
            ff = Dense(embed_dim)(ff)
            ff = Dropout(dropout)(ff)
            x_emb = LayerNormalization(epsilon=1e-6)(x_emb + ff)

        # Extract center position: (batch*115, embed_dim)
        center_features = x_emb[:, center_idx, :]

        # Reshape back: (batch, 115, embed_dim)
        center_features = Kops.reshape(center_features, [n_batch, n_species, embed_dim])
        return center_features

    left_center = encode_half(x_left)
    right_center = encode_half(Kops.flip(x_right, axis=2))  # reverse complement

    # Concat forward + reverse center features: (batch, 115, 2*embed_dim)
    merged = Kops.concatenate([left_center, right_center], axis=-1)

    # Species attention (SE block)
    # Reshape to 4D for the existing species_attention function
    merged = Kops.expand_dims(merged, axis=2)  # (batch, 115, 1, 2*embed_dim)
    merged = species_attention(merged)
    merged = Dropout(dropout)(merged)
    merged = Kops.squeeze(merged, axis=2)  # (batch, 115, 2*embed_dim)

    # --- GCN on phylogenetic tree ---
    x = FixedAdjacencyGraphConv(gcn_units, A, activation="relu")(merged)
    x = Dropout(dropout)(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.4)(x)

    # --- Dual prediction heads ---
    nucleotide_head = Dense(dense_units // 2, activation="relu")(x)
    nucleotide_head = Dropout(dropout)(nucleotide_head)
    nucleotide_output = Dense(5, activation="softmax", name="nucleotide")(nucleotide_head)

    binary_head = Dense(dense_units // 2, activation="relu")(x)
    binary_head = Dropout(dropout)(binary_head)
    binary_output = Dense(1, activation="sigmoid", name="binary")(binary_head)

    model = Model(inputs=inputs, outputs=[nucleotide_output, binary_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"nucleotide": nucleotide_loss, "binary": binary_loss},
        loss_weights={"nucleotide": 1.0, "binary": binary_loss_weight},
        metrics={"nucleotide": ["accuracy"], "binary": ["accuracy"]},
    )
    return model


def build_multitask_hybrid_v4(
    input_shape: tuple,
    A: np.ndarray,
    nucleotide_loss="categorical_crossentropy",
    binary_loss="binary_crossentropy",
    binary_loss_weight: float = 1.0,
    gcn_units: int = 32,
    dense_units: int = 128,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    num_attn_heads: int = 4,
    attn_key_dim: int = 16,
    num_attn_layers: int = 4,
    embed_dim: int = 64,
) -> Model:
    """v4: Higher-capacity Transformer for wide-context (flank>=32) models.

    Identical structure to v3 but with doubled embed_dim (64), doubled attention
    layers (4), and doubled key_dim (16), giving 4x the center-position capacity
    for aggregating signal from longer flanking sequences.
    """
    seq_len = input_shape[-1]
    half = seq_len // 2
    center_idx = half // 2

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    shared_embed = Conv1D(embed_dim, 3, padding="same", activation="gelu")
    shared_norm = LayerNormalization(epsilon=1e-6)

    pos_enc = _sinusoidal_positional_encoding(half, embed_dim)

    def encode_half(x_half):
        batch_shape = Kops.shape(x_half)
        n_batch = batch_shape[0]
        n_species = NUM_NODES

        x_flat = Kops.reshape(x_half, [-1, half, 6])
        x_emb = shared_embed(x_flat)
        x_emb = shared_norm(x_emb)
        x_emb = x_emb + tf.cast(pos_enc, x_emb.dtype)

        for _ in range(num_attn_layers):
            attn_out = MultiHeadAttention(
                num_heads=num_attn_heads,
                key_dim=attn_key_dim,
                dropout=dropout,
            )(x_emb, x_emb)
            x_emb = LayerNormalization(epsilon=1e-6)(x_emb + attn_out)

            ff = Dense(embed_dim * 2, activation="gelu")(x_emb)
            ff = Dense(embed_dim)(ff)
            ff = Dropout(dropout)(ff)
            x_emb = LayerNormalization(epsilon=1e-6)(x_emb + ff)

        center_features = x_emb[:, center_idx, :]
        center_features = Kops.reshape(center_features, [n_batch, n_species, embed_dim])
        return center_features

    left_center = encode_half(x_left)
    right_center = encode_half(Kops.flip(x_right, axis=2))

    merged = Kops.concatenate([left_center, right_center], axis=-1)

    merged = Kops.expand_dims(merged, axis=2)
    merged = species_attention(merged)
    merged = Dropout(dropout)(merged)
    merged = Kops.squeeze(merged, axis=2)

    x = FixedAdjacencyGraphConv(gcn_units, A, activation="relu")(merged)
    x = Dropout(dropout)(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.4)(x)

    nucleotide_head = Dense(dense_units // 2, activation="relu")(x)
    nucleotide_head = Dropout(dropout)(nucleotide_head)
    nucleotide_output = Dense(5, activation="softmax", name="nucleotide")(nucleotide_head)

    binary_head = Dense(dense_units // 2, activation="relu")(x)
    binary_head = Dropout(dropout)(binary_head)
    binary_output = Dense(1, activation="sigmoid", name="binary")(binary_head)

    model = Model(inputs=inputs, outputs=[nucleotide_output, binary_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"nucleotide": nucleotide_loss, "binary": binary_loss},
        loss_weights={"nucleotide": 1.0, "binary": binary_loss_weight},
        metrics={"nucleotide": ["accuracy"], "binary": ["accuracy"]},
    )
    return model


# =====================================================================
# EvoLSTM baseline (single-species, no graph)
# =====================================================================

def build_evolstm_baseline(
    input_length: int,
    loss="binary_crossentropy",
    lstm_units: int = 64,
    num_classes: int = 2,
) -> Model:
    """
    Simple BiLSTM + Attention baseline operating on a single species row.

    Parameters
    ----------
    input_length : sequence length (e.g. 42 for context21)
    """
    inputs = Input(shape=(input_length,), dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = tf.keras.layers.Attention()([x, x])
    x = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="sgd", metrics=["accuracy"])
    return model


# =====================================================================
# Spatial attention (used by Conv2D models)
# =====================================================================

def spatial_attention(x, kernel_size: int = 7):
    """
    Spatial attention via average pooling and Conv2D.
    Learns which spatial positions (sequence positions) are most informative.
    """
    avg_pool = Kops.mean(x, axis=1, keepdims=True)
    spatial = Conv2D(
        1, (kernel_size, kernel_size), strides=1,
        activation="sigmoid", padding="same", data_format="channels_first",
    )(avg_pool)
    return multiply([x, spatial])


# =====================================================================
# Model 5: Conv2D-GCN (2D convolutions on species × position)
# =====================================================================

def build_conv2d_gcn(
    input_shape: tuple,
    A: np.ndarray,
    loss="binary_crossentropy",
    num_classes: int = 2,
    gcn_units: int = 32,
    dense_units: int = 64,
) -> Model:
    """
    Siamese 2-D CNN treating (species, position, one-hot) as an image.

    From ``conservation/train_graphylo_siamese.py``:
    ``model_conv3d_siamese`` / ``model_conv3d_bahdanau_onehot_human``.

    Parameters
    ----------
    input_shape : (115, seq_len)
    A           : adjacency matrix
    loss        : loss function
    """
    seq_len = input_shape[-1]
    half = seq_len // 2

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    shared_conv = Conv2D(
        NUM_NODES, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )
    x_left = shared_conv(x_left)
    x_right = shared_conv(x_right)
    x_right = Kops.flip(x_right, axis=2)
    x = Kops.concatenate([x_left, x_right], axis=-1)

    x = MaxPooling2D((2, 2), data_format="channels_first")(x)
    x = Conv2D(
        NUM_NODES, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), data_format="channels_first")(x)

    x = Reshape((x.shape[1], -1))(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(0.3)(x)
    x = FixedAdjacencyGraphConv(
        gcn_units, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.5)(x)
    logits = Dense(num_classes, name="logits")(x)
    outputs = Activation("softmax", name="softmax")(logits)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model


# =====================================================================
# Model 6: Bahdanau attention (additive attention on Conv2D features)
# =====================================================================

def build_bahdanau_gcn(
    input_shape: tuple,
    A: np.ndarray,
    loss="binary_crossentropy",
    num_classes: int = 2,
    dense_units: int = 256,
) -> Model:
    """
    Conv2D encoder with Bahdanau (additive) attention → GCN classifier.

    From ``conservation/train_graphylo_siamese.py``:
    ``model_conv3d_bahdanau_onehot_human``.

    The query and value branches share Conv2D weights within each half
    (forward / reverse complement), then additive attention is applied.

    Parameters
    ----------
    input_shape : (115, seq_len)
    A           : adjacency matrix
    loss        : loss function
    """
    seq_len = input_shape[-1]
    half = seq_len // 2

    inputs = Input(shape=input_shape, dtype="uint8")
    x = Kops.one_hot(inputs, 6)
    x_left = x[:, :, :half, :]
    x_right = x[:, :, half:, :]

    shared_q = Conv2D(
        32, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )
    shared_v = Conv2D(
        32, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )

    # Query path
    x_q_l = Dropout(0.3)(shared_q(x_left))
    x_q_r = Dropout(0.3)(shared_q(x_right))
    x_q_r = Kops.flip(x_q_r, axis=2)
    x_q = Kops.concatenate([x_q_l, x_q_r], axis=-1)
    x_q = MaxPooling2D((2, 2), data_format="channels_first")(x_q)

    # Value path
    x_v_l = Dropout(0.3)(shared_v(x_left))
    x_v_r = Dropout(0.3)(shared_v(x_right))
    x_v_r = Kops.flip(x_v_r, axis=2)
    x_v = Kops.concatenate([x_v_l, x_v_r], axis=-1)
    x_v = MaxPooling2D((2, 2), data_format="channels_first")(x_v)

    # Additive attention
    x_attn = AdditiveAttention()([x_q, x_v])
    x = Concatenate()([x_q, x_attn])

    x = Conv2D(
        32, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), data_format="channels_first")(x)

    # GCN tail
    x = Reshape((x.shape[1], -1))(x)
    x = FixedAdjacencyGraphConv(
        32, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = FixedAdjacencyGraphConv(
        32, A, activation="relu", kernel_regularizer=l2(5e-4)
    )(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.5)(x)
    logits = Dense(num_classes, name="logits")(x)
    outputs = Activation("softmax", name="softmax")(logits)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model


# =====================================================================
# Factory
# =====================================================================

_REGISTRY = {
    "cnn_gcn": build_cnn_gcn,
    "cnn_gcn_v2": build_cnn_gcn_v2,
    "lstm_gcn": build_lstm_gcn,
    "transformer_gcn": build_transformer_gcn,
    "multitask_transformer_gcn": build_multitask_transformer_gcn,
    "multitask_cnn_gcn_v2": build_multitask_cnn_gcn_v2,
    "multitask_hybrid_v3": build_multitask_hybrid_v3,
    "multitask_hybrid_v4": build_multitask_hybrid_v4,
    "evolstm": build_evolstm_baseline,
    "conv2d_gcn": build_conv2d_gcn,
    "bahdanau_gcn": build_bahdanau_gcn,
}


def build_model(name: str, **kwargs) -> Model:
    """
    Build a model by name.

    Parameters
    ----------
    name : one of "cnn_gcn", "lstm_gcn", "transformer_gcn", "evolstm"
    **kwargs : forwarded to the builder function.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
