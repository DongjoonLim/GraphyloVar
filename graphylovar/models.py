"""
GraphyloVar model architectures.

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
    multiply,
)
from tensorflow.keras.regularizers import l2
from spektral.layers import GCNConv

import numpy as np

from graphylovar.phylogeny import NUM_NODES


# =====================================================================
# Attention modules
# =====================================================================

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
    x = GCNConv(gcn_units, activation="relu")([x, A])
    x = Dropout(dropout)(x)
    x = GCNConv(gcn_units, activation="relu", kernel_regularizer=l2(5e-4))([x, A])
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
                       dropout_rate: float = 0.4):
    """
    Applies embedding + positional encoding + stacked transformer layers.
    """
    embedding = Embedding(input_dim=max_seq_len, output_dim=6)(inputs)
    embed_dim = embedding.shape[-2] * embedding.shape[-1]

    # Positional embeddings
    positions = tf.range(start=0, limit=max_seq_len, delta=1)
    pos_emb = Embedding(input_dim=max_seq_len, output_dim=6)(positions)
    pos_emb = tf.repeat(pos_emb, repeats=[NUM_NODES], axis=0)
    pos_emb = tf.reshape(pos_emb, [-1, embedding.shape[1], embed_dim])

    flattened = tf.reshape(embedding, [-1, embedding.shape[1], embed_dim])
    flattened = tf.add(flattened, pos_emb, name="add_position_embeddings")

    for _ in range(num_layers):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=64)(flattened, flattened)
        attn = Dropout(rate=dropout_rate)(attn)
        attn = LayerNormalization(epsilon=1e-6)(flattened + attn)

        ff = Dense(units=dff, activation="relu")(attn)
        ff = Dense(units=embed_dim)(ff)
        ff = Dropout(rate=dropout_rate)(ff)
        flattened = LayerNormalization(epsilon=1e-6)(attn + ff)

    output = tf.reshape(flattened,
                        [-1, inputs.shape[1], inputs.shape[2], embedding.shape[-1]])
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
    x = tf.one_hot(inputs, 6)
    x_left, x_right = tf.split(x, [half, half], axis=2)

    conv1_l = Conv1D(32, 11, padding="same", activation="relu")
    conv1_r = Conv1D(32, 11, padding="same", activation="relu")
    x_left = conv1_l(x_left)
    x_left = channel_attention(x_left)
    x_right = conv1_r(x_right)
    x_right = tf.reverse(x_right, [2])

    x = tf.concat([x_left, x_right], axis=-1)
    x = Conv1D(32, 11, padding="same", activation="relu")(x)
    x = species_attention(x)
    x = Dropout(0.3)(x)

    outputs = _gcn_classifier(x, A, num_classes=num_classes)
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
    x = tf.one_hot(inputs, 6)
    x_left, x_right = tf.split(x, [half, half], axis=2)

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

    x = tf.concat([x_left, x_right], axis=-1)
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
    x = tf.one_hot(inputs, 6)
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
    avg_pool = tf.reduce_mean(x, axis=1, keepdims=True)
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
    x = tf.one_hot(inputs, 6)
    x_left, x_right = tf.split(x, [half, half], axis=2)

    shared_conv = Conv2D(
        NUM_NODES, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )
    x_left = shared_conv(x_left)
    x_right = shared_conv(x_right)
    x_right = tf.reverse(x_right, [2])
    x = tf.concat([x_left, x_right], axis=-1)

    x = MaxPooling2D((2, 2), data_format="channels_first")(x)
    x = Conv2D(
        NUM_NODES, (10, 6), padding="same", activation="relu",
        data_format="channels_first",
    )(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), data_format="channels_first")(x)

    x = Reshape((x.shape[1], -1))(x)
    x = GCNConv(gcn_units, activation="relu", kernel_regularizer=l2(5e-4))([x, A])
    x = Dropout(0.3)(x)
    x = GCNConv(gcn_units, activation="relu", kernel_regularizer=l2(5e-4))([x, A])
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
    x = tf.one_hot(inputs, 6)
    x_left, x_right = tf.split(x, [half, half], axis=2)

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
    x_q_r = tf.reverse(x_q_r, [2])
    x_q = tf.concat([x_q_l, x_q_r], axis=-1)
    x_q = MaxPooling2D((2, 2), data_format="channels_first")(x_q)

    # Value path
    x_v_l = Dropout(0.3)(shared_v(x_left))
    x_v_r = Dropout(0.3)(shared_v(x_right))
    x_v_r = tf.reverse(x_v_r, [2])
    x_v = tf.concat([x_v_l, x_v_r], axis=-1)
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
    x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x, A])
    x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x, A])
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
    "lstm_gcn": build_lstm_gcn,
    "transformer_gcn": build_transformer_gcn,
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
