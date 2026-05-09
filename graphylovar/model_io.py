# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Helpers for model save/load paths and prediction output normalization.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MODEL_FILE_EXTENSIONS = (".keras", ".h5", ".hdf5")


def normalize_model_path(path: str) -> str:
    """
    Ensure model save paths use an explicit file extension.

    Modern Keras expects `.keras` or `.h5` for full-model checkpoints.
    """
    path = os.fspath(path)
    if path.endswith(MODEL_FILE_EXTENSIONS):
        return path
    return f"{path}.keras"


def resolve_model_path(path: str) -> str:
    """
    Resolve a user-supplied model path, accepting either bare names or files.
    """
    path = os.fspath(path)
    candidates = [path]
    normalized = normalize_model_path(path)
    if normalized not in candidates:
        candidates.append(normalized)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return normalized if path == normalized[:-6] else path


def _hdf5_attr_as_text(value: Any, default: str | None = None) -> str | None:
    """Return an HDF5 attribute as text across old/new h5py conventions."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _hdf5_attr_list_as_text(group: Any, name: str) -> list[str]:
    """Return an HDF5 attribute list as text across old/new h5py conventions."""
    if name not in group.attrs:
        return []
    values = group.attrs[name]
    if isinstance(values, np.ndarray):
        values = values.tolist()
    elif not isinstance(values, (list, tuple)):
        values = [values]
    return [_hdf5_attr_as_text(value, "") or "" for value in values]


def load_hdf5_weights_compat(model: Any, path: str) -> None:
    """
    Load weights from a full-model HDF5 file even when attrs are stored as `str`.

    Some older TensorFlow stacks expect HDF5 attrs like `keras_version` and
    `backend` to be bytes. Newer h5py may surface them as Python strings, which
    breaks the stock loader with `'str' object has no attribute 'decode'`.
    """
    import h5py
    from tensorflow.python.keras.saving import hdf5_format

    with h5py.File(path, mode="r") as handle:
        group = handle["model_weights"] if "model_weights" in handle else handle
        original_keras_version = _hdf5_attr_as_text(group.attrs.get("keras_version"), "1")
        original_backend = _hdf5_attr_as_text(group.attrs.get("backend"))

        filtered_layers = []
        for layer in model.layers:
            weights = hdf5_format._legacy_weights(layer)
            if weights:
                filtered_layers.append(layer)

        layer_names = _hdf5_attr_list_as_text(group, "layer_names")
        filtered_layer_names = []
        for name in layer_names:
            layer_group = group[name]
            weight_names = _hdf5_attr_list_as_text(layer_group, "weight_names")
            if weight_names:
                filtered_layer_names.append(name)

        layer_names = filtered_layer_names
        if len(layer_names) != len(filtered_layers):
            raise ValueError(
                "You are trying to load a weight file containing "
                f"{len(layer_names)} layers into a model with {len(filtered_layers)} layers."
            )

        weight_value_tuples = []
        for index, name in enumerate(layer_names):
            layer_group = group[name]
            weight_names = _hdf5_attr_list_as_text(layer_group, "weight_names")
            weight_values = [np.asarray(layer_group[weight_name]) for weight_name in weight_names]
            layer = filtered_layers[index]
            symbolic_weights = hdf5_format._legacy_weights(layer)
            weight_values = hdf5_format.preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend,
            )
            if len(weight_values) != len(symbolic_weights):
                raise ValueError(
                    f'Layer #{index} (named "{layer.name}" in the current model) '
                    f"expects {len(symbolic_weights)} weights, but the saved weights have "
                    f"{len(weight_values)} elements."
                )
            weight_value_tuples.extend(zip(symbolic_weights, weight_values))

        hdf5_format.K.batch_set_value(weight_value_tuples)


def extract_binary_scores(predictions: Any) -> np.ndarray:
    """
    Convert model outputs into a 1-D variant score array.

    Preference order:
        1. `binary` head from dict outputs
        2. second item from list/tuple outputs
        3. direct array output

    For 2-class softmax outputs, the positive-class column is returned.
    """
    if isinstance(predictions, Mapping):
        if "binary" in predictions:
            predictions = predictions["binary"]
        else:
            predictions = next(iter(predictions.values()))
    elif isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes, np.ndarray)):
        if not predictions:
            raise ValueError("Model returned an empty output sequence")
        predictions = predictions[1] if len(predictions) > 1 else predictions[0]

    scores = np.asarray(predictions)
    if scores.ndim == 2 and scores.shape[1] == 2:
        return scores[:, 1]
    if scores.ndim == 2 and scores.shape[1] == 1:
        return scores[:, 0]
    return scores


def describe_prediction_output(predictions: Any) -> str:
    """Return a compact description of prediction output shapes."""
    if isinstance(predictions, Mapping):
        parts = [f"{key}={np.asarray(value).shape}" for key, value in predictions.items()]
        return ", ".join(parts)
    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes, np.ndarray)):
        return ", ".join(str(np.asarray(value).shape) for value in predictions)
    return str(np.asarray(predictions).shape)
