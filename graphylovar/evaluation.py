"""
Evaluation utilities for GraphyloVar models.

Provides:
    - ROC / AUC computation and plotting
    - Precision-Recall / AUPRC computation and plotting
    - Probability calibration for imbalanced datasets
    - Variant scoring pipeline (TOPMed allele frequency-based evaluation)
    - Side-by-side comparison with baselines (EvoLSTM, etc.)
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ─── Probability calibration ────────────────────────────────────────

def calibrate(pred_prob: float | np.ndarray, under_ratio: float) -> float | np.ndarray:
    """
    Apply probability calibration to adjust for class imbalance.

    When training data is under-sampled for one class, raw predicted
    probabilities are biased.  This applies the standard correction:

    .. math::

        p_{cal} = \\frac{p \\cdot r}{p \\cdot r - p + 1}

    Parameters
    ----------
    pred_prob   : raw predicted probability (or array)
    under_ratio : ratio of minority class in reality vs. in training set

    Returns
    -------
    Calibrated probability.
    """
    return (pred_prob * under_ratio) / (pred_prob * under_ratio - pred_prob + 1)


# ─── ROC ─────────────────────────────────────────────────────────────

def compute_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Compute ROC curve and AUC.

    Parameters
    ----------
    y_true  : binary ground-truth labels (N,) or one-hot (N, 2)
    y_score : predicted probabilities for positive class (N,)

    Returns
    -------
    dict with keys: fpr, tpr, thresholds, auc
    """
    if y_true.ndim == 2:
        y_true = y_true[:, 1]
    if y_score.ndim == 2:
        y_score = y_score[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    area = roc_auc_score(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": area}


def compute_prc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Compute Precision-Recall curve and AUPRC.

    Parameters
    ----------
    y_true  : binary ground-truth labels (N,) or one-hot (N, 2)
    y_score : predicted probabilities for positive class (N,)

    Returns
    -------
    dict with keys: precision, recall, thresholds, auprc
    """
    if y_true.ndim == 2:
        y_true = y_true[:, 1]
    if y_score.ndim == 2:
        y_score = y_score[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    area = auc(recall, precision)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auprc": area,
    }


# ─── Plotting ────────────────────────────────────────────────────────

def plot_roc(
    results: dict | list[dict],
    names: list[str] | None = None,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """
    Plot one or more ROC curves.

    Parameters
    ----------
    results   : single dict from ``compute_roc`` or list of dicts
    names     : legend labels (one per result)
    save_path : if given, save the figure to this path
    title     : plot title

    Returns
    -------
    matplotlib Figure
    """
    if isinstance(results, dict):
        results = [results]
    if names is None:
        names = [f"model_{i}" for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(7, 6))
    for res, name in zip(results, names):
        ax.plot(res["fpr"], res["tpr"], lw=2, label=f'{name} (AUC={res["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prc(
    results: dict | list[dict],
    names: list[str] | None = None,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """
    Plot one or more precision-recall curves.

    Parameters
    ----------
    results   : single dict from ``compute_prc`` or list of dicts
    names     : legend labels
    save_path : if given, save the figure
    title     : plot title

    Returns
    -------
    matplotlib Figure
    """
    if isinstance(results, dict):
        results = [results]
    if names is None:
        names = [f"model_{i}" for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(7, 6))
    for res, name in zip(results, names):
        ax.plot(
            res["recall"], res["precision"], lw=2,
            label=f'{name} (AUPRC={res["auprc"]:.3f})',
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── High-level evaluation ──────────────────────────────────────────

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "GraphyloVar",
    batch_size: int = 256,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Full evaluation: predict, compute ROC + PRC, optionally save plots.

    Parameters
    ----------
    model      : trained Keras model
    X_test     : test features
    y_test     : test labels (one-hot or binary)
    model_name : label for plots
    batch_size : prediction batch size
    save_dir   : directory to save ROC/PRC plots

    Returns
    -------
    dict with keys: auroc, auprc, roc, prc, predictions
    """
    preds = model.predict(X_test, batch_size=batch_size)

    roc_result = compute_roc(y_test, preds)
    prc_result = compute_prc(y_test, preds)

    print(f"[{model_name}]  AUROC: {roc_result['auc']:.4f}  |  AUPRC: {prc_result['auprc']:.4f}")

    if save_dir:
        plot_roc(roc_result, names=[model_name],
                 save_path=os.path.join(save_dir, f"roc_{model_name}.png"))
        plot_prc(prc_result, names=[model_name],
                 save_path=os.path.join(save_dir, f"prc_{model_name}.png"))

    return {
        "auroc": roc_result["auc"],
        "auprc": prc_result["auprc"],
        "roc": roc_result,
        "prc": prc_result,
        "predictions": preds,
    }


def compare_models(
    models: dict[str, tf.keras.Model],
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    save_dir: Optional[str] = None,
) -> dict[str, dict]:
    """
    Evaluate and compare multiple models on the same test data.

    Parameters
    ----------
    models    : dict mapping model_name → compiled Keras model
    X_test    : test features
    y_test    : test labels
    batch_size : prediction batch size
    save_dir  : directory for combined ROC/PRC plots

    Returns
    -------
    dict mapping model_name → evaluation dict
    """
    all_results = {}
    roc_list, prc_list, name_list = [], [], []

    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test, model_name=name,
                             batch_size=batch_size)
        all_results[name] = res
        roc_list.append(res["roc"])
        prc_list.append(res["prc"])
        name_list.append(name)

    if save_dir:
        plot_roc(roc_list, names=name_list,
                 save_path=os.path.join(save_dir, "roc_comparison.png"),
                 title="ROC Comparison")
        plot_prc(prc_list, names=name_list,
                 save_path=os.path.join(save_dir, "prc_comparison.png"),
                 title="PRC Comparison")

    return all_results


# ─── Variant scoring (TOPMed-based evaluation) ──────────────────────

def score_variants(
    model: tf.keras.Model,
    X_variants: np.ndarray,
    y_variants: np.ndarray,
    mask_indices: list[int] | None = None,
    model_name: str = "GraphyloVar",
    batch_size: int = 64,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Score variants from TOPMed and evaluate AUROC / AUPRC.

    Parameters
    ----------
    model         : trained model
    X_variants    : (N, 115, seq_len)  variant feature arrays
    y_variants    : (N,) binary labels (0=common, 1=rare/pathogenic)
    mask_indices  : species indices to zero out (hg38, apes, ancestors)
    model_name    : label for plots
    batch_size    : prediction batch size
    save_dir      : directory for plots

    Returns
    -------
    dict with auroc, auprc, predictions
    """
    from graphylovar.data import mask_species

    X = X_variants.copy()
    X = mask_species(X, indices=mask_indices)

    preds = model.predict(X, batch_size=batch_size)
    if preds.ndim == 2 and preds.shape[1] == 2:
        scores = preds[:, 1]
    else:
        scores = preds.ravel()

    y = y_variants if y_variants.ndim == 1 else y_variants[:, 1]

    roc_result = compute_roc(y, scores)
    prc_result = compute_prc(y, scores)

    print(f"[{model_name}]  AUROC: {roc_result['auc']:.4f}  |  AUPRC: {prc_result['auprc']:.4f}")

    if save_dir:
        plot_roc(roc_result, names=[model_name],
                 save_path=os.path.join(save_dir, f"roc_{model_name}.png"),
                 title=f"ROC – {model_name}")
        plot_prc(prc_result, names=[model_name],
                 save_path=os.path.join(save_dir, f"prc_{model_name}.png"),
                 title=f"PRC – {model_name}")

    return {
        "auroc": roc_result["auc"],
        "auprc": prc_result["auprc"],
        "predictions": scores,
    }
