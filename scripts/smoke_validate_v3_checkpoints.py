#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.model_io import describe_prediction_output, load_hdf5_weights_compat, resolve_model_path
from graphylovar.models import build_model
from graphylovar.phylogeny import NUM_NODES, build_graph


V3_FLANKS = (0, 1, 8, 16, 32, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke validate V3 flank checkpoints for reuse.")
    parser.add_argument("--project_root", default=os.path.expanduser("~/GraphyloVar"))
    parser.add_argument(
        "--output_csv",
        default=os.path.expanduser("~/GraphyloVar/topmed_models/full_streaming_runs/plots/v3_checkpoint_smoke_validation.csv"),
    )
    return parser.parse_args()


def run_name(flank: int) -> str:
    return f"multitask_hybrid_v3_train1-10_val11-12_flank{flank}_v3flank{flank}"


def config_path(project_root: Path, flank: int) -> Path:
    return project_root / "topmed_models" / "full_streaming_runs" / "v3_ablation" / f"{run_name(flank)}_config.json"


def model_path(project_root: Path, flank: int) -> Path:
    return project_root / "topmed_models" / "full_streaming_runs" / "v3_ablation" / run_name(flank)


def classify_model_artifact(path: Path) -> str:
    if path.is_dir() and (path / "saved_model.pb").exists():
        return "savedmodel"
    if path.is_file() and path.suffix == ".keras":
        return "keras"
    if path.exists():
        return "unknown"
    return "missing"


def validate_one(project_root: Path, flank: int) -> dict[str, str]:
    result = {
        "flank": str(flank),
        "run_tag": f"v3flank{flank}",
        "config_path": str(config_path(project_root, flank)),
        "model_path": str(model_path(project_root, flank)),
        "status": "unknown",
        "load_mode": "",
        "prediction_shape": "",
        "detail": "",
    }

    cfg_path = config_path(project_root, flank)
    if not cfg_path.exists():
        result["status"] = "missing-config"
        result["detail"] = "no V3 config file"
        return result

    raw_model_path = model_path(project_root, flank)
    resolved_path = Path(resolve_model_path(str(raw_model_path)))
    result["model_path"] = str(resolved_path)
    artifact_type = classify_model_artifact(resolved_path)
    if artifact_type == "missing":
        result["status"] = "missing-model"
        result["detail"] = "no model checkpoint file"
        return result

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("mixed_precision"):
            mixed_precision.set_global_policy("mixed_float16")
        else:
            mixed_precision.set_global_policy("float32")

        _, adjacency = build_graph()
        flank_context = int(cfg["context_flank"])
        seq_len = (2 * flank_context + 1) * 2
        dummy = np.zeros((2, NUM_NODES, seq_len), dtype=np.uint8)

        if artifact_type == "savedmodel":
            loaded = tf.saved_model.load(str(resolved_path))
            infer = loaded.signatures.get("serving_default")
            if infer is None:
                raise RuntimeError("savedmodel missing serving_default signature")
            predictions = infer(tf.convert_to_tensor(dummy))
            result["load_mode"] = "savedmodel"
            result["prediction_shape"] = describe_prediction_output(predictions)
            result["status"] = "ok-savedmodel"
        else:
            try:
                model = tf.keras.models.load_model(str(resolved_path), compile=True)
                result["load_mode"] = "full-model"
            except Exception:
                model = build_model(
                    name=cfg["model_name"],
                    input_shape=(NUM_NODES, seq_len),
                    A=adjacency,
                    learning_rate=float(cfg["learning_rate"]),
                    binary_loss_weight=float(cfg["binary_loss_weight"]),
                    dense_units=128,
                    gcn_units=32,
                )
                load_hdf5_weights_compat(model, str(resolved_path))
                result["load_mode"] = "weights-compat"

            predictions = model(dummy, training=False)
            result["prediction_shape"] = describe_prediction_output(predictions)
            result["status"] = "ok-keras"
        result["detail"] = "build+load+forward-pass succeeded"
        return result
    except Exception as exc:  # pragma: no cover - operational smoke output
        result["status"] = "failed"
        result["detail"] = str(exc).replace("\n", " ")[:500]
        return result


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    rows = [validate_one(project_root, flank) for flank in V3_FLANKS]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["flank", "run_tag", "config_path", "model_path", "status", "load_mode", "prediction_shape", "detail"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote smoke validation to {output_path}")


if __name__ == "__main__":
    main()
