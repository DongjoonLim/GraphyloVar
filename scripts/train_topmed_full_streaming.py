#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.model_io import load_hdf5_weights_compat, normalize_model_path, resolve_model_path  # noqa: E402
from graphylovar.models import build_model  # noqa: E402
from graphylovar.phylogeny import MASK_INDICES, NUM_NODES, build_graph  # noqa: E402
from graphylovar.topmed import (  # noqa: E402
    extract_batch_examples_from_encoded,
    load_alignment_encoded_cache,
    parse_chromosome_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GraphyloVar on full compact TOPMed data with on-the-fly example extraction")
    parser.add_argument("--compact_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train_chromosomes", default="1-10")
    parser.add_argument("--val_chromosomes", default="11-12")
    parser.add_argument("--test_chromosomes", default="13-22")
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--context_flank", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=9999)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--binary_loss_weight", type=float, default=1.5)
    parser.add_argument("--model_name", default="multitask_cnn_gcn_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--distribution", default="single", choices=["mirrored", "single"])
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--shuffle_chunk_size", type=int, default=100_000)
    parser.add_argument("--skip_test_eval", action="store_true")
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--alignment_cache_dir", default="")
    parser.add_argument("--tensorboard_dir", default="")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--lr_reduce_patience", type=int, default=5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--cosine_lr", action="store_true",
                        help="Use cosine decay LR schedule instead of ReduceLROnPlateau")
    parser.add_argument("--cosine_epochs", type=int, default=200,
                        help="Total epochs for cosine decay denominator")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append_history", action="store_true")
    parser.add_argument(
        "--species_masking",
        choices=["on", "off"],
        default="off",
        help="Mask configured species channels during training/eval windows.",
    )
    return parser


def compact_files(compact_dir: str, chromosome: int) -> tuple[str, str, str]:
    return (
        os.path.join(compact_dir, f"positions_graphylovar_topmed_chr{chromosome}.npy"),
        os.path.join(compact_dir, f"y_graphylovar_topmed_chr{chromosome}.npy"),
        os.path.join(compact_dir, f"metadata_graphylovar_topmed_chr{chromosome}.json"),
    )


def iter_existing_chromosomes(chromosomes: list[int], compact_dir: str) -> list[int]:
    found = []
    for chromosome in chromosomes:
        pos_path, y_path, meta_path = compact_files(compact_dir, chromosome)
        try:
            with open(pos_path, "rb"), open(y_path, "rb"), open(meta_path, "r", encoding="utf-8"):
                found.append(chromosome)
        except OSError:
            continue
    return found


def load_metadata(compact_dir: str, chromosome: int) -> dict:
    _, _, meta_path = compact_files(compact_dir, chromosome)
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_full_compact_data(compact_dir: str, chromosomes: list[int]) -> None:
    invalid = []
    for chromosome in chromosomes:
        meta = load_metadata(compact_dir, chromosome)
        if meta.get("mode") != "compact_full" or meta.get("capped", True):
            invalid.append(chromosome)
    if invalid:
        invalid_str = ", ".join(str(chromosome) for chromosome in invalid)
        raise RuntimeError(f"Compact full preprocessing is missing or invalid for chromosomes: {invalid_str}")


def count_samples(compact_dir: str, chromosomes: list[int]) -> int:
    total = 0
    for chromosome in chromosomes:
        pos_path, _, _ = compact_files(compact_dir, chromosome)
        total += int(np.load(pos_path, mmap_mode="r").shape[0])
    return total


def iter_index_blocks(num_samples: int, shuffle: bool, rng: np.random.Generator, chunk_size: int):
    starts = list(range(0, num_samples, chunk_size))
    if shuffle:
        rng.shuffle(starts)
    for start in starts:
        stop = min(start + chunk_size, num_samples)
        indices = np.arange(start, stop, dtype=np.int64)
        if shuffle:
            rng.shuffle(indices)
        yield indices


def create_dataset(
    compact_dir: str,
    chromosomes: list[int],
    batch_size: int,
    context: int,
    context_flank: int | None,
    shuffle: bool,
    seed: int,
    shuffle_chunk_size: int,
    alignment_cache_dir: str,
        species_masking: bool,
) -> tuple[tf.data.Dataset, int]:
    seq_len = (2 * context_flank + 1) * 2 if context_flank is not None else (context * 4 + 2)

    def generator():
        rng = np.random.default_rng(seed)
        chroms = list(chromosomes)
        while True:
            if shuffle:
                rng.shuffle(chroms)
            for chromosome in chroms:
                pos_path, y_path, _ = compact_files(compact_dir, chromosome)
                meta = load_metadata(compact_dir, chromosome)
                cache_array, cache_meta = load_alignment_encoded_cache(
                    alignment_path=meta["alignment_path"],
                    cache_dir=alignment_cache_dir,
                    chromosome=chromosome,
                )
                print(
                    f"Using encoded alignment cache for chr{chromosome}: {cache_meta['sequence_length']:,} columns",
                    flush=True,
                )
                positions = np.load(pos_path, mmap_mode="r")
                targets = np.load(y_path, mmap_mode="r")
                for block_indices in iter_index_blocks(positions.shape[0], shuffle, rng, shuffle_chunk_size):
                    block_positions = positions[block_indices]
                    block_targets = targets[block_indices]
                    for batch_start in range(0, len(block_positions), batch_size):
                        pos_batch = block_positions[batch_start : batch_start + batch_size].astype(np.int64) - 1
                        target_batch = block_targets[batch_start : batch_start + batch_size]
                        x_batch, valid = extract_batch_examples_from_encoded(
                            encoded_alignment=cache_array,
                            positions_zero_based=pos_batch,
                            context=context,
                            context_flank=context_flank,
                                mask_indices=MASK_INDICES if species_masking else None,
                        )
                        if x_batch.shape[0] == 0:
                            continue
                        y_batch = np.asarray(target_batch[valid], dtype=np.float32)
                        yield x_batch, {
                            "nucleotide": y_batch[:, :5],
                            "binary": y_batch[:, 5:6],
                        }
                del positions, targets, cache_array
                gc.collect()

    output_signature = (
        tf.TensorSpec(shape=(None, NUM_NODES, seq_len), dtype=tf.uint8),
        {
            "nucleotide": tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            "binary": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        },
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, seq_len


def resolve_checkpoint_prefix(save_path: str) -> str | None:
    save_path = resolve_model_path(save_path)
    variables_prefix = os.path.join(save_path, "variables", "variables")
    if os.path.exists(f"{variables_prefix}.index"):
        return variables_prefix
    if os.path.exists(f"{save_path}.index"):
        return save_path
    return None


_HISTORY_COLS = [
    "epoch", "binary_accuracy", "binary_loss", "loss", "lr",
    "nucleotide_accuracy", "nucleotide_loss",
    "val_binary_accuracy", "val_binary_loss", "val_loss",
    "val_nucleotide_accuracy", "val_nucleotide_loss",
]


def _normalize_history_csv(path: str) -> int:
    """Read history CSV robustly, normalize to _HISTORY_COLS, save back. Returns row count."""
    all_lines = open(path).readlines()
    canonical_ncols = len(_HISTORY_COLS)
    good_rows: list[list[str]] = []
    for raw in all_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        try:
            float(parts[0])  # data row
        except ValueError:
            continue  # header row : skip
        # Accept rows with 11 or 12 fields; pad short ones
        if len(parts) < canonical_ncols - 1:
            continue  # too short, corrupt
        while len(parts) < canonical_ncols:
            parts.append("")  # pad missing lr column
        if len(parts) > canonical_ncols:
            parts = parts[:canonical_ncols]  # truncate extra
        good_rows.append(parts)
    if not good_rows:
        return 0
    df = pd.DataFrame(good_rows, columns=_HISTORY_COLS)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["epoch"] = range(len(df))
    df.to_csv(path, index=False)
    return len(df)


def existing_history_rows(history_csv_path: str) -> int:
    if not os.path.exists(history_csv_path):
        return 0
    try:
        return _normalize_history_csv(history_csv_path)
    except Exception:
        return 0


def main() -> None:
    args = build_parser().parse_args()
    if not args.alignment_cache_dir:
        args.alignment_cache_dir = os.path.join(os.path.dirname(args.compact_dir), "topmed_alignment_cache")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_gpus:
        tf.config.experimental.set_memory_growth(device, True)
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    requested_train = parse_chromosome_spec(args.train_chromosomes)
    requested_val = parse_chromosome_spec(args.val_chromosomes)
    requested_test = parse_chromosome_spec(args.test_chromosomes)

    train_chromosomes = iter_existing_chromosomes(requested_train, args.compact_dir)
    val_chromosomes = iter_existing_chromosomes(requested_val, args.compact_dir)
    test_chromosomes = iter_existing_chromosomes(requested_test, args.compact_dir)

    if not train_chromosomes or not val_chromosomes:
        raise RuntimeError("Training and validation chromosomes with compact full TOPMed data are required")

    ensure_full_compact_data(args.compact_dir, sorted(set(train_chromosomes + val_chromosomes + test_chromosomes)))

    print(f"Train chromosomes: {train_chromosomes}")
    print(f"Val chromosomes:   {val_chromosomes}")
    print(f"Test chromosomes:  {test_chromosomes}")

    train_total = count_samples(args.compact_dir, train_chromosomes)
    val_total = count_samples(args.compact_dir, val_chromosomes)
    test_total = count_samples(args.compact_dir, test_chromosomes) if test_chromosomes else 0
    print(f"Samples -> train={train_total:,} val={val_total:,} test={test_total:,}")

    train_ds, seq_len = create_dataset(
        compact_dir=args.compact_dir,
        chromosomes=train_chromosomes,
        batch_size=args.batch_size,
        context=args.context,
        context_flank=args.context_flank,
        shuffle=True,
        seed=args.seed,
        shuffle_chunk_size=args.shuffle_chunk_size,
        alignment_cache_dir=args.alignment_cache_dir,
        species_masking=(args.species_masking == "on"),
    )
    val_ds, _ = create_dataset(
        compact_dir=args.compact_dir,
        chromosomes=val_chromosomes,
        batch_size=args.batch_size,
        context=args.context,
        context_flank=args.context_flank,
        shuffle=False,
        seed=args.seed,
        shuffle_chunk_size=args.shuffle_chunk_size,
        alignment_cache_dir=args.alignment_cache_dir,
        species_masking=(args.species_masking == "on"),
    )

    full_steps_per_epoch = max(1, math.ceil(train_total / args.batch_size))
    full_validation_steps = max(1, math.ceil(val_total / args.batch_size))
    steps_per_epoch = args.steps_per_epoch or full_steps_per_epoch
    validation_steps = args.validation_steps or full_validation_steps

    if args.distribution == "mirrored" and len(physical_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    print(f"Visible GPUs: {len(physical_gpus)}")
    print(f"Distribution strategy: {strategy.__class__.__name__}")
    print(f"Mixed precision policy: {mixed_precision.global_policy().name}")
    print(f"Species masking: {'ON' if args.species_masking == 'on' else 'OFF'}")
    print(
        "Epoch schedule -> "
        f"train_steps={steps_per_epoch:,}/{full_steps_per_epoch:,} "
        f"val_steps={validation_steps:,}/{full_validation_steps:,}"
    )

    os.makedirs(args.model_dir, exist_ok=True)
    run_name = (
        f"{args.model_name}"
        f"_train{args.train_chromosomes.replace(',', '_')}"
        f"_val{args.val_chromosomes.replace(',', '_')}"
        f"_flank{args.context_flank}"
    )
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    save_path = normalize_model_path(os.path.join(args.model_dir, run_name))
    history_csv_path = os.path.join(args.model_dir, f"{run_name}_history.csv")
    config_path = os.path.join(args.model_dir, f"{run_name}_config.json")

    previous_epochs = existing_history_rows(history_csv_path) if args.append_history else 0

    _, A = build_graph()
    with strategy.scope():
        model = build_model(
            name=args.model_name,
            input_shape=(NUM_NODES, seq_len),
            A=A,
            learning_rate=args.learning_rate,
            binary_loss_weight=args.binary_loss_weight,
            dense_units=128,
            gcn_units=32,
        )
        resolved_resume_path = resolve_model_path(save_path)
        if args.resume and os.path.exists(resolved_resume_path):
            try:
                model = tf.keras.models.load_model(resolved_resume_path, compile=True)
                print(f"Resumed full model from {resolved_resume_path}")
            except Exception as exc:
                print(f"Full-model resume failed at {resolved_resume_path}: {exc}")
                if resolved_resume_path.endswith((".keras", ".h5", ".hdf5")):
                    load_hdf5_weights_compat(model, resolved_resume_path)
                    print(f"Resumed weights from {resolved_resume_path} using HDF5 compatibility loader")
                else:
                    raise
        elif args.resume:
            print(f"No checkpoint found for resume at {save_path}; starting fresh")

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    # Monitor nucleotide val loss (not total) : more stable signal than total
    # which is dominated by noisy binary head in short val windows.
    _monitor = "val_nucleotide_loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=_monitor,
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor=_monitor,
            save_best_only=True,
            verbose=1,
        ),
        *(
            [tf.keras.callbacks.LearningRateScheduler(
                (lambda ep, _init_lr=args.learning_rate, _total=args.cosine_epochs:
                 _init_lr * 0.5 * (1 + __import__("math").cos(__import__("math").pi * ep / _total))),
                verbose=0,
            )]
            if args.cosine_lr else
            [tf.keras.callbacks.ReduceLROnPlateau(
                monitor=_monitor,
                factor=args.lr_reduce_factor,
                patience=args.lr_reduce_patience,
                min_lr=args.lr_min,
                verbose=1,
            )]
        ),
        tf.keras.callbacks.CSVLogger(history_csv_path, append=args.append_history),
    ]
    if args.tensorboard_dir:
        tb_run_dir = os.path.join(args.tensorboard_dir, run_name)
        os.makedirs(tb_run_dir, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tb_run_dir,
                histogram_freq=0,
                update_freq="epoch",
                write_graph=False,
            )
        )
        print(f"TensorBoard logging to {tb_run_dir}")

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        initial_epoch=previous_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    summary = {
        "train_chromosomes": train_chromosomes,
        "val_chromosomes": val_chromosomes,
        "test_chromosomes": test_chromosomes,
        "train_samples": train_total,
        "val_samples": val_total,
        "test_samples": test_total,
        "full_steps_per_epoch": full_steps_per_epoch,
        "full_validation_steps": full_validation_steps,
        "effective_steps_per_epoch": steps_per_epoch,
        "effective_validation_steps": validation_steps,
        "context": args.context,
        "context_flank": args.context_flank,
        "alignment_cache_dir": args.alignment_cache_dir,
        "batch_size": args.batch_size,
        "distribution": args.distribution,
        "num_visible_gpus": len(physical_gpus),
        "mixed_precision": args.mixed_precision,
        "learning_rate": args.learning_rate,
        "binary_loss_weight": args.binary_loss_weight,
        "model_name": args.model_name,
        "resumed_from_checkpoint": bool(args.resume),
        "initial_epoch": previous_epochs,
        "epochs_trained": previous_epochs + len(history.history.get("loss", [])),
        "best_val_loss": float(min(history.history.get("val_loss", [float("nan")]))),
    }

    if test_chromosomes and not args.skip_test_eval:
        test_ds, _ = create_dataset(
            compact_dir=args.compact_dir,
            chromosomes=test_chromosomes,
            batch_size=args.batch_size,
            context=args.context,
            context_flank=args.context_flank,
            shuffle=False,
            seed=args.seed,
            shuffle_chunk_size=args.shuffle_chunk_size,
            alignment_cache_dir=args.alignment_cache_dir,
            species_masking=(args.species_masking == "on"),
        )
        test_steps = max(1, math.ceil(test_total / args.batch_size))
        test_metrics = model.evaluate(test_ds, steps=test_steps, verbose=1, return_dict=True)
        summary["test_metrics"] = {k: float(v) for k, v in test_metrics.items()}

    summary_path = Path(args.model_dir) / f"{run_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")

    print(f"Saved history: {history_csv_path}")


if __name__ == "__main__":
    main()
