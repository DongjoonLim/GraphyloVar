#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.topmed import ensure_alignment_encoded_cache, parse_chromosome_spec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build shared encoded alignment caches for TOPMed training")
    parser.add_argument("--compact_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--chromosomes", default="1-12")
    return parser


def metadata_path(compact_dir: str, chromosome: int) -> str:
    return os.path.join(compact_dir, f"metadata_graphylovar_topmed_chr{chromosome}.json")


def main() -> None:
    args = build_parser().parse_args()
    chromosomes = parse_chromosome_spec(args.chromosomes)
    os.makedirs(args.cache_dir, exist_ok=True)

    for chromosome in chromosomes:
        meta_file = metadata_path(args.compact_dir, chromosome)
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Missing metadata for chr{chromosome}: {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        print(f"Preparing encoded cache for chr{chromosome}: {meta['alignment_path']}", flush=True)
        cache_path, cache_meta = ensure_alignment_encoded_cache(
            alignment_path=meta["alignment_path"],
            cache_dir=args.cache_dir,
            chromosome=chromosome,
        )
        print(
            f"Ready chr{chromosome}: {cache_path} ({cache_meta['sequence_length']:,} columns)",
            flush=True,
        )


if __name__ == "__main__":
    main()