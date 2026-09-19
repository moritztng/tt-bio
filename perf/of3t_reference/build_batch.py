#!/usr/bin/env python3
"""Freeze the reference batches: upstream's own WeightedPDBDataset, their collator, fixed order.

Run with PYTHONPATH pointing at an openfold3 0.4.3 checkout. Writes one .pt per step plus a
sha256 for each, so every downstream row consumes bytes it can verify rather than re-deriving a
batch and hoping the pipeline was seeded the same way.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import torch
import yaml


def sha256_of(obj) -> str:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    import pytorch_lightning as pl
    from openfold3.core.data.framework.data_module import DatasetMode
    from openfold3.core.data.framework.data_module import openfold_batch_collator
    from openfold3.projects.of3_all_atom.config.dataset_configs import TrainingDatasetSpec
    from openfold3.core.data.framework.single_datasets.abstract_single import DATASET_REGISTRY

    raw = args.config.read_text().replace("CORPUS", str(args.corpus.resolve()))
    cfg = yaml.safe_load(raw)

    name, entry = next(iter(cfg["dataset_configs"]["train"].items()))
    spec_dict = dict(entry)
    spec_dict["name"] = name
    spec_dict["mode"] = DatasetMode.train.value
    spec_dict["config"]["dataset_paths"] = cfg["dataset_paths"][name]
    spec = TrainingDatasetSpec.model_validate(spec_dict)

    pl.seed_everything(args.seed, workers=True)
    dataset = DATASET_REGISTRY[spec.dataset_class](spec.config)
    print(f"dataset {spec.dataset_class} datapoints={len(dataset)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "dataset_class": spec.dataset_class,
        "n_datapoints": len(dataset),
        "steps": [],
    }
    # Fixed data order. The real sampler is weighted and distributed; a reference needs an order a
    # second machine can reproduce without replaying a sampler's RNG, so the order is pinned here
    # and stated rather than sampled. Walking the datapoint cache in its own order would spend all
    # 20 steps inside two or three entries, because the cache lists every chain and interface of an
    # entry consecutively. Instead: group datapoints by entry, then round-robin over entries in
    # sorted order, taking each entry's next datapoint. Every entry appears, and the step sequence
    # is a pure function of the cache.
    by_entry: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        by_entry.setdefault(dataset.datapoint_cache.iloc[i]["pdb_id"], []).append(i)
    entries = sorted(by_entry)
    order = []
    for round_no in range(args.steps):
        entry = entries[round_no % len(entries)]
        picks = by_entry[entry]
        order.append(picks[(round_no // len(entries)) % len(picks)])
    manifest["entries"] = {k: len(v) for k, v in sorted(by_entry.items())}

    for step in range(args.steps):
        idx = order[step]
        pl.seed_everything(args.seed + step, workers=True)
        batch = openfold_batch_collator([dataset[idx]])
        path = args.out / f"batch_step{step + 1:03d}.pt"
        torch.save(batch, path)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["steps"].append(
            {
                "step": step + 1,
                "datapoint_index": idx,
                "pdb_id": batch["pdb_id"],
                "file": path.name,
                "sha256": h,
                "n_tokens": int(batch["token_mask"].sum().item()),
                "seed": args.seed + step,
            }
        )
        print(f"step {step + 1}: idx={idx} {batch['pdb_id']} "
              f"tokens={manifest['steps'][-1]['n_tokens']} sha256={h[:16]}", flush=True)

    (args.out / "batches_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
