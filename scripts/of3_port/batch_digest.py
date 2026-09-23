#!/usr/bin/env python3
"""Hash the batches an OpenFold3 dataset emits, to show they are reproducible.

`of3t-equivalence` can only compare two stacks step by step if both are fed the same
batch in the same order, so determinism here is a deliverable rather than a detail.
This script builds a dataset from the pinned 4-structure subset
(`build_of3_subset.py`), pulls a fixed number of samples in a fixed order, and prints
a sha256 over every tensor in every batch.

The digest covers dtype, shape and raw bytes of each feature, so a change in any of
them moves it. Keys are hashed in sorted order, and the per-sample digests are folded
in in emission order, so a reordering of the data is also a change.

`--package` chooses which source tree builds the batch:

    tt_bio._vendor.openfold3   our vendored copy    (default)
    openfold3                  upstream, as installed

Running it once per package and diffing the digests is the check that the re-vendored
pipeline still emits upstream's batch. The two trees are genuinely different files on
disk (14 of 108 differ), so this is not a tree compared against itself.

Usage:
    python scripts/of3_port/batch_digest.py --data-dir <dir> --n 4
    python scripts/of3_port/batch_digest.py --data-dir <dir> --n 4 --package openfold3
    python scripts/of3_port/batch_digest.py --data-dir <dir> --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

DEFAULT_PKG = "tt_bio._vendor.openfold3"


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_digest(value) -> tuple[str, str]:
    """(description, sha256) for one feature value."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        v = value.detach().cpu()
        # .numpy() refuses bf16; view the raw storage instead so the bytes hashed are
        # the bytes the model would receive, with no dtype conversion in between.
        raw = v.contiguous().view(torch.uint8) if v.dtype == torch.bfloat16 else v.contiguous()
        buf = raw.numpy().tobytes() if not isinstance(raw, np.ndarray) else raw.tobytes()
        desc = f"torch {v.dtype} {tuple(v.shape)}"
    elif isinstance(value, np.ndarray):
        buf = np.ascontiguousarray(value).tobytes()
        desc = f"numpy {value.dtype} {value.shape}"
    elif isinstance(value, (str, int, float, bool)) or value is None:
        buf = repr(value).encode()
        desc = f"scalar {type(value).__name__}"
    elif isinstance(value, (list, tuple)):
        buf = json.dumps(value, sort_keys=True, default=repr).encode()
        desc = f"{type(value).__name__}[{len(value)}]"
    elif isinstance(value, dict):
        parts = [f"{k}={tensor_digest(value[k])[1]}" for k in sorted(value)]
        buf = "|".join(parts).encode()
        desc = f"dict[{len(value)}]"
    else:
        buf = repr(value).encode()
        desc = f"repr {type(value).__name__}"
    return desc, hashlib.sha256(desc.encode() + b"\0" + buf).hexdigest()


def sample_digest(sample: dict) -> tuple[str, list[tuple[str, str, str]]]:
    rows = []
    h = hashlib.sha256()
    for key in sorted(sample):
        desc, d = tensor_digest(sample[key])
        rows.append((key, desc, d))
        h.update(f"{key}\0{d}\0".encode())
    return h.hexdigest(), rows


# The loss-weight overrides each stage applies to its `weighted-pdb` dataset, read off
# their yamls by stage_loss_coverage.py. initial_training leaves the defaults alone;
# the finetunes raise `bond` and drop `smooth_lddt`; finetune_3 is confidence-only.
STAGE_LOSS_OVERRIDES = {
    "initial_training": {},
    "finetune_1": {"bond": 4.0, "smooth_lddt": 0.0},
    "finetune_2": {"bond": 4.0, "smooth_lddt": 0.0},
    "finetune_3": {"mse": 0.0, "distogram": 0.0, "smooth_lddt": 0.0, "pae": 1e-4},
}
STAGE_CROP = {"initial_training": 384, "finetune_1": 640, "finetune_2": 768, "finetune_3": 768}


def build_dataset(pkg: str, data_dir: Path, n_templates: int,
                  token_budget: int | None = None, split: str = "val",
                  stage: str | None = None):
    """Instantiate their dataset over the corpus `build_of3_subset.py` fetched.

    `val` gives ValidationPDBDataset over the 4 pinned path-diverse structures;
    `train` gives WeightedPDBDataset over the 8-structure sample, which is the class
    the (stage, dataset) pairs in `stage_loss_coverage.py` actually name and the only
    one cropping is reachable from.
    """
    mod = "pdb" if split == "train" else "validation"
    single = importlib.import_module(f"{pkg}.core.data.framework.single_datasets.{mod}")
    dataset_configs = importlib.import_module(f"{pkg}.projects.of3_all_atom.config.dataset_configs")

    root = data_dir / "pdb_training_set"
    std = root / "preprocessed_pdb_data" / "standard"
    if split == "train":
        cache = data_dir / "training_cache_with_templates_subset_8.json"
        tmpl_sub, cls_name, ds_name = "train_template_cache", "WeightedPDBDataset", "weighted-pdb"
    else:
        cache = data_dir / "validation_cache_with_templates_subset_4.json"
        tmpl_sub, cls_name, ds_name = "val_template_cache", "ValidationPDBDataset", "val-weighted-pdb"
    if not cache.is_file():
        raise SystemExit(f"missing {cache} -- run build_of3_subset.py --split {split} first")

    # Mirrors upstream pdb_subset_helpers._dataset_paths_entry + build_runner_yaml_config.
    # Exactly one alignment path may be set and exactly one template path, so the "none"
    # strings upstream writes into its yaml are passed as None here.
    paths = {
        "dataset_cache_file": str(cache),
        "alignment_array_directory": str(root / "alignment_arrays"),
        "target_structures_directory": str(std / "structure_files"),
        "target_structure_file_format": "npz",
        "reference_molecule_directory": str(std / "reference_mols"),
        "template_cache_directory": str(root / "templates" / tmpl_sub),
        "template_structure_array_directory": str(root / "templates" / "template_structure_arrays"),
        "template_file_format": "npz",
    }
    # Their `register_dataset_config` decorator returns None, so the decorated class
    # names (`ValidationPDBConfig`, ...) are all bound to None at module level and the
    # registry is the only way to reach them. Upstream defect; worked around, not fixed
    # here, so the vendored file stays byte-identical to theirs.
    config_cls = dataset_configs.DATASET_CONFIG_REGISTRY.get(cls_name)

    if token_budget is None:
        crop = {"token_crop": {"enabled": False}}
    else:
        # The stage configs crop to 384 (initial_training), 640 (finetune_1) and
        # 768 (finetune_2, finetune_3). Cropping is the stochastic step the seed has
        # to control, so it is worth digesting on its own.
        crop = {
            "token_crop": {
                "enabled": True,
                "token_budget": token_budget,
                "crop_weights": {"contiguous": 0.2, "spatial": 0.4, "spatial_interface": 0.4},
            },
            "chain_crop": {"enabled": True},
        }
    kwargs = dict(
        name=ds_name,
        debug_mode=True,
        dataset_paths=dataset_configs.TrainingDatasetPaths(**paths),
        template={"n_templates": n_templates, "take_top_k": split != "train"},
        crop=crop,
    )
    if split == "train":
        # WeightedPDBDataset walks a datapoint cache of chains and interfaces; ordering
        # it makes the digest a property of the corpus rather than of a draw.
        kwargs["sample_in_order"] = True
        if stage:
            kwargs["loss"] = {"loss_weights": STAGE_LOSS_OVERRIDES[stage]}
    else:
        kwargs["sample_in_order"] = True
        kwargs["msa"] = {"subsample_main": False}
    cfg = config_cls(**kwargs)
    return getattr(single, cls_name)(cfg)


def install_retry_guard(ds) -> dict:
    """Make their silent sample-substitution visible.

    Both dataset classes catch every exception in `__getitem__`, log a warning and
    return `self.__getitem__(random.randint(0, len(self)-1))`. A sample that fails to
    featurize is therefore replaced by a RANDOM other one, and the caller is never
    told. For a digest that is the worst possible failure: the run still completes and
    still reproduces (the retry draws from the seeded global RNG), so a corrupt batch
    looks exactly like a clean one. This counts the re-entries so the run can refuse to
    report a digest that is not the batch it claims to be.
    """
    cls = type(ds)
    state = {"depth": 0, "retries": 0}
    original = cls.__getitem__

    def counting(self, index):
        state["depth"] += 1
        if state["depth"] > 1:
            state["retries"] += 1
        try:
            return original(self, index)
        finally:
            state["depth"] -= 1

    cls.__getitem__ = counting
    state["_restore"] = lambda: setattr(cls, "__getitem__", original)
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--package", default=DEFAULT_PKG)
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--stage", choices=tuple(STAGE_LOSS_OVERRIDES), default=None,
                    help="Apply that stage's weighted-pdb loss overrides and crop.")
    ap.add_argument("--n", type=int, default=4, help="Samples to draw (default: all 4).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=None,
                    help="Enable token cropping to this budget (384/640/768 upstream).")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--per-feature", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    if args.stage and args.token_budget is None:
        args.token_budget = STAGE_CROP[args.stage]
    ds = build_dataset(args.package, args.data_dir, args.n_templates, args.token_budget,
                       args.split, args.stage)
    print(f"package {args.package}   dataset {type(ds).__name__}   len {len(ds)}   "
          f"seed {args.seed}" + (f"   stage {args.stage} crop {args.token_budget}" if args.stage else ""))

    guard = install_retry_guard(ds)
    overall = hashlib.sha256()
    out = {"package": args.package, "split": args.split, "seed": args.seed,
           "token_budget": args.token_budget, "samples": []}
    for i in range(min(args.n, len(ds))):
        seed_everything(args.seed + i)
        sample = ds[i]
        d, rows = sample_digest(sample)
        overall.update(d.encode())
        nk = len(rows)
        print(f"  sample {i}  keys {nk:3d}  sha256 {d}")
        if args.per_feature:
            for k, desc, kd in rows:
                print(f"      {k:44s} {desc:34s} {kd[:16]}")
        out["samples"].append({"index": i, "n_keys": nk, "sha256": d,
                               "features": [{"key": k, "desc": dsc, "sha256": kd} for k, dsc, kd in rows]})

    guard["_restore"]()
    out["retries"] = guard["retries"]
    out["batch_digest"] = overall.hexdigest()
    if guard["retries"]:
        print(f"REFUSING to report a digest: {guard['retries']} sample(s) failed to "
              f"featurize and were silently replaced by a random other sample.")
        return 1
    print(f"retries 0 (no sample was silently substituted)")
    print(f"batch-digest sha256 {out['batch_digest']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
