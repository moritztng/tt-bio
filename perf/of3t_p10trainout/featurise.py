#!/usr/bin/env python3
"""Dump the graded corpus once, as the `.pt` batches `OpenFold3Dataset` reads.

Both arms of this row have to see the SAME crop of the same target in the same order. A
featuriser called per arm would not give that: upstream crops and samples its MSA under an RNG,
so two calls at the same index are two different training examples, and an A/B whose two arms
saw different data is not an A/B. So the corpus is materialised to disk once here and both arms
replay the files.

The featuriser is upstream's own, reached the way `perf/of3t_auxheads/bond_coverage.py` reaches
it (`batch_digest.build_dataset` plus that file's `collate1` against a rank template), because a
second featurisation is the one thing this campaign refuses to write.

    featurise.py --data-dir <datasets> --cache-file <subset.json> --split train --crop 384 \
                 --stage initial_training --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "of3_port"))
sys.path.insert(0, str(REPO / "perf" / "of3t_reference"))

import batch_digest as BD                                             # noqa: E402
from perf.of3t_auxheads.bond_coverage import collate1                 # noqa: E402


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--cache-file", required=True, type=Path)
    ap.add_argument("--package", default="tt_bio._vendor.openfold3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--indices", default="",
                    help="comma-separated datapoints to dump, in step order")
    ap.add_argument("--draw", type=int, default=0,
                    help="draw this many datapoints in step order from upstream own "
                         "`datapoint_probabilities`, which is the weighted-PDB sampler a real "
                         "step draws from. The 8-target train subset is 292 datapoints, not 8, "
                         "and they are not equiprobable: 2wig holds 228 of them and 1wyc one, "
                         "so dumping the cache in index order would train almost entirely on "
                         "one structure. With replacement, like the sampler")
    ap.add_argument("--draw-seed", type=int, default=20260926)
    ap.add_argument("--rank-template", type=Path,
                    help="a batch upstream's collator produced; decides which features already "
                         "carry the batch axis")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    BD.seed_everything(a.seed)
    ds = BD.build_dataset(a.package, a.data_dir, 4, token_budget=a.crop, split=a.split,
                          stage=a.stage, cache_file=a.cache_file)
    guard = BD.install_retry_guard(ds)
    dc = ds.datapoint_cache
    n = len(dc)
    if a.draw:
        import numpy as np
        w = np.asarray(dc["datapoint_probabilities"], np.float64) \
            if "datapoint_probabilities" in dc.columns else np.ones(n)
        order = [int(i) for i in np.random.default_rng(a.draw_seed).choice(
            n, size=a.draw, replace=True, p=w / w.sum())]
    elif a.indices:
        order = [int(x) for x in a.indices.split(",") if x != ""]
    else:
        order = list(range(n))
    idx = sorted(set(order))
    tmpl = torch.load(a.rank_template, weights_only=False) if a.rank_template else None
    a.out.mkdir(parents=True, exist_ok=True)

    # `order` is the step order and is the contract between the two arms: they replay this
    # list, they do not re-draw it. `entries` is the unique files, because the sampler draws
    # with replacement and a repeated datapoint is the same crop, not a new one.
    manifest = {"cache_file": str(a.cache_file), "split": a.split, "stage": a.stage,
                "crop": a.crop, "seed": a.seed, "draw_seed": a.draw_seed,
                "package": a.package, "datapoints_in_cache": n,
                "order": order, "entries": []}
    for i in idx:
        dp = ds.datapoint_cache.iloc[i]
        before = guard["retries"]
        sample = ds[i]
        if guard["retries"] != before:
            raise SystemExit(f"index {i} was silently substituted; it is not the target it "
                             f"claims to be")
        batch = collate1(sample, tmpl)
        gt = batch.get("ground_truth")
        if not gt:
            raise SystemExit(f"index {i} carries no ground_truth; a batch with no deposited "
                             f"structure cannot grade a training outcome")
        pdb = str(dp["pdb_id"])
        out = a.out / f"{i:02d}_{pdb}.pt"
        torch.save(batch, out)
        tok = int(batch["token_mask"].sum())
        manifest["entries"].append({
            "index": i, "pdb_id": pdb,
            "chain_or_interface": str(dp.get("preferred_chain_or_interface", "")),
            "file": out.name, "sha256": _sha(out), "bytes": out.stat().st_size,
            "real_tokens": tok, "crop_tokens": int(batch["token_mask"].shape[-1]),
        })
        print(f"[{time.time()-t0:6.0f}s] {i:2d} {pdb} -> {out.name}  {tok} real tokens",
              flush=True)

    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{time.time()-t0:.0f}s] {len(manifest['entries'])} batches, "
          f"{len(order)} steps of order -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
