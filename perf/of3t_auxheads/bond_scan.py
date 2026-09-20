#!/usr/bin/env python3
"""PROTOCOL SS6, the `bond` hole: find a training target that actually carries an inter-token bond.

`bond` fires nowhere in this campaign for two independent reasons, both visible in
BUNDLE-MIN-043 s own batch: `initial_training` sets `loss_weights.bond = 0.0`, and 5nw3 s
crop carries `token_bonds` all-zero, so even at weight 4.0 the term would have nothing to
sum over. The first is a stage choice and `finetune_1` fixes it; the second is a dataset
property and no stage config can.

This walks the 8-structure training corpus `build_of3_subset.py` fetched, at `finetune_1` s
crop and loss weights, and reports for each sample how many inter-token bonds its crop
carries. A target with a non-zero count is the one the coverage probe runs on; if none has
one, that is the honest answer and it gets reported as the gap it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "of3_port"))

import batch_digest as BD  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--stage", default="finetune_1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--pkg", default="tt_bio._vendor.openfold3")
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()

    import torch

    BD.seed_everything(20260920)
    ds = BD.build_dataset(a.pkg, a.data_dir, a.n_templates,
                          token_budget=BD.STAGE_CROP[a.stage], split="train", stage=a.stage)
    guard = BD.install_retry_guard(ds)
    rows = []
    for i in range(min(a.n, len(ds))):
        s = ds[i]
        tb = s["token_bonds"]
        tm = s["token_mask"]
        lw = {k: float(v) for k, v in s["loss_weights"].items()} if "loss_weights" in s else {}
        nnz = int((tb != 0).sum())
        # token_bonds is symmetric with no self-bonds in their featuriser; count PAIRS.
        rows.append({
            "index": i,
            "n_tokens_real": int(tm.sum()),
            "n_tokens_padded": int(tm.numel()),
            "token_bonds_nnz": nnz,
            "token_bonds_pairs": nnz // 2,
            "is_ligand": int(s["is_ligand"].sum()),
            "is_atomized": int(s["is_atomized"].sum()),
            "loss_weight_bond": lw.get("bond"),
        })
        print(json.dumps(rows[-1]), flush=True)

    out = {"stage": a.stage, "crop": BD.STAGE_CROP[a.stage], "pkg": a.pkg,
           "retries": guard["retries"], "rows": rows,
           "with_bond": [r["index"] for r in rows if r["token_bonds_nnz"] > 0]}
    print(json.dumps({"with_bond": out["with_bond"], "retries": guard["retries"]}, indent=1))
    if a.json_out:
        a.json_out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
