#!/usr/bin/env python3
"""PROTOCOL SS6, step 1: does the featuriser turn a polymer-ligand covale into a mask entry?

`of3t-auxheads` established the term is polymer-ligand (`core/loss/diffusion.py:205-210`
builds `bond_mask = token_bonds * (is_polymer[..., None, :] * is_ligand[..., None])`) and
that 0 of 8 corpus targets carry one, so `bond_loss` read 0.0 and the term moved no
gradient at all. `of3t-orchestrator/bondcov` then found targets that satisfy the predicate
at mmCIF level. Between those two there is one unverified step: mmCIF annotation is not a
feature tensor, and a covale row that the featuriser drops is worth exactly nothing.

This closes that step without touching the model. It walks a corpus built with
`build_of3_subset.py --ids`, featurises every datapoint, and reports for each the
`token_bonds` non-zero count, how many bonded pairs have one polymer and one ligand
partner, and the `bond_mask` non-zero count computed with upstream's own expression.

Nothing is synthesised. The bond either survives featurization or it does not, and a zero
here is the finding (PROTOCOL SS6: the featuriser cannot express the predicate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "of3_port"))

import batch_digest as BD  # noqa: E402


def upstream_bond_mask(s: dict) -> torch.Tensor:
    """`core/loss/diffusion.py` bond_loss, the token-level half, verbatim."""
    is_polymer = s["is_protein"] + s["is_dna"] + s["is_rna"]
    return s["token_bonds"] * (is_polymer[..., None, :] * s["is_ligand"][..., None])


def describe(s: dict, idx: int) -> dict:
    tb = s["token_bonds"]
    is_polymer = s["is_protein"] + s["is_dna"] + s["is_rna"]
    is_ligand = s["is_ligand"]
    bm = upstream_bond_mask(s)
    pairs = (tb != 0).nonzero()
    # A bond is polymer-ligand in EITHER orientation; the loss only keeps one of the two
    # because token_bonds is symmetric, so the mask count is half the unordered count.
    cross = [(int(i), int(j)) for i, j in pairs
             if (float(is_polymer[i]) > 0) != (float(is_polymer[j]) > 0)
             and (float(is_ligand[i]) > 0) != (float(is_ligand[j]) > 0)]
    hits = [(int(i), int(j)) for i, j in (bm != 0).nonzero()]
    return {
        "index": idx,
        "n_tokens_real": int(s["token_mask"].sum()),
        "n_polymer_tokens": int((is_polymer > 0).sum()),
        "n_ligand_tokens": int((is_ligand > 0).sum()),
        "token_bonds_nnz": int((tb != 0).sum()),
        "token_bonds_pairs": int((tb != 0).sum()) // 2,
        "polymer_ligand_pairs_either_orientation": len(cross) // 2,
        "bond_mask_nnz": int((bm != 0).sum()),
        "bond_mask_entries": hits[:64],
        "loss_weight_bond": float(s["loss_weights"]["bond"]),
        "partners": [
            {"i": i, "j": j,
             "i_is_polymer": bool(is_polymer[i] > 0), "i_is_ligand": bool(is_ligand[i] > 0),
             "j_is_polymer": bool(is_polymer[j] > 0), "j_is_ligand": bool(is_ligand[j] > 0)}
            for i, j in hits[:16]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--cache-file", type=Path, required=True)
    ap.add_argument("--stage", default="finetune_1")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--max-index", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    BD.seed_everything(a.seed)
    ds = BD.build_dataset("openfold3", a.data_dir, 4, token_budget=a.crop,
                          split="train", stage=a.stage, cache_file=a.cache_file)
    guard = BD.install_retry_guard(ds)
    n = len(ds) if a.max_index is None else min(len(ds), a.max_index)
    print(f"{type(ds).__name__}: {len(ds)} datapoints, walking {n}, "
          f"stage {a.stage} crop {a.crop}", flush=True)

    # `datapoint_cache` is their own (pdb_id, chain-or-interface) table, so the triple
    # a coverage claim names comes from the dataset rather than from the walk order.
    dp = ds.datapoint_cache
    rows = []
    for i in range(n):
        before = guard["retries"]
        s = ds[i]
        sub = guard["retries"] - before
        r = describe(s, i)
        r["pdb_id"] = str(dp.iloc[i]["pdb_id"])
        r["datapoint"] = str(dp.iloc[i]["preferred_chain_or_interface"])
        r["silent_substitutions"] = sub
        rows.append(r)
        print(f"[{time.time()-t0:.0f}s] idx {i} {r['pdb_id']} {r['datapoint']}: "
              f"tokens {r['n_tokens_real']} "
              f"(polymer {r['n_polymer_tokens']}, ligand {r['n_ligand_tokens']})  "
              f"token_bonds nnz {r['token_bonds_nnz']}  "
              f"polymer-ligand pairs {r['polymer_ligand_pairs_either_orientation']}  "
              f"bond_mask nnz {r['bond_mask_nnz']}  subs {sub}", flush=True)

    fired = [r for r in rows if r["bond_mask_nnz"] > 0]
    report = {
        "instrument": "PROTOCOL SS6 step 1: bond_mask on a polymer-ligand covale target",
        "package": "openfold3",
        "data_dir": str(a.data_dir), "cache_file": str(a.cache_file),
        "stage": a.stage, "dataset": "weighted-pdb", "crop": a.crop, "seed": a.seed,
        "n_datapoints": len(ds), "n_walked": n,
        "n_with_nonzero_bond_mask": len(fired),
        "rows": rows,
        "total_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\n{len(fired)} of {n} datapoints carry a non-zero bond_mask -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
