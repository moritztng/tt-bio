#!/usr/bin/env python3
"""A control's own control, without spending a model forward on 0 == 0.

`covpath_gradient.py` differences two arms that see the same batch and differ only in one
feature edit. On a target the path cannot act on, that edit has nothing to do, so the two arms
would be bit-identical and the measured contribution would read zero by construction rather
than by measurement. The instrument refuses those runs outright ("no nucleotide token on this
batch; the control cannot move"), which is right for a coverage claim and leaves the null
unstated.

This states the null where it is actually decidable: apply the edit to a batch the path cannot
act on and compare every tensor in the feature dict. If one moves, the edit is doing something
other than what it claims and the positive reading is not attributable to the path.

  --edit nucleotide  on a protein-only target: there is no is_dna/is_rna token to relabel
  --edit templates   on a target whose crop drew no template: the masks are already zero
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "of3_port"))

import batch_digest as BD  # noqa: E402
from covpath_gradient import fold_nucleotide_into_protein, zero_template_masks  # noqa: E402

EDITS = {"nucleotide": fold_nucleotide_into_protein, "templates": zero_template_masks}


def walk(d, prefix=""):
    # `ref_space_uid_to_perm` is keyed by int, so the name has to be stringified.
    for k in sorted(d, key=str):
        v = d[k]
        if torch.is_tensor(v):
            yield prefix + str(k), v
        elif isinstance(v, dict):
            yield from walk(v, prefix + str(k) + ".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="openfold3")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--cache-file", required=True, type=Path)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--edit", default="nucleotide", choices=sorted(EDITS))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    BD.seed_everything(a.seed)
    ds = BD.build_dataset(a.package, a.data_dir, 4, token_budget=a.crop, split="train",
                          stage=a.stage, cache_file=a.cache_file)
    guard = BD.install_retry_guard(ds)
    dp = ds.datapoint_cache.iloc[a.index]
    s = ds[a.index]
    if guard["retries"]:
        raise SystemExit("silent sample substitution; not the target it claims to be")

    tok = s["token_mask"].bool()
    counts = {k: int((s[k][tok] > 0).sum())
              for k in ("is_protein", "is_dna", "is_rna", "is_ligand") if k in s}
    tpl = int((s["template_pseudo_beta_mask"] != 0).sum()) \
        if "template_pseudo_beta_mask" in s else None

    before = copy.deepcopy(s)
    note = EDITS[a.edit](s)

    after = dict(walk(s))
    moved, compared = [], 0
    for name, v in walk(before):
        compared += 1
        w = after.get(name)
        if w is None or not torch.equal(v, w):
            moved.append(name)

    rep = {
        "instrument": "one feature edit applied to a batch the path cannot act on, tensor by tensor",
        "edit_applied": a.edit,
        "stage": a.stage, "dataset": "weighted-pdb", "crop": a.crop, "index": a.index,
        "seed": a.seed,
        "target": {"pdb_id": str(dp["pdb_id"]),
                   "datapoint": str(dp["preferred_chain_or_interface"]),
                   "n_tokens_real": int(tok.sum()), "molecule_type_tokens": counts,
                   "template_pseudo_beta_mask_nnz": tpl},
        "edit": note,
        "tensors_compared": compared,
        "tensors_moved": moved,
        "n_tensors_moved": len(moved),
        "verdict": "NO-OP" if not moved else "THE EDIT MOVES SOMETHING IT SHOULD NOT",
        "total_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"{a.edit} on {dp['pdb_id']} {counts} tpl={tpl}: "
          f"{len(moved)} of {compared} tensors moved -> {rep['verdict']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
