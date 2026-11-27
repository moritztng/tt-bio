#!/usr/bin/env python3
"""Upstream's own bf16 floor on the loss-parity batches: what a bf16 forward of the SAME model
does to each G3 loss term, with nothing ported.

Reruns `call_nn` from each dump in ``--ref`` (same inputs, same RNG state) under
``torch.autocast("cpu", torch.bfloat16)`` with fp32 weights, scores the outputs with
upstream's fp32 `compute_loss`, and compares every optimisation term at the RECORDED winners
with the recorded fp32 terms. The fp32 arm (``--arm fp32``) is the control: it must land at the
replay floor (2.5e-6), or the harness, not the dtype, is what moved.

This sets the scale a TT bf16 reading is judged against. It does not move the bar.
Prints one JSON line a batch and a summary with the worst rel per term.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loss_parity_reference import build_model, load_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--arm", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--only", default=None, help="comma-separated dump names, e.g. batch08")
    a = ap.parse_args()

    import torch
    from rfproteina.utils.torch import minimum_loss_attempt_indices  # noqa: F401  (import check)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    _, model = build_model(Path(os.environ["RFD4_REPO"]).expanduser(), torch.float32)
    fm = model.fm
    only = set(a.only.split(",")) if a.only else None
    worst, rows = {}, []
    for path in sorted(a.ref.glob("batch*.npz")):
        if only and path.stem not in only:
            continue
        batch, meta, _, rec = load_batch(path, torch.float32)
        win = torch.as_tensor(meta["winners"])
        torch.set_rng_state(meta["rng_state"])
        with torch.no_grad():
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.arm == "bf16"):
                out = model.call_nn(batch, n_recycle=meta["depth"])
            out = {k: ({m: v.float() for m, v in d.items()} if isinstance(d, dict) else d.float())
                   for k, d in out.items()}
            losses = fm.compute_loss(batch=batch, nn_out=out)
            ours = {k: float(v.mean()) for k, v in fm.optimization_terms({k: v[win] for k, v in losses.items()}).items()}
            ref = {k: float(v.mean()) for k, v in fm.optimization_terms(
                {k: torch.as_tensor(v)[win] for k, v in rec.items()}).items()}
        ours["total"], ref["total"] = sum(ours.values()), sum(ref.values())
        rel = {k: (0.0 if ours[k] == ref[k] else abs(ours[k] - ref[k]) / abs(ref[k]) if ref[k] else float("inf"))
               for k in ref}
        for k, v in rel.items():
            worst[k] = max(worst.get(k, 0.0), v)
        rows.append(dict(batch=path.stem, arm=a.arm, rel=rel, ours=ours, ref=ref))
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps(dict(summary=True, arm=a.arm, batches=len(rows), worst_rel=worst)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
