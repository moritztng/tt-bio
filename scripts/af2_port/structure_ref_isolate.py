"""Is the host structure module's own float32 math the defect, or only the messenger?

Pass 10 crossed two trunk arms' `single` and `pair` through one host-torch structure module and
found the 9 remaining failing taps follow the single input, ignore the pair input, and are not
even monotone in the single. That kills pair-track precision work, and it leaves one reading
untested: the module may be adding error of its own. Every cell in that experiment was driven
from a trunk arm, so every cell carried a nonzero input error.

This drives the module from JAX's own `single` and `pair`, captured whole by
`capture_ref_structure.py`. The input error is then exactly zero and whatever comes out is the
module's own arithmetic against JAX's. Two readings, and they are mutually exclusive:

* the JAX-input cell scores clean and the trunk-input cells do not -- the module is a pure
  amplifier, the 9 taps are trunk error, and pass 10's non-monotonicity is the whole story;
* the JAX-input cell fails too -- the residue is a host reference-implementation difference and
  no amount of trunk precision can reach it.

Host only, no card. The full capture is not committed (~40 MB), so the run is reproducible from
`capture_ref_structure.py` rather than from the repo.

    PYTHONPATH=. env/bin/python3 scripts/af2_port/structure_ref_isolate.py \\
        --full /tmp/af2_ref_structure_full.npz torch=/tmp/torch_r0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structure_isolate import SCORED  # noqa: E402
from tap_gate import ARTIFACTS, DEFAULT_PARAMS, load_inputs, score_one  # noqa: E402

#: The module's two inputs, as the full capture names them.
SINGLE_TAP = "evoformer#0/single"
PAIR_TAP = "evoformer#0/pair"


def _full(cap: dict, tap: str) -> torch.Tensor:
    shape = tuple(int(x) for x in cap[f"{tap}/shape"])
    return torch.from_numpy(cap[f"{tap}/full"].reshape(shape).astype(np.float32))


def _dtype(cap: dict, tap: str) -> str:
    key = f"{tap}/dtype"
    return bytes(cap[key]).decode() if key in cap else "?"


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    """Whole-array agreement. `rel_rms` is the residual as a fraction of the reference's own rms,
    which is the number that says how much of the signal the disagreement is."""
    d = got - ref
    rms = float(np.sqrt((ref * ref).mean()))
    return {"n": int(ref.size),
            "pcc": float(np.corrcoef(ref, got)[0, 1]) if ref.std() > 1e-12 else 1.0,
            "rel_rms": float(np.sqrt((d * d).mean()) / max(rms, 1e-30)),
            "max_abs": float(np.abs(d).max()),
            "bit_exact": bool(np.array_equal(ref, got))}


def control(cap: dict, ref: dict) -> dict:
    """The full capture has to reproduce the committed subsample element for element.

    Same fixture, same config, same recycle count, so any disagreement here means the two
    captures are not the same run and nothing below can be trusted.
    """
    rows = []
    for key in sorted(k[:-len("/full")] for k in cap if k.endswith("/full")):
        if f"{key}/val" in ref:
            want = ref[f"{key}/val"].astype(np.float64)
            got = cap[f"{key}/full"].astype(np.float64)[ref[f"{key}/idx"]]
        elif f"{key}/full" in ref:
            want = ref[f"{key}/full"].astype(np.float64)
            got = cap[f"{key}/full"].astype(np.float64)
        else:
            continue
        rows.append({"tap": key, "n": int(want.size),
                     "max_abs": float(np.abs(got - want).max()),
                     "bit_exact": bool(np.array_equal(got, want))})
    return {"taps": len(rows), "all_bit_exact": all(r["bit_exact"] for r in rows),
            "worst": max((r["max_abs"] for r in rows), default=0.0), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="capture_ref_structure.py artifact")
    ap.add_argument("arms", nargs="*", metavar="NAME=PATH",
                    help="trunk dumps from `tap_gate.py --dump-trunk`, scored the same way")
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--recycle", type=int, default=0)
    args = ap.parse_args()

    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict

    suffix = "" if args.stage == "complex" else "_monomer"
    feats, _ = load_inputs(Path(ARTIFACTS / f"ref_inputs{suffix}.npz"))
    with np.load(ARTIFACTS / f"ref_taps{suffix}.npz", allow_pickle=False) as npz:
        ref = {k: npz[k] for k in npz.files}
    with np.load(args.full, allow_pickle=False) as npz:
        cap = {k: npz[k] for k in npz.files}

    inputs = {"jax": {"single": _full(cap, SINGLE_TAP), "pair": _full(cap, PAIR_TAP)}}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        with np.load(path) as npz:
            inputs[name] = {k: torch.from_numpy(npz[k]) for k in ("single", "pair")}

    model = load_af2_model(load_af2_state_dict(args.params), template=args.stage == "complex")
    model.eval()

    scored = [name.format(r=args.recycle) for name in SCORED
              if f"{name.format(r=args.recycle)}/full" in cap]
    cells = []
    for name, arm in inputs.items():
        with torch.no_grad():
            out = model.structure(arm["single"], arm["pair"], feats)
            plddt = model.heads["plddt"](out["representations/structure_module"])
        got = {f"structure_module#{args.recycle}/{k}": v for k, v in out.items()}
        got[f"predicted_lddt_head#{args.recycle}/logits"] = plddt
        rows = []
        for tap in scored:
            want = cap[f"{tap}/full"].astype(np.float64)
            have = got[tap].reshape(-1).float().numpy().astype(np.float64)
            row = {"tap": tap} | _metrics(want, have)
            row["verdict"] = score_one(ref, tap, got[tap])["verdict"]
            rows.append(row)
        cells.append({
            "arm": name,
            "in_single": _metrics(cap[f"{SINGLE_TAP}/full"].astype(np.float64),
                                  arm["single"].reshape(-1).float().numpy().astype(np.float64)),
            "in_pair": _metrics(cap[f"{PAIR_TAP}/full"].astype(np.float64),
                                arm["pair"].reshape(-1).float().numpy().astype(np.float64)),
            "failed": sum(1 for r in rows if r["verdict"] != "PASS"),
            "pcc_min": min(r["pcc"] for r in rows),
            "rel_rms_max": max(r["rel_rms"] for r in rows),
            "rows": rows})

    print(json.dumps({
        "mode": "af2ig_structure_ref_isolate", "stage": args.stage, "recycle": args.recycle,
        "control": control(cap, ref),
        "jax_dtype": {t: _dtype(cap, t) for t in [SINGLE_TAP, PAIR_TAP] + scored},
        "scored": scored, "cells": cells}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
