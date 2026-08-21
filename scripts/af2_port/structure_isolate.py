"""Is the structure module a second amplifier, and which of its two inputs drives it?

Pass 9 left 9 failing taps and every one is a structure-module tap. The module is float32 host
torch in both arms, so it cannot hold a device term -- but that leaves two readings. Either the
module is only reporting the trunk error it was handed, or its own float32 math differs from
JAX's. The capture stores `single` and `pair` subsampled, so the module cannot be driven from the
reference directly; what it can be driven from is either arm's own trunk output, and crossing the
two inputs says which channel the module's output error follows.

Four cells per pair of arms: (single, pair) from the same arm on the diagonal, and the two
crossings off it. If the device arm's structure error tracks its pair and not its single, the
lever is pair growth; if the two crossings both land near the diagonal, the module amplifies
whatever it is given and is not itself the defect.

    PYTHONPATH=. env/bin/python3 scripts/af2_port/tap_gate.py --recycles 0 \\
        --dump-trunk /tmp/torch_r0.npz > /dev/null
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. env/bin/python3 scripts/af2_port/tap_gate.py --device \\
        --recycles 0 --dump-trunk /tmp/device_r0.npz > /dev/null
    PYTHONPATH=. env/bin/python3 scripts/af2_port/structure_isolate.py \\
        torch=/tmp/torch_r0.npz device=/tmp/device_r0.npz
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

from tap_gate import ARTIFACTS, DEFAULT_PARAMS, load_inputs, score_one  # noqa: E402

#: The taps the pass 9 device leg failed on, plus the two the filter actually reads.
SCORED = ("structure_module#{r}/final_atom_positions",
          "structure_module#{r}/final_atom14_positions",
          "structure_module#{r}/final_affines",
          "structure_module#{r}/traj",
          "structure_module#{r}/representations/structure_module",
          "structure_module#{r}/sidechains/angles_sin_cos",
          "predicted_lddt_head#{r}/logits")

#: The module's own inputs, scored the same way so the transfer ratio has a denominator.
INPUTS = {"single": "linear/single_activations#0/out", "pair": "evoformer#0/pair"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", metavar="NAME=PATH",
                    help="trunk dumps from `tap_gate.py --dump-trunk`")
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

    arms = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        with np.load(path) as npz:
            arms[name] = {k: torch.from_numpy(npz[k]) for k in ("single", "pair")}

    model = load_af2_model(load_af2_state_dict(args.params), template=args.stage == "complex")
    model.eval()

    # The inputs, scored against the reference. This is the denominator of the transfer ratio.
    inputs = {}
    for name, arm in arms.items():
        for key, tap in INPUTS.items():
            inputs[f"{name}/{key}"] = score_one(ref, tap, arm[key])

    cells = []
    for s_name in arms:
        for p_name in arms:
            with torch.no_grad():
                out = model.structure(arms[s_name]["single"], arms[p_name]["pair"], feats)
                plddt = model.heads["plddt"](out["representations/structure_module"])
            got = {f"structure_module#{args.recycle}/{k}": v for k, v in out.items()}
            got[f"predicted_lddt_head#{args.recycle}/logits"] = plddt
            rows = [score_one(ref, name.format(r=args.recycle), got[name.format(r=args.recycle)])
                    for name in SCORED if f"{name.format(r=args.recycle)}/shape" in ref]
            cells.append({
                "single": s_name, "pair": p_name,
                "in_single": 1.0 - inputs[f"{s_name}/single"]["pcc"],
                "in_pair": 1.0 - inputs[f"{p_name}/pair"]["pcc"],
                "failed": sum(1 for r in rows if r["verdict"] != "PASS"),
                "pcc_min": min(r["pcc"] for r in rows if "pcc" in r),
                "rows": rows})

    print(json.dumps({"mode": "af2ig_structure_isolate", "stage": args.stage,
                      "recycle": args.recycle, "arms": sorted(arms),
                      "inputs": inputs, "cells": cells}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
