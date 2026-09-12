#!/usr/bin/env python3
"""cdk2x2_298 structural control for TT_BIO_DEVICE_CONDITIONING.

The device conditioning is not bit-exact -- bf16 device math against fp32 torch, and the fused
bias stack on top -- so a CIF hash cannot score it. `cdk2x2_512` cannot score it either: it is
CDK2 fused to a truncated copy of itself and its unconstrained hinge saturates RMSD for any
change whatever the cause (memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`).
`cdk2x2_298` is the same family with one real domain and no hinge, and the thresholds in
`state/answered/4649-decision.md` are stated against it: <=0.35 A pass, 0.35-0.60 A hold,
>0.60 A reject.

Three folds in one process on one device open: base, base again, and the arm. The repeated base
gives the A/A floor in the same session, which is what makes the arm's number readable -- a
nonzero A/A would mean the comparison is measuring the box, not the change. Score with
`perf/b2x-flag-levers/score298.py <cifdir> --ref base_0`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
FIX = REPO / "perf" / "size512" / "fixtures"

FLAG = "TT_BIO_DEVICE_CONDITIONING"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=298)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    assert FLAG not in os.environ, f"{FLAG} is set in the environment; this script owns it"
    tgt, a3m = FIX / f"cdk2x2_{a.size}.yaml", FIX / f"cdk2x2_{a.size}.a3m"
    msa_dir = Path(__file__).resolve().parent / f".msa_{a.size}"
    one_fold, meta, _state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "host": os.uname().nodename, "size": a.size, "msa_dir": str(msa_dir),
                   "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS},
           "folds": []}
    a.cifdir.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    print("=== cold fold (discarded) ===", flush=True)
    one_fold()

    for tag, val in (("base_0", "0"), ("base_1", "0"), ("devcond_0", "1")):
        os.environ[FLAG] = val
        t0 = time.perf_counter()
        _t, m = one_fold()
        wall = time.perf_counter() - t0
        dest = a.cifdir / f"{a.size}_{tag}"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(struct_dir, dest)
        out["folds"].append({"tag": tag, FLAG: val, "wall_s": round(wall, 4),
                             "plddt": m.get("plddt"), "dir": dest.name})
        print(f"  {tag:10s} {FLAG}={val}  {wall:.3f} s  plddt {m.get('plddt')}", flush=True)
        a.out.write_text(json.dumps(out, indent=1))
    os.environ.pop(FLAG, None)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
