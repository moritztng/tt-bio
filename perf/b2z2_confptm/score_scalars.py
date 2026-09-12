#!/usr/bin/env python3
"""The quantity a confidence-head change actually moves, on its own.

`score_conf.py` reads the CIF, and the CIF carries coordinates and the per-atom pLDDT. This
port touches neither: pLDDT comes from `to_plddt_logits(s)`, which stays in torch. What it does
move is pae, pde and the pTM family, and none of those reach a file unless `--write_pae` is on.
So intercept the head's own return dict and diff it: base against base for the floor, arm against
base for the change. Three folds, one process, one device open, exactly like `control298.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
FIX = REPO / "perf" / "size512" / "fixtures"

SCALARS = ("ptm", "iptm", "ligand_iptm", "protein_iptm",
           "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde")
MATRICES = ("pae", "pde", "plddt")


def capture(out_dict):
    import torch
    rec = {k: round(float(out_dict[k].reshape(-1)[0]), 6) for k in SCALARS if k in out_dict}
    for k in MATRICES:
        if k in out_dict:
            rec[f"{k}_"] = torch.as_tensor(out_dict[k]).float().clone()
    return rec


def diff(ref, arm):
    d = {f"d_{k}": round(arm[k] - ref[k], 6) for k in SCALARS if k in ref and k in arm}
    for k in MATRICES:
        a, b = ref.get(f"{k}_"), arm.get(f"{k}_")
        if a is None or b is None:
            continue
        e = (a - b).abs()
        d[f"{k}_max_abs"] = round(float(e.max()), 6)
        d[f"{k}_mean_abs"] = round(float(e.mean()), 6)
        d[f"{k}_mean_ref"] = round(float(a.mean()), 6)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--flag", default="TT_BIO_DEVICE_CONF_HEADS")
    a = ap.parse_args()
    FLAG = a.flag

    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    import tt_bio.boltz2 as boltz2
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    assert FLAG not in os.environ, f"{FLAG} is set in the environment; this script owns it"
    grabbed: list = []
    orig = boltz2.ConfidenceHeads.forward

    def wrapped(self, *args, **kw):
        out = orig(self, *args, **kw)
        grabbed.append(capture(out))
        return out

    boltz2.ConfidenceHeads.forward = wrapped

    tgt, a3m = FIX / f"cdk2x2_{a.size}.yaml", FIX / f"cdk2x2_{a.size}.a3m"
    msa_dir = Path(__file__).resolve().parent / f".msa_{a.size}"
    one_fold, meta, _state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "host": os.uname().nodename, "size": a.size, "flag": FLAG,
                   "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS},
           "folds": [], "aa": None, "arm": None}

    print("=== cold fold (discarded) ===", flush=True)
    one_fold()
    grabbed.clear()

    keep = {}
    for tag, val in (("base_0", "0"), ("base_1", "0"), ("arm_0", "1")):
        os.environ[FLAG] = val
        t0 = time.perf_counter()
        one_fold()
        keep[tag] = grabbed[-1]
        out["folds"].append({"tag": tag, FLAG: val,
                             "wall_s": round(time.perf_counter() - t0, 4),
                             **{k: v for k, v in keep[tag].items() if not k.endswith("_")}})
        print(f"  {tag:8s} {FLAG}={val}  "
              + " ".join(f"{k}={keep[tag][k]}" for k in SCALARS if k in keep[tag]), flush=True)
    os.environ.pop(FLAG, None)

    out["aa"] = diff(keep["base_0"], keep["base_1"])
    out["arm"] = diff(keep["base_0"], keep["arm_0"])
    print("  A/A floor:", json.dumps(out["aa"]), flush=True)
    print("  ARM      :", json.dumps(out["arm"]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
