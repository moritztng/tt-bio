#!/usr/bin/env python3
"""of3t-stackbound: COTANGENT_COMPLETE on the 4hhb frame, then the one correction (A42).

  CONTROL  the float64 trunk replay driven by the captured (cot_s, cot_z) against the float64
           full-model reference's own pairformer_stack section, block 47 and all 2,736 tensors,
           each against the 1e-12 bar fixed in PREREGISTERED.md before either file existed.
           of3t-recut's `fit`, imported, not copied.
  OUT      cot_external_sb.pt = (cot_s, cot_z - delta), delta the replay's own
           cot_z_correction. Written ONLY if the control passes: an ungated frame may not drive
           an arm (A40).

usage: frame_check.py   (paths are this row's; see frame.sh)
"""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_recut"))
from n384_check import fit  # noqa: E402

O = Path("/home/ttuser/of3t_stackbound")
REF = O / "ref_f64" / "grads_f64.pt"
ARM = O / "ref_f64_trunk_sb_corrected.pt"
CAP = O / "cot_model_sb.pt"
BND = O / "boundary_model_sb.pt"
PRE = "pairformer_stack.blocks."
BAR = 1e-12


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()


def main() -> int:
    ref = torch.load(REF, map_location="cpu", weights_only=False)
    ref = ref.get("grads", ref)
    arm = torch.load(ARM, map_location="cpu", weights_only=False)
    rep = json.loads((HERE / "REF_F64_TRUNK_SB_CORRECTED.json").read_text())
    if arm["injection_convention"] != "graph-cut-external":
        raise SystemExit(f"{ARM} carries {arm['injection_convention']!r}")
    g = arm["grads"]
    b47 = sorted(k for k in ref if k.startswith(f"{PRE}47.") and ref[k] is not None)
    allk = sorted(k for k in ref if k.startswith(PRE) and ref[k] is not None)
    out = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
           "row": "of3t-stackbound", "device_involved": False,
           "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
           "boundary_version": "upstream OpenFold3 0.4.3 (of3pkg043), 4hhb, 384 real tokens",
           "bar": BAR, "bar_fixed_in": "perf/of3t_stackbound/PREREGISTERED.md",
           "inputs": {k: {"path": str(p), "sha256": sha(p)}
                      for k, p in (("float64_reference", REF), ("replay", ARM),
                                   ("capture_cotangent", CAP), ("boundary", BND))},
           "injection": rep.get("injection"),
           "block47": fit(g, ref, b47), "all_trunk_tensors": fit(g, ref, allk)}
    out["block47_clears"] = out["block47"]["rel_l2_as_is"] <= BAR
    out["all_trunk_clears"] = out["all_trunk_tensors"]["rel_l2_as_is"] <= BAR
    out["COTANGENT_COMPLETE"] = bool(out["block47_clears"] and out["all_trunk_clears"]
                                     and out["all_trunk_tensors"]["n_absent_from_arm"] == 0)
    if out["COTANGENT_COMPLETE"]:
        cot_s, cot_z = torch.load(CAP, map_location="cpu", weights_only=False)["cot"]
        cot_s, cot_z = cot_s.to(torch.float64), cot_z.to(torch.float64)
        delta = arm["cot_z_correction"].to(torch.float64)
        assert tuple(delta.shape) == tuple(cot_z.shape), (delta.shape, cot_z.shape)
        ext = cot_z - delta
        p = O / "cot_external_sb.pt"
        torch.save({"cot": (cot_s, ext), "injection_convention": "graph-cut-external",
                    "from": {"capture": str(CAP), "correction_from": str(ARM)}}, p)
        out["cot_external"] = {
            "path": str(p), "sha256": sha(p), "bytes": p.stat().st_size,
            "norms": {"cot_s": float(cot_s.norm()), "cot_z_hooked": float(cot_z.norm()),
                      "delta": float(delta.norm()), "cot_z_external": float(ext.norm()),
                      "duplicate_share_of_the_hooked_cot_z":
                          float(delta.norm()) / float(cot_z.norm())}}
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()
    (HERE / "COTANGENT_COMPLETE.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if out["COTANGENT_COMPLETE"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
