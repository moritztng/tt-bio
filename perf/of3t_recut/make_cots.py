#!/usr/bin/env python3
"""of3t-recut job 3: the two cotangent files the device arms are driven from.

`delta` is the double-counted route, `d<cot_s, s_out>/d(z_out)`, taken from the f64 reference
arm's own graph. It is a property of the reference's downstream, not of the arm being scored, so
every arm is corrected with this one tensor rather than with its own.

  cot_external.pt    (cot_s, cot_z - delta)   the injection the fixed instrument would use. Drives
                                              the END TO END control arm.
  cot_delta_only.pt  (0, delta)               drives the SHORTCUT arm. The parameter gradient is
                                              linear in the injected cotangent
                                              (perf/of3t_frameself/TWOBASIS.json,
                                              6.435383259300361e-15), so
                                              g_corrected = g(cot_s, cot_z) - g(0, delta) and
                                              every banked arm is rescored with one extra arm
                                              instead of a re-run.

Neither file is written into a concluded row's namespace and neither replaces the capture.
"""
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import torch

O = Path("/home/ttuser/of3t_recut")
CAP = Path("/home/ttuser/of3t_modelframe/cot_model_n384.pt")
CAP_SHA = "4e66d1ef45da2eec18fdff9489d68e980df928dc43d10fd4af82d14ec67141ac"
ARM = O / "ref_f64_model_n384_corrected.pt"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()


def main() -> int:
    if sha(CAP) != CAP_SHA:
        raise SystemExit(f"{CAP} is not the capture this row scores")
    cot_s, cot_z = torch.load(CAP, map_location="cpu", weights_only=False)["cot"]
    cot_s = cot_s.to(torch.float64)
    cot_z = cot_z.to(torch.float64)
    arm = torch.load(ARM, map_location="cpu", weights_only=False)
    if arm["injection_convention"] != "graph-cut-external":
        raise SystemExit(f"{ARM} carries {arm['injection_convention']!r}")
    delta = arm["cot_z_correction"].to(torch.float64)
    if tuple(delta.shape) != tuple(cot_z.shape):
        raise SystemExit(f"correction {tuple(delta.shape)} vs cot_z {tuple(cot_z.shape)}")

    ext = cot_z - delta
    torch.save({"cot": (cot_s, ext),
                "injection_convention": "graph-cut-external",
                "from": {"capture": str(CAP), "correction_from": str(ARM)}},
               O / "cot_external.pt")
    torch.save({"cot": (torch.zeros_like(cot_s), delta),
                "injection_convention": "delta-only (the double-counted route alone)",
                "from": {"capture": str(CAP), "correction_from": str(ARM)}},
               O / "cot_delta_only.pt")

    rep = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
           "row": "of3t-recut", "defect": "D242", "device_involved": False,
           "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
           "capture": {"path": str(CAP), "sha256": CAP_SHA},
           "correction_from": {"path": str(ARM),
                               "convention": arm["injection_convention"]},
           "norms": {
               "cot_s": float(cot_s.norm()),
               "cot_z_hooked": float(cot_z.norm()),
               "delta": float(delta.norm()),
               "cot_z_external": float(ext.norm()),
               "duplicate_share_of_the_hooked_cot_z": float(delta.norm())
                                                      / float(cot_z.norm()),
               "hooked_over_external": float(cot_z.norm()) / float(ext.norm()),
           },
           "out": {}}
    for nm in ("cot_external.pt", "cot_delta_only.pt"):
        rep["out"][nm] = {"path": str(O / nm), "sha256": sha(O / nm),
                          "bytes": (O / nm).stat().st_size}
    Path("perf/of3t_recut/COTANGENTS.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
