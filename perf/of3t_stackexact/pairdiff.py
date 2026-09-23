#!/usr/bin/env python3
"""of3t-modelever pairdiff.py, with this row's arm reports added to the lookup. Two trunk arms against each other, in one metric, so a floor and a separation sit side by side.

    pairdiff.py <label> <a.pt> <b.pt> [--out report.json]

Per-parameter bit identity (as perf/of3t_verbinstall/armdiff.py), plus the concatenated
rel = ||a-b|| / ||b||, norm ratio and cos over the common tensors. On an A/A pair that rel IS the
floor; on the lever pair it is the separation, and the two are only comparable because they are
the same statistic over the same tensor set. Says nothing about accuracy. CPU only.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import socket
import subprocess
import sys

import torch


def _grads(d):
    for k in ("grads", "gradients", "ours", "g"):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return d if isinstance(d, dict) else {}


# Every device arm this can be asked about, with the provenance its runner wrote.
ARM_REPORTS = ("perf/of3t_stackexact/DEV_*.json", "perf/of3t_modelever/DEV_*.json", "perf/of3t_recut/DEV_RENORM_MODEL_N384_*.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()


def arm_origin(pt):
    """Host, board and card of the arm that PRODUCED `pt` (D155), read from its runner's
    provenance and tied to it by the digest, not the path. A bit-identity claim is a claim about
    device output, so the host of the comparison is not the host of the claim."""
    digest = sha(pt)
    for rep in sorted(f for g in ARM_REPORTS for f in glob.glob(g)):
        pv = json.load(open(rep)).get("provenance", {})
        if (pv.get("out") or {}).get("path") == os.path.abspath(pt):
            return {"pt": pt, "sha256": digest, "report": rep,
                    "sha256_matches_report": pv["out"]["sha256"] == digest,
                    "host": pv.get("host"), "board": pv.get("board_class"), "card": pv.get("card"),
                    "aiclk_mhz_sampled_DURING": pv.get("aiclk_mhz_sampled_DURING")
                                                or pv.get("aiclk_mhz_during_the_run"),
                    "host_quiet": (pv.get("host_quiet") or "").splitlines()[-1:] or None}
    raise SystemExit(f"STOP: no arm report records producing {pt}; an unstamped bit-identity "
                     f"claim is what D155/D249 refuse")


def main() -> int:
    label, pa, pb = sys.argv[1:4]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
    a = _grads(torch.load(pa, map_location="cpu"))
    b = _grads(torch.load(pb, map_location="cpu"))
    common = sorted(k for k in set(a) & set(b)
                    if torch.is_tensor(a[k]) and torch.is_tensor(b[k]) and a[k].shape == b[k].shape)
    bit, e2, aa, bb, ab = 0, 0.0, 0.0, 0.0, 0.0
    for k in common:
        x, y = a[k].double(), b[k].double()
        bit += int(torch.equal(x, y))
        e2 += float(((x - y) ** 2).sum()); aa += float((x * x).sum())
        bb += float((y * y).sum()); ab += float((x * y).sum())
    rel = math.sqrt(e2 / bb) if bb else None
    r = math.sqrt(aa / bb) if bb else None
    cos = ab / math.sqrt(aa * bb) if aa and bb else None
    rep = {"label": label, "a": pa, "b": pb,
           "compared": len(common), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
           "bit_identical": bit, "differing": len(common) - bit,
           "all_bit_identical": bit == len(common) and set(a) == set(b),
           "concatenated_rel_a_vs_b": rel, "norm_ratio_a_over_b": r, "cos": cos,
           "arms": {"a": arm_origin(pa), "b": arm_origin(pb)},
           "device_involved": False,
           "why_no_aiclk": "the COMPARISON is CPU only. The arms it compares ran on a card, and "
                           "their host, board, card and DURING-sampled AICLK are under `arms`",
           "comparison_environment": {
               "host": socket.gethostname(), "board": None, "card": None,
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip() or None}}
    assert all(v["sha256_matches_report"] for v in rep["arms"].values()), rep["arms"]
    print("PAIRDIFF " + json.dumps(rep))
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
