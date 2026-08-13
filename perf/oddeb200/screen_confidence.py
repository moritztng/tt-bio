#!/usr/bin/env python3
"""Decompose `top:confidence` (1.568 s/fold), the last unpriced region in the ledger.

The state doc called this "the next CPU-only screen, needs no box" on the strength of
protenix.py:1287-1316 looking like host torch. That reading is wrong: ConfidenceHead.confidence
runs `so, zo = self.pf(T(s_t), T(z))`, a full Pairformer stack ON DEVICE, in the middle of the
host work. So the region is a host/device mix and the screen needs the card.

What this measures, per fold:
  total        wall of ConfidenceHead.confidence
  pf_device    the device Pairformer inside it, synchronised so the number is honest
               (PLAYBOOKS ACCELERATE rule 1: syncs are ADDED to measure)
  host         total - pf_device, i.e. the F.linear / cdist / one-hot block the doc priced
               at ~15 GFLOP of single-node host compute

That split decides the owed question: if the region is dominated by pf_device there is nothing
for a host-side rewrite to win and `confidence_device` (TT_PROTENIX_CONF_DEVICE=1, ships OFF,
gated on unverified plDDT PCC) is the only lever; if it is dominated by host, the host path is
worth a pass.

Runs the real 512 aa fold, so the shapes are production shapes.
"""
import json, sys, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/opendde-beat-b200")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "gpu_vs_tt"))

import tt_baseline as TB
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.protenix import ConfidenceHead

REC = []
_orig_conf = ConfidenceHead.confidence


def timed_conf(self, *a, **k):
    pfcls = type(self.pf)
    orig_pf = pfcls.__call__
    acc = [0.0, 0]

    def timed_pf(inst, *aa, **kk):
        t = time.perf_counter()
        try:
            return orig_pf(inst, *aa, **kk)
        finally:
            ttnn.synchronize_device(T.get_device())
            acc[0] += time.perf_counter() - t
            acc[1] += 1

    pfcls.__call__ = timed_pf
    t0 = time.perf_counter()
    try:
        return _orig_conf(self, *a, **k)
    finally:
        pfcls.__call__ = orig_pf
        tot = time.perf_counter() - t0
        REC.append({"total_s": round(tot, 4), "pf_device_s": round(acc[0], 4),
                    "host_s": round(tot - acc[0], 4), "pf_calls": acc[1]})


ConfidenceHead.confidence = timed_conf

FIX = WT / "perf" / "size512" / "fixtures"
one_fold, meta, state = TB.build_fold(
    "opendde", WT / ".msa_om512_512", FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_512.a3m")

folds = []
for i in range(3):                      # fold 0 is cold; 1 and 2 are warm
    t, m = one_fold()
    folds.append({"fold": i, "kind": "cold" if i == 0 else "warm",
                  "fold_s": round(t, 3), "plddt": m.get("plddt"),
                  "confidence": REC[-1] if REC else None})
    print(f"fold {i} {folds[-1]['kind']:4} {t:8.3f} s  confidence {REC[-1]}", flush=True)

warm = [f for f in folds if f["kind"] == "warm"]
cw = [f["confidence"] for f in warm]
summary = {
    "host": meta.get("machine", "tt-quietbox2"), "card": meta.get("visible_devices"),
    "ttnn": meta.get("ttnn_version"), "model": "opendde", "n_tokens": 512,
    "folds": folds,
    "warm_confidence_total_s": round(sum(c["total_s"] for c in cw) / len(cw), 4),
    "warm_confidence_pf_device_s": round(sum(c["pf_device_s"] for c in cw) / len(cw), 4),
    "warm_confidence_host_s": round(sum(c["host_s"] for c in cw) / len(cw), 4),
    "warm_fold_s": round(sum(f["fold_s"] for f in warm) / len(warm), 3),
}
summary["pf_device_share"] = round(
    summary["warm_confidence_pf_device_s"] / summary["warm_confidence_total_s"], 4)
out = WT / "perf" / "oddeb200" / "screen_confidence.json"
out.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
T.cleanup()
