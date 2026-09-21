#!/usr/bin/env python3
"""Attribute OpenDDE's two caught L1 circular-buffer overflows to a call site.

Three passes of this row have reported these throws and none has said WHERE they come from.
Pass 2 guessed the pair projection's L1-output leg and pass 5 withdrew that guess. This stops
guessing: it instruments BOTH ends of the catch.

  THROWERS  -- a wrapper on the ttnn entry points that build a program records a traceback
               whenever one raises with "circular buffers" in the message, then re-raises so the
               shipped handler still sees it and the fold is unchanged.
  RECORDERS -- every place tt_bio writes down a caught L1 refusal (`_L1_OUT_REFUSED`,
               `_PM_OVER_L1`, `_GATE_OVER_L1`, `_l1_out_narrow`, `SG.note_l1_refusal`) is wrapped
               to record a traceback too, so if a throw is swallowed without a thrower match the
               recorder still names the handler.

Instrumenting only one end is how the earlier guess went wrong: the handler that CATCHES is not
the site that ASKS, and the size-limit question is about the asker.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))
sys.path.insert(0, str(ROOT / "perf" / "allm_safety"))

import card_guard  # noqa: E402
card_guard.preflight()

HITS = {"throw": [], "record": []}
MARK = "circular buffers"


def _frames():
    """Caller frames inside tt_bio, innermost last -- the model code, not the ttnn plumbing."""
    out = []
    for f in traceback.extract_stack()[:-2]:
        if "/tt_bio/" in f.filename and "device_lease" not in f.filename:
            out.append(f"{Path(f.filename).name}:{f.lineno} {f.name}")
    return out[-12:]


def wrap_thrower(mod, name):
    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_l1_wrapped", False):
        return
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:                                   # noqa: BLE001
            if MARK in str(e):
                HITS["throw"].append({"op": name, "msg": str(e)[:200], "frames": _frames()})
            raise
    w._l1_wrapped = True
    setattr(mod, name, w)


class TracingSet(set):
    def __init__(self, src, label):
        super().__init__(src)
        self._label = label
    def add(self, x):
        HITS["record"].append({"recorder": self._label, "key": repr(x)[:160],
                               "frames": _frames()})
        return super().add(x)


def wrap_recorder_fn(mod, name, label):
    fn = getattr(mod, name, None)
    if fn is None:
        return
    def w(*a, **k):
        HITS["record"].append({"recorder": label, "key": repr(a[:1])[:160], "frames": _frames()})
        return fn(*a, **k)
    setattr(mod, name, w)


def main():
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    for n in ("linear", "matmul"):
        wrap_thrower(ttnn, n)
    if hasattr(ttnn, "experimental"):
        wrap_thrower(ttnn.experimental, "minimal_matmul")
    SG = getattr(TS, "SG", None)
    if SG is not None:
        wrap_thrower(SG, "sdpa")
        wrap_recorder_fn(SG, "note_l1_refusal", "SG.note_l1_refusal")

    T._L1_OUT_REFUSED = TracingSet(T._L1_OUT_REFUSED, "tenstorrent._L1_OUT_REFUSED")
    TS._PM_OVER_L1 = TracingSet(TS._PM_OVER_L1, "triatt_sdpa._PM_OVER_L1")
    TS._GATE_OVER_L1 = TracingSet(TS._GATE_OVER_L1, "triatt_sdpa._GATE_OVER_L1")
    wrap_recorder_fn(T, "_l1_out_narrow", "tenstorrent._l1_out_narrow")

    model, size = "opendde", 512
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, model)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        model, ROOT / f".msa_allmsafety_{model}_{size}",
        fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m")
    print(f"=== {model} {size}: cold fold, recycling={B.RECYCLING_STEPS} "
          f"sampling={B.SAMPLING_STEPS} ===", flush=True)
    s, _ = one_fold()
    print(f"  cold {s:.2f}s  throws_caught={len(HITS['throw'])} "
          f"records={len(HITS['record'])}", flush=True)

    out = ROOT / "perf" / "allm_safety" / "out" / "l1_attribution.json"
    out.write_text(json.dumps({"model": model, "size": size,
                               "recycling_steps": B.RECYCLING_STEPS,
                               "sampling_steps": B.SAMPLING_STEPS,
                               "chip": os.environ.get("TT_VISIBLE_DEVICES"),
                               "grid": list(T.COMPUTE_GRID_MAIN), **HITS}, indent=1))
    for h in HITS["throw"]:
        print(f"\nTHROW in ttnn.{h['op']}: {h['msg'][:110]}")
        for f in h["frames"]:
            print(f"    {f}")
    for h in HITS["record"][:6]:
        print(f"\nRECORDED by {h['recorder']} key={h['key'][:80]}")
        for f in h["frames"]:
            print(f"    {f}")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
