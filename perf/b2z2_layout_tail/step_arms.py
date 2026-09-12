#!/usr/bin/env python3
"""Both layout levers on the diffusion step they were built for, interleaved in one process.

  base        main's path
  padtail     TT_BIO_HEAD_PAD_TAIL -- the tail carries the head split's zero pad lanes instead of
              spending four programs stripping them
  fusedqkv    TT_BIO_DIT_FUSED_QKV -- the qkv projection writes the head split itself, so
              nlp_create_qkv_heads never runs
  both        the two together, which is the number that matters

Every arm is a module-global flip on one grabbed `Diffusion.__call__`, so no arm can inherit
another's compiled program and the four are genuinely the same work. Arm order rotates block to
block so no arm keeps the lead. The step grab and the fence are
`perf/b2z2_step_fusion/step_probe.py`'s.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location(
    "_step_probe", ROOT / "perf" / "b2z2_step_fusion" / "step_probe.py")
SP = importlib.util.module_from_spec(_spec)
sys.modules["_step_probe"] = SP
_spec.loader.exec_module(SP)

ARMS = ("base", "padtail", "fusedqkv", "both")
OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def set_arm(T, Q, arm):
    T._HEAD_PAD_TAIL = arm in ("padtail", "both")
    Q._WIDE_ENABLED = arm in ("fusedqkv", "both")


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    OUT_PATH = a.out
    a.out.parent.mkdir(parents=True, exist_ok=True)
    SP.OUT, SP.OUT_PATH = OUT, OUT_PATH

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio import triatt_qkv as Q
    import tt_baseline as B

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": open("/proc/loadavg").read().split()[:3]}
    dump()

    set_arm(T, Q, "base")
    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    call = lambda: g["obj"](*g["args"], **g["kwargs"])                    # noqa: E731
    fence = SP.make_fence(ttnn, dev)

    # parity first, while nothing is timed: every arm against base, and each arm's own repeat
    par = {}
    set_arm(T, Q, "base")
    base_out = ttnn.to_torch(call()).clone()
    base_rep = ttnn.to_torch(call()).clone()
    par["base_deterministic"] = bool(torch.equal(base_out, base_rep))
    for arm in ARMS[1:]:
        set_arm(T, Q, arm)
        got = ttnn.to_torch(call()).clone()
        d = (base_out.float() - got.float()).abs()
        par[arm] = {"bit_exact": bool(d.max() == 0), "max_abs": float(d.max()),
                    "mean_abs": float(d.mean())}
    par["served"] = {"fused_qkv_calls": Q.WIDE_STATS[0], "declined": Q.WIDE_STATS[1],
                     "rejects": {str(k): v for k, v in Q.WIDE_REJECTS.items()}}
    OUT["parity"] = par
    dump()
    print("  parity:", json.dumps(par, indent=1), flush=True)

    for arm in ARMS:                       # warm every arm's programs before any of them is timed
        set_arm(T, Q, arm)
        for _ in range(3):
            call()
    ttnn.synchronize_device(dev)
    fence()

    walls = {arm: [] for arm in ARMS}
    order = []
    for b in range(a.blocks):
        rot = ARMS[b % len(ARMS):] + ARMS[:b % len(ARMS)]
        for arm in rot:
            set_arm(T, Q, arm)
            t = time.perf_counter()
            for _ in range(a.reps):
                call()
            ttnn.synchronize_device(dev)
            ms = 1e3 * (time.perf_counter() - t) / a.reps
            walls[arm].append(ms)
            order.append((arm, round(ms, 4)))
        fence()
    set_arm(T, Q, "base")

    med = {k: st.median(v) for k, v in walls.items()}
    base = walls["base"]
    h = len(base) // 2
    OUT["ab"] = {
        "ms": {k: round(v, 4) for k, v in med.items()},
        "ratio_vs_base": {k: round(med["base"] / v, 5) for k, v in med.items()},
        "delta_ms": {k: round(med["base"] - v, 4) for k, v in med.items()},
        "aa_floor": round(st.median(base[:h]) / st.median(base[h:]), 5),
        "spread_pct": {k: round(100 * (max(v) - min(v)) / st.median(v), 3)
                       for k, v in walls.items()},
        "walls": {k: [round(x, 4) for x in v] for k, v in walls.items()},
        "order": order, "reps": a.reps, "blocks": a.blocks,
    }
    dump()
    for k in ARMS:
        print(f"  {k:9s} {med[k]:8.4f} ms   {med['base'] / med[k]:.5f}x", flush=True)
    print(f"  A/A {OUT['ab']['aa_floor']}", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
