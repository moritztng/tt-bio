#!/usr/bin/env python3
"""The interleaved step A/B for the `ttnn.mac` fusion, plus its program count and its parity.

One process, one grabbed `Diffusion.__call__`, arms alternating base/mac/base/mac/... so drift
shows as an A/A spread instead of folding into the ratio (`_L1_OUT_RUNG` rewrites a baseline
mid-run, so batching the arms is not allowed here).

The gate is a module global read at call time, so an arm is a flip of `T._MAC_FUSE` -- with one
thing to undo by hand: `AdaLN`'s atom-level memo caches `(s_scale, s_bias)` for the whole rollout
and the fused arm stores a SIGMOID'd `s_scale` there. Flipping without dropping the memo mixes the
two conventions and is silently wrong, so every AdaLN's memo is cleared on every flip.

Parity is checked on the step's own output tensor with `torch.equal`, which is the strongest
statement available without a fold, and a negative control (one arm with the memo NOT cleared)
must fail for the check to mean anything.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "b2z2_step_fusion"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))


def adalns(T):
    out = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, T.AdaLN):
                out.append(obj)
        except ReferenceError:
            continue
    return out


def set_arm(T, on, memos):
    T._MAC_FUSE = on
    for m in memos:
        m._s_memo = None
        m._s_memo_src = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import step_probe
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from itemize import top_level_spans

    step_probe.OUT_PATH = a.out.with_suffix(".grab.json")
    step_probe.OUT = {"env": {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "mode": "step_ab", "size": a.size,
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "ttnn": getattr(ttnn, "__file__", "?"),
        "flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_BIO_")},
        "loadavg": open("/proc/loadavg").read().split()[:3]}}
    OUT = {"env": step_probe.OUT["env"]}
    dev, g = step_probe.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    memos = adalns(T)
    OUT["n_adaln"] = len(memos)
    print(f"  {len(memos)} AdaLN modules reachable", flush=True)

    call = lambda: g["obj"](*g["args"], **g["kwargs"])          # noqa: E731

    # ---- program count, both arms ----------------------------------------------------------
    OUT["programs"] = {}
    for name, on in (("base", False), ("mac", True)):
        set_arm(T, on, memos)
        call()
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        call()
        ttnn.synchronize_device(dev)
        ops, _ = top_level_spans(ttnn.graph.end_graph_capture())
        names = [o["name"] for o in ops]
        disp = [n for n in names if n != "ttnn.deallocate"]
        OUT["programs"][name] = {"top_level": len(names), "dispatching": len(disp),
                                 "mac": sum(n == "ttnn.mac" for n in names)}
        print(f"  {name}: {len(names)} top-level ttnn calls, {len(disp)} dispatching, "
              f"{OUT['programs'][name]['mac']} mac", flush=True)

    # ---- parity on the step output, and its negative control --------------------------------
    set_arm(T, False, memos)
    ref = ttnn.to_torch(call()).float()
    set_arm(T, True, memos)
    got = ttnn.to_torch(call()).float()
    OUT["parity"] = {"bit_exact": bool(torch.equal(ref, got)),
                     "max_abs": round(float((got - ref).abs().max()), 8),
                     "ref_absmax": round(float(ref.abs().max()), 6)}
    T._MAC_FUSE = False                       # negative control: flip WITHOUT clearing the memo
    bad = ttnn.to_torch(call()).float()
    OUT["parity"]["negative_control_differs"] = not bool(torch.equal(ref, bad))
    OUT["parity"]["negative_control_max_abs"] = round(float((bad - ref).abs().max()), 8)
    print(f"  parity bit_exact={OUT['parity']['bit_exact']} max_abs={OUT['parity']['max_abs']} "
          f"neg_control_differs={OUT['parity']['negative_control_differs']}", flush=True)

    # ---- interleaved timing -----------------------------------------------------------------
    fence = step_probe.make_fence(ttnn, dev)
    walls = {"base": [], "mac": []}
    set_arm(T, False, memos)
    for _ in range(3):
        call()
    ttnn.synchronize_device(dev)
    for r in range(a.rounds):
        for name, on in (("base", False), ("mac", True)):
            set_arm(T, on, memos)
            for _ in range(2):
                call()
            ttnn.synchronize_device(dev)
            fence()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                call()
            ttnn.synchronize_device(dev)
            ms = 1e3 * (time.perf_counter() - t0) / a.reps
            walls[name].append(round(ms, 4))
            print(f"  round {r} {name:5s} {ms:8.4f} ms", flush=True)
    med = {k: st.median(v) for k, v in walls.items()}
    OUT["step"] = {"ms": {k: round(v, 4) for k, v in med.items()},
                   "all": walls,
                   "spread_pct": {k: round(100 * (max(v) - min(v)) / st.median(v), 3)
                                  for k, v in walls.items()},
                   "ratio": round(med["base"] / med["mac"], 5),
                   "aa_floor_base": round(max(walls["base"]) / min(walls["base"]), 5),
                   "aa_floor_mac": round(max(walls["mac"]) / min(walls["mac"]), 5)}
    print(f"  STEP base {med['base']:.4f} ms  mac {med['mac']:.4f} ms  "
          f"RATIO {OUT['step']['ratio']:.5f}x  A/A base {OUT['step']['aa_floor_base']:.5f}x",
          flush=True)
    a.out.write_text(json.dumps(OUT, indent=1))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
