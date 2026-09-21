"""Price D56's lever at the op, where host load cannot reach it.

The step-scope ON/OFF pair is smaller than the host it is measured on: two identical taped
steps in one process came out 100.58 s apart. So this measures the thing the lever actually
adds -- one `ttnn.sum(y, precise)` and one `ttnn.divide` per softmax backward -- at the
shapes the model's own softmaxes produce, alternating ON and OFF blocks in one process with
a device sync around every block.

Shapes are CAPTURED, not assumed: `ttnn.softmax` and `ttnn.softmax_in_place` are wrapped for
one taped trunk forward (a few seconds) and every output shape and its count is recorded.
That covers the softmax nodes the tape holds; `triangle_attention`'s chunked recompute calls
the same helper on shapes of its own, so the per-call figures here are reported per shape and
the step total is given as a RANGE over the shapes seen rather than as one number.

    renorm_micro.py --tokens 384 --iters 40 --blocks 4 --out <json>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402


def capture_shapes(held, out, dev):
    """One taped trunk cycle with the softmax verbs wrapped, for the shape histogram.

    The forward is `fullstep.trunk_forward`, imported rather than re-written, so the inputs
    are rehydrated exactly as the timed arms rehydrate them.
    """
    import ttnn
    from tt_bio import autograd as ag
    from perf.of3t_stepfloor.fullstep import trunk_forward
    seen = Counter()
    real_sm, real_ip = ttnn.softmax, ttnn.softmax_in_place

    def wrap(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            try:
                seen[tuple(int(d) for d in r.shape)] += 1
            except Exception:
                pass
            return r
        return inner

    ttnn.softmax, ttnn.softmax_in_place = wrap(real_sm), wrap(real_ip)
    try:
        t0 = time.perf_counter()
        with ag.tape():
            trunk_forward(held["trunk"][0], held, 1, taped=True)
        ttnn.synchronize_device(dev)
        out["shape_capture_s"] = round(time.perf_counter() - t0, 3)
        ag.release_pins()
    finally:
        ttnn.softmax, ttnn.softmax_in_place = real_sm, real_ip
    return seen


def time_helper(shape, iters, on, dev):
    """ms per `softmax_bw_inner` call at one shape, with the lever forced on or off."""
    import torch
    import ttnn
    from tt_bio import autograd as ag
    y = ttnn.from_torch(torch.rand(*shape), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    g = ttnn.from_torch(torch.rand(*shape), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    prev = ag.SOFTMAX_BW_RENORM
    ag.SOFTMAX_BW_RENORM = on
    try:
        ag.softmax_bw_inner(y, g)                    # warm the kernels, not timed
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            ag.softmax_bw_inner(y, g)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) * 1e3 / iters
    finally:
        ag.SOFTMAX_BW_RENORM = prev
        ttnn.deallocate(y)
        ttnn.deallocate(g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=4, help="ON/OFF alternations per shape")
    ap.add_argument("--calls", type=int, default=4251,
                    help="renorm calls in one taped step, from the step artifact")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import os
    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:],
           "env": {"tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                   "loadavg_start": os.getloadavg(),
                   "calls_per_step": a.calls},
           "config": {"crop": a.tokens, "iters": a.iters, "blocks": a.blocks}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        import ttnn
        from tt_bio.tenstorrent import get_device
        held, _meta = S.capture(a.tokens, out)
        dev = get_device()
        out["env"]["arch"] = str(dev.arch())
        seen = capture_shapes(held, out, dev)
        out["shapes_seen"] = [{"shape": list(s), "count": n} for s, n in seen.most_common()]
        dump()

        rows = []
        for shape, count in seen.most_common():
            on_ms, off_ms = [], []
            try:
              for b in range(a.blocks):
                # alternate, and alternate the ORDER too, so a monotone drift in host load
                # cannot land on one arm
                first_on = (b % 2 == 0)
                for on in ((True, False) if first_on else (False, True)):
                    ms = time_helper(shape, a.iters, on, dev)
                    (on_ms if on else off_ms).append(ms)
            except Exception as e:
                # One shape that will not fit is a row of its own, not the end of the sweep.
                rows.append({"shape": list(shape), "count_in_forward": count,
                             "failed": f"{type(e).__name__}: {e}"[:300]})
                out["per_shape"] = rows
                dump()
                print(f"{tuple(shape)}  x{count}  FAILED {type(e).__name__}", flush=True)
                continue
            row = {"shape": list(shape), "count_in_forward": count,
                   "on_ms": [round(x, 4) for x in on_ms],
                   "off_ms": [round(x, 4) for x in off_ms],
                   "on_median_ms": round(statistics.median(on_ms), 4),
                   "off_median_ms": round(statistics.median(off_ms), 4)}
            row["delta_ms"] = round(row["on_median_ms"] - row["off_median_ms"], 4)
            row["step_s_if_every_call_this_shape"] = round(row["delta_ms"] * a.calls / 1e3, 3)
            rows.append(row)
            out["per_shape"] = rows
            dump()
            print(f"{tuple(shape)}  x{count}  ON {row['on_median_ms']:.3f} ms  "
                  f"OFF {row['off_median_ms']:.3f} ms  delta {row['delta_ms']:+.3f} ms  "
                  f"=> {row['step_s_if_every_call_this_shape']:+.2f} s per step", flush=True)

        if any('delta_ms' in r for r in rows):
            ds = [r["step_s_if_every_call_this_shape"] for r in rows
                  if "step_s_if_every_call_this_shape" in r]
            out["step_cost_bound_s"] = {"min": min(ds), "max": max(ds),
                                        "note": "the lever's step cost lies between these, "
                                                "because every one of the %d calls takes one "
                                                "of the shapes measured" % a.calls}
        out["env"]["loadavg_end"] = os.getloadavg()
        out["env"]["aiclk_during"] = clk.summary()
        out["env"]["aiclk_line"] = clk.line()
        dump()
        print(out["env"]["aiclk_line"], flush=True)
        print("BOUND:", json.dumps(out.get("step_cost_bound_s")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
