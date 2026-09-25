#!/usr/bin/env python3
"""The exact softmax's crossing, split three ways at the one seam that can split it.

`of3t-xcost` attributed the crop-384 backward on the additive axis (verb SELF-time by verb,
shape and dtype, which sums to `verb_wall_s`). One tensor class, the triangle attention score
block `[384,4,384,384]` FLOAT32, is 437.986 s of 705.78 s -- 62.06 %:

    to_torch          216 calls  153.854 s
    from_torch        216 calls   97.214 s
    softmax_in_place  108 calls  186.918 s   self, i.e. the host float64 arithmetic

Two of those three terms are already split without a card
(`perf/of3t_xcost/hostprice.py`, `perf/of3t_xcost/layoutprice.py`, both run with no device
open). What is left is one number and it is the largest single term in the backward:

    to_torch   153.854 s banked  -  46.96 s host-only  =  106.89 s
    from_torch  97.214 s banked  -  35.59 s host-only  =   61.62 s
                                                          -------
                                                          168.51 s, 23.87 % of the backward

and THREE different defects produce it. It is a DMA running at 2.32 GB/s over 391.4 GB, or it
is the blocking `to_torch` being charged for the `_scores` matmul draining behind it
(`a-blocking-op-is-charged-for-the-queue-drained-behind-it`: `of3t-zerosfill` was briefed at
35.1 % off a self-time table and was worth 1.0518x), or it is the device-side buffer
allocation. Those have three different fixes and one of them is "nothing, the seconds are real
device work".

So this instruments the ONE function both paths funnel through --
`tt_bio.autograd.host_f64_softmax_values`, reached by `_exact_softmax_raw` (the 108 recompute
calls inside `triangle_attention`'s own backward, `tt_bio/autograd.py:2005`) and by
`host_f64_softmax` (the 111 taped ones) -- and times its three legs separately.

ARMS. Each is a break control and must MOVE the number or its suspect is not the cause.

  base    production. down / math / up timed separately, nothing else changed.
  xsync   `ttnn.synchronize_device(dev)` immediately BEFORE the `to_torch`, timed as its own
          leg. If `down_s` collapses and `drain_s` picks it up, the crossing was never the
          cost: it was the matmul that produced the scores, and candidates 2 and 4 shrink to
          their layout-and-copy halves. If `down_s` holds, the link is the defect.
  rm      candidate 4: untilize on the DEVICE and cross in ROW_MAJOR, mirror on the way back.
          Bit-exact by construction and CHECKED, not assumed -- the first `--check` calls run
          both routes and compare with `torch.equal`. `layoutprice.py` sized this at 44.90 s
          off the host; this is whether the device pays it back.

The arms differ only inside this one function, so the route set, the call count and the
arithmetic are identical across them by construction, and `EXACT_SOFTMAX_STATS` is printed for
each so a silent decline shows up as a count rather than as a fast arm.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-xcost \
      python3 perf/of3t_xcost/crossprice.py --arm base \
        --out perf/of3t_xcost/out/CROSS_384_base.json

Run base, then xsync, then rm, in three separate processes -- one device context each. The
comparison is base-vs-xsync on `down_s` and base-vs-rm on `down_s + up_s`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402
from perf.of3t_perf import step as S                                     # noqa: E402

NS = time.perf_counter_ns
LEGS: dict = {}          # (phase, shape, dtype) -> [n, down_ns, math_ns, up_ns, drain_ns, els]
PHASE = ["forward"]


def _key(v):
    try:
        return (PHASE[0], tuple(int(d) for d in v.shape), str(v.dtype).replace("DataType.", ""))
    except Exception:                                                    # noqa: BLE001
        return (PHASE[0], (), "-")


def _rec(k, down, math, up, drain, els):
    r = LEGS.get(k)
    if r is None:
        r = LEGS[k] = [0, 0, 0, 0, 0, 0]
    r[0] += 1
    r[1] += down
    r[2] += math
    r[3] += up
    r[4] += drain
    r[5] += els


def make_shim(ag, ttnn, torch, arm, dev, check, report):
    """Replace `host_f64_softmax_values` with a leg-timed version of itself.

    The seam is a module GLOBAL, looked up at call time by both `_exact_softmax_raw` and
    `host_f64_softmax`, so rebinding the module attribute is seen by both. (The `_EXACT_OPS`
    dispatch dict captured the exact *verbs* at import and would NOT see a rebinding -- that is
    why the seam is this function and not `_exact_softmax_raw`.)
    """
    checked = [0]

    def shim(v, dim: int = -1):
        drain = 0
        if arm == "xsync":
            t = NS()
            ttnn.synchronize_device(dev)
            drain = NS() - t

        t0 = NS()
        if arm == "rm" and v.layout == ttnn.TILE_LAYOUT:
            rm = ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT)
            xt = ttnn.to_torch(rm)
            if rm is not v:                 # never free the caller's own tensor
                ttnn.deallocate(rm)
        else:
            xt = ttnn.to_torch(v)
        t1 = NS()

        y64 = torch.softmax(xt.double(), dim=dim)
        yf = y64.float()
        t2 = NS()

        if arm == "rm" and v.layout == ttnn.TILE_LAYOUT:
            up = ttnn.from_torch(yf, layout=ttnn.ROW_MAJOR_LAYOUT, device=v.device(),
                                 dtype=v.dtype)
            y = ttnn.to_layout(up, ttnn.TILE_LAYOUT, memory_config=v.memory_config())
            if y is not up:
                ttnn.deallocate(up)
        else:
            y = ttnn.from_torch(yf, layout=v.layout, device=v.device(), dtype=v.dtype,
                                memory_config=v.memory_config())
        t3 = NS()

        _rec(_key(v), t1 - t0, t2 - t1, t3 - t2, drain, int(y64.numel()))

        if arm == "rm" and checked[0] < check:
            checked[0] += 1
            ref_x = ttnn.to_torch(v)
            ref_y = ttnn.from_torch(yf, layout=v.layout, device=v.device(), dtype=v.dtype,
                                    memory_config=v.memory_config())
            report.append({
                "call": checked[0], "shape": list(int(d) for d in v.shape),
                "dtype": str(v.dtype).replace("DataType.", ""),
                "down_bit_identical": bool(torch.equal(ref_x, xt)),
                "up_bit_identical": bool(torch.equal(ttnn.to_torch(ref_y), ttnn.to_torch(y))),
                "down_max_abs_diff": float((ref_x - xt).abs().max()),
            })
            ttnn.deallocate(ref_y)
        return y64, y

    return shim


def _rows():
    out = []
    for (ph, shp, dt), (n, dn, mt, up, dr, els) in LEGS.items():
        out.append({"phase": ph, "shape": list(shp), "dtype": dt, "n": n,
                    "elements": els,
                    "down_s": round(dn / 1e9, 4), "math_s": round(mt / 1e9, 4),
                    "up_s": round(up / 1e9, 4), "drain_s": round(dr / 1e9, 4),
                    "total_s": round((dn + mt + up + dr) / 1e9, 4),
                    "down_ms_per_call": round(dn / n / 1e6, 2),
                    "math_ms_per_call": round(mt / n / 1e6, 2),
                    "up_ms_per_call": round(up / n / 1e6, 2),
                    "drain_ms_per_call": round(dr / n / 1e6, 2)})
    out.sort(key=lambda r: -r["total_s"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--arm", default="base", choices=("base", "xsync", "rm"))
    ap.add_argument("--check", type=int, default=2,
                    help="rm arm: how many calls also run the base route for a bit-exact check")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "arm": a.arm,
           "env": {"host": socket.gethostname(),
                   "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
                   "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "loadavg": os.getloadavg()},
           "config": {"crop": a.tokens, "cycles": a.cycles},
           "banked_for_comparison": {
               "backward_s": 705.78,
               "score_to_torch_s": 153.854, "score_from_torch_s": 97.214,
               "score_softmax_self_s": 186.918,
               "host_only_to_torch_s": 46.96, "host_only_from_torch_s": 35.59,
               "dma_or_drain_s": 168.51,
               "source": "perf/of3t_bwattrib/out/hist_384_base.json (arm base, pc card 0, "
                         "Blackhole p150a, AICLK 1350 median sampled DURING, commit d4eb1f0cf) "
                         "and perf/of3t_xcost/out/LAYOUTPRICE.json (no device opened)"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))   # noqa: E731
    dump()

    with during() as clk:
        try:
            import torch
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            S.declare_weights(trunk, out)

            report: list = []
            ag.host_f64_softmax_values = make_shim(ag, ttnn, torch, a.arm, dev, a.check, report)
            out["env"]["exact_training_ops"] = list(ag.exact_training_ops())

            snap_args, snap_kwargs = held["trunk_snap"]
            args_ = S._rehydrate(snap_args, dev)
            kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                       if k != "progress_fn"}
            trunk.num_cycles = a.cycles
            gc.collect()

            t0 = time.perf_counter()
            with ag.tape():
                _s, z = trunk(*args_, **kwargs_)
                ttnn.synchronize_device(dev)
            out["forward"] = {"s": round(time.perf_counter() - t0, 2),
                              "legs": _rows(),
                              "exact_softmax_stats": dict(ag.EXACT_SOFTMAX_STATS)}
            dump()

            if not isinstance(z, ag.Tensor):
                raise SystemExit("the trunk output is not taped; nothing to differentiate")

            LEGS.clear()
            PHASE[0] = "backward"
            zr = z.value
            seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
            before = dict(ag.EXACT_SOFTMAX_STATS)
            rs = TT.recompute_scope()
            rs.__enter__()
            t0 = time.perf_counter()
            try:
                ag.backward([z], [seed])
                ttnn.synchronize_device(dev)
                out["backward"] = {"ok": True}
            except Exception as e:                                       # noqa: BLE001
                out["backward"] = {"ok": False,
                                   "error_head": str(e).split("backtrace")[0][:900],
                                   "error": traceback.format_exc()[-4000:]}
            finally:
                rs.__exit__(None, None, None)
            wall = time.perf_counter() - t0
            rows = _rows()
            legs = {k: round(sum(r[k] for r in rows), 3)
                    for k in ("down_s", "math_s", "up_s", "drain_s")}
            out["backward"].update({
                "s": round(wall, 2), "legs": rows, "leg_totals_s": legs,
                "legs_share_of_backward": {k: round(v / wall, 4) for k, v in legs.items()},
                "exact_softmax_stats_delta": {
                    k: ag.EXACT_SOFTMAX_STATS.get(k, 0) - before.get(k, 0)
                    for k in ag.EXACT_SOFTMAX_STATS}})
            if a.arm == "rm":
                out["bit_exactness"] = {
                    "checks": report,
                    "all_bit_identical": bool(report) and all(
                        r["down_bit_identical"] and r["up_bit_identical"] for r in report),
                    "note": "a layout conversion moves bytes; at a shape that is a multiple of "
                            "the 32x32 tile there is no pad and no rounding. If this is False "
                            "the arm is void whatever its seconds say."}
        except Exception as e:                                           # noqa: BLE001
            out["fatal"] = {"error": str(e)[:900], "trace": traceback.format_exc()[-4000:]}
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line()
    out["env"]["loadavg_end"] = os.getloadavg()
    dump()
    print(json.dumps({"arm": a.arm,
                      "backward_s": out.get("backward", {}).get("s"),
                      "leg_totals_s": out.get("backward", {}).get("leg_totals_s"),
                      "bit_exactness": out.get("bit_exactness", {}).get("all_bit_identical"),
                      "clock": out["env"].get("aiclk_line")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
