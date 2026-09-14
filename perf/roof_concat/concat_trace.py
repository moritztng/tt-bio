#!/usr/bin/env python3
"""The same layer A/B with the host taken out of the loop, for a box that will not go quiet.

qb2 runs this campaign's release gates at loadavg 25-38. A synchronize-per-block arm reports
max(host dispatch, device execution), so co-tenant pressure on the host adds the SAME additive
term to both arms and drags their ratio toward 1 -- which biases a layout lever toward NO-GO for
a reason that has nothing to do with the lever. Replaying a captured trace removes host dispatch
entirely: what is left is the device executing the same programs in the same order.

`roof-difftx-arith-efficiency` established on this shape that trace and synced agree when the box
IS quiet (624.2 vs 625.8 us for the whole layer on Blackhole), so this is the same measurement,
not a different one. Arms alternate inside every block, fixed order, minimum over blocks, and the
shipped arm is captured twice to carry an own-session A/A floor.
"""
from __future__ import annotations

import argparse, json, platform, sys, time
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from concat_ladder import weights                                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "trace.json")
    ap.add_argument("--seq", type=int, nargs="+", default=[320, 512, 768, 1024])
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=7)
    a = ap.parse_args()

    dev = T.get_device(trace_region_size=1 << 29)
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    dim, H = a.dim, a.heads
    torch.manual_seed(0)
    wd = weights(dim, 2 * dim)
    mods = {}
    for name, on in (("ship", False), ("concat", True)):
        T._APB_CONCAT_HEADS = on
        mods[name] = T.DiffusionTransformerLayer(dim, H, False, dict(wd), kc)
    assert not mods["ship"].attn_pair_bias._concat_heads
    assert mods["concat"].attn_pair_bias._concat_heads

    out = {"host": platform.node(), "arch": str(dev.arch()), "reps": a.reps,
           "blocks": a.blocks, "dim": dim, "heads": H,
           "loadavg_start": open("/proc/loadavg").read().split()[:3], "rungs": []}

    for S in a.seq:
        def t(shape, sc=1.0):
            return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16) * sc,
                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x, s = t((1, S, dim)), t((1, S, dim))
        z = t((1, H, S, S), 0.1)
        rec = {"S": S, "trace_us": {}, "errors": {},
               "loadavg": open("/proc/loadavg").read().split()[:3]}
        traces, held = {}, {}
        for arm in ("ship", "concat", "ship_AA"):
            fn = (lambda m=mods[arm.split("_")[0]]: m(x, s, z))
            try:
                for _ in range(3):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(dev)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                held[arm] = [fn() for _ in range(a.reps)]
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                traces[arm] = tid
            except Exception as e:                                           # noqa: BLE001
                rec["errors"][arm] = "%s: %s" % (type(e).__name__,
                                                 str(e).splitlines()[0][:240])
                print("SKIP %s S=%d %s" % (arm, S, rec["errors"][arm]), flush=True)
        best = {n: None for n in traces}
        for _ in range(a.blocks):
            for arm, tid in traces.items():
                t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / a.reps
                best[arm] = dt if best[arm] is None else min(best[arm], dt)
        for arm, tid in traces.items():
            rec["trace_us"][arm] = best[arm] * 1e6
            ttnn.release_trace(dev, tid)
            for h in held[arm]:
                try:
                    ttnn.deallocate(h)
                except Exception:                                            # noqa: BLE001
                    pass
        if "ship" in best and "concat" in best:
            rec["layer_ratio"] = best["ship"] / best["concat"]
        if "ship" in best and "ship_AA" in best:
            rec["AA_pct"] = 100 * abs(best["ship_AA"] - best["ship"]) / best["ship"]
        out["rungs"].append(rec)
        print("S=%-5d ship %8.2f us  concat %8.2f us  ratio %s  A/A %s %%  load %s"
              % (S, rec["trace_us"].get("ship", float("nan")),
                 rec["trace_us"].get("concat", float("nan")),
                 ("%.4f" % rec["layer_ratio"]) if "layer_ratio" in rec else "-",
                 ("%.3f" % rec["AA_pct"]) if "AA_pct" in rec else "-",
                 rec["loadavg"][0]), flush=True)
        for v in (x, s, z):
            ttnn.deallocate(v)
        a.out.write_text(json.dumps(out, indent=1))
    out["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
