#!/usr/bin/env python3
"""Is there a ONE-PROGRAM ternary elementwise on this ttnn, and is it faster than the two
programs the diffusion step spends today?

`chains.py` finds the step's 339 BinaryNg programs are 143 chains of length 2 plus 53 singletons,
and that every chain is the same ternary shape: `f(x) * y (+|*) z` with f in {id, sigmoid, silu}.
Fusing one costs nothing if the fused form still dispatches two programs, so that is measured here
first, by graph capture, before anything is timed.

Off-fold, production shapes and dtypes taken from the site map. `b2z2-step-program-fusion` found
its own off-fold probe over-predicted the step by 38 %, so a number here is a SCREEN, not a result.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

OUT: dict = {}


def count_programs(ttnn, dev, fn):
    from itemize import top_level_spans
    fn()
    ttnn.synchronize_device(dev)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    fn()
    ttnn.synchronize_device(dev)
    nodes = ttnn.graph.end_graph_capture()
    ops, _ = top_level_spans(nodes)
    return [o["name"] for o in ops]


def timeit(ttnn, dev, fn, reps=50, blocks=5):
    for _ in range(5):
        fn()
    ttnn.synchronize_device(dev)
    walls = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        walls.append((time.perf_counter() - t0) / reps)
    return round(1e6 * st.median(walls), 2), [round(1e6 * w, 2) for w in walls]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device(trace_region_size=512 << 20)
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "ttnn": getattr(ttnn, "__file__", "?")}

    def mk(shape):
        t = torch.randn(*shape) * 0.5
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev), t

    results = {}
    for shape in ([1, 140, 32, 128], [1, 140, 32, 256]):
        key = "x".join(map(str, shape))
        x, tx = mk(shape)
        y, ty = mk(shape)
        z, tz = mk(shape)
        r = {}

        # --- the incumbent: two programs, sigmoid fused into operand b of the multiply --------
        def chain_sig():
            t = ttnn.multiply(x, y, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            o = ttnn.add(t, z)
            ttnn.deallocate(t)
            return o

        def chain_plain():
            t = ttnn.multiply(x, y)
            o = ttnn.add(t, z)
            ttnn.deallocate(t)
            return o

        def chain_mulmul():
            t = ttnn.multiply(x, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            o = ttnn.multiply(t, z)
            ttnn.deallocate(t)
            return o

        cands = {
            "chain_mul_add": chain_plain,
            "chain_sigmul_add": chain_sig,
            "chain_silumul_mul": chain_mulmul,
            "mac": lambda: ttnn.mac(x, y, z),
            "addcmul": lambda: ttnn.addcmul(z, x, y, 1.0),
        }
        for name, fn in cands.items():
            try:
                progs = count_programs(ttnn, dev, fn)
            except Exception as e:                                  # noqa: BLE001
                r[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                continue
            us, allus = timeit(ttnn, dev, fn)
            o = fn()
            r[name] = {"programs": len(progs), "codes": progs, "us": us, "us_all": allus,
                       "out_shape": list(o.shape)}
            ttnn.deallocate(o)
            print(f"  {key:16s} {name:20s} {len(progs)} programs {progs}  {us:8.2f} us", flush=True)

        # --- bit-exactness of the one-program candidates against the chain they would replace --
        ref = ttnn.to_torch(chain_plain()).float()
        for name in ("mac", "addcmul"):
            if "error" in r.get(name, {}):
                continue
            got = ttnn.to_torch(cands[name]()).float()
            r[name]["bit_exact_vs_chain_mul_add"] = bool(torch.equal(got, ref))
            r[name]["max_abs_vs_chain"] = round(float((got - ref).abs().max()), 8)
            exact = (tx * ty + tz)
            r[name]["max_abs_vs_fp32"] = round(float((got - exact).abs().max()), 8)
            print(f"  {key:16s} {name:20s} bit-exact vs chain "
                  f"{r[name]['bit_exact_vs_chain_mul_add']}  maxabs {r[name]['max_abs_vs_chain']}"
                  f"  vs fp32 {r[name]['max_abs_vs_fp32']}", flush=True)
        r["chain_vs_fp32"] = round(float((ref - (tx * ty + tz)).abs().max()), 8)
        results[key] = r
        for t in (x, y, z):
            ttnn.deallocate(t)

    OUT["results"] = results
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(OUT, indent=1))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
