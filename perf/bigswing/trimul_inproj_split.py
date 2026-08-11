#!/usr/bin/env python3
"""§56 measured the matmul. Production also has to split the result, and the split is not free.

`perf/bigswing/trimul_inproj_width.py` measured 2.033x for grouping the trimul in-projection weights
(G=1 -> 4, bit-exact, `max_abs` 0.0), and the fold A/B measured **0.9724x -- a 2.8 % regression**,
reproducible to 2 ms across arms and 40x the A/A floor, with the whole delta inside
`body:TriangleMultiplication`.

The difference between the two units is one op. The loop does not consume the fused matmul result; it
consumes four `[1,S,S,C]` pieces of it, and at grouping G the split is `ttnn.chunk(..., chunks=4*G)`.
§56 timed the matmul and stopped. This times the mandatory unit -- matmul plus split -- at each G,
which is what the fold actually runs.

Two arms per G, so the split can be priced on its own:
  mm     : the matmuls alone, reproducing §56
  mm+cut : the matmuls and their 4G-way split, which is what the loop needs before its first multiply
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--groups", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio.tenstorrent import get_device

    torch.manual_seed(0)
    dev = get_device()
    S, K, C, P = a.seq, a.cz, a.c, a.pairs
    NW = 4 * C * P

    A = ttnn.from_torch(torch.randn(1, S, S, K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wt = torch.randn(K, NW, dtype=torch.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)

    groups = [g for g in (int(x) for x in a.groups.split(",")) if P % g == 0]
    W = {g: [ttnn.from_torch(wt[:, i * 4 * C * g:(i + 1) * 4 * C * g].contiguous(),
                             layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
             for i in range(P // g)] for g in groups}

    def run(g, cut):
        """One trimul's in-projection at grouping g, with or without the split the loop needs."""
        out = []
        for w in W[g]:
            f = ttnn.experimental.minimal_matmul(
                input_tensor=A, weight_tensor=w, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16, compute_kernel_config=ckc)
            if not cut:
                out.append(f)
                continue
            out.extend(ttnn.chunk(f, chunks=4 * g, dim=-1))
            ttnn.deallocate(f)
        return out

    res = {"host": "qb2", "chip": 0, "seq": S, "K": K, "C": C, "n_pairs": P, "reps": a.reps,
           "shape": f"[1,{S},{S},{K}] @ [{K},{4*C}g] then chunk(4g)", "arms": []}
    import importlib.metadata as im
    res["ttnn"] = im.version("ttnn")

    for g in groups:                                   # warm every kernel shape before timing
        for cut in (False, True):
            for t in run(g, cut):
                ttnn.deallocate(t)
    ttnn.synchronize_device(dev)

    keys = [(g, cut) for g in groups for cut in (False, True)]
    times = {k: [] for k in keys}
    for _ in range(a.reps):
        for k in keys:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = run(*k)
            ttnn.synchronize_device(dev)
            times[k].append((time.perf_counter() - t0) * 1000.0)
            for t in outs:
                ttnn.deallocate(t)

    base = st.median(times[(1, True)])
    for (g, cut) in keys:
        ms = sorted(times[(g, cut)])
        med = st.median(ms)
        res["arms"].append({
            "G": g, "split": cut, "pieces": 4 * g if cut else 0, "matmuls": P // g,
            "ms_median": round(med, 4), "spread_ms": round(ms[-1] - ms[0], 4),
            "split_only_ms": round(med - st.median(times[(g, False)]), 4) if cut else None,
            "vs_G1_with_split": round(base / med, 4) if cut else None,
        })
        print(f"G={g} split={int(cut)} {P//g:2d} mm  {med:8.3f} ms" +
              (f"   split alone {med - st.median(times[(g, False)]):7.3f} ms"
               f"   {base/med:6.4f}x vs G=1" if cut else ""), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
