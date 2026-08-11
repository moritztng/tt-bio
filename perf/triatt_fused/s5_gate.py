#!/usr/bin/env python3
"""S5: what does the head-contiguous work split cost, before K2 spends it?

K2 needs one head per core so a core's mask is fixed for its whole lifetime. The factory's own split
saturates batch first and hands every core all 8 heads, so this is a required change, not a tuning
knob. It is separable from the mask work and it is measured here on its own, with the wheel's
kernels UNMODIFIED, so K2's later number is not carrying this cost silently.

PREDICTION, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md §11):

    Bit-exact -- the same work is done, only distributed differently, and flash attention's
    accumulation is per (batch, head, q_chunk), none of which is split across cores in either
    arrangement.

    Slower by roughly the core loss. 512 batches x 8 heads = 4096 units. Today: 110 cores x
    ceil(512/110)=5 batches x 8 heads = 40 units each. Head-contiguous: 13 x 8 = 104 cores x
    ceil(512/13)=40 batches x 1 head = 40 units each, with 6 cores idle. Same units per core, 5.5 %
    fewer cores, so I predict 1.00-1.07x slower, i.e. 6.53 -> 6.53-6.99 ms, and most likely near
    the top of that band because the per-core unit count is identical and only the core count drops.

    Anything outside that band means the split changes something other than core count -- DRAM
    locality of the q/k/v reads is the obvious candidate, since a core now walks 40 consecutive
    batches of one head instead of 5 batches of all heads.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import sdpa_generic as SG

RES = {"predictions": __doc__, "arms": []}


def timed(fn, dev, warm=2, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return st.median(ts), (max(ts) - min(ts)) / st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--qc", type=int, default=512)
    ap.add_argument("--kc", type=int, default=256)
    ap.add_argument("--out", default="perf/triatt_fused/s5_gate.json")
    args = ap.parse_args()
    S, H, D = args.n, 8, 32

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    ckc = (ttnn.MathFidelity.HiFi2, True, False, False)   # the op's own default, see s4_gate
    RES["meta"] = {"n": S, "q_chunk": args.qc, "k_chunk": args.kc, "grid": list(grid),
                   "loadavg": os.getloadavg(), "card": os.environ.get("TT_VISIBLE_DEVICES")}
    print(json.dumps(RES["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    q = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    k = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    v = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5
    pc = T._sdpa_program_config(args.qc, args.kc)

    def native():
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale, program_config=pc)

    ref = ttnn.to_torch(native())

    num_cores = grid[0] * grid[1]
    hc_split = (num_cores // H, H, 1)
    RES["meta"]["head_contiguous_split"] = list(hc_split)
    RES["meta"]["cores_used"] = hc_split[0] * hc_split[1] * hc_split[2]

    outs = {name: ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        for name in ("stock", "headcontig")}

    def run(name, split):
        return SG.sdpa(dev, q, k, v, bias, outs[name], args.qc, args.kc, grid, ckc, scale,
                       split=split)

    for name, split in (("stock", None), ("headcontig", hc_split)):
        p = SG.plan(q, k, v, bias, q, args.qc, args.kc, grid, ckc, scale, split)
        info = {kk: p[kk] for kk in ("batch_pf", "nh_pf", "q_pf", "batch_per_core",
                                     "nh_per_core", "q_per_core")}
        try:
            run(name, split)
            got = ttnn.to_torch(outs[name])
            eq = bool(torch.equal(got, ref))
            row = {"arm": name, "split": info, "equal": eq}
            if not eq:
                d = (got.float() - ref.float()).abs()
                row["max_abs_err"] = float(d.max())
                row["mismatch_frac"] = float((d > 0).float().mean())
        except Exception as e:  # noqa: BLE001
            row = {"arm": name, "split": info, "error": repr(e)[:400]}
        RES["arms"].append(row)
        print(json.dumps(row), flush=True)

    if any("error" in r for r in RES["arms"]):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(RES, indent=1))
        return

    times = {}
    for rnd in range(2):
        for name, split in (("stock", None), ("headcontig", hc_split)):
            ms, aa = timed(lambda n=name, s=split: run(n, s), dev)
            times.setdefault(name, []).append(ms * 1e3)
            print(json.dumps({"arm": name, "round": rnd, "ms": ms * 1e3, "aa_spread": aa}),
                  flush=True)
    ms_nat, _ = timed(native, dev)
    a, b = st.median(times["stock"]), st.median(times["headcontig"])
    RES["summary"] = {"native_ms": ms_nat * 1e3, "stock_split_ms": a, "headcontig_ms": b,
                      "headcontig_over_stock": b / a,
                      "all_equal": all(r.get("equal") for r in RES["arms"])}
    RES["meta"]["loadavg_end"] = os.getloadavg()
    print(json.dumps(RES["summary"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
