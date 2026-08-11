#!/usr/bin/env python3
"""K2: the SDPA mask read once per head instead of once per (batch, head, k_chunk).

PREDICTION, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md §4 and §11):

    Bit-exact. Nothing about the arithmetic changes -- the same mask tiles are added to the same
    QK tiles in the same order. Only where the bytes come from changes, and the ordering within
    add_block_inplace is untouched.

    6.53 -> 2.37-2.75 ms, i.e. the measured bias-off 2.222 ms plus ~0.15 ms for the one-time
    56 MiB mask read plus 0-0.35 for the L1-accumulate. That is the §4 prediction verbatim; the
    per-call number there was 2.56. Against the 6.53 ms the transcription measures with the stock
    reader, that is 2.4-2.8x on the op.

    Anything above 3.0 ms means the mask was not actually the cost, and the 84.2 % read-share
    attribution the whole of K2 rests on is wrong.

Preconditions the hoisted fill assumes, asserted here before the define is set: one head per core,
one q chunk per core, a batch-broadcast mask, and no padded mask.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import sdpa_generic as SG

KDIR = REPO / "tt_bio" / "kernels" / "triatt_sdpa"
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
    ap.add_argument("--out", default="perf/triatt_fused/s6_gate.json")
    args = ap.parse_args()
    S, H, D = args.n, 8, 32

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    ckc = (ttnn.MathFidelity.HiFi2, True, False, False)
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

    hc = (grid[0] * grid[1] // H, H, 1)
    p = SG.plan(q, k, v, bias, q, args.qc, args.kc, grid, ckc, scale, hc)
    RES["plan"] = {kk: p[kk] for kk in ("k_num_chunks", "Sq_chunk_t", "Sk_chunk_t", "nh_per_core",
                                        "q_per_core", "batch_per_core", "bcast_batch",
                                        "bcast_heads", "use_padded_mask", "mask_tiles")}
    print(json.dumps(RES["plan"]), flush=True)
    # what the hoisted fill assumes
    assert p["nh_per_core"] == 1, p
    assert p["q_per_core"] == 1, p
    assert p["bcast_batch"], "mask must be batch-broadcast for the hoist to be valid"
    assert not p["use_padded_mask"], p
    persistent_tiles = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    RES["meta"]["persistent_mask_tiles"] = persistent_tiles
    RES["meta"]["persistent_mask_kib"] = persistent_tiles * 2048 // 1024
    RES["meta"]["stock_mask_cb_tiles"] = p["mask_tiles"]
    print(json.dumps({"persistent_mask_tiles": persistent_tiles,
                      "stock_mask_cb_tiles": p["mask_tiles"]}), flush=True)

    outs = {n: ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        for n in ("base", "k2")}

    def base():
        return SG.sdpa(dev, q, k, v, bias, outs["base"], args.qc, args.kc, grid, ckc, scale,
                       split=hc, kernel_dir=KDIR)

    def k2():
        return SG.sdpa(dev, q, k, v, bias, outs["k2"], args.qc, args.kc, grid, ckc, scale,
                       split=hc, kernel_dir=KDIR, mask_cb_tiles=persistent_tiles,
                       defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]})

    for name, fn in (("base", base), ("k2", k2)):
        try:
            fn()
            got = ttnn.to_torch(outs[name])
            row = {"arm": name, "equal": bool(torch.equal(got, ref))}
            if not row["equal"]:
                d = (got.float() - ref.float()).abs()
                row["max_abs_err"] = float(d.max())
                row["mismatch_frac"] = float((d > 0).float().mean())
        except Exception as e:  # noqa: BLE001
            row = {"arm": name, "error": repr(e)[:500]}
        RES["arms"].append(row)
        print(json.dumps(row), flush=True)

    if any("error" in r for r in RES["arms"]):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(RES, indent=1))
        return

    times = {}
    for rnd in range(2):
        for name, fn in (("native", native), ("base", base), ("k2", k2)):
            ms, aa = timed(fn, dev)
            times.setdefault(name, []).append(ms * 1e3)
            print(json.dumps({"arm": name, "round": rnd, "ms": ms * 1e3, "aa_spread": aa}),
                  flush=True)
    med = {n: st.median(t) for n, t in times.items()}
    RES["summary"] = dict(med, k2_over_base=med["k2"] / med["base"],
                          base_over_k2=med["base"] / med["k2"],
                          all_equal=all(r.get("equal") for r in RES["arms"]))
    RES["meta"]["loadavg_end"] = os.getloadavg()
    print(json.dumps(RES["summary"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
