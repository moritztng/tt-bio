#!/usr/bin/env python3
"""K1b: the tail of the sub-block stays head-major, so nlp_concat_heads never runs.

PREDICTION, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md 11, item 1):

    torch.equal on the gate projection and on the final `out` at every size, and the head-major
    tail beats the stock tail by the whole 0.748 ms/call of nlp_concat_heads at 512 aa
    (0.70-0.80 ms), with the gate multiply and the `out` projection each within their own A/A
    spread of the stock ones. Anything that makes multiply_ or `out` slower on a [S, 8, S, 32]
    tensor than on the [S, S, 256] one it is a reshape of would be a surprise: the tile count,
    the tile contents and the op are identical, only the logical shape differs.

    Wider than that, or not torch.equal, and K1b stops.

Stock tail:        nlp_concat_heads(o) -> squeeze -> multiply_(oc, g) -> minimal_matmul(.., w_o)
Head-major tail:   multiply_(o, g_hm) -> generic minimal_matmul with a head-major in0 read

`o` comes out of the SDPA already head-major, so the stock tail's first op exists only to undo that.

SECOND PREDICTION, WRITTEN AFTER THE 512 RUN AND BEFORE THE SWEEP. At 512 the head-major `out`
read measured 0.848 ms against the stock 0.796, +6.6 %, outside its own 2.6 % A/A, while
`nlp_concat_heads` (0.738) and the gate write (-0.005) went exactly as predicted. The mechanism I
claim is a DRAM BANK CONFLICT, and this card has MEASURED 8 banks (dram_grid_size 8x1). The in0
reader walks j = 0..7 within one K block; in the stock layout those are 8 CONSECUTIVE tile ids, one
per bank. Head-major they are strided by HEAD_MAJOR_IN0_MT tiles, so the number of distinct banks
an 8-tile K block touches is 8 / gcd(mt, 8):

    n     298  320  384  512  576  640
    mt     10   10   12   16   18   20
    mt%8    2    2    4    0    2    4
    banks   4    4    2    1    4    2

So I predict the `out` regression is WORST at 512 (1 bank, the only size where mt is a multiple of
8), roughly half as bad at 384 and 640 (2 banks), and mildest at 298, 320 and 576 (4 banks). If the
regression is instead flat across all six sizes, the bank-conflict story is WRONG and the cost is
something else -- say per-tile address arithmetic -- and the residual must be renamed.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import mm_generic as G

KDIR = REPO / "tt_bio" / "kernels" / "triatt"
RES = {"predictions": __doc__, "sizes": {}}


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


def run_size(dev, ckc, S, C=256, H=8, D=32):
    row = {"n": S}
    torch.manual_seed(0)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = dram(torch.randn(S, S, C).to(torch.bfloat16))
    w_g = dram(torch.randn(C, H * D).to(torch.bfloat16))
    w_o = dram(torch.randn(C, C).to(torch.bfloat16))
    # `o`, as the SDPA leaves it: head-major [S, H, S, D]
    o_t = torch.randn(S, H, S, D).to(torch.bfloat16)
    Spad = int(x.padded_shape[-2])
    mt = Spad // 32
    row["head_major_mt"] = mt

    blk = T._MM_BLOCK[(H * D) // 32]
    grid = tuple(T.COMPUTE_GRID_MAIN)
    gckc = G.ckc_args(ckc)
    cfg_g = T._qkv_mm_config(x, w_g)

    # ---- stock tail -------------------------------------------------------------------------
    def stock_g():
        return ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w_g, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg_g)

    def stock_tail():
        o = dram(o_t)
        oc = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(
            o, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)
        ttnn.deallocate(o)
        gated = ttnn.multiply_(oc, stock_g(), input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        out = T._pair_proj_minimal_matmul(gated, w_o, ckc, ttnn.bfloat16)
        ttnn.deallocate(gated)
        return out

    ref_out = ttnn.to_torch(stock_tail())
    g_ref = ttnn.to_torch(stock_g()).reshape(S, S, H, D).permute(0, 2, 1, 3).contiguous()

    # ---- head-major tail --------------------------------------------------------------------
    def hm_g(dst):
        return G.generic_minimal_matmul(dev, x, w_g, dst, (blk, grid), gckc,
                                        {"HEAD_MAJOR_OUT_MT": mt}, KDIR)

    def hm_out(src, dst):
        return G.generic_minimal_matmul(dev, src, w_o, dst, (blk, grid), gckc,
                                        {"HEAD_MAJOR_IN0_MT": mt}, KDIR, m_k=(S * Spad, C))

    g_hm = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    out_hm = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, S, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)

    def hm_tail():
        o = dram(o_t)
        hm_g(g_hm)
        gated = ttnn.multiply_(o, g_hm, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        hm_out(gated, out_hm)
        ttnn.deallocate(gated)
        return out_hm

    try:
        hm_g(g_hm)
        row["g_equal"] = bool(torch.equal(ttnn.to_torch(g_hm), g_ref))
        hm_tail()
        got = ttnn.to_torch(out_hm)
        row["out_equal"] = bool(torch.equal(got, ref_out))
        if not row["out_equal"]:
            d = (got.float() - ref_out.float()).abs()
            row["out_max_abs_err"] = float(d.max())
            row["out_mismatch_frac"] = float((d > 0).float().mean())
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:300]
        RES["sizes"][S] = row
        print(json.dumps(row), flush=True)
        return row

    # ---- the three ops, separately, plus the whole tail ----------------------------------------
    o_res = dram(o_t)
    gated_plain = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(
        o_res, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)
    arms = {
        "stock_concat_heads": lambda: ttnn.experimental.nlp_concat_heads(
            o_res, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        "stock_g": stock_g,
        "hm_g": lambda: hm_g(g_hm),
        "stock_out": lambda: T._pair_proj_minimal_matmul(gated_plain, w_o, ckc, ttnn.bfloat16),
        "hm_out": lambda: hm_out(g_hm, out_hm),
        "stock_tail": stock_tail,
        "hm_tail": hm_tail,
    }
    for label, fn in arms.items():
        ms, aa = timed(fn, dev)
        row[label + "_ms"] = ms * 1e3
        row[label + "_aa"] = aa
    row["tail_saving_ms_per_call"] = row["stock_tail_ms"] - row["hm_tail_ms"]
    row["tail_speedup"] = row["stock_tail_ms"] / row["hm_tail_ms"]

    for t in (x, w_g, w_o, g_hm, out_hm, o_res, gated_plain):
        ttnn.deallocate(t)
    RES["sizes"][S] = row
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="298,320,384,512,576,640")
    ap.add_argument("--out", default="perf/triatt_fused/s3_gate.json")
    args = ap.parse_args()

    dev = T.get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    RES["meta"] = {"grid": list(T.COMPUTE_GRID_MAIN), "loadavg": os.getloadavg(),
                   "card": os.environ.get("TT_VISIBLE_DEVICES")}
    print(json.dumps(RES["meta"]), flush=True)

    for S in [int(s) for s in args.sizes.split(",")]:
        try:
            run_size(dev, ckc, S)
        except Exception as e:  # noqa: BLE001
            RES["sizes"][S] = {"n": S, "error": repr(e)[:300]}
            print(json.dumps(RES["sizes"][S]), flush=True)

    RES["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
