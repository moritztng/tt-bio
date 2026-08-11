#!/usr/bin/env python3
"""S4, the kill gate for K2: does the SDPA transcription reproduce the native op?

PREDICTION, WRITTEN BEFORE THE RUN. S0 did the same thing for `minimal_matmul` and landed 1.018x
and torch.equal, so the same standard applies here:

    torch.equal against ttnn.transformer.scaled_dot_product_attention, and within 5 % of its
    measured time. Anything else and K2's route through generic_op is dead, because the mask patch
    is this same descriptor plus a work-split change and two kernel edits.

The derived constants are asserted against the factory's own arithmetic first, so a mismatch shows
up as a named assertion rather than as a wrong number or a hang. For the 512 aa shipped config
(q_chunk 512, k_chunk 256, head_dim 32, 8 heads, batch 512, 11x10 grid, fp32_dest_acc):

    dst_size 4, qk subblock (2,2), out subblock (4,1), mask CB 256 tiles = 512 KiB,
    batch_parallel_factor 110, nh_parallel_factor 1 -> every core carries all 8 heads.
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
    ap.add_argument("--kernel-dir", default=None,
                    help="patched kernel tree; default is the wheel's own")
    ap.add_argument("--out", default="perf/triatt_fused/s4_gate.json")
    args = ap.parse_args()
    S, H, D = args.n, 8, 32

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    # The fold calls SDPA with NO compute_kernel_config, so the op takes its own default:
    # init_device_compute_kernel_config(arch, nullopt, HiFi2, /*approx*/ true, /*fp32_acc*/ false,
    # /*l1_acc*/ false) -- sdpa.cpp:33-34 against compute_kernel_config.hpp:40-47. NOT the trunk's
    # HiFi4/fp32_dest_acc config, which is what a first guess reaches for and which changes
    # dst_size from 8 to 4 and with it every subblock and granularity.
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

    p = SG.plan(q, k, v, bias, q, args.qc, args.kc, grid, ckc, scale)
    RES["plan"] = {kk: (vv if isinstance(vv, (int, float, bool)) else str(vv))
                   for kk, vv in p.items()}
    print(json.dumps({kk: RES["plan"][kk] for kk in
                      ("dst_size", "qk_sb_h", "qk_sb_w", "out_sb_h", "out_sb_w", "mask_tiles",
                       "batch_pf", "nh_pf", "q_pf", "batch_per_core", "nh_per_core",
                       "q_per_core", "Sq_chunk_t", "Sk_chunk_t", "q_buffer_factor")}), flush=True)
    # the constants derived by hand in the state doc, asserted so drift is named not silent
    assert (p["dst_size"], p["qk_sb_h"], p["qk_sb_w"]) == (8, 2, 4), p
    assert (p["out_sb_h"], p["out_sb_w"]) == (8, 1), p
    assert p["mask_tiles"] == 256 and p["nh_per_core"] == H, p
    assert p["mask_tiles"] * 2048 == 512 * 1024, p

    pc = T._sdpa_program_config(args.qc, args.kc)

    def native():
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale, program_config=pc)

    ref = ttnn.to_torch(native())

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)

    kdir = Path(args.kernel_dir) if args.kernel_dir else None
    RES["meta"]["kernel_dir"] = str(kdir)

    def generic():
        return SG.sdpa(dev, q, k, v, bias, out, args.qc, args.kc, grid, ckc, scale,
                       kernel_dir=kdir)

    try:
        generic()
        got = ttnn.to_torch(out)
        RES["equal"] = bool(torch.equal(got, ref))
        if not RES["equal"]:
            d = (got.float() - ref.float()).abs()
            RES["max_abs_err"] = float(d.max())
            RES["mismatch_frac"] = float((d > 0).float().mean())
            num = (got.float() * ref.float()).sum()
            RES["pcc_like"] = float(num / (got.float().norm() * ref.float().norm() + 1e-30))
    except Exception as e:  # noqa: BLE001
        RES["error"] = repr(e)[:600]
        print(json.dumps({"error": RES["error"]}), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(RES, indent=1))
        return
    print(json.dumps({kk: RES[kk] for kk in
                      ("equal", "max_abs_err", "mismatch_frac", "pcc_like") if kk in RES}),
          flush=True)

    for rnd in range(2):
        for label, fn in (("native", native), ("generic_op", generic)):
            ms, aa = timed(fn, dev)
            row = {"arm": label, "round": rnd, "ms": ms * 1e3, "aa_spread": aa}
            RES["arms"].append(row)
            print(json.dumps(row), flush=True)

    nat = st.median([r["ms"] for r in RES["arms"] if r["arm"] == "native"])
    gen = st.median([r["ms"] for r in RES["arms"] if r["arm"] == "generic_op"])
    RES["summary"] = {"native_ms": nat, "generic_ms": gen, "ratio": gen / nat,
                      "gate_pass": bool(RES.get("equal") and gen <= nat * 1.05)}
    RES["meta"]["loadavg_end"] = os.getloadavg()
    print(json.dumps(RES["summary"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
