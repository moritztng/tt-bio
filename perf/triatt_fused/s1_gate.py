#!/usr/bin/env python3
"""S1/K1a: the head-major writer. Does the qkv matmul write q, k, v straight into head-major
layout, bit-exactly, at the same speed -- deleting nlp_create_qkv_heads entirely?

PREDICTION, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md 5-6):

    torch.equal against nlp_create_qkv_heads(minimal_matmul(...)) at every size, and a time within
    0.15 ms of S0's 2.209 ms. Wider than 0.15 ms means the per-tile destination scatter costs
    something, which would contradict the read of write_block_sync_split, and K1 stops there.
    The prize is the whole 2.096 ms/call of nlp_create_qkv_heads.

Three writes replace one: the matmul's 24 output tile-columns are split into q/k/v by N chunk and
placed at (batch, head, row) instead of (row, head). head_dim is 32 = one tile, so no element moves
inside a tile and the transaction count and size are unchanged.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import ttnn
from tt_bio import tenstorrent as T
import generic_mm as G

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
    w = dram(torch.randn(C, 3 * H * D).to(torch.bfloat16))
    cfg = T._qkv_mm_config(x, w)
    blk = T._MM_BLOCK[(3 * H * D) // 32]
    gcfg = (blk, tuple(T.COMPUTE_GRID_MAIN))
    gckc = (ttnn.MathFidelity.HiFi4, False, True, False)
    row["mm_config"] = str(cfg)

    def native_mm():
        return ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg)

    def native_chain():
        qkv = native_mm()
        u = ttnn.unsqueeze(qkv, 1)
        r = ttnn.experimental.nlp_create_qkv_heads(
            u, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
            memory_config=u.memory_config())
        ttnn.deallocate(qkv)
        return r

    qkv_ref = native_mm()
    u = ttnn.unsqueeze(qkv_ref, 1)
    ref = [ttnn.to_torch(t) for t in ttnn.experimental.nlp_create_qkv_heads(
        u, num_heads=H, num_kv_heads=H, transpose_k_heads=False, memory_config=u.memory_config())]
    ttnn.deallocate(qkv_ref)
    row["ref_shapes"] = [list(t.shape) for t in ref]

    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    # Row tiles per batch: the PADDED sequence length, not S. At 298 aa the tile grid is 10 rows
    # deep, not 9.3, and a floor here writes every batch but the first to the wrong address.
    mt = int(x.padded_shape[-2]) // 32
    assert mt * 32 == int(outs[0].padded_shape[-2]), (mt, outs[0].padded_shape)
    defines = {"HEAD_MAJOR_MT": mt}
    row["head_major_mt"] = mt

    def head_major():
        return G.generic_minimal_matmul(dev, x, w, outs, gcfg, gckc, defines, KDIR)

    try:
        head_major()
        got = [ttnn.to_torch(o) for o in outs]
        row["equal"] = [bool(torch.equal(g, r)) for g, r in zip(got, ref)]
        row["all_equal"] = all(row["equal"])
        if not row["all_equal"]:
            row["max_abs_err"] = [float((g.float() - r.float()).abs().max()) for g, r in zip(got, ref)]
    except Exception as e:  # noqa: BLE001 -- a failed size must not lose the file
        row["error"] = repr(e)[:300]
        RES["sizes"][S] = row
        print(json.dumps(row), flush=True)
        return row

    for label, fn in (("native_mm", native_mm), ("head_major_mm", head_major),
                      ("native_mm_plus_heads", native_chain)):
        ms, aa = timed(fn, dev)
        row[label + "_ms"] = ms * 1e3
        row[label + "_aa"] = aa
    row["saving_ms_per_call"] = row["native_mm_plus_heads_ms"] - row["head_major_mm_ms"]
    row["speedup_vs_chain"] = row["native_mm_plus_heads_ms"] / row["head_major_mm_ms"]
    row["delta_vs_native_mm_ms"] = row["head_major_mm_ms"] - row["native_mm_ms"]

    for t in (x, w, *outs):
        ttnn.deallocate(t)
    RES["sizes"][S] = row
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="298,320,384,512,576,640")
    ap.add_argument("--out", default="perf/triatt_fused/s1_gate.json")
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
