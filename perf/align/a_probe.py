#!/usr/bin/env python3
"""p2-alignment probes: what the LOGICAL length of a contracted axis costs, where else the trunk
pays it, what mechanism it is, and what an aligned fill would cost.

Every arm holds the PADDED shape fixed and varies only the logical shape, so the bytes, the tile
count, the grid and the program config are identical across an A/B and the only difference is the
metadata ttnn dispatches on.

    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... python3 perf/align/a_probe.py <cmd> --out x.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tt_bio.tenstorrent as T  # noqa: E402

CKC = None


def ckc():
    global CKC
    if CKC is None:
        CKC = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
    return CKC


def L1():
    return ttnn.L1_MEMORY_CONFIG


def DRAM():
    return ttnn.DRAM_MEMORY_CONFIG


def mk(dev, shape, mc, seed=0, fill=None):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32) if fill is None \
        else torch.full(shape, fill, dtype=torch.float32)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=mc)


def timeit(dev, fn, reps=20, warm=3):
    for _ in range(warm):
        o = fn()
        del o
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(reps)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    del outs
    return dt * 1e6            # us/call


# --------------------------------------------------------------------------------------------- #
# roofs, measured on this card this pass
# --------------------------------------------------------------------------------------------- #
def cmd_roofs(dev, a):
    out = {}
    # DRAM read / write / combined, unary clone
    for tag, src, dst, mb in (("dram_read", DRAM(), L1(), 8), ("dram_write", L1(), DRAM(), 32),
                              ("dram_rw", DRAM(), DRAM(), 32)):
        n = mb * 1024 * 1024 // 2
        rows = n // 512
        try:
            x = mk(dev, (1, 1, rows, 512), src)
            us = timeit(dev, lambda: ttnn.clone(x, memory_config=dst), reps=10)
            gb = (mb * (2 if tag == "dram_rw" else 1)) / 1024.0
            out[tag] = {"us": us, "GB_s": gb / (us * 1e-6), "MB": mb}
            ttnn.deallocate(x)
        except Exception as e:                                   # noqa: BLE001
            out[tag] = {"error": str(e)[:200]}
    # square compute roof, DRAM out
    for nsq in (4096,):
        try:
            x = mk(dev, (nsq, nsq), DRAM())
            y = mk(dev, (nsq, nsq), DRAM(), seed=1)
            us = timeit(dev, lambda: ttnn.matmul(x, y, compute_kernel_config=ckc()), reps=6)
            out[f"square_{nsq}"] = {"us": us, "TFLOPs": 2 * nsq ** 3 / (us * 1e-6) / 1e12}
            ttnn.deallocate(x); ttnn.deallocate(y)
        except Exception as e:                                   # noqa: BLE001
            out[f"square_{nsq}"] = {"error": str(e)[:200]}
    # the contraction's own class: K=320, output width nt=10, L1 in and L1 out. Best of a config
    # search -- a bare ttnn.matmul at this shape lands on a single-core path and is 100x off.
    best = None
    M, Mt, Nt = 10240, 320, 10
    x = mk(dev, (M, 320), L1())
    y = mk(dev, (320, 320), L1(), seed=1)
    for gx, gy in ((11, 10), (10, 10), (8, 8)):
        for bw in (1, 2, 5, 10):
            try:
                pc = _pc(gx, gy, in0_block_w=bw, mt=Mt, nt=Nt)
                us = timeit(dev, lambda: ttnn.matmul(x, y, compute_kernel_config=ckc(),
                                                     memory_config=L1(), program_config=pc,
                                                     dtype=ttnn.bfloat16), reps=10)
                tf = 2 * M * 320 * 320 / (us * 1e-6) / 1e12
                print(f"  K320 nt10 L1  grid {gx}x{gy} in0_block_w={bw:2d} -> {us:8.2f} us "
                      f"{tf:6.2f} TFLOP/s")
                if best is None or tf > best["TFLOPs"]:
                    best = {"us": us, "TFLOPs": tf, "grid": [gx, gy], "in0_block_w": bw, "M": M}
            except Exception as e:                               # noqa: BLE001
                print(f"  K320 nt10 L1  grid {gx}x{gy} bw={bw}: {str(e)[:80]}")
    out["K320_nt10_L1_best"] = best
    ttnn.deallocate(x); ttnn.deallocate(y)
    # same class with a DRAM output, one column of charter 4.6
    x = mk(dev, (M, 320), DRAM())
    y = mk(dev, (320, 320), DRAM(), seed=1)
    bestd = None
    for gx, gy in ((11, 10),):
        for bw in (1, 2, 5, 10):
            try:
                pc = _pc(gx, gy, in0_block_w=bw, mt=Mt, nt=Nt)
                us = timeit(dev, lambda: ttnn.matmul(x, y, compute_kernel_config=ckc(),
                                                     memory_config=DRAM(), program_config=pc,
                                                     dtype=ttnn.bfloat16), reps=10)
                tf = 2 * M * 320 * 320 / (us * 1e-6) / 1e12
                if bestd is None or tf > bestd["TFLOPs"]:
                    bestd = {"us": us, "TFLOPs": tf, "grid": [gx, gy], "in0_block_w": bw}
            except Exception as e:                               # noqa: BLE001
                pass
    out["K320_nt10_DRAM_best"] = bestd
    ttnn.deallocate(x); ttnn.deallocate(y)
    out["grid"] = list(T.COMPUTE_GRID_MAIN)
    return out


# --------------------------------------------------------------------------------------------- #
# the four-arm A/B, the fold's own program config on every arm
# --------------------------------------------------------------------------------------------- #
ARMS = [
    ("320x320", (320, 320), (320, 320)),
    ("298x320", (298, 320), (320, 320)),
    ("320x298", (320, 298), (298, 320)),
    ("298x298", (298, 298), (298, 320)),
    ("fold_298x298x298", (298, 298), (298, 298)),
]


def _run_arm(dev, ashape, bshape, batch=32, pc=None, mc=None, reps=20):
    mc = mc or L1()
    a = mk(dev, (1, batch) + ashape, mc)
    b = mk(dev, (1, batch) + bshape, mc, seed=1)
    us = timeit(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc(), memory_config=mc,
                                         program_config=pc, dtype=ttnn.bfloat16), reps=reps)
    pad_a, pad_b = list(a.padded_shape), list(b.padded_shape)
    ttnn.deallocate(a); ttnn.deallocate(b)
    return us, pad_a, pad_b


def cmd_ab4(dev, a):
    pc = T._triangle_mul_program_config(10)
    res = []
    for name, ash, bsh in ARMS:
        us, pa, pb = _run_arm(dev, ash, bsh, pc=pc)
        res.append({"arm": name, "logical_a": list(ash), "logical_b": list(bsh),
                    "padded_a": pa, "padded_b": pb, "us": us})
        print(f"  {name:18s} a{list(ash)} b{list(bsh)} padded {pa} -> {us:8.2f} us")
    base = next(r["us"] for r in res if r["arm"] == "320x320")
    for r in res:
        r["ratio_vs_aligned"] = r["us"] / base
    return {"arms": res, "program_config": "triangle_mul(10): in0_block_w=10, per_core_M=per_core_N=1"}


# --------------------------------------------------------------------------------------------- #
# core-grid ladder (utilisation) on both arms
# --------------------------------------------------------------------------------------------- #
def _pc(gx, gy, in0_block_w=10, mt=10, nt=10):
    per_core_M = -(-mt // gy)
    per_core_N = -(-nt // gx)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=in0_block_w,
        out_subblock_h=1, out_subblock_w=1, out_block_h=per_core_M, out_block_w=per_core_N,
        per_core_M=per_core_M, per_core_N=per_core_N, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)


def cmd_cores(dev, a):
    gx, gy = T.COMPUTE_GRID_MAIN
    res = []
    for g in [(gx, gy), (10, 10), (8, 8), (5, 5), (4, 4)]:
        for name, ash, bsh in (ARMS[0], ARMS[3]):
            try:
                pc = _pc(g[0], g[1])
                us, _, _ = _run_arm(dev, ash, bsh, pc=pc)
            except Exception as e:                               # noqa: BLE001
                us = None
                print(f"  grid {g} {name}: {str(e)[:90]}")
            mt = nt = 10
            cores = min(g[0], nt) * min(g[1], mt) if us else None
            res.append({"grid": list(g), "arm": name, "us": us, "cores_engaged": cores,
                        "grid_cores": g[0] * g[1]})
            if us:
                print(f"  grid {g[0]}x{g[1]} ({cores} cores engaged) {name:9s} -> {us:8.2f} us")
    return {"ladder": res, "grid_main": [gx, gy]}


# --------------------------------------------------------------------------------------------- #
# mechanism: in0_block_w ladder and batch ladder
# --------------------------------------------------------------------------------------------- #
def cmd_ladder(dev, a):
    out = {"in0_block_w": [], "batch": []}
    gx, gy = T.COMPUTE_GRID_MAIN
    for bw in (1, 2, 5, 10):
        row = {"in0_block_w": bw}
        for name, ash, bsh in (ARMS[0], ARMS[3]):
            us, _, _ = _run_arm(dev, ash, bsh, pc=_pc(gx, gy, in0_block_w=bw))
            row[name] = us
        row["delta_us"] = row["298x298"] - row["320x320"]
        row["ratio"] = row["298x298"] / row["320x320"]
        print(f"  in0_block_w={bw:2d}  aligned {row['320x320']:8.2f}  unaligned "
              f"{row['298x298']:8.2f}  delta {row['delta_us']:7.2f} us  ratio {row['ratio']:.3f}")
        out["in0_block_w"].append(row)
    for batch in (8, 16, 32, 64):
        row = {"batch": batch}
        for name, ash, bsh in (ARMS[0], ARMS[3]):
            us, _, _ = _run_arm(dev, ash, bsh, batch=batch, pc=_pc(gx, gy))
            row[name] = us
        row["delta_us"] = row["298x298"] - row["320x320"]
        row["ratio"] = row["298x298"] / row["320x320"]
        print(f"  batch={batch:3d}  aligned {row['320x320']:8.2f}  unaligned "
              f"{row['298x298']:8.2f}  delta {row['delta_us']:7.2f} us  ratio {row['ratio']:.3f}")
        out["batch"].append(row)
    return out


# --------------------------------------------------------------------------------------------- #
# the other candidate sites
# --------------------------------------------------------------------------------------------- #
def cmd_sites(dev, a):
    out = {}
    # 1. SDPA @1629, tri-attention: q/k/v padded [298, 8, 320, 32]
    for tag, S in (("sdpa_logical_298", 298), ("sdpa_logical_320", 320)):
        try:
            q = mk(dev, (298, 8, S, 32), DRAM())
            k = mk(dev, (298, 8, S, 32), DRAM(), seed=1)
            v = mk(dev, (298, 8, S, 32), DRAM(), seed=2)
            pc = T._tri_att_sdpa_program_config(S, S)
            us = timeit(dev, lambda: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=0.176, program_config=pc), reps=6, warm=2)
            out[tag] = {"us": us, "padded": list(q.padded_shape), "logical": list(q.shape),
                        "q_chunk": pc.q_chunk_size, "k_chunk": pc.k_chunk_size}
            print(f"  {tag:18s} padded {list(q.padded_shape)} -> {us:8.2f} us")
            for t in (q, k, v):
                ttnn.deallocate(t)
        except Exception as e:                                   # noqa: BLE001
            out[tag] = {"error": str(e)[:250]}
            print(f"  {tag}: {str(e)[:150]}")
    # 2. attn@v inside AttentionPairBias's fp32-softmax attention, site 378: [1,16,S,S] @ [1,16,S,32]
    for tag, S in (("attnv_logical_298", 298), ("attnv_logical_320", 320)):
        try:
            us, pa, pb = _run_arm(dev, (S, S), (S, 32), batch=16, pc=None, mc=DRAM(), reps=20)
            out[tag] = {"us": us, "padded_a": pa, "padded_b": pb}
            print(f"  {tag:18s} padded {pa} @ {pb} -> {us:8.2f} us")
        except Exception as e:                                   # noqa: BLE001
            out[tag] = {"error": str(e)[:250]}
    # 3. control: the pair-track projection, K=256 (aligned), rows logically 298 vs 320
    for tag, S in (("proj_rows_298", 298), ("proj_rows_320", 320)):
        try:
            x = mk(dev, (1, 298, S, 256), DRAM())
            w = mk(dev, (256, 256), DRAM(), seed=1)
            cfg = T._pair_proj_config(x, w)
            us = timeit(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc(),
                                                 program_config=cfg, dtype=ttnn.bfloat16),
                        reps=8, warm=2)
            out[tag] = {"us": us, "padded": list(x.padded_shape), "logical": list(x.shape),
                        "cfg": cfg is not None}
            print(f"  {tag:18s} padded {list(x.padded_shape)} -> {us:8.2f} us")
            ttnn.deallocate(x); ttnn.deallocate(w)
        except Exception as e:                                   # noqa: BLE001
            out[tag] = {"error": str(e)[:250]}
            print(f"  {tag}: {str(e)[:150]}")
    # 4. softmax over a logically-298 last axis, fixed padded 320
    for tag, S in (("softmax_298", 298), ("softmax_320", 320)):
        try:
            x = mk(dev, (1, 32, 320, S), L1())
            us = timeit(dev, lambda: ttnn.softmax(x, dim=-1, memory_config=L1()), reps=10)
            out[tag] = {"us": us, "padded": list(x.padded_shape)}
            print(f"  {tag:18s} -> {us:8.2f} us")
            ttnn.deallocate(x)
        except Exception as e:                                   # noqa: BLE001
            out[tag] = {"error": str(e)[:250]}
    return out


# --------------------------------------------------------------------------------------------- #
# parity: is the aligned fill bit-exact?
# --------------------------------------------------------------------------------------------- #
def cmd_exact(dev, a):
    """Production arm: operands logically 298 (tail = whatever the producer left there).
    Aligned arm:    the same 298x298 data, tail EXPLICITLY zeroed, relabelled logically 320.
    """
    g = torch.Generator().manual_seed(7)
    at = torch.randn(1, 32, 298, 298, generator=g, dtype=torch.float32)
    bt = torch.randn(1, 32, 298, 298, generator=g, dtype=torch.float32)
    pc = T._triangle_mul_program_config(10)

    a_prod = ttnn.from_torch(at, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=L1())
    b_prod = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=L1())
    o_prod = ttnn.matmul(a_prod, b_prod, compute_kernel_config=ckc(), memory_config=L1(),
                         program_config=pc, dtype=ttnn.bfloat16)
    r_prod = ttnn.to_torch(o_prod)[..., :298, :298]

    a_pad = torch.zeros(1, 32, 320, 320, dtype=torch.float32)
    b_pad = torch.zeros(1, 32, 320, 320, dtype=torch.float32)
    a_pad[..., :298, :298] = at
    b_pad[..., :298, :298] = bt
    a_al = ttnn.from_torch(a_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=L1())
    b_al = ttnn.from_torch(b_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=L1())
    o_al = ttnn.matmul(a_al, b_al, compute_kernel_config=ckc(), memory_config=L1(),
                       program_config=pc, dtype=ttnn.bfloat16)
    r_al = ttnn.to_torch(o_al)[..., :298, :298]

    eq = torch.equal(r_prod, r_al)
    d = (r_prod.float() - r_al.float())
    res = {"torch_equal": bool(eq), "max_abs": float(d.abs().max()),
           "rmsd": float(d.pow(2).mean().sqrt()),
           "n_diff": int((d != 0).sum()), "n_total": int(d.numel()),
           "prod_logical": list(o_prod.shape), "aligned_logical": list(o_al.shape),
           "prod_padded": list(o_prod.padded_shape)}
    print(f"  torch.equal={eq}  max_abs={res['max_abs']:.3e}  rmsd={res['rmsd']:.3e}  "
          f"{res['n_diff']}/{res['n_total']} elements differ")

    # what does the PRODUCTION arm leave in the padded tail of its own output, and what does it
    # read from the padded tail of its operands? Both matter for where a fill has to live.
    full = ttnn.to_torch(o_prod)
    tail_out = full[..., 298:, :].abs().max().item(), full[..., :, 298:].abs().max().item()
    res["prod_output_tail_absmax"] = tail_out
    print(f"  production output padded tail |max| rows={tail_out[0]:.3e} cols={tail_out[1]:.3e}")
    for t in (a_prod, b_prod, o_prod, a_al, b_al, o_al):
        ttnn.deallocate(t)
    return res


# --------------------------------------------------------------------------------------------- #
# what does an aligned fill cost, and where can it live?
# --------------------------------------------------------------------------------------------- #
def cmd_fill(dev, a):
    out = {}
    at = torch.randn(1, 32, 298, 298, dtype=torch.float32)
    x = ttnn.from_torch(at, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=L1())
    print(f"  operand logical {list(x.shape)} padded {list(x.padded_shape)}")
    # A. metadata-only relabel: does ttnn.reshape to the padded shape cost anything?
    try:
        us = timeit(dev, lambda: ttnn.reshape(x, (1, 32, 320, 320)), reps=50)
        y = ttnn.reshape(x, (1, 32, 320, 320))
        out["reshape_relabel"] = {"us": us, "logical": list(y.shape),
                                  "padded": list(y.padded_shape)}
        print(f"  ttnn.reshape relabel      -> {us:8.3f} us  logical now {list(y.shape)}")
    except Exception as e:                                       # noqa: BLE001
        out["reshape_relabel"] = {"error": str(e)[:250]}
        print(f"  ttnn.reshape relabel: {str(e)[:200]}")
    # B. an explicit zeroing multiply by a [1,1,320,320] mask, in place, L1
    try:
        m = torch.zeros(1, 1, 320, 320, dtype=torch.float32)
        m[..., :298, :298] = 1.0
        mask = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=L1())
        mask320 = ttnn.reshape(mask, (1, 1, 320, 320))
        x320 = ttnn.reshape(x, (1, 32, 320, 320))
        us = timeit(dev, lambda: ttnn.multiply(x320, mask320, memory_config=L1()), reps=20)
        out["mask_multiply_L1"] = {"us": us, "MB": 32 * 320 * 320 * 2 / 1e6}
        print(f"  mask multiply (L1, 32ch)  -> {us:8.3f} us")
    except Exception as e:                                       # noqa: BLE001
        out["mask_multiply_L1"] = {"error": str(e)[:250]}
        print(f"  mask multiply: {str(e)[:200]}")
    # C. the same fill applied once to the pair tensor z instead, [1,298,320,256] DRAM
    try:
        z = mk(dev, (1, 298, 298, 256), DRAM())
        zm = torch.zeros(1, 1, 320, 1, dtype=torch.float32)
        zm[:, :, :298] = 1.0
        zmask = ttnn.from_torch(zm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=DRAM())
        z320 = ttnn.reshape(z, (1, 298, 320, 256))
        us = timeit(dev, lambda: ttnn.multiply(z320, zmask, memory_config=DRAM()), reps=8, warm=2)
        out["z_fill_once_DRAM"] = {"us": us, "MB": 298 * 320 * 256 * 2 / 1e6}
        print(f"  z fill once (DRAM 48.8MB) -> {us:8.3f} us")
        ttnn.deallocate(z)
    except Exception as e:                                       # noqa: BLE001
        out["z_fill_once_DRAM"] = {"error": str(e)[:250]}
        print(f"  z fill once: {str(e)[:200]}")
    ttnn.deallocate(x)
    return out


CMDS = {"roofs": cmd_roofs, "ab4": cmd_ab4, "cores": cmd_cores, "ladder": cmd_ladder,
        "sites": cmd_sites, "exact": cmd_exact, "fill": cmd_fill}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=sorted(CMDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = T.get_device()
    print(f"grid {T.COMPUTE_GRID_MAIN}")
    t0 = time.time()
    res = CMDS[a.cmd](dev, a)
    res["_cmd"] = a.cmd
    res["_grid"] = list(T.COMPUTE_GRID_MAIN)
    res["_wall_s"] = time.time() - t0
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)
    T.cleanup()


if __name__ == "__main__":
    main()
