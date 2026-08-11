#!/usr/bin/env python3
"""Screens for the four levers the completed tape names, plus the roofs re-measured pipelined in
this session so the floor and the shares use the same card in the same state.

Everything here is a SCREEN (per-op, off-fold). No number from it is a result.
"""
import json, statistics as st, time
from pathlib import Path
import sys

import torch
import ttnn

HERE = Path(__file__).resolve().parent
import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.reblock_permute as RB  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, CORE_GRID_MAIN, get_device  # noqa: E402

N, C_Z, HID = 512, 256, 256
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
OUT = {"n": N, "grid": list(COMPUTE_GRID_MAIN)}


def pipe(fn, warm=3, inflight=6, live_mib=0):
    # An L1 destination cannot hold `inflight` copies; cap so the live set stays under 100 MiB.
    if live_mib:
        inflight = max(1, min(inflight, int(100 // live_mib)))
    """Pipelined: no per-call sync. This is the roof methodology the predecessor used."""
    for _ in range(warm):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    best = None
    for _ in range(3):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(inflight)]
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / inflight
        best = ms if best is None else min(best, ms)
        for o in outs:
            if o is not None:
                ttnn.deallocate(o)
    return best


def serial(fn, warm=3, reps=5):
    for _ in range(warm):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if r is not None:
            ttnn.deallocate(r)
    return st.median(ts)


def mib(*shape):
    v = 2
    for d in shape:
        v *= d
    return v / 2 ** 20


# ------------------------------------------------------------------ 1. roofs, pipelined, this card
print("=== roofs, pipelined, card 0, this session ===")
roofs = {}
for name, mc_in, mc_out, mbytes in (("read DRAM->L1", DRAM, L1, 24),
                                    ("write L1->DRAM", L1, DRAM, 24),
                                    ("combined DRAM->DRAM", DRAM, DRAM, 48)):
    rows = int(mbytes * 2 ** 20 / 2 / 1024)
    s = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=mc_in)
    ms = pipe(lambda: ttnn.clone(s, memory_config=mc_out), live_mib=(mbytes if mc_out == L1 else 0))
    n_gb = mbytes / 1024 * (2 if "combined" in name else 1)
    roofs[name] = round(n_gb / (ms / 1e3), 1)
    print(f"  {name:24s} {mbytes} MiB  {ms:.4f} ms  {roofs[name]:.1f} GB/s")
    ttnn.deallocate(s)
# the shape the trimul actually moves, pipelined, both destinations
for nm, sh in (("chunk[1,32,512,512]", (1, 32, N, N)), ("chunkG8[1,256,512,512]", (1, 256, N, N))):
    s = ttnn.from_torch(torch.zeros(*sh), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    for tag, mc in (("->DRAM", DRAM), ("->L1", L1)):
        if tag == "->L1" and mib(*sh) > 120:
            continue
        ms = pipe(lambda: ttnn.clone(s, memory_config=mc), live_mib=(mib(*sh) if tag == "->L1" else 0))
        roofs[f"clone {nm}{tag}"] = round(mib(*sh) / 1024 / (ms / 1e3), 1)
        print(f"  clone {nm}{tag:7s} {ms:.4f} ms  {roofs[f'clone {nm}{tag}']:.1f} GB/s each way")
    ttnn.deallocate(s)
# square compute roof
for k in (2048, 4096):
    a = ttnn.from_torch(torch.zeros(k, k), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ms = pipe(lambda: ttnn.matmul(a, a, compute_kernel_config=ckc, memory_config=DRAM,
                                  core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16), inflight=4)
    roofs[f"compute {k}^3"] = round(2 * k ** 3 / 1e12 / (ms / 1e3), 2)
    print(f"  compute {k}^3  {ms:.4f} ms  {roofs[f'compute {k}^3']:.2f} TFLOP/s")
    ttnn.deallocate(a)
OUT["roofs"] = roofs

# ------------------------------------------------------------------ 2. the gates: is it the sigmoid?
print("\n=== gate screen: is the a/b gate SFPU-bound on the sigmoid? ===")
gates = {}
for C in (32, 256):
    p = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    g = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    # `multiply_` mutates, so re-make the destination each call.
    def mk(act):
        def f():
            pp = ttnn.clone(p, memory_config=DRAM)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            kw = dict(input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]) if act else {}
            r = ttnn.multiply_(pp, g, **kw)
            ttnn.synchronize_device(dev)
            ms = (time.perf_counter() - t0) * 1e3
            ttnn.deallocate(r)
            return ms
        return f
    for act in (True, False):
        f = mk(act)
        for _ in range(3):
            f()
        ms = st.median([f() for _ in range(5)])
        gates[f"C={C} sigmoid={act}"] = round(ms, 4)
        thru = mib(1, N, N, C) * 1.5 / 1024 / (ms / 1e3)   # 2 reads + 1 write
        print(f"  multiply_ C={C} sigmoid={str(act):5s} {ms:.4f} ms  {thru:.1f} GB/s (r+w/1.5)")
    # sigmoid folded into the matmul instead
    w = ttnn.from_torch(torch.randn(C_Z, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    x = ttnn.from_torch(torch.randn(1, N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    base = serial(lambda: ttnn.experimental.minimal_matmul(
        x, w, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc))
    print(f"  minimal_matmul [.,{C_Z}]@[{C_Z},{C}] {base:.4f} ms  (for reference)")
    gates[f"C={C} minimal_matmul_ms"] = round(base, 4)
    for t in (p, g, w, x):
        ttnn.deallocate(t)
OUT["gates"] = gates

# ------------------------------------------------------------------ 3. reblock on an L1 destination
print("\n=== reblock_permute with the L1 window forced open at N=512 ===")
rb = {}
old = (RB.L1_N_MIN, RB.L1_N_MAX)
for C in (32, 64, 256):
    s = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ref = ttnn.to_torch(s).permute(0, 3, 1, 2)
    for tag, mc in (("->DRAM", DRAM), ("->L1", L1)):
        if tag == "->L1" and mib(1, N, N, C) > 120:
            continue
        RB.L1_N_MIN, RB.L1_N_MAX = 0, 4096
        el = RB.eligible(s, mc)
        if not el:
            print(f"  C={C}{tag}: still not eligible ({dict(RB.REJECTS)})")
            continue
        lm = mib(1, N, N, C) if tag == "->L1" else 0
        base = pipe(lambda: ttnn.permute(s, (0, 3, 1, 2), memory_config=mc), live_mib=lm)
        got = pipe(lambda: RB.reblock_permute(s, mc), live_mib=lm)
        r = RB.reblock_permute(s, mc)
        ok = torch.equal(ttnn.to_torch(r), ref)
        ttnn.deallocate(r)
        rb[f"C={C}{tag}"] = dict(permute_ms=round(base, 4), reblock_ms=round(got, 4),
                                 x=round(base / got, 4), equal=bool(ok),
                                 reblock_gbs=round(mib(1, N, N, C) / 1024 / (got / 1e3), 1))
        print(f"  C={C}{tag:7s} permute {base:.4f} -> reblock {got:.4f} ms  {base/got:.3f}x  "
              f"{rb[f'C={C}{tag}']['reblock_gbs']:.1f} GB/s  torch.equal={ok}")
    ttnn.deallocate(s)
RB.L1_N_MIN, RB.L1_N_MAX = old
OUT["reblock_l1"] = rb

# ------------------------------------------------------------------ 4. tail: row-block + L1 output
print("\n=== tail screen: row-blocked output projections with L1 results ===")
tail = {}
w_g = ttnn.from_torch(torch.randn(C_Z, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                      dtype=ttnn.bfloat16)
w_p = ttnn.from_torch(torch.randn(C_Z, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                      dtype=ttnn.bfloat16)
zf = ttnn.from_torch(torch.randn(1, N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                     dtype=ttnn.bfloat16, memory_config=DRAM)
xf = ttnn.from_torch(torch.randn(1, N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                     dtype=ttnn.bfloat16, memory_config=DRAM)


def tail_today():
    xn = ttnn.layer_norm(xf, epsilon=1e-5, compute_kernel_config=ckc)
    po = T._trimul_out_proj(xn, w_p, ckc)
    ttnn.deallocate(xn)
    zn = ttnn.layer_norm(zf, epsilon=1e-5, compute_kernel_config=ckc)
    go = T._trimul_out_proj(zn, w_g, ckc)
    ttnn.deallocate(zn)
    r = ttnn.multiply_(po, go, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(go)
    return r


def tail_rowblocked(R):
    blocks = []
    for s in range(0, N, R):
        e = min(s + R, N)
        xr = ttnn.layer_norm(xf[:, s:e], epsilon=1e-5, compute_kernel_config=ckc)
        pb = T._pair_proj_linear(xr, w_p, ckc, ttnn.bfloat16, l1_out=True)
        ttnn.deallocate(xr)
        zr = ttnn.layer_norm(zf[:, s:e], epsilon=1e-5, compute_kernel_config=ckc)
        gb = T._pair_proj_linear(zr, w_g, ckc, ttnn.bfloat16, l1_out=True)
        ttnn.deallocate(zr)
        blocks.append(ttnn.clone(ttnn.multiply_(
            pb, gb, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
            memory_config=DRAM))
        ttnn.deallocate(gb)
        ttnn.deallocate(pb)
    r = ttnn.concat(blocks, 1, memory_config=DRAM)
    for b in blocks:
        ttnn.deallocate(b)
    return r


base = serial(tail_today)
ref = ttnn.to_torch(tail_today())
tail["today_ms"] = round(base, 4)
print(f"  today (full-size, _trimul_out_proj)          {base:.4f} ms")
for R in (128, 256):
    try:
        ms = serial(lambda: tail_rowblocked(R))
        r = tail_rowblocked(R)
        got = ttnn.to_torch(r)
        ttnn.deallocate(r)
        eq = torch.equal(got, ref)
        pcc = float(torch.corrcoef(torch.stack(
            [got.flatten().float(), ref.flatten().float()]))[0, 1])
        tail[f"rowblock_{R}_ms"] = round(ms, 4)
        tail[f"rowblock_{R}_equal"] = bool(eq)
        tail[f"rowblock_{R}_pcc"] = round(pcc, 8)
        print(f"  row-blocked R={R}, L1 projection results     {ms:.4f} ms  {base/ms:.3f}x  "
              f"torch.equal={eq} pcc={pcc:.8f}")
    except Exception as e:
        tail[f"rowblock_{R}_err"] = str(e)[:200]
        print(f"  row-blocked R={R}: {type(e).__name__} {str(e)[:140]}")
OUT["tail"] = tail

Path(HERE / "levers_512_qb2c0.json").write_text(json.dumps(OUT, indent=2))
print("\nWROTE " + str(HERE / "levers_512_qb2c0.json"))
