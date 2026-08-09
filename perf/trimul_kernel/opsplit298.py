#!/usr/bin/env python3
"""Where a 298-aa trimul spends its 7.24 ms, and what happens if you take the layout tax out.

W2 of the PERF WAR. `layer_split.py --n 320` ranks the Pairformer block's parts but times each
trimul sub-op in isolation, so the shares do not add up to a trimul. This replays the real
`TriangleMultiplication.__call__` op for op, times every one of them with a device sync on both
sides, and checks the replay is bit-identical to the module it copies (torch.equal). Only then
are the shares a decomposition rather than a sample.

Every op also gets its analytic byte traffic and FLOP count, so each line lands on a named roof:
DRAM read 410.9 GB/s, DRAM write 277.6 GB/s, HiFi4 dense bf16 137.1 TFLOP/s (WARROOM ground
truth, re-measured per card). The L1-resident ops are additionally compared against a same-shape
ttnn.clone measured on this card, because their roof is L1 bandwidth, not DRAM.

Variants (`--variant`, repeatable) each turn one thing off and are checked for bit-exactness
against the baseline replay:
  base          the production path
  norealloc     drop the ttnn.reallocate after every channel-move permute
  oneconcat     collect the per-channel output chunks and concat once (O(n) not O(n^2) copies)
  both          norealloc + oneconcat

    TT_VISIBLE_DEVICES=1 python3 perf/trimul_kernel/opsplit298.py --n 320
    TT_VISIBLE_DEVICES=1 python3 perf/trimul_kernel/opsplit298.py --n 320 \
        --variant base --variant norealloc --variant oneconcat --variant both
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

READ_ROOF_GBS = 410.9
WRITE_ROOF_GBS = 277.6
HIFI4_TFLOPS = 137.1


class Tape:
    """Per-op timing with a device sync on both sides, tagged by op class."""

    def __init__(self, dev, on: bool):
        self.dev = dev
        self.on = on
        self.rows = []

    def __call__(self, tag, fn, *, rbytes=0, wbytes=0, flops=0):
        if not self.on:
            return fn()
        ttnn.synchronize_device(self.dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(self.dev)
        self.rows.append(dict(tag=tag, ms=(time.perf_counter() - t0) * 1e3,
                              rbytes=rbytes, wbytes=wbytes, flops=flops))
        return out


def _tile_bytes(shape, dtype_bytes=2):
    n = 1
    for d in shape:
        n *= d
    return n * dtype_bytes


def trimul_replay(tm, x, tape, *, realloc=True, oneconcat=False):
    """Op-for-op copy of TriangleMultiplication.__call__, L1 path, mask=None, non-fast.

    Kept deliberately literal so a divergence from tt_bio shows up as a torch.equal failure
    rather than as a plausible-looking number.
    """
    ckc = tm.compute_kernel_config
    x_in = x
    B, H, _, c_z = x.shape
    pair_b = _tile_bytes((B, H, H, c_z))

    x_norm_in = tape("layer_norm.in", lambda: ttnn.layer_norm(
        x, weight=tm.in_norm_weight, bias=tm.in_norm_bias, epsilon=1e-5,
        compute_kernel_config=ckc), rbytes=pair_b, wbytes=pair_b)
    mc = T._triangle_mul_memory_config(H)
    C = T._trimul_chunk_size(H, tm._hidden)
    gp_in_chunks = tm._gp_in_chunks(C)
    n_pairs = len(gp_in_chunks)
    seq_len_tiles = (H + 31) // 32
    program_config = T._triangle_mul_program_config(seq_len_tiles)

    chunk_b = _tile_bytes((B, H, H, C))
    fused_b = 4 * chunk_b
    out_chunks = [] if oneconcat else None

    for i in range(n_pairs):
        gp_in_fused = tape("minimal_matmul.gp_in", lambda: ttnn.experimental.minimal_matmul(
            x_norm_in, gp_in_chunks[i], memory_config=mc, dtype=T._dtype(),
            compute_kernel_config=ckc),
            rbytes=pair_b, wbytes=fused_b, flops=2 * B * H * H * c_z * 4 * C)
        parts = tape("chunk4.split", lambda: ttnn.chunk(gp_in_fused, chunks=4, dim=-1),
                     rbytes=fused_b, wbytes=fused_b)
        g_in_a, g_in_b, p_in_a, p_in_b = parts
        ttnn.deallocate(gp_in_fused)
        a_chunk = tape("gate.sigmoid_mul", lambda: ttnn.multiply_(
            p_in_a, g_in_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
            rbytes=2 * chunk_b, wbytes=chunk_b)
        b_chunk = tape("gate.sigmoid_mul", lambda: ttnn.multiply_(
            p_in_b, g_in_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
            rbytes=2 * chunk_b, wbytes=chunk_b)
        ttnn.deallocate(g_in_a)
        ttnn.deallocate(g_in_b)

        dims_a = (0, 3) + ((2, 1) if tm.ending else (1, 2))
        dims_b = (0, 3) + ((1, 2) if tm.ending else (2, 1))
        a_p = tape(f"permute.in{dims_a}", lambda: ttnn.permute(a_chunk, dims_a, memory_config=mc),
                   rbytes=chunk_b, wbytes=chunk_b)
        ttnn.deallocate(a_chunk)
        if realloc:
            a_r = tape("reallocate", lambda: ttnn.reallocate(a_p, memory_config=mc),
                       rbytes=chunk_b, wbytes=chunk_b)
            ttnn.deallocate(a_p)
            a_p = a_r
        b_p = tape(f"permute.in{dims_b}", lambda: ttnn.permute(b_chunk, dims_b, memory_config=mc),
                   rbytes=chunk_b, wbytes=chunk_b)
        ttnn.deallocate(b_chunk)
        if realloc:
            b_r = tape("reallocate", lambda: ttnn.reallocate(b_p, memory_config=mc),
                       rbytes=chunk_b, wbytes=chunk_b)
            ttnn.deallocate(b_p)
            b_p = b_r

        x_chunk = tape("matmul.triangle", lambda: ttnn.matmul(
            a_p, b_p, compute_kernel_config=ckc, memory_config=mc,
            program_config=program_config, dtype=ttnn.bfloat16),
            rbytes=2 * chunk_b, wbytes=chunk_b, flops=2 * B * C * H * H * H)
        ttnn.deallocate(a_p)
        ttnn.deallocate(b_p)
        x_chunk = tape("permute.out(0,2,3,1)", lambda: ttnn.permute(
            x_chunk, (0, 2, 3, 1), memory_config=mc), rbytes=chunk_b, wbytes=chunk_b)

        if out_chunks is not None:
            moved = tape("clone.to_dram", lambda: ttnn.clone(
                x_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                rbytes=chunk_b, wbytes=chunk_b)
            ttnn.deallocate(x_chunk)
            out_chunks.append(moved)
        elif i == 0:
            x = tape("clone.to_dram", lambda: ttnn.clone(
                x_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                rbytes=chunk_b, wbytes=chunk_b)
            ttnn.deallocate(x_chunk)
        else:
            x_old = x
            acc = (i + 1) * C
            x = tape(f"concat.acc{acc}", lambda: ttnn.concat([x_old, x_chunk], dim=-1),
                     rbytes=acc * chunk_b // C, wbytes=acc * chunk_b // C)
            ttnn.deallocate(x_old)
            ttnn.deallocate(x_chunk)

    if out_chunks is not None:
        x = tape("concat.once", lambda: ttnn.concat(out_chunks, dim=-1),
                 rbytes=pair_b, wbytes=pair_b)
        for t in out_chunks:
            ttnn.deallocate(t)

    x = tape("layer_norm.out", lambda: ttnn.layer_norm(
        x, weight=tm.out_norm_weight, bias=tm.out_norm_bias, epsilon=1e-5,
        compute_kernel_config=ckc), rbytes=pair_b, wbytes=pair_b)
    p_out = tape("linear.p_out", lambda: ttnn.linear(
        x, tm.out_p_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(),
        compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN),
        rbytes=pair_b, wbytes=pair_b, flops=2 * B * H * H * c_z * c_z)
    ttnn.deallocate(x)
    g_out = tape("linear.g_out", lambda: ttnn.linear(
        x_norm_in, tm.g_out_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(),
        compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN),
        rbytes=pair_b, wbytes=pair_b, flops=2 * B * H * H * c_z * c_z)
    ttnn.deallocate(x_norm_in)
    out = tape("gate.out_sigmoid_mul", lambda: ttnn.multiply_(
        p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
        rbytes=2 * pair_b, wbytes=pair_b)
    return out


def timeit(dev, fn, warm, iters, pipe):
    for _ in range(warm):
        r = fn()
        ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ser = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ser.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(pipe)]
    ttnn.synchronize_device(dev)
    pip = (time.perf_counter() - t0) * 1e3 / pipe
    for o in outs:
        ttnn.deallocate(o)
    return sorted(ser)[len(ser) // 2], pip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--variant", action="append", default=None)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--pipe", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    variants = args.variant or ["base", "norealloc", "oneconcat", "both"]
    VAR = {"base": dict(realloc=True, oneconcat=False),
           "norealloc": dict(realloc=False, oneconcat=False),
           "oneconcat": dict(realloc=True, oneconcat=True),
           "both": dict(realloc=False, oneconcat=True)}

    N = args.n
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    tm = layer.triangle_multiplication_start
    torch.manual_seed(0)
    z_h = torch.randn(1, N, N, c_z)
    z = ttnn.from_torch(z_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    C = T._trimul_chunk_size(N, tm._hidden)
    mc = T._triangle_mul_memory_config(N)
    print(f"N={N} c_z={c_z} hidden={tm._hidden} chunk={C} n_pairs={tm._hidden // C} "
          f"buffer={mc.buffer_type} grid={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}")

    # the module itself: the number every share is a share OF
    ref = tm(z, None)
    ref_h = ttnn.to_torch(ref)
    ttnn.deallocate(ref)
    mod_ser, mod_pipe = timeit(dev, lambda: tm(z, None), args.warm, args.iters, args.pipe)
    print(f"module trimul: serial {mod_ser:.3f} ms  pipe {mod_pipe:.3f} ms")

    # L1 copy roof on THIS card, at the chunk shape the layout ops actually move
    chunk_l1 = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    cl_ser, cl_pipe = timeit(dev, lambda: ttnn.clone(chunk_l1, memory_config=ttnn.L1_MEMORY_CONFIG),
                             args.warm, args.iters, args.pipe)
    cb = _tile_bytes((1, N, N, C))
    print(f"L1->L1 clone of the chunk shape [1,{N},{N},{C}] ({cb / 2**20:.1f} MiB): "
          f"{cl_pipe:.3f} ms = {cb / (cl_pipe * 1e-3) / 1e9:.1f} GB/s each way")
    ttnn.deallocate(chunk_l1)

    results = {"n": N, "c_z": c_z, "chunk": C, "buffer": str(mc.buffer_type),
               "module_serial_ms": round(mod_ser, 4), "module_pipe_ms": round(mod_pipe, 4),
               "l1_clone_chunk_ms": round(cl_pipe, 4),
               "l1_clone_gbs_each_way": round(cb / (cl_pipe * 1e-3) / 1e9, 1),
               "variants": []}

    base_out = None
    for name in variants:
        kw = VAR[name]
        # correctness: one untimed replay, compared to the module's own output
        silent = Tape(dev, on=False)
        out = trimul_replay(tm, z, silent, **kw)
        out_h = ttnn.to_torch(out)
        ttnn.deallocate(out)
        eq_mod = bool(torch.equal(out_h, ref_h))
        maxabs = float((out_h.float() - ref_h.float()).abs().max())
        if base_out is None:
            base_out = out_h
        eq_base = bool(torch.equal(out_h, base_out))

        # timing: replay with no per-op syncs, same protocol as the module
        r_ser, r_pipe = timeit(dev, lambda: trimul_replay(tm, z, Tape(dev, on=False), **kw),
                               args.warm, args.iters, args.pipe)
        # attribution: one taped replay
        tape = Tape(dev, on=True)
        out = trimul_replay(tm, z, tape, **kw)
        ttnn.deallocate(out)
        agg = collections.OrderedDict()
        for r in tape.rows:
            a = agg.setdefault(r["tag"], dict(tag=r["tag"], n=0, ms=0.0, rbytes=0, wbytes=0, flops=0))
            a["n"] += 1
            a["ms"] += r["ms"]
            for k in ("rbytes", "wbytes", "flops"):
                a[k] += r[k]
        total = sum(a["ms"] for a in agg.values())
        rows = sorted(agg.values(), key=lambda a: -a["ms"])
        print(f"\n--- {name}: replay serial {r_ser:.3f} ms  pipe {r_pipe:.3f} ms  "
              f"(taped sum {total:.3f} ms)  bit-exact vs module={eq_mod} vs base={eq_base} "
              f"maxabs={maxabs:.3g}")
        print(f"{'op':28s} {'n':>3s} {'ms':>8s} {'%':>6s} {'GB/s r':>8s} {'GB/s w':>8s} "
              f"{'TFLOP/s':>8s} {'%roof':>7s}")
        for a in rows:
            s = a["ms"] * 1e-3
            rg = a["rbytes"] / s / 1e9
            wg = a["wbytes"] / s / 1e9
            tf = a["flops"] / s / 1e12
            if a["flops"]:
                pr = 100 * tf / HIFI4_TFLOPS
                roof = "cmp"
            else:
                pr = 100 * max(rg / READ_ROOF_GBS, wg / WRITE_ROOF_GBS)
                roof = "dram"
            a["gbs_r"], a["gbs_w"], a["tflops"], a["pct_roof"], a["roof"] = (
                round(rg, 1), round(wg, 1), round(tf, 2), round(pr, 1), roof)
            a["ms"] = round(a["ms"], 4)
            a["share"] = round(a["ms"] / total, 4)
            print(f"{a['tag']:28s} {a['n']:3d} {a['ms']:8.3f} {100 * a['share']:5.1f}% "
                  f"{rg:8.1f} {wg:8.1f} {tf:8.2f} {pr:6.1f}%")
        results["variants"].append(dict(
            name=name, replay_serial_ms=round(r_ser, 4), replay_pipe_ms=round(r_pipe, 4),
            taped_sum_ms=round(total, 4), bit_exact_vs_module=eq_mod, bit_exact_vs_base=eq_base,
            max_abs_vs_module=maxabs, speedup_vs_module=round(mod_pipe / r_pipe, 4), ops=rows))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
