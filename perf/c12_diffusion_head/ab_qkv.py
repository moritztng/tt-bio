#!/usr/bin/env python3
"""The AttentionPairBias qkv projection: three arms, interleaved, at the executed shapes.

`c12-tail-classes-screen` priced this site's head split at 0.20470 s (token transformer, 4800
programs) + 0.00581 s (trunk, 264 programs) per 512 aa fold, 1350 MHz. The head-major writer
deletes both. The lever is two changes and they have to be separated or neither reading means
anything:

    A0  ttnn.linear(core_grid) + nlp_create_qkv_heads     what ships today
    A1  minimal_matmul(config) + nlp_create_qkv_heads     op class changed, split still separate
    B   head-major minimal_matmul                         the split deleted

    torch.equal(A1, B)   the bit-exactness claim. A pure tile re-point moves no element, so this
                         must hold at every converted signature; if it does not, the transcription
                         is wrong and that is a hard stop.
    A1 - B               the prize this row exists for.
    A0 - A1              the op-class change, which is NOT bit-exact and owes an Angstrom reading
                         against the seed-scatter floor before any default flips.

`--plan` needs no device: it derives the descriptor geometry from the shapes and checks the guards
the head-major writer depends on. Everything else opens one card and must be run under a card
lease with both chips of the pair checked before and after.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent

# (name, batch, seq, c_in, n_heads, head_dim, padded_head_dim, calls_per_fold, in_situ_s)
# From perf/c12_tail_screen/leads.json + screen.json operands, 512 aa, 1350 MHz.
SIGS = [
    ("dit_token", 1, 512, 768, 16, 48, 64, 4800, 0.20470),
    ("apb_trunk", 1, 512, 384, 16, 32, 32, 264, 0.00581),
]

# The atom block, which is two projections and a deleted pad and slice rather than one projection.
# (name, B, K, W, ATOM_DIM, D_S, n_heads, head_dim, calls_per_fold, in_situ_s)
# in_situ_s is the create signature only; the 0.01309 s concat is the tail and is not this arm.
ATOM = ("atom_block", 1, 140, 32, 128, 128, 4, 32, 1200, 0.09016)


def plan(grid=(11, 10)):
    """The descriptor geometry for each signature, from shapes alone."""
    sys.path.insert(0, str(OUT.parents[1]))
    from tt_bio.mm_generic import _div_up, _round_up
    from tt_bio.tenstorrent import _MM_BLOCK, _MM_DEFAULT

    gx, gy = grid
    rows = []
    for name, b, s, c_in, heads, hd, phd, calls, in_situ in SIGS:
        kt, nt = _div_up(c_in, 32), _div_up(3 * heads * phd, 32)
        mt = _div_up(b * s, 32)
        blk = _MM_BLOCK.get((kt, nt))
        assert blk is not None, f"{name}: no block entry at (kt, nt) = ({kt}, {nt})"
        M_blk, K_blk, N_blk, sh, sw = blk
        M, N = b * s, 3 * heads * phd
        transpose = M > N
        in0_axis, in1_axis = (gx, gy) if transpose else (gy, gx)
        padded_mt, padded_nt = _round_up(mt, in0_axis), _round_up(nt, in1_axis)
        padded_kt = _round_up(kt, K_blk)
        dt = phd // 32
        d1 = heads * dt                      # N tiles in one output chunk
        rows.append({
            "sig": name, "calls_per_fold": calls, "in_situ_s": in_situ,
            "M": M, "K": c_in, "N": N, "M_tiles": mt, "K_tiles": kt, "N_tiles": nt,
            "block": list(blk), "is_MM_DEFAULT": blk is _MM_DEFAULT,
            "transpose_core_grid": transpose,
            "padded_M_tiles": padded_mt, "padded_N_tiles": padded_nt, "padded_K_tiles": padded_kt,
            "M_tiles_per_core": padded_mt // in0_axis, "N_tiles_per_core": padded_nt // in1_axis,
            "M_blocks_per_core": _div_up(padded_mt // in0_axis, M_blk),
            "N_blocks_per_core": _div_up(padded_nt // in1_axis, N_blk),
            "K_blocks": padded_kt // K_blk,
            "HEAD_MAJOR_MT": s // 32, "HEAD_MAJOR_DT": dt,
            "N_tiles_per_chunk": nt // 3, "n_heads_times_DT": d1,
            # The guards the head-major writer depends on.
            "chunks_divide_N": nt % 3 == 0,
            "chunk_is_heads_times_DT": nt // 3 == d1,
            "padded_head_dim_is_whole_tiles": phd % 32 == 0,
            "head_dim_is_whole_tiles": hd % 32 == 0,
        })
    return rows


def atom_plan(grid=(11, 10)):
    """The atom block's two descriptors, from shapes alone."""
    sys.path.insert(0, str(OUT.parents[1]))
    from tt_bio.mm_generic import _div_up, _round_up
    from tt_bio.tenstorrent import _MM_BLOCK, _MM_DEFAULT

    _n, b, k, w, adim, d_s, heads, hd, calls, in_situ = ATOM
    gx, gy = grid
    rows = []
    for part, m_rows, n_cols, chunks in (("q", b * k * w, heads * hd, 1),
                                         ("kv", b * k * adim, 2 * heads * hd, 2)):
        kt, nt, mt = _div_up(d_s, 32), _div_up(n_cols, 32), _div_up(m_rows, 32)
        blk = _MM_BLOCK.get((kt, nt))
        assert blk is not None, f"atom {part}: no block entry at ({kt}, {nt})"
        M_blk, K_blk, N_blk, _sh, _sw = blk
        transpose = m_rows > n_cols
        in0_axis, in1_axis = (gx, gy) if transpose else (gy, gx)
        dt = hd // 32
        mt_per_window = (w if part == "q" else adim) // 32
        rows.append({
            "part": part, "M": m_rows, "K": d_s, "N": n_cols,
            "M_tiles": mt, "K_tiles": kt, "N_tiles": nt, "n_chunks": chunks,
            "block": list(blk), "is_MM_DEFAULT": blk is _MM_DEFAULT,
            "transpose_core_grid": transpose,
            "M_tiles_per_core": _round_up(mt, in0_axis) // in0_axis,
            "N_tiles_per_core": _round_up(nt, in1_axis) // in1_axis,
            "K_blocks": _round_up(kt, K_blk) // K_blk,
            "HEAD_MAJOR_MT": mt_per_window, "HEAD_MAJOR_DT": dt,
            # At MT = 1 and DT = 1 the head-major id is the plain writer's, so q needs no define.
            "needs_no_define": mt_per_window == 1 and dt == 1,
            "chunks_divide_N": nt % chunks == 0,
            "chunk_is_heads_times_DT": nt // chunks == heads * dt,
            "guards_pass": nt % chunks == 0 and nt // chunks == heads * dt and hd % 32 == 0,
        })
    return {"calls_per_fold": calls, "in_situ_s": in_situ, "parts": rows}


def _plan_main(args):
    rows = plan()
    ok = True
    for r in rows:
        good = (r["chunks_divide_N"] and r["chunk_is_heads_times_DT"]
                and r["padded_head_dim_is_whole_tiles"])
        ok &= good
        print(f"{r['sig']:<11} M={r['M']:<6} K={r['K']:<5} N={r['N']:<5} "
              f"mt/kt/nt={r['M_tiles']}/{r['K_tiles']}/{r['N_tiles']:<4} "
              f"blk={tuple(r['block'])} default={r['is_MM_DEFAULT']} "
              f"transpose={r['transpose_core_grid']} "
              f"MT={r['HEAD_MAJOR_MT']} DT={r['HEAD_MAJOR_DT']} "
              f"chunk={r['N_tiles_per_chunk']}=={r['n_heads_times_DT']} guards={good}")
        # head_dim 48 pads to 64: the PADDED width is what the writer addresses, and that is the
        # precondition. The logical 48 is not, and the tail that has to unpad it is lead L2.
        if not r["head_dim_is_whole_tiles"]:
            print(f"{'':<11} logical head_dim is not a whole tile -- padded form is, which is the "
                  "precondition; the MERGE at this site is L2 and is not this row")
    atom = atom_plan()
    for r in atom["parts"]:
        ok &= r["guards_pass"]
        print(f"atom_{r['part']:<6} M={r['M']:<6} K={r['K']:<5} N={r['N']:<5} "
              f"mt/kt/nt={r['M_tiles']}/{r['K_tiles']}/{r['N_tiles']:<4} "
              f"blk={tuple(r['block'])} default={r['is_MM_DEFAULT']} "
              f"transpose={r['transpose_core_grid']} "
              f"MT={r['HEAD_MAJOR_MT']} DT={r['HEAD_MAJOR_DT']} chunks={r['n_chunks']} "
              f"no_define={r['needs_no_define']} guards={r['guards_pass']}")
    total = sum(r["in_situ_s"] for r in rows) + atom["in_situ_s"]
    progs = sum(r["calls_per_fold"] for r in rows) + atom["calls_per_fold"]
    print(f"\nin situ {total:.5f} s over {progs} programs, "
          f"1350 MHz -> {total * 1350e6 / 1e6:.1f} Mcycles")
    (OUT / "plan.json").write_text(json.dumps({"all_guards_pass": bool(ok), "grid": [11, 10],
                                               "signatures": rows, "atom": atom}, indent=1) + "\n")
    print(f"{'PASS' if ok else 'FAIL'} -- {OUT / 'plan.json'}")
    return 0 if ok else 1


# --- the device arms -----------------------------------------------------------------------------

def _clock_mhz():
    """AICLK sampled now. Every number this script prints carries it."""
    import subprocess
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=30).stdout
        d = json.loads(out)
        return [int(c["board_info"]["aiclk"]) for c in d.get("device_info", [])]
    except Exception as e:                                                   # noqa: BLE001
        return f"unavailable: {e}"


def _arms(device, sig, reps):
    import torch
    import ttnn
    sys.path.insert(0, str(OUT.parents[1]))
    from tt_bio import triatt_qkv as TQ
    from tt_bio.tenstorrent import CORE_GRID_MAIN, _qkv_mm_config

    name, b, s, c_in, heads, hd, phd, _calls, _in_situ = sig
    torch.manual_seed(0)
    x_t = torch.randn(b, s, c_in, dtype=torch.bfloat16)
    w_t = torch.randn(c_in, 3 * heads * phd, dtype=torch.bfloat16) * 0.02
    bias_t = torch.randn(3 * heads * phd, dtype=torch.bfloat16) * 0.02
    mk = dict(layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16,
              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.from_torch(x_t, **mk)
    w = ttnn.from_torch(w_t, **mk)
    bias = ttnn.from_torch(bias_t, **mk)
    # The repo's own trunk/diffusion config (tt_bio/af2.py:compute_kernel_config), arch-correct.
    # fp32_dest_acc_en=True is what makes `_MM_DEFAULT` the op's actual default blocking, which is
    # the whole basis for A1 being byte-identical to an unconfigured minimal_matmul.
    cls = (ttnn.types.WormholeComputeKernelConfig
           if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = _qkv_mm_config(x, w)
    assert cfg is not None, f"{name}: no block config"

    def a0():
        qkv = ttnn.linear(x, w, bias=bias, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        return q, k, v

    def a1():
        qkv = ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, bias_tensor=bias, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        return q, k, v

    def arm_b():
        out = TQ.qkv_heads(x, w, ckc, heads, phd, ttnn.bfloat16, cfg, bias=bias,
                           allow_m_le_n=True, site="apb")
        assert out is not None, f"{name}: head-major declined -- {TQ.APB_REJECTS}"
        return out

    arms = {"A0_linear_split": a0, "A1_mm_split": a1, "B_head_major": arm_b,
            "AA_control": a0}
    # torch.equal(A1, B) BEFORE any timing, so a wrong transcription never reaches a number.
    ref = [ttnn.to_torch(t) for t in a1()]
    got = [ttnn.to_torch(t) for t in arm_b()]
    equal = all(torch.equal(r, g) for r, g in zip(ref, got))
    maxabs = max(float((r.float() - g.float()).abs().max()) for r, g in zip(ref, got))

    times: dict = {k: [] for k in arms}
    for rep in range(reps + 1):                    # rep 0 is the cold rep, discarded PER ARM
        for key, fn in arms.items():
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            outs = fn()
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) * 1e3
            for o in outs:
                ttnn.deallocate(o)
            if rep:
                times[key].append(dt)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    return {"sig": name, "torch_equal_A1_vs_B": equal, "max_abs_A1_vs_B": maxabs,
            "ms_per_call": med, "reps": reps,
            "aa_floor_pct": abs(med["AA_control"] - med["A0_linear_split"])
            / med["A0_linear_split"] * 100.0,
            "B_over_A1": med["A1_mm_split"] / med["B_head_major"],
            "A1_over_A0": med["A0_linear_split"] / med["A1_mm_split"],
            "raw_ms": times}


def _atom_arms(device, reps):
    """The atom block's three arms. A1 vs B is the tile re-point; A0 vs A1 is the op class."""
    import torch
    import ttnn
    sys.path.insert(0, str(OUT.parents[1]))
    from tt_bio import triatt_qkv as TQ
    from tt_bio.tenstorrent import CORE_GRID_MAIN, _qkv_mm_config

    _n, B, K, W, ADIM, D_S, H, HD, _calls, _in_situ = ATOM
    torch.manual_seed(0)
    mk = dict(layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16,
              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s = ttnn.from_torch(torch.randn(B, K, W, D_S, dtype=torch.bfloat16), **mk)
    s_kv = ttnn.from_torch(torch.randn(B, K, ADIM, D_S, dtype=torch.bfloat16), **mk)
    w_q = ttnn.from_torch(torch.randn(D_S, H * HD, dtype=torch.bfloat16) * 0.02, **mk)
    b_q = ttnn.from_torch(torch.randn(H * HD, dtype=torch.bfloat16) * 0.02, **mk)
    w_kv = ttnn.from_torch(torch.randn(D_S, 2 * H * HD, dtype=torch.bfloat16) * 0.02, **mk)
    cls = (ttnn.types.WormholeComputeKernelConfig
           if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg_q, cfg_kv = _qkv_mm_config(s, w_q), _qkv_mm_config(s_kv, w_kv)
    assert cfg_q is not None and cfg_kv is not None, "atom: no block config"

    def split(q, kv):
        """The shipped pad -> reshape -> split -> reshape -> slice, shared by A0 and A1."""
        q = ttnn.pad(q, [[0, 0], [0, 0], [0, ADIM - W], [0, 0]], 0.0)
        q = ttnn.reshape(q, (B * K, 1, ADIM, -1))
        kv = ttnn.reshape(kv, (B * K, 1, ADIM, -1))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q, kv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
        _b, h, sq, d = q.shape
        q, k, v = (ttnn.reshape(x, (B, K * h, sq, d)) for x in (q, k, v))
        return q[:, :, :W, :], k, v

    def a0():
        return split(
            ttnn.linear(s, w_q, bias=b_q, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                        dtype=ttnn.bfloat16),
            ttnn.linear(s_kv, w_kv, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                        dtype=ttnn.bfloat16))

    def a1():
        return split(
            ttnn.experimental.minimal_matmul(
                input_tensor=s, weight_tensor=w_q, bias_tensor=b_q, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=cfg_q),
            ttnn.experimental.minimal_matmul(
                input_tensor=s_kv, weight_tensor=w_kv, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=cfg_kv))

    def arm_b():
        out = TQ.atom_qkv_heads(s, w_q, b_q, s_kv, w_kv, ckc, H, HD, ttnn.bfloat16, cfg_q, cfg_kv)
        assert out is not None, f"atom: head-major declined -- {TQ.ATOM_REJECTS}"
        return out

    arms = {"A0_linear_split": a0, "A1_mm_split": a1, "B_head_major": arm_b, "AA_control": a0}
    ref = [ttnn.to_torch(x) for x in a1()]
    got = [ttnn.to_torch(x) for x in arm_b()]
    equal = all(r.shape == g.shape and torch.equal(r, g) for r, g in zip(ref, got))
    maxabs = max(float((r.float() - g.float()).abs().max())
                 for r, g in zip(ref, got) if r.shape == g.shape) if equal else float("nan")

    times: dict = {k: [] for k in arms}
    for rep in range(reps + 1):
        for key, fn in arms.items():
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            outs = fn()
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) * 1e3
            for o in outs:
                ttnn.deallocate(o)
            if rep:
                times[key].append(dt)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    return {"sig": "atom_block", "torch_equal_A1_vs_B": equal, "max_abs_A1_vs_B": maxabs,
            "ms_per_call": med, "reps": reps,
            "aa_floor_pct": abs(med["AA_control"] - med["A0_linear_split"])
            / med["A0_linear_split"] * 100.0,
            "B_over_A1": med["A1_mm_split"] / med["B_head_major"],
            "A1_over_A0": med["A0_linear_split"] / med["A1_mm_split"],
            "raw_ms": times}


def _device_main(args):
    import ttnn
    dev = ttnn.open_device(device_id=args.device_id)
    try:
        clk_before = _clock_mhz()
        rows = [_arms(dev, s, args.reps) for s in SIGS] + [_atom_arms(dev, args.reps)]
        clk_after = _clock_mhz()
    finally:
        ttnn.close_device(dev)
    rec = {"clock_MHz_before": clk_before, "clock_MHz_after": clk_after,
           "device_id": args.device_id, "signatures": rows}
    for r in rows:
        print(f"{r['sig']:<11} torch.equal(A1,B)={r['torch_equal_A1_vs_B']} "
              f"maxabs={r['max_abs_A1_vs_B']:.3e} "
              + " ".join(f"{k}={v:.4f}ms" for k, v in r["ms_per_call"].items())
              + f" B/A1={r['B_over_A1']:.4f}x A1/A0={r['A1_over_A0']:.4f}x "
              f"A/A={r['aa_floor_pct']:.2f}%")
    out = OUT / args.out
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"\nclock {clk_before} -> {clk_after} MHz; {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="geometry and guards only, no device")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--out", default="ab_qkv.json")
    args = ap.parse_args()
    return _plan_main(args) if args.plan else _device_main(args)


if __name__ == "__main__":
    sys.exit(main())
