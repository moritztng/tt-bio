#!/usr/bin/env python3
"""Replay the six unhinted batched-matmul sites of RFD3 and ESMFold2, and decide each one.

Handled sites (G3's audit, `state/perfwar-attention-site-audit.md` section 4):

    rfd3/model.py:491   trunk Pairformer  q @ k^T
    rfd3/model.py:501   trunk Pairformer  attn @ v
    rfd3/model.py:1331  RFD3AtomBlock     q @ k^T  (dense-bias branch)
    rfd3/model.py:1346  RFD3AtomBlock     q @ k^T  (sparse-bias branch -- same matmul)
    rfd3/model.py:1358  RFD3AtomBlock     attn @ v
    esmfold2.py:71      token AttentionPairBias  q @ k^T   (fp32)
    esmfold2.py:75      token AttentionPairBias  attn @ v  (fp32)

RFD3AtomBlock is instantiated twice with different head geometry (atom encoder/decoder
n_head=4 head_dim=32; token DiT n_head=16 head_dim=48), so 1331/1346/1358 each carry two
shape classes and get two verdicts.

Arms per shape, all on synthetic DRAM-interleaved operands of the exact padded shape:

    auto        exactly as the model calls it today
    grid        + core_grid= the active grid. ttnn routes this to
                create_matmul_program_config, whose batched-b branch returns
                MatmulMultiCoreReuseProgramConfig{per_core_M=Mt, per_core_N=Nt,
                in0_block_w=1} -- unless can_cbs_fit_in_l1(Mt,Nt,1) fails, in which case it
                silently returns the naive config and the arm is a no-op.
    reuse/pcm/w the same factory with per_core_M swept over the divisors of Mt that clear
                G1's correctness predicate (per_core_M == Mt, or block count <= cores), and
                in0_block_w in {1, 2}. Splitting M is what raises the engaged-core count
                above the batch size.

Every arm is compared to `auto` with torch.equal on the same operands: an arm that is not
bit-identical is reported as such and is not a candidate for shipping.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:perfwar-rfd3-esmfold2-sites \
        python3 -u perf/attn_sites/rfd3_esm_replay.py --out perf/attn_sites/rfd3_esm_qb2c0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

TILE = 32
BF16, FP32 = "bf16", "fp32"
TSIZE = {BF16: 2048, FP32: 4096}
TT = {BF16: ttnn.bfloat16, FP32: ttnn.float32}


def shapes(token_len: int, n_atom: int, d_batch: int):
    """The six sites at the 298 aa priority size.

    `token_len` is the design/sequence length in tokens, `n_atom` the atom count the
    RFD3 atom encoder/decoder runs at, `d_batch` the diffusion batch. Logical shapes are
    given; ttnn pads the last two dims up to a tile, which is what decides Mt/Kt/Nt.
    """
    L, A, D = token_len, n_atom, d_batch
    esm_L = L + (-L) % 32  # esmfold2 buckets the token dim to PAD_MULTIPLE=32
    return [
        # site, model, instance, batch, M, K, N, dtype
        ("rfd3:491", "rfd3", "trunk pairformer q@kT", 16, L, 24, L, BF16),
        ("rfd3:501", "rfd3", "trunk pairformer attn@v", 16, L, L, 24, BF16),
        ("rfd3:1331", "rfd3", "token DiT q@kT", D * 16, L, 48, esm_L, BF16),
        ("rfd3:1358", "rfd3", "token DiT attn@v", D * 16, L, esm_L, 48, BF16),
        ("rfd3:1331", "rfd3", "atom block q@kT", D * 4, A, 32, A + (-A) % 32, BF16),
        ("rfd3:1358", "rfd3", "atom block attn@v", D * 4, A, A + (-A) % 32, 32, BF16),
        ("esmfold2:71", "esmfold2", "token attn-pair-bias q@kT", 16, esm_L, 64, esm_L, FP32),
        ("esmfold2:75", "esmfold2", "token attn-pair-bias attn@v", 16, esm_L, esm_L, 64, FP32),
    ]


def tiles(x):
    return (x + TILE - 1) // TILE


def subblock(m, n, fp32_dest_acc=True):
    """Largest legal (h, w): h divides m, w divides n, h*w <= the DEST tile cap."""
    cap = 4 if fp32_dest_acc else 8
    best = (1, 1)
    for h in range(1, m + 1):
        for w in range(1, n + 1):
            if m % h == 0 and n % w == 0 and h * w <= cap and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def cb_bytes(pcm, pcn, bw, ta, tb, tinterm):
    """ttnn's own get_estimated_size_of_cbs, for DRAM-interleaved operands (no in2, no bias).

    matmul_utilities.cpp: in0 and in1 are double-buffered by MCAST_INPUT_BUFFERING_DEPTH=2,
    the output CB is sized with in0's tile size, and the interm CB is fp32 whenever
    fp32_dest_acc_en is on.
    """
    return pcm * bw * 2 * ta + pcn * bw * 2 * tb + pcm * pcn * ta + pcm * pcn * tinterm


def legal_reuse(mt, kt, nt, batch, cores, l1_budget, ta, tb, tinterm):
    """Every (per_core_M, in0_block_w) worth timing.

    G1's correctness predicate first: the reuse factory's dataflow kernels advance a whole
    batch stride per per-core loop iteration while the factory hands them a per-core block
    count, so the result is only right when per_core_M == Mt (one block is one batch
    element) or the total block count is <= the core count (each core gets exactly one
    block and the bad increment never runs). per_core_N is not a knob:
    matmul_device_operation.cpp asserts N == per_core_N for this factory.
    """
    out = []
    for pcm in sorted((d for d in range(1, mt + 1) if mt % d == 0), reverse=True):
        blocks = batch * (mt // pcm)
        if not (pcm == mt or blocks <= cores):
            continue
        for bw in ([1, 2] if kt % 2 == 0 and kt > 1 else [1]):
            if cb_bytes(pcm, nt, bw, ta, tb, tinterm) < l1_budget:
                out.append((pcm, bw, blocks))
    return out


def naive_config(mt, kt, nt, m_pad, n_pad, dev_x, dev_y, ta, tb, tinterm, l1_budget):
    """Predict which factory ttnn picks with no hint, and its in0_block_w.

    matmul_program_config.cpp v0.68.0, create_simple_matmul_program_config (:1051):
    per_core_M = per_core_N = the largest square factor in {16,8,4,2,1} whose CBs fit L1 at
    in0_block_w=2; num_blocks_{y,x} = ceil({Mt,Nt}/that); then

        both blocks == 1                  -> MatmulMultiCore (no reuse, batch not split)
        num_blocks_y == 1 or is_wide      -> mcast-1d, mcast_in0
        num_blocks_x == 1 or is_tall      -> mcast-1d, mcast_in1
        otherwise (all-DRAM-interleaved)  -> mcast-2d on the FULL 13x10 device grid

    in0_block_w is the only config field that can regroup the K accumulation (RFD3 p15), so
    it is the only field that decides whether a hinted arm can be bit-exact.
    """
    pcf = 1
    for cand in (16, 8, 4, 2):
        if cb_bytes(cand, cand, 2, ta, tb, tinterm) < l1_budget:
            pcf = cand
            break
    by, bx = -(-mt // pcf), -(-nt // pcf)
    ratio = max(m_pad, n_pad) // max(1, min(m_pad, n_pad))
    narrow = ratio > 8 or m_pad <= TILE or n_pad <= TILE  # all_dram -> the <= TILE clause is live
    wide = narrow and n_pad > m_pad
    tall = narrow and not wide
    if by == 1 and bx == 1 and not (wide or tall):
        return {"factory": "multicore", "in0_block_w": None, "cores": None, "pcf": pcf}
    grid = dev_x * dev_y
    if by == 1 or wide:
        # get_mcast_1d_config, mcast_in0: per_core_N = ceil(ceil(N/grid)/32), per_core_M = Mt
        pcn = max(1, -(-(-(-n_pad // grid)) // TILE))
        return {"factory": "mcast1d_in0", "in0_block_w": 2 if kt % 2 == 0 else 1,
                "cores": min(grid, -(-nt // pcn)), "pcf": pcf}
    if bx == 1 or tall:
        # get_mcast_1d_config, mcast_in1: per_core_M = ceil(ceil(M/grid)/32), per_core_N = Nt
        pcm = max(1, -(-(-(-m_pad // grid)) // TILE))
        return {"factory": "mcast1d_in1", "in0_block_w": 2 if kt % 2 == 0 else 1,
                "cores": min(grid, -(-mt // pcm)), "pcf": pcf}
    # mcast-2d on the full device grid: per_core_M = ceil(Mt/dev_y), per_core_N = ceil(Nt/dev_x)
    pcm, pcn = -(-mt // dev_y), -(-nt // dev_x)
    return {"factory": "mcast2d", "in0_block_w": kt // dev_x if kt % dev_x == 0 else 1,
            "cores": min(grid, -(-mt // pcm) * -(-nt // pcn)), "pcf": pcf}


def timed(fn, dev, warm=3, reps=20):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(reps)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    del outs
    return dt


def run(row, dev, ckc, roofs, cores, l1_budget):
    site, model, inst, batch, m, k, n, dt = row
    mt, kt, nt = tiles(m), tiles(k), tiles(n)
    m_pad, k_pad, n_pad = mt * TILE, kt * TILE, nt * TILE
    ta = tb = to = TSIZE[dt]
    tinterm = TSIZE[FP32]  # fp32_dest_acc_en=True on both models

    torch.manual_seed(0)
    ha = torch.randn(1, batch, m, k)
    hb = torch.randn(1, batch, k, n)
    a = ttnn.from_torch(ha, dtype=TT[dt], layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(hb, dtype=TT[dt], layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    flops = 2.0 * batch * mt * kt * nt * TILE ** 3
    byts = batch * (mt * kt * ta + kt * nt * tb + mt * nt * to)
    comp_roof = roofs["compute_bf16_TFLOPs"] if dt == BF16 else roofs["compute_fp32_TFLOPs"]
    rd, wr = roofs["dram_read_GBs"] * 1e9, roofs["dram_write_GBs"] * 1e9
    t_mem = max(batch * mt * kt * ta / rd, batch * kt * nt * tb / rd + batch * mt * nt * to / wr)
    t_comp = flops / (comp_roof * 1e12)

    pred = naive_config(mt, kt, nt, m_pad, n_pad, T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y,
                        ta, tb, tinterm, l1_budget)
    rec = {"site": site, "model": model, "instance": inst, "batch": batch,
           "m": m, "k": k, "n": n, "mt": mt, "kt": kt, "nt": nt, "dtype": dt,
           "flops": flops, "bytes": byts, "ai": flops / byts,
           "t_comp_us": t_comp * 1e6, "t_mem_us": t_mem * 1e6,
           "naive_predicted": pred, "arms": []}

    def mm(**kw):
        return lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, dtype=TT[dt], **kw)

    ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc, dtype=TT[dt]))
    t_auto = timed(mm(), dev)
    rec["arms"].append({"arm": "auto", "us": t_auto * 1e6, "exact": True, "cores": None})

    def add(name, fn, cores_pred):
        try:
            out = ttnn.to_torch(fn())
            exact = bool(torch.equal(out, ref))
            us = timed(fn, dev) * 1e6
        except Exception as exc:  # a rejected config raises before it can run
            rec["arms"].append({"arm": name, "error": str(exc)[:200]})
            return
        rec["arms"].append({"arm": name, "us": us, "exact": exact, "cores": cores_pred,
                            "speedup": t_auto * 1e6 / us})

    grid_fits = cb_bytes(mt, nt, 1, ta, tb, tinterm) < l1_budget
    add("grid" + ("" if grid_fits else "(NO-OP: cbs do not fit, ttnn falls back)"),
        mm(core_grid=T.CORE_GRID_MAIN), batch if grid_fits else None)

    for pcm, bw, blocks in legal_reuse(mt, kt, nt, batch, cores, l1_budget, ta, tb, tinterm):
        h, w = subblock(pcm, nt)
        cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y),
            in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
            per_core_M=pcm, per_core_N=nt)
        add(f"reuse/pcm={pcm}/bw={bw}", mm(program_config=cfg), min(blocks, cores))

    ttnn.deallocate(a)
    ttnn.deallocate(b)
    best = max((x for x in rec["arms"] if "us" in x), key=lambda x: t_auto * 1e6 / x["us"])
    rec["best"] = best["arm"]
    rec["best_speedup"] = t_auto * 1e6 / best["us"]
    rec["best_exact"] = best["exact"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roofs", default=None, help="roofs JSON; omit to skip the % columns")
    ap.add_argument("--token-len", type=int, default=298)
    ap.add_argument("--n-atom", type=int, default=1192)
    ap.add_argument("--d-batch", type=int, default=1)
    args = ap.parse_args()

    roofs = {"compute_bf16_TFLOPs": float("nan"), "compute_fp32_TFLOPs": float("nan"),
             "dram_read_GBs": float("nan"), "dram_write_GBs": float("nan")}
    if args.roofs:
        roofs.update(json.load(open(args.roofs)))

    dev = get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    cores = T.CORE_GRID_MAIN.x * T.CORE_GRID_MAIN.y
    # ttnn's budget is get_max_l1_space() = the lowest occupied compute L1 address (or the full
    # per-core L1 when nothing is resident, which is our case -- every operand is DRAM-interleaved)
    # minus the L1 allocator base. MeshDevice does not expose either, so take Blackhole's per-core
    # L1 directly. Only used to predict which branch ttnn takes; the timings decide.
    l1_budget = 1464 * 1024
    print(f"# grid {T.CORE_GRID_MAIN.x}x{T.CORE_GRID_MAIN.y} = {cores} cores, "
          f"L1/core {l1_budget} B (ttnn's budget is this minus the allocator base)")

    rows = []
    for row in shapes(args.token_len, args.n_atom, args.d_batch):
        rec = run(row, dev, ckc, roofs, cores, l1_budget)
        rows.append(rec)
        print(f"{rec['site']:14s} {rec['instance']:32s} B={rec['batch']:5d} "
              f"Mt/Kt/Nt={rec['mt']}/{rec['kt']}/{rec['nt']} "
              f"naive={rec['naive_predicted']['factory']}(bw={rec['naive_predicted']['in0_block_w']}) "
              f"auto={rec['arms'][0]['us']:8.1f} us  best={rec['best']} "
              f"x{rec['best_speedup']:.2f} exact={rec['best_exact']}")
        for arm in rec["arms"][1:]:
            if "us" in arm:
                print(f"    {arm['arm']:34s} {arm['us']:8.1f} us  x{arm['speedup']:.2f}  "
                      f"exact={arm['exact']}  cores~{arm['cores']}")
            else:
                print(f"    {arm['arm']:34s} REJECTED {arm['error'][:90]}")
        Path(args.out).write_text(json.dumps({"roofs": roofs, "cores": cores,
                                              "l1_size_per_core": l1_budget,
                                              "token_len": args.token_len,
                                              "n_atom": args.n_atom,
                                              "d_batch": args.d_batch,
                                              "rows": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
