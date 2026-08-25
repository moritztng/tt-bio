#!/usr/bin/env python3
"""p128 -- region 2's step 0: three answers about `fc1` in one device session.

`state/rfd3-fusion-programme.md` §13.6. Region 2's route branches on facts nobody has recorded:

**(a) Where does the silu round?** If `ttnn.linear(x, w, activation="silu")` applies its silu in the
matmul kernel, on the fp32 accumulator before the pack, then an in-kernel activation can reproduce it
and §6's `generic_op` screen is the right screen. If instead ttnn runs the matmul and a separate
eltwise silu over the packed bf16 output, the shipped value is rounded twice and NO in-kernel
activation can ever be bit-exact against it -- but then splitting the call is not a precision change,
and region 2 needs no kernel at all. §13.4 argues the second case from the silu's measured cost
(114 % / 109 % of a separate pass's roof) and this settles it by comparison.

**(b) Which half of P3.10.3's conjunction failed?** `_calibrate_linear` keeps a candidate only if it
is bitwise equal to the default AND at least `_TUNE_MIN_GAIN` faster, and it logs only the winner.
P3.10.3 recorded DEFAULT for both activated `fc1` shapes, which cannot distinguish "no candidate was
bit-exact" from "the bit-exact ones were not 1.05x". Region 2's residency half needs a pinned
bit-exact config at ANY speed, so that distinction decides whether the half is alive.

**(c) What does `fc1` cost at the shapes the model calls TODAY?** p46 measured
`(1,685,704,128)` and `_PAIR_TRANSITION_L1` has since made the shipped call `(1,64,704,128)` eleven
times over, so every number in §13.2 is historical (`perf-page-cell-is-historical-not-live-baseline`).

Reuses `_mm_candidates`, `_mm_maxabs`, `_mm_time` and `_mm_random_like` from `tt_bio.rfd3.model`
rather than re-implementing any of them; the only new code is a field-for-field rebuild of each
candidate with `fused_activation` set, because the program config is immutable once constructed.

**Card control first, as the protocol requires.** pc card 0 miscomputes some ttnn matmuls at a low
location-keyed rate (`pc-card0-512aa-fold-nondeterminism`), so an exactness verdict off a
miscomputing run means nothing. Every key runs the shipped activated call three times and the three
results are compared to each other before any verdict below is allowed to stand.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import ttnn

from tt_bio.rfd3 import model as M
from tt_bio.tenstorrent import get_device

C_Z = 128
# The live `fc1` keys. R3 is the largest fixture pc can host (§10.1); R4's chunk shapes are included
# because they are the census fixture's and the matmul does not need the fold to exist.
KEYS = [
    ("R3 body",  (1, 64, 544, C_Z), 512), ("R3 body",  (1, 64, 544, C_Z), 256),
    ("R3 tail",  (1,  2, 544, C_Z), 512), ("R3 tail",  (1,  2, 544, C_Z), 256),
    ("R4 body",  (1, 64, 704, C_Z), 512), ("R4 body",  (1, 64, 704, C_Z), 256),
    ("R4 tail",  (1, 45, 704, C_Z), 512), ("R4 tail",  (1, 45, 704, C_Z), 256),
]


def with_silu(pc):
    """`_mm_candidates`' config, rebuilt with the fused activation. The config is immutable."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
        in0_block_w=pc.in0_block_w, out_subblock_h=pc.out_subblock_h,
        out_subblock_w=pc.out_subblock_w, out_block_h=pc.out_block_h,
        out_block_w=pc.out_block_w, per_core_M=pc.per_core_M, per_core_N=pc.per_core_N,
        fuse_batch=pc.fuse_batch, mcast_in0=pc.mcast_in0,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU))


def one_key(dev, rung, xshape, hidden, ckc):
    x = ttnn.from_torch(torch.randn(xshape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn((C_Z, hidden), dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    kw = dict(compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    fused = lambda: ttnn.linear(x, w, activation="silu", core_grid=None, **kw)
    plain = lambda: ttnn.linear(x, w, core_grid=None, **kw)

    # --- card control: three shipped calls, compared to each other ------------------------
    reps = [fused() for _ in range(3)]
    control = [M._mm_maxabs(reps[0], r) for r in reps[1:]]
    ref = reps[0]
    for r in reps[1:]:
        ttnn.deallocate(r)
    row = {"rung": rung, "x": list(xshape), "hidden": hidden,
           "control_maxabs": control, "control_unanimous": all(c == 0.0 for c in control)}

    # --- (a) does splitting the activation reproduce the shipped value? -------------------
    mm = plain()
    split = ttnn.silu(mm)
    row["split_maxabs"] = M._mm_maxabs(split, ref)
    row["split_bit_exact"] = row["split_maxabs"] == 0.0
    row["split_torch_equal"] = bool(torch.equal(ttnn.to_torch(split), ttnn.to_torch(ref)))
    ttnn.deallocate(split)
    ttnn.deallocate(mm)

    # --- (c) the live cost of the shipped call, and of its unactivated sibling ------------
    row["ms_activated_default"] = M._mm_time(fused) * 1e3
    row["ms_plain_default"] = M._mm_time(plain) * 1e3

    # --- (b) the full candidate table for the ACTIVATED call ------------------------------
    rx, rw = M._mm_random_like(x, 0), M._mm_random_like(w, 1)
    rref = ttnn.linear(rx, rw, activation="silu", core_grid=None, **kw)
    cands = []
    for base in M._mm_candidates(x, w, dev.compute_with_storage_grid_size()):
        pc = with_silu(base)
        rec = {"in0_block_w": pc.in0_block_w, "out_block_w": pc.out_block_w,
               "out_subblock_w": pc.out_subblock_w, "per_core_M": pc.per_core_M,
               "per_core_N": pc.per_core_N}
        try:
            rec["maxabs_random"] = M._mm_maxabs(
                ttnn.linear(rx, rw, program_config=pc, **kw), rref)
            rec["maxabs_live"] = M._mm_maxabs(ttnn.linear(x, w, program_config=pc, **kw), ref)
            rec["bit_exact"] = rec["maxabs_random"] == 0.0 and rec["maxabs_live"] == 0.0
            rec["ms"] = M._mm_time(lambda: ttnn.linear(x, w, program_config=pc, **kw)) * 1e3
        except Exception as e:                      # illegal L1 / subblock combos are expected
            rec["error"] = type(e).__name__
        cands.append(rec)
    row["candidates"] = cands

    # --- (d) the split route, priced at the live shape ------------------------------------
    # If (a) says the split is bit-exact, the plain matmul becomes tunable -- which is the whole
    # blockage §6 named -- and its output becomes eligible for L1, which the activated call could
    # never be (`_tuned_linear`: an L1 output re-blocks K under the heuristic, so residency is only
    # safe behind a PINNED config). Both halves are priced here against the shipped activated call.
    pref = plain()
    rpref = ttnn.linear(rx, rw, core_grid=None, **kw)
    pcands, best, best_ms = [], None, None
    for base in M._mm_candidates(x, w, dev.compute_with_storage_grid_size()):
        rec = {"in0_block_w": base.in0_block_w, "out_block_w": base.out_block_w,
               "out_subblock_w": base.out_subblock_w, "per_core_M": base.per_core_M}
        try:
            rec["maxabs_random"] = M._mm_maxabs(
                ttnn.linear(rx, rw, program_config=base, **kw), rpref)
            rec["maxabs_live"] = M._mm_maxabs(ttnn.linear(x, w, program_config=base, **kw), pref)
            rec["bit_exact"] = rec["maxabs_random"] == 0.0 and rec["maxabs_live"] == 0.0
            rec["ms"] = M._mm_time(lambda: ttnn.linear(x, w, program_config=base, **kw)) * 1e3
            if rec["bit_exact"] and (best_ms is None or rec["ms"] < best_ms):
                best, best_ms = base, rec["ms"]
        except Exception as e:
            rec["error"] = type(e).__name__
        pcands.append(rec)
    row["plain_candidates"] = pcands
    row["n_plain_bit_exact"] = sum(1 for c in pcands if c.get("bit_exact"))
    row["ms_plain_pinned"] = best_ms
    row["plain_pinned_gain"] = row["ms_plain_default"] / best_ms if best_ms else None
    row["plain_pinned_cfg"] = ({"in0_block_w": best.in0_block_w, "out_block_w": best.out_block_w,
                                "out_subblock_w": best.out_subblock_w,
                                "per_core_M": best.per_core_M} if best else None)

    # the silu leg, and then the same chain with both intermediates resident in L1
    row["ms_silu_dram"] = M._mm_time(lambda: ttnn.silu(pref)) * 1e3
    L1 = ttnn.L1_MEMORY_CONFIG
    if best is not None:
        mm_l1 = ttnn.linear(x, w, program_config=best, memory_config=L1, **kw)
        row["pinned_l1_maxabs"] = M._mm_maxabs(mm_l1, pref)
        row["ms_plain_pinned_l1"] = M._mm_time(
            lambda: ttnn.linear(x, w, program_config=best, memory_config=L1, **kw)) * 1e3
        s_l1 = ttnn.silu(mm_l1, memory_config=L1)
        row["silu_l1_maxabs"] = M._mm_maxabs(s_l1, ref)
        row["ms_silu_l1"] = M._mm_time(lambda: ttnn.silu(mm_l1, memory_config=L1)) * 1e3
        row["ms_split_chain_l1"] = row["ms_plain_pinned_l1"] + row["ms_silu_l1"]
        row["chain_gain_vs_shipped"] = row["ms_activated_default"] / row["ms_split_chain_l1"]
        row["chain_delta_ms"] = row["ms_activated_default"] - row["ms_split_chain_l1"]
        ttnn.deallocate(s_l1)
        ttnn.deallocate(mm_l1)
    ttnn.deallocate(rpref)
    ttnn.deallocate(pref)
    exact = [c for c in cands if c.get("bit_exact")]
    row["n_candidates"] = len(cands)
    row["n_bit_exact"] = len(exact)
    row["best_bit_exact_ms"] = min((c["ms"] for c in exact), default=None)
    row["best_bit_exact_gain"] = (row["ms_activated_default"] / row["best_bit_exact_ms"]
                                 if exact else None)
    row["bit_exact_in0_block_w"] = sorted({c["in0_block_w"] for c in exact})
    for t in (rx, rw, rref, ref, x, w):
        ttnn.deallocate(t)
    return row


def main():
    dev = get_device()
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = [one_key(dev, r, s, h, ckc) for r, s, h in KEYS]
    out = {"host": "pc", "card": 0, "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "0.68.0",
           "grid": [dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y],
           "rows": rows}
    dest = pathlib.Path("perf/p128")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fc1_config_census.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\ngrid {out['grid']}\n")
    print(f"{'rung':9} {'x':22} {'hid':>4} {'ctl':>4} {'split':>6} "
          f"{'act ms':>7} {'plain ms':>9} {'act-cfg exact':>14} "
          f"{'plain exact':>12} {'pinned ms':>10} {'gain':>6} {'+L1 ms':>7} {'silu ms':>8} "
          f"{'chain ms':>9} {'vs shipped':>11}")
    for r in rows:
        print(f"{r['rung']:9} {str(tuple(r['x'])):22} {r['hidden']:4} "
              f"{'ok' if r['control_unanimous'] else 'BAD':>4} "
              f"{str(r['split_bit_exact'] and r['split_torch_equal']):>6} "
              f"{r['ms_activated_default']:7.3f} {r['ms_plain_default']:9.3f} "
              f"{('%d/%d' % (r['n_bit_exact'], r['n_candidates'])):>14} "
              f"{('%d/%d' % (r['n_plain_bit_exact'], len(r['plain_candidates']))):>12} "
              f"{(('%10.3f' % r['ms_plain_pinned']) if r['ms_plain_pinned'] else '%10s' % '-')} "
              f"{(('%5.2fx' % r['plain_pinned_gain']) if r['plain_pinned_gain'] else '%6s' % '-')} "
              f"{(('%7.3f' % r['ms_plain_pinned_l1']) if 'ms_plain_pinned_l1' in r else '%7s' % '-')} "
              f"{(('%8.3f' % r['ms_silu_l1']) if 'ms_silu_l1' in r else '%8s' % '-')} "
              f"{(('%9.3f' % r['ms_split_chain_l1']) if 'ms_split_chain_l1' in r else '%9s' % '-')} "
              f"{(('%10.2fx' % r['chain_gain_vs_shipped']) if 'chain_gain_vs_shipped' in r else '%11s' % '-')}")
    print("\nwrote perf/p128/fc1_config_census.json")


if __name__ == "__main__":
    main()
