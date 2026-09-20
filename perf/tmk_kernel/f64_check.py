#!/usr/bin/env python3
"""Float64 verification of the trimul contraction path, and of the reference itself.

K-E: every result checked against a float64 reference that is itself validated against float64
central finite differences, never against another approximation. This kernel family has three
recorded silent correctness failures, including a `perfwar-trimul-kernel` commit that passed every
module-level and block-level benchmark while breaking every real 298 aa fold.

What is checked, in order:

1. **The reference is validated before it is used as one.** The forward is
   `out[c,i,j] = sum_k (p_a[i,k,c]*sigmoid(g_a[i,k,c])) * (p_b[j,k,c]*sigmoid(g_b[j,k,c]))`.
   Its analytic gradient of `J = sum(out * w)` with respect to `blk` is compared against float64
   CENTRAL finite differences at randomly chosen entries. A reference that passes this is
   implementing the function we think it is; one that does not cannot verify anything.
2. **The shipped gated move** against the reference s gated-and-permuted operand.
3. **The ungated `reblock_permute`** against `ttnn.permute`, which is bit-exact by construction
   (a pure index reordering) and so is checked with `torch.equal`, not a tolerance.
4. **The full composite** against the float64 contraction.

Small shapes on purpose: a float64 host contraction at 512 aa is 17.2 GFLOP in numpy and the point
here is correctness, not throughput. The permutation and the gate are shape-independent.
"""
from __future__ import annotations

import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch, ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE.parent))
import tt_bio  # noqa: E402
assert str(REPO) in tt_bio.__file__
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import reblock_permute as RB  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
S, C = 128, 32
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_f64(blk, C):
    """blk is [S,S,4C] float64 in the role order the engine uses. Returns out [C,S,S] float64."""
    roles = tt.gp_roles()
    q = {r: blk[:, :, i * C:(i + 1) * C] for i, r in enumerate(roles)}
    a = np.transpose(q["p_a"] * sig(q["g_a"]), (2, 0, 1))      # [C,S,S] indexed (c,i,k)
    b = np.transpose(q["p_b"] * sig(q["g_b"]), (2, 0, 1))      # [C,S,S] indexed (c,j,k)
    return np.einsum("cik,cjk->cij", a, b)


def validate_reference(blk, C, w, n_probe=12, h=1e-5):
    """Central finite differences against the analytic gradient of J = sum(out*w)."""
    roles = tt.gp_roles()
    q = {r: blk[:, :, i * C:(i + 1) * C] for i, r in enumerate(roles)}
    sa, sb = sig(q["g_a"]), sig(q["g_b"])
    a = np.transpose(q["p_a"] * sa, (2, 0, 1))
    b = np.transpose(q["p_b"] * sb, (2, 0, 1))
    # dJ/da[c,i,k] = sum_j w[c,i,j] b[c,j,k];  dJ/db[c,j,k] = sum_i w[c,i,j] a[c,i,k]
    ga = np.einsum("cij,cjk->cik", w, b)
    gb = np.einsum("cij,cik->cjk", w, a)
    grad = {}
    grad["p_a"] = np.transpose(ga, (1, 2, 0)) * sa
    grad["g_a"] = np.transpose(ga, (1, 2, 0)) * q["p_a"] * sa * (1.0 - sa)
    grad["p_b"] = np.transpose(gb, (1, 2, 0)) * sb
    grad["g_b"] = np.transpose(gb, (1, 2, 0)) * q["p_b"] * sb * (1.0 - sb)
    rng = np.random.default_rng(7)
    worst, rows = 0.0, []
    for _ in range(n_probe):
        r = roles[rng.integers(len(roles))]
        i, k, c = (int(rng.integers(blk.shape[0])), int(rng.integers(blk.shape[1])),
                   int(rng.integers(C)))
        col = roles.index(r) * C + c
        up, dn = blk.copy(), blk.copy()
        up[i, k, col] += h; dn[i, k, col] -= h
        fd = (np.sum(forward_f64(up, C) * w) - np.sum(forward_f64(dn, C) * w)) / (2 * h)
        an = grad[r][i, k, c]
        rel = abs(fd - an) / max(abs(an), abs(fd), 1e-12)
        worst = max(worst, rel)
        rows.append({"role": r, "idx": [i, k, c], "analytic": an, "central_fd": fd,
                     "rel": rel})
    return worst, rows


def main():
    dev = tt.get_device(); torch.manual_seed(0)
    rng = np.random.default_rng(0)
    blk64 = rng.standard_normal((S, S, 4 * C)) * 0.05
    w = rng.standard_normal((C, S, S))
    out = {"host": TAG, "S": S, "C": C}

    worst, rows = validate_reference(blk64, C, w)
    out["reference_validation"] = {"worst_rel_vs_central_fd": worst, "n_probe": len(rows),
                                   "h": 1e-5, "probes": rows[:4]}
    print(f"  REFERENCE vs central finite differences: worst rel {worst:.3e} over {len(rows)} probes",
          flush=True)
    if worst > 1e-6:
        print("  REFERENCE FAILED ITS OWN VALIDATION -- nothing below is admissible", flush=True)

    ref = forward_f64(blk64, C)

    blk_t = torch.from_numpy(blk64).to(torch.bfloat16).reshape(1, S, S, 4 * C)
    blk = ttnn.from_torch(blk_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a_out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, S, S]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    b_out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, S, S]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    RB.reblock_permute_gated(blk, tt.gp_off("p_a", C), tt.gp_off("g_a", C), C, out=a_out)
    RB.reblock_permute_gated(blk, tt.gp_off("p_b", C), tt.gp_off("g_b", C), C, out=b_out)

    roles = tt.gp_roles()
    q = {r: blk64[:, :, i * C:(i + 1) * C] for i, r in enumerate(roles)}
    a_ref = np.transpose(q["p_a"] * sig(q["g_a"]), (2, 0, 1))
    a_dev = ttnn.to_torch(a_out).float().numpy()[0]
    out["gated_move_a"] = {"max_abs": float(np.abs(a_dev - a_ref).max()),
                           "rel_l2": float(np.linalg.norm(a_dev - a_ref) / np.linalg.norm(a_ref))}
    print("  GATED MOVE a vs float64: max_abs %.3e  rel_l2 %.3e" % (out["gated_move_a"]["max_abs"], out["gated_move_a"]["rel_l2"]), flush=True)

    # the ungated tile-granular move is a pure index reordering -> torch.equal, not a tolerance
    pair = ttnn.from_torch(torch.from_numpy(q["p_a"].copy()).to(torch.bfloat16)
                           .reshape(1, S, S, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rp = ttnn.to_torch(RB.reblock_permute(pair, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    pm = ttnn.to_torch(ttnn.permute(pair, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG))
    out["reblock_permute_bitexact_vs_ttnn_permute"] = bool(torch.equal(rp, pm))
    print("  REBLOCK_PERMUTE torch.equal vs ttnn.permute: %s" % out["reblock_permute_bitexact_vs_ttnn_permute"], flush=True)

    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 10), in0_block_w=S // 32, out_subblock_h=1,
        out_subblock_w=1, out_block_h=max(1, -(-(S // 32) // 10)),
        out_block_w=max(1, -(-(S // 32) // 11)), per_core_M=max(1, -(-(S // 32) // 10)),
        per_core_N=max(1, -(-(S // 32) // 11)), transpose_mcast=False, fused_activation=None,
        fuse_batch=False)
    dev_out = ttnn.to_torch(ttnn.matmul(a_out, b_out, compute_kernel_config=KC,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc,
                                        dtype=ttnn.bfloat16, transpose_b=True)).float().numpy()[0]
    out["composite"] = {"max_abs": float(np.abs(dev_out - ref).max()),
                        "rel_l2": float(np.linalg.norm(dev_out - ref) / np.linalg.norm(ref)),
                        "ref_absmax": float(np.abs(ref).max())}
    print("  COMPOSITE vs float64: max_abs %.3e  rel_l2 %.3e  (|ref|max %.3e)" % (out["composite"]["max_abs"], out["composite"]["rel_l2"], out["composite"]["ref_absmax"]), flush=True)
    (HERE / f"f64_check_{TAG}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
