#!/usr/bin/env python3
"""Materialise one arm of the W6 fold-parity gate into tt_bio/tenstorrent.py.

W6 (`wk/perfwar-attention-block-fusion`) shipped four independent changes in one file. A single
pass/fail over all four tells whoever lands them nothing, so this script rebuilds the file from
the pristine base and applies exactly the change(s) an arm names:

  BASE    current main, untouched
  C1      full-length SDPA chunk in the 256<seq<=384 band
  C2      pair-tensor dim0/dim1 permute into L1 (+ the DRAM pins that keep the consumer out of L1)
  C3      ttnn.linear(core_grid=...) -> ttnn.experimental.minimal_matmul at 3 tri-attention sites
  C4      SwiGLU silu moved off the linear epilogue onto the multiply that already reads x_1
  CTRL    control arm: the band special-case DELETED, so N=320 takes the file's own capped
          default (256/256). Same equivalence class as C1, opposite end of it. The BASE-vs-CTRL
          distance is the reduction-order band the shipped file already produces at this size,
          and every other arm is judged against it.
  ALL     C1+C2+C3+C4 == W6's branch, byte-for-byte (asserted by --verify)
  ALLFIX  ALL plus the two one-line defect fixes (see D1/D2 below)

Two defects in W6's diff, both found by reading it, both one line:

  D1  `@lru_cache(maxsize=None)` on `_transpose_memory_config(t: ttnn.Tensor)`.
      `ttnn.Tensor.__hash__ is object.__hash__` and `__eq__` is a lambda returning a tensor, so
      the cache keys on object identity: every call is a miss (zero benefit) AND lru_cache holds
      a strong reference to every tensor ever passed for the life of the process. Three call
      sites x 48 blocks x 10 recycles is up to 1440 retained pair tensors per 298-aa fold.
      Fix: key on (shape, dtype) instead of the tensor.
  D2  the same edit moved that decorator OFF `_triangle_mul_program_config`, silently dropping
      memoisation from the trimul program-config builder. Fix: put it back.

Usage:
    python3 perf/w6_gate/arm.py --verify              # card-free self-check, no writes
    python3 perf/w6_gate/arm.py --arm ALLFIX          # write the arm into the worktree
    python3 perf/w6_gate/arm.py --arm BASE            # restore
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TARGET = REPO / "tt_bio" / "tenstorrent.py"
# PINNED, deliberately. This was "origin/main", and origin/main moved under the campaign: a
# sibling worktree fetched at 2026-08-09 16:37:31 UTC and fast-forwarded it from 96482b1e to
# 83499742 (W2's trimul merge, which carries "a one-block accumulator concat aliased its input,
# and it broke every real fold" -- a fix that changes fold output). Every arm measured in this
# gate was built from 96482b1e, so the base stays pinned to it or the arms stop being comparable
# with each other. Re-point it and re-run the whole sweep; do not mix bases.
BASE_REF = "96482b1e"
W6_REF = "origin/wk/perfwar-attention-block-fusion"


def _show(ref: str, path: str = "tt_bio/tenstorrent.py") -> str:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"git show {ref}:{path} failed: {out.stderr.strip()}")
    return out.stdout


# --------------------------------------------------------------------------------------------
# C1 — SDPA chunk config in the 256-384 band
# --------------------------------------------------------------------------------------------
SDPA_BASE = '''    # Microbench M7/M7b/M7c (Blackhole 13x10, tri-att shape batch=seq, h=8, d=32):
    # the 256-cap is optimal at >=512 (0.59x regression at 64) and 128 is best at
    # <=128, but in the 256<seq<=384 band (298-aa proteins pad to 320, 2 chunks of
    # 256+64 padded) q_chunk=k_chunk=64 is 2.45x faster. Chunking only changes the
    # online-softmax reduction order (measured PCC 0.9999 vs the 256 config).
    if 256 < q_len <= 384 and 256 < k_len <= 384:
        return _sdpa_program_config(q_chunk_size=64, k_chunk_size=64)
'''

SDPA_C1 = '''    # The 256-cap in _capped_sdpa_chunk_size is what makes the 256<seq<=384 band bad: it
    # splits 320 into 256+64, and the ragged pair is worse than any uniform split. The band
    # was therefore given q=k=64, but a full-length single chunk is better still. Measured on
    # a Blackhole P300c chip at the 298-aa tri-att shape (batch=320, h=8, seq=320, d=32),
    # median of 5 synced calls: 320/320 1.843 ms, 128/320 2.444, 64/64 2.895 (the config this
    # replaces), 128/128 3.007, 256/256 6.944, 32/32 4.679. So 1.57x over 64/64 and 3.8x over
    # the capped 256/256. Chunking only changes the online-softmax reduction order, same
    # equivalence class as the sizes already shipped here (max abs delta 0.00098 vs 32/32,
    # one bf16 ulp at these magnitudes).
    if 256 < q_len <= 384 and 256 < k_len <= 384 and q_len % 32 == 0 and k_len % 32 == 0:
        return _sdpa_program_config(q_chunk_size=q_len, k_chunk_size=k_len)
'''

SDPA_CTRL = '''    # CONTROL ARM (perf/w6_gate): the band special-case is deleted, so N=320 falls through to
    # _sdpa_program_config_for_lengths and takes the file's own capped default, 256/256. Not a
    # candidate for merge -- it exists to measure how far apart two configs this file already
    # emits can put a fold, which is the band every other arm is judged against.
'''

# --------------------------------------------------------------------------------------------
# C2 — pair-tensor dim0/dim1 permute into L1
# --------------------------------------------------------------------------------------------
TRANSPOSE_FN = '''def _transpose_memory_config(t: ttnn.Tensor) -> ttnn.MemoryConfig:
    """L1 for a pair-tensor dim0/dim1 transpose when it fits, else DRAM.

    ttnn's dim0/dim1 permute is a real element transpose, not a tile-block copy: tiling
    covers the last two dims, so swapping the untiled batch dim with the tile-row dim moves
    single rows between tiles. Its writes are therefore row-granular scatter, and DRAM
    punishes that. Measured on a Blackhole P300c chip at the 298-aa pair shape
    320x320x256 bf16 (median of 9 synced calls): 1.479 ms to DRAM = 70.9 GB/s, against
    0.281 ms for a plain ttnn.clone of the same tensor = 373.3 GB/s, so the transpose runs
    at 19% of the copy roof. Into L1 the same permute is 0.600 ms, 2.47x. ttnn.permute,
    ttnn.transpose(0,1) and the 4-D (0,2,1,3) form all land on the same kernel and the same
    1.48 ms, so this is the only lever short of a new kernel.

    A memory config cannot change a value: verified bit-identical (torch.equal) against the
    DRAM permute.
    """
    try:
        per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception:
        return ttnn.DRAM_MEMORY_CONFIG
    shape = [int(d) for d in t.shape]
    if len(shape) < 2:
        return ttnn.DRAM_MEMORY_CONFIG
    volume = 1
    for d in shape[:-2]:
        volume *= d
    volume *= ((shape[-2] + 31) // 32) * 32 * ((shape[-1] + 31) // 32) * 32
    elem = 4 if t.dtype == ttnn.float32 else 2
    # 2.5x headroom: the consumer still needs its circular buffers on every core.
    if 2.5 * volume * elem <= per_core * COMPUTE_GRID_MAIN[0] * COMPUTE_GRID_MAIN[1]:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


'''

# W6 reuses the decorator that sat on _triangle_mul_program_config, which is defect D2.
C2_ANCHOR_BASE = '''@lru_cache(maxsize=None)
def _triangle_mul_program_config('''
C2_ANCHOR_W6 = '@lru_cache(maxsize=None)\n' + TRANSPOSE_FN + 'def _triangle_mul_program_config('
C2_ANCHOR_FIX = ('@lru_cache(maxsize=None)\n' + TRANSPOSE_FN
                 + '@lru_cache(maxsize=None)\ndef _triangle_mul_program_config(')

PERM1_BASE = '''                if self.ending:
                    blk = ttnn.permute(blk, (1, 0, 2))
                return ttnn.layer_norm(
                    blk,
                    weight=self.layer_norm_weight,
                    bias=self.layer_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )
'''
PERM1_C2 = '''                if self.ending:
                    blk = ttnn.permute(blk, (1, 0, 2),
                                       memory_config=_transpose_memory_config(blk))
                return ttnn.layer_norm(
                    blk,
                    weight=self.layer_norm_weight,
                    bias=self.layer_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
'''

PERM2_BASE = '''            if self.ending:
                x = ttnn.permute(x, (1, 0, 2))  # THIS CAUSES CACHE -> RESHAPE PROBLEM
            x = ttnn.layer_norm(
                x,
                weight=self.layer_norm_weight,
                bias=self.layer_norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
'''
PERM2_C2 = '''            if self.ending:
                x = ttnn.permute(x, (1, 0, 2), memory_config=_transpose_memory_config(x))
            # Explicit DRAM: for the ending variant x is the L1 transpose result
            # (_transpose_memory_config) and ttnn would otherwise inherit L1 here and again
            # for the qkv projection, whose 157 MB does not fit.
            x = ttnn.layer_norm(
                x,
                weight=self.layer_norm_weight,
                bias=self.layer_norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
'''

PERM3_BASE = '''        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.reshape(x, (1, *x.shape))
'''
PERM3_C2 = '''        if self.ending:
            x = ttnn.permute(x, (1, 0, 2), memory_config=_transpose_memory_config(x))
        x = ttnn.reshape(x, (1, *x.shape))
'''

# --------------------------------------------------------------------------------------------
# C3 — ttnn.linear(core_grid=...) -> minimal_matmul
# --------------------------------------------------------------------------------------------
MM1_BASE = '''                b = ttnn.linear(
                    xc,
                    self.bias_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    dtype=ttnn.bfloat16,
                    core_grid=CORE_GRID_MAIN,
                )
'''
MM1_C3 = '''                b = ttnn.experimental.minimal_matmul(
                    input_tensor=xc,
                    weight_tensor=self.bias_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    dtype=ttnn.bfloat16,
                )
'''

MM2_BASE = '''            triangle_bias = ttnn.linear(
                x,
                self.bias_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                core_grid=CORE_GRID_MAIN,
            )
'''
MM2_C3 = '''            triangle_bias = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.bias_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
'''

MM3_BASE = '''            x_out = ttnn.linear(
                o_in,
                self.o_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=_dtype(),
                core_grid=CORE_GRID_MAIN,
            )
'''
MM3_C3 = '''            x_out = ttnn.experimental.minimal_matmul(
                input_tensor=o_in,
                weight_tensor=self.o_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=_dtype(),
            )
'''

# --------------------------------------------------------------------------------------------
# C4 — SwiGLU silu off the linear epilogue, onto the multiply
# --------------------------------------------------------------------------------------------
SILU1_BASE = '''            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                activation="silu",
                compute_kernel_config=self.compute_kernel_config,
'''
SILU1_C4 = '''            # The silu rides on the multiply below, which already reads x_1: as a linear
            # epilogue it costs 0.170 ms against a 0.100 ms matmul, 1.7x the matmul itself.
            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                compute_kernel_config=self.compute_kernel_config,
'''

SILU2_BASE = '''            x = ttnn.multiply_(x_1, x_2)
'''
SILU2_C4 = '''            x = ttnn.multiply_(
                x_1, x_2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
'''

# --------------------------------------------------------------------------------------------
# D1 fix — key the transpose memory config on (shape, dtype), not on the tensor
# --------------------------------------------------------------------------------------------
D1_BROKEN = '''@lru_cache(maxsize=None)
def _transpose_memory_config(t: ttnn.Tensor) -> ttnn.MemoryConfig:'''
D1_FIXED = '''def _transpose_memory_config(t: ttnn.Tensor) -> ttnn.MemoryConfig:'''

CHANGES = {
    "C1":   [(SDPA_BASE, SDPA_C1)],
    "CTRL": [(SDPA_BASE, SDPA_CTRL)],
    "C2":   [(C2_ANCHOR_BASE, C2_ANCHOR_W6), (PERM1_BASE, PERM1_C2),
             (PERM2_BASE, PERM2_C2), (PERM3_BASE, PERM3_C2)],
    "C3":   [(MM1_BASE, MM1_C3), (MM2_BASE, MM2_C3), (MM3_BASE, MM3_C3)],
    "C4":   [(SILU1_BASE, SILU1_C4), (SILU2_BASE, SILU2_C4)],
}
# C2 with both defects fixed: the helper keeps no cache, _triangle_mul_program_config keeps its.
CHANGES["C2FIX"] = [(C2_ANCHOR_BASE, C2_ANCHOR_FIX.replace(D1_BROKEN, D1_FIXED)),
                    (PERM1_BASE, PERM1_C2), (PERM2_BASE, PERM2_C2), (PERM3_BASE, PERM3_C2)]

ARMS = {
    "BASE":   [],
    "C1":     ["C1"],
    "C2":     ["C2"],
    "C3":     ["C3"],
    "C4":     ["C4"],
    "CTRL":   ["CTRL"],
    "ALL":    ["C1", "C2", "C3", "C4"],
    "ALLFIX": ["C1", "C2FIX", "C3", "C4"],
    "C2FIX":  ["C2FIX"],
}


def build(arm: str) -> str:
    src = _show(BASE_REF)
    for name in ARMS[arm]:
        for old, new in CHANGES[name]:
            n = src.count(old)
            if n != 1:
                sys.exit(f"arm {arm}: change {name}: anchor occurs {n} times, expected 1:\n"
                         f"--- anchor ---\n{old}")
            src = src.replace(old, new)
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--verify", action="store_true",
                    help="card-free: build every arm and assert ALL == W6's branch byte-for-byte")
    a = ap.parse_args()

    if a.verify:
        for arm in sorted(ARMS):
            build(arm)
            print(f"  {arm:<7} builds")
        got, want = build("ALL"), _show(W6_REF)
        if got != want:
            import difflib
            sys.stdout.writelines(difflib.unified_diff(
                want.splitlines(True), got.splitlines(True), "W6-branch", "arm-ALL", n=1))
            sys.exit("FAIL: arm ALL is not W6's branch")
        print("VERIFY OK: arm ALL == " + W6_REF + ":tt_bio/tenstorrent.py, byte-for-byte")
        fix = build("ALLFIX")
        assert "@lru_cache(maxsize=None)\ndef _transpose_memory_config" not in fix, \
            "D1 fix did not land: the helper is still cached on a tensor argument"
        assert "@lru_cache(maxsize=None)\ndef _triangle_mul_program_config" in fix, \
            "D2 fix did not land: the trimul program config is still uncached"
        # net decorator count is unchanged: one moves off the helper, one goes back on the trimul
        assert fix.count("@lru_cache") == _show(BASE_REF).count("@lru_cache"), \
            "ALLFIX changed the lru_cache count -- it should only move one and restore one"
        print("VERIFY OK: arm ALLFIX carries both defect fixes and nothing else")
        return 0

    if not a.arm:
        sys.exit("need --arm or --verify")
    TARGET.write_text(build(a.arm))
    print(f"wrote {TARGET} = arm {a.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
