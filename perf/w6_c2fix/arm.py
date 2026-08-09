#!/usr/bin/env python3
"""Materialise BASE or C2FIX into tt_bio/tenstorrent.py, rebased onto the current main tip.

W11 (`perf/w6_gate/arm.py`, branch `wk/perfwar-w6-fold-parity-gate`) split W6's four bundled
changes and found exactly one that survives a fold gate: C2, the pair-tensor dim0/dim1 permute
into L1, and only with the D2 defect fixed. This script is that arm and nothing else, rebuilt
from `main` at BASE_REF instead of W11's pinned 96482b1e, because main has since taken W2's
trimul merge and L1's pair-track program config and the baseline moved with them.

  BASE    current main, untouched
  C2FIX   the permute-into-L1 helper + its three call sites, with:
          - NO @lru_cache on `_transpose_memory_config` (W11 defect D1). ttnn.Tensor hashes by
            object identity, so every call would miss AND the cache would pin every pair tensor
            it ever saw for the life of the process.
          - @lru_cache(maxsize=None) KEPT on `_triangle_mul_program_config` (W11 defect D2).
            W6 inserted the helper directly under that decorator, silently stealing it, and
            `_configure_active_compute_grid` calls `.cache_clear()` on it at every device open.
            Any grid other than the module default takes that branch, so W6's branch cannot
            open a Blackhole at all.

Not included, deliberately: C1 (SDPA band, a measured 0.87x regression -- its guard tests the
ttnn logical length 298, not the padded 320 it was benchmarked at, so it deletes the shipped
band config instead of replacing it), C3 (minimal_matmul) and C4 (SwiGLU silu), both held.

    python3 perf/w6_c2fix/arm.py --verify     # card-free self-check, no writes
    python3 perf/w6_c2fix/arm.py --arm C2FIX  # write the arm into the worktree
    python3 perf/w6_c2fix/arm.py --arm BASE   # restore
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TARGET = REPO / "tt_bio" / "tenstorrent.py"
# The main tip this leg re-measured against. Pinned so BASE and C2FIX stay comparable if main
# moves under the run; re-point it and re-run the whole sweep, do not mix bases.
BASE_REF = "a8ccd374"
W6_REF = "origin/wk/perfwar-attention-block-fusion"


def _show(ref: str, path: str = "tt_bio/tenstorrent.py") -> str:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"git show {ref}:{path} failed: {out.stderr.strip()}")
    return out.stdout


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

    Not cached: ttnn.Tensor hashes by object identity, so an lru_cache here would miss on
    every call and pin every tensor it ever saw for the life of the process.
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

# The helper goes ABOVE the decorator, so _triangle_mul_program_config keeps its own cache.
ANCHOR_BASE = '''@lru_cache(maxsize=None)
def _triangle_mul_program_config('''
ANCHOR_C2FIX = TRANSPOSE_FN + ANCHOR_BASE

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
PERM1_C2FIX = '''                if self.ending:
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
PERM2_C2FIX = '''            if self.ending:
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
PERM3_C2FIX = '''        if self.ending:
            x = ttnn.permute(x, (1, 0, 2), memory_config=_transpose_memory_config(x))
        x = ttnn.reshape(x, (1, *x.shape))
'''

CHANGES = [(ANCHOR_BASE, ANCHOR_C2FIX), (PERM1_BASE, PERM1_C2FIX),
           (PERM2_BASE, PERM2_C2FIX), (PERM3_BASE, PERM3_C2FIX)]

ARMS = {"BASE": False, "C2FIX": True}


def build(arm: str) -> str:
    src = _show(BASE_REF)
    if not ARMS[arm]:
        return src
    for old, new in CHANGES:
        n = src.count(old)
        if n != 1:
            sys.exit(f"arm {arm}: anchor occurs {n} times, expected 1:\n--- anchor ---\n{old}")
        src = src.replace(old, new)
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--verify", action="store_true", help="card-free structural self-check")
    a = ap.parse_args()

    if a.verify:
        base, fix = build("BASE"), build("C2FIX")
        assert base == _show(BASE_REF), "BASE is not main@" + BASE_REF
        assert "@lru_cache(maxsize=None)\ndef _transpose_memory_config" not in fix, \
            "D1: the helper must not be cached on a tensor argument"
        assert "@lru_cache(maxsize=None)\ndef _triangle_mul_program_config" in fix, \
            "D2: the trimul program config must keep its cache"
        assert fix.count("@lru_cache") == base.count("@lru_cache"), \
            "the lru_cache count moved -- C2FIX adds and removes none"
        assert fix.count("_transpose_memory_config(") == 4, \
            "expected one definition and three call sites"
        # The helper body must be W6's, character for character, minus the docstring note.
        w6 = _show(W6_REF)
        w6_fn = w6[w6.index("def _transpose_memory_config"):w6.index("def _triangle_mul_program_config")]
        fix_fn = fix[fix.index("def _transpose_memory_config"):fix.index("@lru_cache(maxsize=None)\ndef _triangle_mul_program_config")]
        w6_code = [l for l in w6_fn.splitlines() if l.strip() and not l.strip().startswith("#")]
        fix_code = [l for l in fix_fn.splitlines() if l.strip() and not l.strip().startswith("#")]
        w6_code = w6_code[w6_code.index('    """') + 1:] if '    """' in w6_code else w6_code
        fix_code = fix_code[fix_code.index('    """') + 1:] if '    """' in fix_code else fix_code
        assert w6_code[-len(w6_code) + w6_code.index("    try:"):] == \
            fix_code[fix_code.index("    try:"):], "helper body diverged from W6's"
        # Nothing outside the four hunks moved: undo them and the file is BASE again. This is
        # the check that keeps C1/C3/C4 out, and it does not depend on marker strings -- main
        # already uses minimal_matmul and UnaryOpType.SILU in unrelated places.
        undo = fix
        for old, new in reversed(CHANGES):
            undo = undo.replace(new, old)
        assert undo == base, "C2FIX touches something outside its four hunks"
        # C1 specifically: W6's guard must be absent and the shipped band config intact.
        assert "if 256 < q_len <= 384 and 256 < k_len <= 384:" in fix, "C1 leaked in"
        print(f"VERIFY OK: BASE == main@{BASE_REF}; C2FIX = W6's C2 helper + 3 call sites, "
              f"D1 dropped, D2 kept, no C1/C3/C4")
        return 0

    if not a.arm:
        sys.exit("need --arm or --verify")
    TARGET.write_text(build(a.arm))
    print(f"wrote {TARGET} = arm {a.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
