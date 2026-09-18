#!/usr/bin/env python3
"""Shared-memory occupancy of Anthropic's shipped triangle-multiplication tiles, against ours.

Reads the constants out of their own source rather than restating them:

  * the tile table and the three SMEM formulas come from
    `common/opt_core/opt_core/kernels/trimul/native/pkg/v5/python/trimul_native/kernel.py`
    (`TILE_TABLE`, `k1_smem`, `k3_smem`, `k3w_smem`), whose docstrings say they equal the
    `K1Cfg::SMEM` / `K3Cfg::SMEM` / `K3WCfg::SMEM` constexprs in `csrc/tmn_kernels.cuh` and
    `csrc/tmn_k3_wide.cuh`;
  * the budget they compare against is `SMEM_LIMIT = 232448` (`csrc/tmn_kernels.cuh:40`),
    the sm_90 per-block opt-in dynamic shared memory, and 166912 on sm_80
    (`python/trimul_native/sm80_ops.py:113`).

Point the script at a checkout with `--kit <path to kernel.py>`.  It imports nothing from torch:
the head of that file up to `def lookup(` is pure arithmetic.

Our side is read from tt_bio so the two halves of the ratio have the same provenance rule.
"""
import argparse
import re
import sys
from pathlib import Path

SM90_SMEM = 232448    # csrc/tmn_kernels.cuh:40 -- "sm_90 max dynamic shared memory per block (227 KB)"
SM80_SMEM = 166912    # python/trimul_native/sm80_ops.py:113
DEFAULT_KIT = Path.home() / ".coworker/artifacts/anthro-kernel-read/kit/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/python/trimul_native/kernel.py"


def load_table(path: Path):
    """Exec the torch-free head of their kernel.py (constants + the SMEM formulas)."""
    src = path.read_text()
    head = src[: src.index("def lookup(")]
    head = "\n".join(l for l in head.splitlines() if not l.lstrip().startswith(("import ", "from ")))
    g = {"__name__": "tmn_kernel_head"}
    exec(compile(head, str(path), "exec"), g)   # noqa: S102 -- their own constants, read as data
    return g


def ours():
    """Our per-core L1, from tt_bio, with the line that states it."""
    root = Path(__file__).resolve().parents[2] / "tt_bio"
    out = {}
    for fname, name in (("sdpa_generic.py", "L1_PER_CORE"), ("sdpa_generic.py", "PROGRAM_RESERVE"),
                        ("softmax_generic.py", "L1_PER_CORE")):
        for i, line in enumerate((root / fname).read_text().splitlines(), 1):
            m = re.match(rf"{name}\s*=\s*(\d+)", line)
            if m:
                out.setdefault(f"{fname}:{name}", (int(m.group(1)), i))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    a = ap.parse_args()
    if not a.kit.is_file():
        print(f"no kit at {a.kit}", file=sys.stderr)
        return 2
    g = load_table(a.kit)
    T, k1s, k3s, k3ws, wide = g["TILE_TABLE"], g["k1_smem"], g["k3_smem"], g["k3w_smem"], g["k3_is_wide"]

    print(f"THEIRS -- shipped sm_90a trimul tiles vs SMEM_LIMIT {SM90_SMEM} B (227 KiB per CTA)\n")
    print(f"{'c_z':>4} {'c_h':>4} {'z':>2} {'K1 (BI,BJ,NSLOT,SKCH)':>22} {'tok':>4} {'K1 smem':>8} {'%':>6}  "
          f"{'K3 cfg':>20} {'K3 smem':>8} {'%':>6}")
    worst = 0
    for (arch, cz, ch, form), ent in sorted(T.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3])):
        k1, k3 = ent["k1"], ent["k3"]
        s1 = k1s(cz, ch, form == "f", *k1)
        mode = ent.get("k3_mode", "f" if form == "f" else "b")
        s3 = (k3ws(cz, ch, "f" if form == "f" else "b", k3[0], k3[1], k3[2], k3[3])
              if wide(k3) else k3s(cz, ch, mode, *k3))
        worst = max(worst, s1, s3)
        print(f"{cz:4d} {ch:4d} {form:>2} {str(k1):>22} {k1[0]*k1[1]:4d} {s1:8d} {100*s1/SM90_SMEM:5.1f}%  "
              f"{str(k3):>20} {s3:8d} {100*s3/SM90_SMEM:5.1f}%")
    print(f"\nhighest shipped occupancy: {worst} B = {100*worst/SM90_SMEM:.1f}% of the 227 KiB budget")

    print("\nby_n refinements (their tile IS a function of sequence length for this op):")
    for (arch, cz, ch, form), ent in sorted(T.items()):
        if ent.get("by_n"):
            print(f"  c_z={cz} c_h={ch} {form}: {ent['by_n']}")

    print(f"\nOURS -- per-Tensix-core L1")
    o = ours()
    for k, (v, line) in sorted(o.items()):
        print(f"  {k}:{line} = {v} B = {v/1024:.0f} KiB")
    l1 = o["sdpa_generic.py:L1_PER_CORE"][0]
    res = o["sdpa_generic.py:PROGRAM_RESERVE"][0]
    usable = l1 - res
    print(f"  usable per core = {l1} - {res} = {usable} B = {usable/1024:.1f} KiB")
    print(f"\nRATIO per worker: {usable} / {SM90_SMEM} = {usable/SM90_SMEM:.2f}x  (vs sm_80: {usable/SM80_SMEM:.2f}x)")

    print("\nWHY THEIR BOUNDARY SITS WHERE IT DOES -- the trimul contraction operand pair")
    print("  X[c,i,j] = sum_k a[c,i,k] b[c,j,k]: one output pair row needs all of a[:,i,:] and b[:,j,:].")
    print(f"  {'N':>5} {'c_h':>5} {'one a row':>11} {'a+b pair':>10} {'% their CTA':>12} {'% our core':>11}")
    for N in (256, 384, 512, 768, 1024):
        for chh in (128, 256):
            row = chh * N * 2
            print(f"  {N:5d} {chh:5d} {row:11d} {2*row:10d} {200.0*row/SM90_SMEM:11.1f}% {200.0*row/usable:10.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
