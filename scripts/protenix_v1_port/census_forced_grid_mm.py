#!/usr/bin/env python3
"""Census every ttnn matmul a fold actually issues, and flag the ones in the deadlock regime.

The >=512-token hang was a deadlock in ttnn's MULTICAST matmul program, which passing
``core_grid`` selects, and it fired on a projection whose OUTPUT was only 4 tiles wide
(tt_bio/protenix.py's pair-conditioning; see perf/pxv1/hang_watcher_capture.txt). Five call
sites in that one function shared the exposed shape and only the one caught hanging had been
fixed -- so the obvious next question is whether any OTHER call site in the codebase is in the
same regime. ``tt_bio`` forces ``core_grid=CORE_GRID_MAIN`` at 82 places, 61 of them in
tenstorrent.py, which every model shares.

Static grep cannot answer it: most of those sites are inside generic helpers whose widths come
from the caller. So this wraps ``ttnn.linear``/``ttnn.matmul`` before tt_bio is imported, runs a
real fold in-process, and reports every distinct (M, K, N, core_grid) actually issued.

EXPOSED = a core_grid was passed AND the output is <= PAIRCOND_MM_NARROW_MAX_TILES tiles wide
AND M is large (a multicast over many rows is what starves). Those three together are what the
watcher caught; any one alone is common and harmless.

    TT_VISIBLE_DEVICES=0 ... python3 scripts/protenix_v1_port/census_forced_grid_mm.py \
        [model] [target.yaml]
"""
import collections
import sys
from pathlib import Path

import ttnn

RECORDS = collections.Counter()
_orig_linear = ttnn.linear
_orig_matmul = ttnn.matmul


def _shape(t):
    try:
        return tuple(int(d) for d in t.shape)
    except Exception:
        return ("?",)


def _wrap(fn, name):
    def inner(a, b, *args, **kw):
        sa, sb = _shape(a), _shape(b)
        # M is the ROW COUNT the matmul actually sees, i.e. every leading dim multiplied, not
        # sa[-2]. For a (1, N, N, c) pair tensor sa[-2] is N while the real row count is N*N --
        # reading sa[-2] understated the pair track by 512x and put its calls in the wrong
        # bucket entirely on the first run of this script.
        try:
            m = 1
            for d in sa[:-1]:
                m *= int(d)
        except Exception:
            m = "?"
        k = sa[-1] if sa else "?"
        n = sb[-1] if len(sb) >= 1 else "?"
        # dtype matters: the op that actually deadlocked was FLOAT32, while most trunk
        # matmuls are bfloat16, and fp32 takes different packing/kernels. Without this column
        # the census reports 20 "exposed" shapes and cannot say which share the failing op's
        # arithmetic.
        dt = str(kw.get("dtype", "")).replace("DataType.", "") or "inherit"
        RECORDS[(name, m, k, n,
                 "core_grid" in kw and kw["core_grid"] is not None, dt)] += 1
        return fn(a, b, *args, **kw)
    return inner


ttnn.linear = _wrap(_orig_linear, "linear")
ttnn.matmul = _wrap(_orig_matmul, "matmul")

# tt_bio must be imported AFTER the wrap so its module-level `from ttnn import ...` (if any)
# and its call sites resolve through the patched attributes.
from tt_bio import weights                                          # noqa: E402
from tt_bio.main import _read_bio_chains, _read_bio_constraints     # noqa: E402
from tt_bio.protenix import PAIRCOND_MM_NARROW_MAX_TILES, Protenix  # noqa: E402
from tt_bio.protenix_data import build_complex_features             # noqa: E402

BIG_M = 4096   # a multicast over this many rows or more; the hang was at M = 512*512 = 262144


def main(model, target):
    chains = _read_bio_chains(Path(target))
    bonds = _read_bio_constraints(Path(target))
    specs = [(seq, None, mt) for _cid, seq, _spec, mt in chains]
    ids = [cid for cid, _s, _sp, _mt in chains]
    feats = build_complex_features(specs, mol_dir=str(weights.fetch("mols")),
                                   chain_ids=ids, bonds=bonds)
    m = Protenix.load_from_checkpoint(str(weights.fetch(model)))
    print(f"model={model} target={target} c_z={m.trunk.C_Z}", flush=True)
    m.fold(feats, n_step=6, n_sample=1, seed=0)

    rows = sorted(RECORDS.items(), key=lambda kv: -kv[1])
    exposed = [(kk, c) for kk, c in rows
               if kk[4] and isinstance(kk[3], int) and isinstance(kk[1], int)
               and kk[3] // 32 <= PAIRCOND_MM_NARROW_MAX_TILES and kk[1] >= BIG_M]
    fp32_exposed = [(kk, c) for kk, c in exposed if "FLOAT32" in kk[5]]
    print(f"\ndistinct matmul shapes issued: {len(rows)}   total calls: {sum(RECORDS.values())}")
    print(f"forced-core_grid shapes: {sum(1 for kk, _ in rows if kk[4])}")
    print(f"\nEXPOSED (core_grid forced, out <= {PAIRCOND_MM_NARROW_MAX_TILES} tiles, "
          f"M >= {BIG_M}): {len(exposed)}")
    for (op, mm, k, n, grid, dt) in [kk for kk, _ in exposed]:
        c = RECORDS[(op, mm, k, n, grid, dt)]
        print(f"  {op:7s} M={mm:8} K={k:5} N={n:5} (out {n // 32} tiles)  {dt:9s} x{c}")
    print(f"\n  ...of which FLOAT32 (the failing op's arithmetic): {len(fp32_exposed)}")
    for (op, mm, k, n, grid, dt) in [kk for kk, _ in fp32_exposed]:
        print(f"    {op:7s} M={mm:8} K={k:5} N={n:5} (out {n // 32} tiles)  x"
              f"{RECORDS[(op, mm, k, n, grid, dt)]}")
    print("\nall forced-core_grid shapes, for reference:")
    for (op, mm, k, n, grid, dt), c in rows:
        if grid:
            nt = f"{n // 32}t" if isinstance(n, int) else "?"
            print(f"  {op:7s} M={mm:8} K={k:5} N={n:5} (out {nt:4s}) {dt:9s} x{c}")
    return 0


if __name__ == "__main__":
    mdl = sys.argv[1] if len(sys.argv) > 1 else "protenix-v1"
    tgt = sys.argv[2] if len(sys.argv) > 2 else "perf/size512/fixtures/cdk2x2_512.yaml"
    sys.exit(main(mdl, tgt))
