"""Project opendde-abag trunk DRAM against a 12 GiB Wormhole chip, at FULL MSA depth.

Supersedes ``analyze_mfeat_panel.py``'s "which targets are runnable" verdict, which
answered a different question: it ranked targets under a *capped* MSA and concluded
the token count was the only binding axis. At the campaign's real config -- uncapped
depth, matching the shipped Blackhole dataset -- depth is binding again, and the
limiter is not the size of any one tensor but **how many copies of the MSA
representation the trunk holds at once**.

The model
---------
Two padded axes, both read off ``tt_bio/tenstorrent.py``, not guessed:

* ``T = pad(tokens, PAIRFORMER_PAD_MULTIPLE=64)``  -- ``MSAModule`` pads before upload.
* ``D = pad(depth,  MSA_PAD_MULTIPLE=1024)``       -- same place; padded rows are
  masked by ``msa_mask`` and excluded from the ``OuterProductMean`` divisor
  (``n_msa``), so the padding costs DRAM and nothing else.

``m_feat = D * T * c_m(128) * 2`` bytes is one copy of the MSA representation.
``z      = T * T * c_z(384) * 2`` bytes is one copy of the pair representation
(opendde's c_z is 384, not Protenix-v2's 256).

Peak is then ``msa_copies * m_feat + pair_copies * z + floor``, where
``msa_copies`` is set by the allocator pattern in force:

  4.0  today, MSA row-chunking OFF (tokens <= the small-grid chunking threshold):
       ``Transition`` does ``ttnn.chunk(x, ...)`` then a list comprehension then
       one ``concat``, so the source, the chunk copies, the swiglu outputs and the
       concat output are all live at once (tenstorrent.py Transition.__call__).
  3.0  today, MSA row-chunking ON: ``MSALayer`` holds the whole source ``m`` for the
       full loop (it is only deallocated *after* it), while ``m_acc`` grows to the
       same size by running ``concat``, whose final step allocates a third copy.
  2.0  after freeing ``m`` before a single terminal concat (bit-exact reordering).
  1.25 after ``OuterProductMean`` consumes the row chunks directly, so the full-depth
       ``m`` is never materialised -- only its c=32 a/b projections are.

``floor`` (resident weights + host-staged constants + whatever else is live) and
``pair_copies`` are NOT derived here. Pass the values a ``TT_BIO_DRAM_PEAK`` trace
measures; the defaults are a stated guess and are labelled as such in the output.

This is a projection. It ranks targets and sizes the fix; it does not replace a
measurement, and no target should be called excluded on its output alone.
"""

from __future__ import annotations

import argparse
import json
import pathlib

C_M, C_Z, DTYPE_BYTES = 128, 384, 2
SEQ_PAD_MULTIPLE = 64      # tenstorrent.PAIRFORMER_PAD_MULTIPLE
MSA_PAD_MULTIPLE = 1024    # tenstorrent.MSA_PAD_MULTIPLE
GIB = 2 ** 30

# msa_copies for each allocator pattern; see the module docstring for the
# code path each one names.
PATTERNS = {
    "now-unchunked": 4.0,
    "now-chunked": 3.0,
    "fix-ab": 2.0,
    "fix-d": 1.25,
}


def pad(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def footprint(row: dict) -> tuple[int, int, float, float]:
    t = pad(row["tokens"], SEQ_PAD_MULTIPLE)
    d = pad(row["depth"], MSA_PAD_MULTIPLE)
    return t, d, d * t * C_M * DTYPE_BYTES / GIB, t * t * C_Z * DTYPE_BYTES / GIB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--panel", type=pathlib.Path, default=here / "mfeat_panel_v2.json",
                    help="per-target projection written by project_mfeat.py")
    ap.add_argument("--chip_gib", type=float, default=12.0,
                    help="usable device DRAM (Wormhole Galaxy 12.0, Blackhole p150a 31.88)")
    ap.add_argument("--pair_copies", type=float, default=4.0,
                    help="live copies of the pair representation at peak (MEASURE THIS)")
    ap.add_argument("--floor_gib", type=float, default=3.0,
                    help="resident weights + everything not scaling with D or T (MEASURE THIS)")
    ap.add_argument("--top", type=int, default=10, help="worst-N targets to list")
    args = ap.parse_args()

    rows = json.loads(args.panel.read_text())
    sized = []
    for r in rows:
        t, d, mf, z = footprint(r)
        sized.append({**r, "T": t, "D": d, "m_feat": mf, "z": z})
    sized.sort(key=lambda r: -r["m_feat"])

    print(f"panel {len(sized)} targets | chip {args.chip_gib:.2f} GiB | "
          f"pair_copies {args.pair_copies} floor {args.floor_gib:.1f} GiB "
          f"(both ASSUMED unless measured)\n")

    print(f"{'target':<7}{'tok':>5}{'T':>6}{'depth':>7}{'D':>7}{'m_feat':>8}{'z':>7}"
          + "".join(f"{k:>14}" for k in PATTERNS))
    for r in sized[: args.top]:
        base = args.pair_copies * r["z"] + args.floor_gib
        cells = "".join(
            f"{mult * r['m_feat'] + base:>11.2f} {'ok' if mult * r['m_feat'] + base <= args.chip_gib else 'OOM':>2}"
            for mult in PATTERNS.values())
        print(f"{r['target']:<7}{r['tokens']:>5}{r['T']:>6}{r['depth']:>7}{r['D']:>7}"
              f"{r['m_feat']:>8.2f}{r['z']:>7.2f}" + cells)

    print("\ncoverage at full (uncapped, unpaired) MSA depth:")
    for name, mult in PATTERNS.items():
        fit = [r for r in sized
               if mult * r["m_feat"] + args.pair_copies * r["z"] + args.floor_gib <= args.chip_gib]
        missing = [r["target"] for r in sized if r not in fit]
        print(f"  {name:<14} {len(fit):>3}/{len(sized)}"
              + (f"   OOM: {' '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}" if missing else ""))


if __name__ == "__main__":
    main()
