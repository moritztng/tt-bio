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
``pair_copies`` are not observable from the panel. They were previously passed as a
stated guess of 4.0 and 3.0 GiB -- **and that guess is refuted by the campaign's own
results**: it predicts 9n8n, 9ly6 and 9lof run out of DRAM, and all three fold.

So they are now calibrated instead. 164 targets have a measured outcome, which pins
the pair down without needing a trace:

* every target that is not capacity-bound must FIT (the 159 that produced structures,
  plus 9q7y, 9ivj and 9mns -- those three fail while dozens of *larger* targets fold
  and their refused sizes match no tensor, so their failure is not footprint; see
  ``classify_blocked_targets.py``);
* 9j4c and 9i3p must NOT fit;
* ``floor > 0``, since resident weights cannot occupy negative DRAM.

That leaves a feasible band rather than a point, and the band is reported instead of a
single number, because pretending to a precision the data does not support is how the
4.0/3.0 guess got in. Predictions are quoted across the whole band, and a conclusion
that does not hold across all of it is labelled conditional.

This is still a projection. It ranks targets and sizes the fix; it does not replace a
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


# Measured outcomes the calibration is anchored on. CAPACITY_BOUND are the only two targets that
# must NOT fit: each sits above every target that folds AND its refused byte count decomposes into
# a real tensor. The other three missing targets fail for reasons that are not footprint, so the
# model has to place them inside the budget like any target that folds -- see
# classify_blocked_targets.py, which derives this split rather than assuming it.
CAPACITY_BOUND = ("9j4c", "9i3p")
NOT_FOOTPRINT = ("9q7y", "9ivj", "9mns")
CURRENT_PATTERN = "now-chunked"     # the path the measured outcomes were produced on


def pad(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def calibrate(sized: list[dict], chip_gib: float, msa_copies: float,
              step: float = 0.1, limit: float = 24.0) -> list[tuple[float, float, float]]:
    """Feasible (pair_copies, floor_lo, floor_hi) under the measured outcomes.

    floor is constrained to (floor_lo, floor_hi]: strictly above what would let a
    capacity-bound target fit, at or below what keeps every other target fitting.
    """
    must_fit = [r for r in sized if r["target"] not in CAPACITY_BOUND]
    must_not = [r for r in sized if r["target"] in CAPACITY_BOUND]
    if not must_not:
        return []
    out = []
    for i in range(int(limit / step) + 1):
        p = round(i * step, 3)
        hi = max(msa_copies * r["m_feat"] + p * r["z"] for r in must_fit)
        lo = min(msa_copies * r["m_feat"] + p * r["z"] for r in must_not)
        f_lo, f_hi = max(0.0, chip_gib - lo), chip_gib - hi
        if f_hi > f_lo:
            out.append((p, f_lo, f_hi))
    return out


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
    ap.add_argument("--pair_copies", type=float,
                    help="live copies of the pair representation at peak; omit to calibrate "
                         "against the measured outcomes instead of guessing")
    ap.add_argument("--floor_gib", type=float,
                    help="resident weights + everything not scaling with D or T; omit to calibrate")
    ap.add_argument("--top", type=int, default=10, help="worst-N targets to list")
    args = ap.parse_args()

    rows = json.loads(args.panel.read_text())
    sized = []
    for r in rows:
        t, d, mf, z = footprint(r)
        sized.append({**r, "T": t, "D": d, "m_feat": mf, "z": z})
    sized.sort(key=lambda r: -r["m_feat"])

    band = calibrate(sized, args.chip_gib, PATTERNS[CURRENT_PATTERN])
    if args.pair_copies is None or args.floor_gib is None:
        if not band:
            raise SystemExit("no (pair_copies, floor) reproduces the measured outcomes -- the "
                             "linear model itself is wrong, not its parameters")
        # Midpoint of the feasible band: a representative, not a measurement.
        p, f_lo, f_hi = band[len(band) // 2]
        pair_copies = args.pair_copies if args.pair_copies is not None else p
        floor_gib = args.floor_gib if args.floor_gib is not None else (f_lo + f_hi) / 2
        source = (f"CALIBRATED against {len(sized)} measured outcomes; feasible pair_copies "
                  f"{band[0][0]:g}..{band[-1][0]:g}, this run uses the band midpoint")
    else:
        pair_copies, floor_gib = args.pair_copies, args.floor_gib
        ok = any(abs(p - pair_copies) < 1e-9 and f_lo < floor_gib <= f_hi
                 for p, f_lo, f_hi in band)
        source = ("supplied" if ok else
                  "supplied, and REFUTED by the measured outcomes -- this pair mispredicts at "
                  "least one target that is known to fold")
    args.pair_copies, args.floor_gib = pair_copies, floor_gib

    print(f"panel {len(sized)} targets | chip {args.chip_gib:.2f} GiB | "
          f"pair_copies {pair_copies:g} floor {floor_gib:.2f} GiB\n  ({source})\n")

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

    # A prediction that only holds at one end of the feasible band is not a prediction, so quote
    # each blocked target across the whole band rather than at the representative point above.
    if band:
        by_target = {r["target"]: r for r in sized}
        print(f"\nwould a copy-count fix clear the two capacity-bound targets? "
              f"(checked across the whole feasible band, not one point)")
        for name, mult in PATTERNS.items():
            if mult >= PATTERNS[CURRENT_PATTERN]:
                continue
            verdicts = []
            for t in CAPACITY_BOUND:
                r = by_target.get(t)
                if r is None:
                    continue
                peaks = [mult * r["m_feat"] + p * r["z"] + f_hi for p, _, f_hi in band]
                if max(peaks) <= args.chip_gib:
                    verdicts.append(f"{t} clears ({min(peaks):.2f}-{max(peaks):.2f} GiB)")
                elif min(peaks) <= args.chip_gib:
                    verdicts.append(f"{t} CONDITIONAL ({min(peaks):.2f}-{max(peaks):.2f} GiB)")
                else:
                    verdicts.append(f"{t} still OOM ({min(peaks):.2f}-{max(peaks):.2f} GiB)")
            print(f"  {name:<14} " + "; ".join(verdicts))


if __name__ == "__main__":
    main()
