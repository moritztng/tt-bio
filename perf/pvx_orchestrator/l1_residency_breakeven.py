#!/usr/bin/env python3
"""Why the L1-residency levers fire on Boltz-2 and refuse on Protenix-v2, in one inequality.

`pvx-eligibility` counted the bucket-2 levers and three of the seven were left with their blocking
clause named but not resolved: residual-into-L1 (80 served / 1048 refused on protenix-v2),
trimul-out-into-L1 (0 / 1048) and the Blackhole transition row raise (110 / 526). This prices all
three against the SHIPPED constants and finds one mechanism under all of them.

It is not a Protenix predicate. Every one of these levers asks the same question -- does a
512x512xc_z bf16 pair tensor fit the grid's L1 banks -- and the break-even channel lands strictly
between Boltz-2's c_z=128 and Protenix-v2's c_z=256 on both shipped grids. Equivalently it is a
SIZE threshold: Boltz-2 loses residual-into-L1 itself above 667 aa.

Constants read off tt_bio/tenstorrent.py rather than a device, so this runs on a CPU host:
  1532416  get_max_worker_l1_unreserved_size() on Blackhole      (:4266, used by _l1_fits)
  1461760  the L1 allocator's per-bank figure                    (:4267, used by _trimul_l1_fits)
   655360  _PAIR_L1_CONSUMER_RESERVE                             (:521)
      0.5  _TRIMUL_TAIL_L1_SHARE                                 (:647)
"""
import math

UNRESERVED, BANK, RESERVE, SHARE, ELEM = 1_532_416, 1_461_760, 640 * 1024, 0.5, 2
GRIDS = ((13 * 10, "13x10"), (11 * 10, "11x10"))
MODELS = ((128, "boltz2"), (256, "protenix-v2"))


def budgets(cores):
    """(residual, trimul-out) L1 budgets in bytes for a grid of `cores` banks."""
    return (UNRESERVED - RESERVE) * cores, SHARE * BANK * cores


def main():
    for cores, grid in GRIDS:
        res, tri = budgets(cores)
        print(f"\n{grid} ({cores} banks): residual budget {res / 2**20:.1f} MiB, "
              f"trimul-out budget {tri / 2**20:.1f} MiB")
        print(f"  break-even channel at 512 aa: residual c_z={res / (512 * 512 * ELEM):.1f}, "
              f"trimul-out c_z={tri / (512 * 512 * ELEM):.1f}"
              "   (hardcoded twin: _BH_TRANSITION_L1_ROWS_MAX_C = 128)")
        for c, who in MODELS:
            nb = 512 * 512 * c * ELEM
            print(f"  {who:12s} c_z={c:3d}  {nb / 2**20:6.1f} MiB at 512 aa   "
                  f"residual {'FITS   ' if nb <= res else 'REFUSES'}  "
                  f"trimul-out {'FITS   ' if nb <= tri else 'REFUSES'}   "
                  f"residual holds to N<={math.isqrt(int(res // (c * ELEM)))} aa")

    res, _ = budgets(130)
    lo, hi = (math.sqrt(res / (c * ELEM)) for c in (256, 128))
    print(f"\nSame lever, stated as a size: it holds to {lo:.0f} aa at c_z=256 and to {hi:.0f} aa "
          f"at c_z=128.\nProtenix-v2's scored cell misses by {(512 - lo) / lo:+.1%} of N. Boltz-2 "
          f"loses it too, at {hi:.0f} aa.")


if __name__ == "__main__":
    main()
