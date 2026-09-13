#!/usr/bin/env python3
"""The op ratio's own floor, from the same draws and the same estimator.

`qchunk_bh.py` prints the same-config pairs as a min/max spread, which is the widest possible
reading of a null and not the one the headline uses. The headline is a MEDIAN of paired ratios, so
the floor has to be the 95 % band of a median of the same number of same-config paired ratios.
The declining site (atom, where both arms get a byte-identical program config) is the control: its
paired ratio must sit inside its own floor, or the instrument is not reading the lever.
"""
import json, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_band  # noqa: E402


def main():
    src = Path(sys.argv[1])
    d = json.loads(src.read_text())
    for rec in d["ops"]:
        draws = [(a, t) for a, t in rec.get("draws_us", [])]
        if not draws:
            continue
        ab, aa = [], []
        for (a1, t1), (a2, t2) in zip(draws, draws[1:]):
            (aa if a1 == a2 else ab).append((t1 / t2) if a1 != "ship" else (t2 / t1))
        stat = st.median(ab)
        lo, hi = boot_band(aa, len(ab))
        rec["floor_matched_95"] = [round(lo, 5), round(hi, 5)]
        rec["paired_median_ratio"] = round(stat, 5)
        rec["clears_floor"] = bool(stat > hi or stat < lo)
        print(f"{rec['site']:12s} {rec['shipped_chunk']:4d}->{rec['chunk']:4d} "
              f"paired {stat:7.4f}x  floor [{lo:.4f},{hi:.4f}]  "
              f"{'CLEARS' if rec['clears_floor'] else 'inside the floor (null)'}  "
              f"n_ab={len(ab)} n_aa={len(aa)}  equal={rec['torch_equal']}")
    src.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
