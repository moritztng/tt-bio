#!/usr/bin/env python3
"""Read the GroEL size ladder 512 / 1024 / 1536 against plans/groel_1024.txt.

Written and verified BEFORE the 1024 cells land, so the test is not chosen after seeing the
numbers -- the same discipline every other cell in this row was read under. `--selfcheck`
reproduces the four published GroEL 2x2 contrasts from the banked rows, which is what makes the
enumeration below trustworthy on a cell nobody has seen yet.

    python3 perf/mgxaccuracy/ladder_read.py results/size_1536.jsonl [results/groel1024.jsonl ...]
    python3 perf/mgxaccuracy/ladder_read.py --selfcheck results/size_1536.jsonl

pc has no scipy, so the p is counted rather than approximated: C(16,8) = 12870 splits, the same
exact two-sided Mann-Whitney the 2x2 and the onetree control were read with.
"""
import argparse
import json
import pathlib
import statistics as st
from itertools import combinations

TARGET = "groel_ring4_2096"
ENGINE = "groel-onetree-ff5435cba"
SIZES = (512, 768, 1024, 1536)
OFFSETS = (0, 560)
PERMISSIVE = 4.0          # BoltzGen's own designability bar, unchanged throughout this row
STRICT = 2.0


def u_stat(a, b) -> float:
    """#{(x,y) in a x b : x < y}, halves for ties. a is the group whose median is reported
    MINUS b's, so a perfect 'a entirely above b' reads U = 0 of len(a)*len(b) -- the same
    orientation the 2x2 table published."""
    return sum((x < y) + 0.5 * (x == y) for x in a for y in b)


def exact_two_sided(a, b) -> float:
    """Exact two-sided p by enumerating every way to split the pooled values into groups of
    len(a) and len(b). Ties are handled by the 0.5 in u_stat, so this is exact only in the
    no-tie case and conservative-by-a-hair otherwise; scRMSD values are floats and do not tie."""
    pool = list(a) + list(b)
    na = len(a)
    obs = u_stat(a, b)
    lo = hi = tot = 0
    for idx in combinations(range(len(pool)), na):
        s = set(idx)
        ua = [pool[i] for i in idx]
        ub = [pool[i] for i in range(len(pool)) if i not in s]
        u = u_stat(ua, ub)
        tot += 1
        lo += u <= obs
        hi += u >= obs
    return min(1.0, 2.0 * min(lo, hi) / tot)


def load(paths):
    cells, refused = {}, []
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("side") != "device" or "scrmsd" not in d:
                continue
            if TARGET not in str(d.get("target", "")):
                continue
            key = (d.get("target_res"), d.get("crop_offset", 0))
            # The tree is part of a cell's identity: a ladder rung measured on a different
            # engine is not comparable with the rungs it is read against, which is the whole
            # reason the 1024 cells were run on ff5435cba rather than on main.
            if d.get("engine") != ENGINE:
                refused.append((key, f"engine {d.get('engine')!r} != {ENGINE!r}"))
                continue
            cells[key] = d
    return cells, refused


def frac(v, bar):
    return 100.0 * sum(x <= bar for x in v) / len(v)


def contrast(cells, big, small, off, label):
    a, b = cells[(big, off)]["scrmsd"], cells[(small, off)]["scrmsd"]
    eff = st.median(a) - st.median(b)
    u = u_stat(a, b)
    p = exact_two_sided(a, b)
    print(f"  {label:<22} {eff:+8.3f} A   U = {u:5.1f} of {len(a)*len(b):<3}  "
          f"exact p = {p:.4f}  {'rejects' if p <= 0.05 else 'does not reject'}")
    return eff, u, p


def selfcheck(cells) -> int:
    """The 2x2's four published contrasts, recomputed. Any mismatch means this script's
    enumeration does not agree with the one the banked verdict was read off, and the ladder
    numbers it would print could not be trusted."""
    want = {"size at offset   0": (+14.754, 0.0, 0.0002),
            "size at offset 560": (+12.647, 8.0, 0.0104),
            "crop at  512": (+0.887, 21.0, 0.2786),
            "crop at 1536": (-1.220, 47.0, 0.1304)}
    got = {}
    print("SELFCHECK against results/groel_2x2_result.txt\n")
    for off in OFFSETS:
        got[f"size at offset {off:>3}"] = contrast(cells, 1536, 512, off, f"size at offset {off:>3}")
    for size in (512, 1536):
        a, b = cells[(size, 560)]["scrmsd"], cells[(size, 0)]["scrmsd"]
        eff, u, p = st.median(a) - st.median(b), u_stat(a, b), exact_two_sided(a, b)
        print(f"  {'crop at ' + str(size):<22} {eff:+8.3f} A   U = {u:5.1f} of 64   "
              f"exact p = {p:.4f}")
        got[f"crop at {size:>4}"] = (eff, u, p)
    bad = 0
    print()
    for k, (e, u, p) in want.items():
        ge, gu, gp = got[k]
        ok = abs(ge - e) < 0.001 and gu == u and abs(gp - p) < 0.00005
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {k}: published {e:+.3f}/{u}/{p:.4f}  "
              f"got {ge:+.3f}/{gu}/{gp:.4f}")
    print("\nSELFCHECK PASS" if not bad else f"\nSELFCHECK FAIL ({bad})")
    return bad


def verdict_one(cells, off, rung=1024):
    """The pre-registered clauses of plans/groel_<rung>.txt, for ONE crop.

    Each rung carries its OWN clause bodies and they are not shared: 1024's USABLE clause
    requires the upper contrast to reject and 768's does not, because at 1024 the upper
    neighbour was the only measured point above it and at 768 it is not. Adding a rung must
    therefore ADD a branch, never repoint the existing one -- the 1024 verdict is published
    (results/groel_1024_result.txt) and this function has to keep reproducing it exactly.
    """
    lo, hi = {1024: (512, 1536), 768: (512, 1024)}[rung]
    v = cells[(rung, off)]["scrmsd"]
    med, f4 = st.median(v), sum(x <= PERMISSIVE for x in v)
    print(f"\nOFFSET {off}   rung {rung}")
    up = contrast(cells, hi, rung, off, f"{rung} -> {hi}")
    dn = contrast(cells, rung, lo, off, f"{lo:>4} -> {rung}")
    if rung == 1024:
        if med <= PERMISSIVE and f4 >= 4 and up[2] <= 0.05:
            v_ = "USABLE AT 1024"
        elif med > PERMISSIVE and f4 <= 1 and dn[2] <= 0.05:
            v_ = "COLLAPSED BY 1024"
        else:
            v_ = "INTERMEDIATE"
    else:
        # plans/groel_768.txt, clause bodies disjoint by construction.
        if med <= PERMISSIVE and f4 >= 4:
            v_ = "USABLE AT 768"
        elif med > PERMISSIVE and f4 <= 1 and dn[2] <= 0.05:
            v_ = "COLLAPSED BY 768"
        else:
            v_ = "INTERMEDIATE"
    print(f"  median {med:.3f} A, {f4} of {len(v)} under {PERMISSIVE} A  ->  {v_}")
    return v_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--selfcheck", action="store_true",
                    help="recompute the banked 2x2 contrasts and stop")
    args = ap.parse_args()

    cells, refused = load(args.jsonl)
    for key, why in refused:
        print(f"  refused {key}: {why}")
    if args.selfcheck:
        need = [(s, o) for s in (512, 1536) for o in OFFSETS]
        if any(k not in cells for k in need):
            print(f"selfcheck needs the four 2x2 cells; missing {[k for k in need if k not in cells]}")
            return 1
        return selfcheck(cells)

    print(f"target {TARGET}   engine {ENGINE}   metric designfolding-bb_rmsd\n")
    print("  size  offset   n   median     range              <=2A    <=4A")
    for s in SIZES:
        for o in OFFSETS:
            d = cells.get((s, o))
            if d is None:
                print(f"  {s:>4}  {o:>6}   -   MISSING")
                continue
            v = d["scrmsd"]
            print(f"  {s:>4}  {o:>6}  {len(v):>2}  {st.median(v):>7.3f}   "
                  f"{min(v):6.3f}-{max(v):6.3f}    {frac(v, STRICT):5.1f}%  {frac(v, PERMISSIVE):5.1f}%")

    # A rung is readable only when BOTH its crops and both crops of each neighbour it is
    # contrasted against are in. Every plan in this row requires the two crops to agree, which
    # a single crop cannot establish, so a half-landed rung is skipped rather than half-read.
    rc = 1
    for rung, (lo, hi) in ((768, (512, 1024)), (1024, (512, 1536))):
        need = [(s, o) for s in (lo, rung, hi) for o in OFFSETS]
        gaps = [k for k in need if k not in cells]
        if gaps:
            print(f"\nrung {rung}: not readable, {len(gaps)} cell(s) missing {gaps}")
            continue
        vs = [verdict_one(cells, o, rung) for o in OFFSETS]
        print(f"\nRUNG {rung} VERDICT: ", end="")
        print(vs[0] if vs[0] == vs[1] else
              f"INTERMEDIATE (the two crops disagree: {vs[0]} at 0, {vs[1]} at 560)")
        rc = 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
