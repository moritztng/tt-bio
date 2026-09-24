#!/usr/bin/env python3
"""Read the FAS single-chain ladder against plans/chain_vs_residues.txt.

Written and verified BEFORE the FAS cells land, the same discipline every other rung in this
row was read under, and written as a SEPARATE script rather than as a flag on ladder_read.py.
That is deliberate: ladder_read.py carries the published GroEL verdicts, its `TARGET` and
`OFFSETS` constants are load-bearing for them, and the FAS clauses are not the GroEL clauses.
Adding a branch there would put two verdict systems in one function for no gain. This file
imports ladder_read's exact enumeration so the arithmetic is literally the same code.

    python3 perf/mgxaccuracy/fas_read.py results/fas_ladder.jsonl
    python3 perf/mgxaccuracy/fas_read.py --selfcheck results/size_1536.jsonl \
        results/groel768.jsonl results/groel1024.jsonl

`--selfcheck` re-derives three published GroEL contrasts through THIS file's code path, which
is what makes its enumeration trustworthy on cells nobody has seen.
"""
import argparse
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ladder_read import u_stat, exact_two_sided, PERMISSIVE, STRICT  # noqa: E402

TARGET = "fas_chainA_1962"
ENGINE = "groel-onetree-ff5435cba"   # the tree label, not the target; see plans/groel_1024.txt
RUNGS = (512, 1024, 1536)          # the pre-registered ladder of plans/chain_vs_residues.txt
SHOW = (512, 768, 1024, 1536)      # what the table prints; 768 is plans/fas_768.txt, an
                                   # ADDITION that is reported separately and enters no clause
                                   # of chain_vs_residues.txt


def load(paths, target, engine):
    cells, refused = {}, []
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("side") != "device" or "scrmsd" not in d:
                continue
            if target not in str(d.get("target", "")):
                continue
            if d.get("engine") != engine:
                refused.append(((d.get("target_res"), d.get("crop_offset", 0)),
                                f"engine {d.get('engine')!r} != {engine!r}"))
                continue
            cells[(d.get("target_res"), d.get("crop_offset", 0))] = d
    return cells, refused


def frac(v, bar):
    return 100.0 * sum(x <= bar for x in v) / len(v)


def contrast(a, b, label):
    eff, u, p = st.median(a) - st.median(b), u_stat(a, b), exact_two_sided(a, b)
    print(f"  {label:<22} {eff:+8.3f} A   U = {u:5.1f} of {len(a)*len(b):<3}  "
          f"exact p = {p:.4f}  {'rejects' if p <= 0.05 else 'does not reject'}")
    return eff, u, p


def selfcheck(paths) -> int:
    """Three published GroEL contrasts, recomputed through this file. Any mismatch means the
    enumeration reached through this import is not the one the banked verdicts were read with."""
    from ladder_read import load as groel_load
    cells, _ = groel_load(paths)
    want = {"size at offset   0": (+14.754, 0.0, 0.0002),
            " 512 -> 768 off 0": (+14.374, 0.0, 0.0002),
            " 512 ->1024 off 0": (+7.866, 7.0, 0.0070)}
    need = [(512, 0), (768, 0), (1024, 0), (1536, 0)]
    missing = [k for k in need if k not in cells]
    if missing:
        print(f"selfcheck needs the offset-0 GroEL cells; missing {missing}")
        return 1
    got = {
        "size at offset   0": contrast(cells[(1536, 0)]["scrmsd"], cells[(512, 0)]["scrmsd"],
                                       "size at offset   0"),
        " 512 -> 768 off 0": contrast(cells[(768, 0)]["scrmsd"], cells[(512, 0)]["scrmsd"],
                                      " 512 -> 768 off 0"),
        " 512 ->1024 off 0": contrast(cells[(1024, 0)]["scrmsd"], cells[(512, 0)]["scrmsd"],
                                      " 512 ->1024 off 0"),
    }
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck(a.jsonl)

    cells, refused = load(a.jsonl, TARGET, ENGINE)
    for key, why in refused:
        print(f"  refused {key}: {why}")
    print(f"target {TARGET}   engine {ENGINE}   metric designfolding-bb_rmsd   "
          f"offset {a.offset}\n")
    print("  size   n   median     range              <=2A    <=4A")
    for s in SHOW:
        d = cells.get((s, a.offset))
        if d is None:
            print(f"  {s:>4}   -   MISSING")
            continue
        v = d["scrmsd"]
        print(f"  {s:>4}  {len(v):>2}  {st.median(v):>7.3f}   {min(v):6.3f}-{max(v):6.3f}    "
              f"{frac(v, STRICT):5.1f}%  {frac(v, PERMISSIVE):5.1f}%")

    missing = [s for s in RUNGS if (s, a.offset) not in cells]
    if missing:
        print(f"\nrungs missing: {missing} -- no verdict. plans/chain_vs_residues.txt reads "
              f"512->1536 and needs 1024 beside it to tell a monotone ladder from one odd cell.")
        return 1

    print()
    top = cells[(1536, a.offset)]["scrmsd"]
    bot = cells[(512, a.offset)]["scrmsd"]
    mid = cells[(1024, a.offset)]["scrmsd"]
    main_c = contrast(top, bot, " 512 -> 1536")
    contrast(mid, bot, " 512 -> 1024")
    contrast(top, mid, "1024 -> 1536")
    f4 = sum(x <= PERMISSIVE for x in top)
    print(f"\n  1536 single-chain cell: median {st.median(top):.3f} A, {f4} of {len(top)} "
          f"under {PERMISSIVE} A")

    # plans/chain_vs_residues.txt, clause bodies verbatim.
    if main_c[2] <= 0.05 and f4 <= 1:
        v = "RESIDUE COUNT IS THE CARRIER"
        why = ("quality falls with residue count on ONE chain throughout, so the row's "
               "published size axis means what it says and the doc's residue rule stands")
    elif main_c[2] > 0.05 and f4 >= 4:
        v = "CHAIN COUNT IS THE CARRIER"
        why = ("a single chain survives to 1536 where every multi-chain 1536 cell in this row "
               "returned nothing under 4 A; the doc's rule becomes a chain rule outright")
    else:
        v = "INTERMEDIATE"
        why = ("report all three medians, all three fractions and both contrasts; the doc says "
               "both residue count and chain count cost something, with the numbers for each")
    print(f"\nFAS LADDER VERDICT: {v}\n  -> {why}")
    # plans/fas_768.txt, reported separately and entering none of the clauses above.
    d768 = cells.get((768, a.offset))
    if d768 is not None:
        v768 = d768["scrmsd"]
        m768, f768 = st.median(v768), sum(x <= PERMISSIVE for x in v768)
        print("\n--- plans/fas_768.txt, an addition: a single chain at the residue count where "
              "GroEL collapsed ---")
        c768 = contrast(v768, bot, " 512 ->  768")
        if m768 <= PERMISSIVE and f768 >= 4:
            v = "SURVIVES AT 768"
        elif m768 > PERMISSIVE and f768 <= 1 and c768[2] <= 0.05:
            v = "COLLAPSED AT 768"
        else:
            v = "INTERMEDIATE"
        print(f"  median {m768:.3f} A, {f768} of {len(v768)} under {PERMISSIVE} A  ->  {v}")
        print("  (GroEL at 768 residues, two chains, was 0 of 8 under 4 A on both crops.)")

    print("\nRead WITHIN this target only. FAS is a third protein, so a difference between its "
          "ladder and\nGroEL's or 1GPB's is a target difference until shown otherwise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
