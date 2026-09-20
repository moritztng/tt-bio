#!/usr/bin/env python3
"""Attribute every size-ladder FAIL finding to the flag it belongs to, or to main.

WHY THIS EXISTS. The size-ladder arm is a change detector against a recorded baseline, and the
p300c baseline fragments were recorded 2026-09-17/18 while main has taken 78 tt_bio commits since.
So the first row to run the arm inherits every other row's lever drift, and a prose claim of "most
of this red is not mine" is unreadable. This classifies each finding per entry, from the census the
engine itself computes, so the attribution can be re-derived instead of trusted.

THE ONE CODE FACT IT TURNS ON. TT_BIO_SDPA_WIDE_K changes the k-chunk pick only at padded lengths
whose shipped chunk does not divide them. scripts/sdpa_wide_k_census.py asserts every OTHER length
returns exactly the shipped k_chunk with the flag on, so at an unaffected rung both arms run the
same pick and the fold is bit-identical. A finding at such a rung cannot be caused by this flag --
with one exception that is kept explicit: the census records the flag's RESOLVED VALUE, and that
changes everywhere. That is finding class 1 in release_gate.py's own list ("a default flip").

Usage:  attribute_sizeladder_drift.py <gate log> [--pad-multiple 32]
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import sdpa_wide_k_census as C  # noqa: E402

FLAG = "SDPA_WIDE_K"

# `    FAIL boltz2/768 QKV_MM_CONFIG: frac 1.000 -> 0.000 (went dark)`
CELL = re.compile(r"^\s+FAIL (?P<model>[a-z0-9-]+)/(?P<rung>\d+) (?P<flag>[A-Z0-9_]+): (?P<detail>.+)$")
# `    FAIL boltz2 256->512: exponent 1.42 -> 2.44 outside ±0.50`
SPAN = re.compile(r"^\s+FAIL (?P<model>[a-z0-9-]+) (?P<lo>\d+)->(?P<hi>\d+): (?P<detail>.+)$")
# `    FAIL boltz2 per-rung base->run: ...`
RUNS = re.compile(r"^\s+FAIL (?P<model>[a-z0-9-]+) per-rung base->run: (?P<detail>.+)$")

UNMEASURED = "unmeasured: the census observed no calls, re-run this rung"
NEEDS_CENSUS = ("unattributable from the log: went dark with no clause named -- read "
                "served/declined in the census before crediting this to anyone")
FLIP_BOOKKEEPING = "flip: resolved-value only, bit-identical at this rung"
FLIP_REACHABLE = "FLIP: REACHABLE -- this rung is one of the affected lengths, review it"
NOT_FLIP = "main: the flip changes no bytes at this rung"
AMBIGUOUS = "AMBIGUOUS: other lever at a rung the flip can reach"


def affected_set(max_len: int) -> set:
    """The padded lengths where the flag changes the pick, from the engine, not from a list."""
    return {r["padded"] for r in C.census(max_len) if r["affected"]}


def classify(flag: str, detail: str, rungs: set, affected: set) -> str:
    reaches = bool(rungs & affected)
    # A rung the census saw no calls of belongs to neither side. Crediting it to main says a
    # merged commit changed behaviour there, and the evidence says nothing ran to change.
    # Matches both the gate's new wording and the old "went dark" with no clause named.
    if "observed NO calls" in detail:
        return UNMEASURED
    # Pre-fix logs say only "(went dark)". A lever that really declines names the clause it
    # declined on, so a bare one is EITHER a decline with an empty rejects dict OR a rung the
    # census never observed -- and the log cannot tell those apart. Refuse to guess: say so and
    # send the reader to the census, rather than crediting a merged commit with a change that
    # may never have been executed.
    if "went dark" in detail and " on " not in detail:
        return NEEDS_CENSUS
    if flag == FLAG:
        if detail.startswith("resolved ") and not reaches:
            return FLIP_BOOKKEEPING
        return FLIP_REACHABLE
    return AMBIGUOUS if reaches else NOT_FLIP


def findings(log_path: str, affected: set) -> list:
    out = []
    for line in open(log_path, errors="replace"):
        m = CELL.match(line)
        if m:
            rung = int(m["rung"])
            out.append((m["model"], str(rung), m["flag"], m["detail"],
                        classify(m["flag"], m["detail"], {rung}, affected)))
            continue
        m = RUNS.match(line)
        if m:
            rs = {int(x) for x in re.findall(r"\b(\d+) [\d.]+->", m["detail"])}
            out.append((m["model"], ",".join(map(str, sorted(rs))) or "-", "runtime",
                        m["detail"][:60], classify("runtime", m["detail"], rs, affected)))
            continue
        m = SPAN.match(line)
        if m:
            rs = {int(m["lo"]), int(m["hi"])}
            out.append((m["model"], f"{m['lo']}->{m['hi']}", "exponent", m["detail"],
                        classify("exponent", m["detail"], rs, affected)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--max-len", type=int, default=1536)
    a = ap.parse_args()

    affected = affected_set(a.max_len)
    print(f"affected padded lengths (computed from tt_bio at this commit): "
          f"{sorted(affected)}")

    rows = findings(a.log, affected)
    if not rows:
        print("NO FINDINGS PARSED -- refusing to report a clean attribution off an empty parse")
        return 1

    # A rung is compared against the census by its padded length, so the two have to be the same
    # number. Every ladder rung is a multiple of the pad multiple, which makes padded == rung --
    # assert it rather than assume it, because a rung that padded UP would be compared against the
    # wrong census row and the whole table would be keyed to the wrong lengths.
    rungs_seen = sorted({int(r[1]) for r in rows if r[1].isdigit()})
    pad = C.T.PAIRFORMER_PAD_MULTIPLE
    off = [r for r in rungs_seen if r % pad]
    if off:
        print(f"REFUSING: rungs {off} are not multiples of the {pad}-token pad, so padded != rung "
              f"and the census rows they would be compared against are the wrong ones")
        return 1
    print(f"ladder rungs that appear in findings: {rungs_seen}")
    print(f"of those, affected by the flag: {sorted(set(rungs_seen) & affected)}")
    print()
    print(f"{'model':<13}{'rung':<10}{'lever':<30}{'verdict'}")
    for model, rung, flag, detail, verdict in rows:
        print(f"{model:<13}{rung:<10}{flag:<30}{verdict}")

    print()
    tally = {}
    for r in rows:
        tally[r[4]] = tally.get(r[4], 0) + 1
    for k in sorted(tally, key=lambda k: -tally[k]):
        print(f"{tally[k]:4d}  {k}")

    # NEGATIVE CONTROL. The classifier must be reading the census, not the flag's name. Pretend one
    # unaffected rung IS affected and require at least one verdict to move; if nothing moves, the
    # affected set is not what the classification turns on and the table above means nothing.
    victim = next((r for r in rows if r[1].isdigit() and int(r[1]) not in affected
                   and r[4] not in (UNMEASURED, NEEDS_CENSUS)), None)
    if victim is None:
        print("\nNEGATIVE CONTROL: no unaffected-rung finding to perturb -- control not run")
        return 1
    v_rung = int(victim[1])
    moved = classify(victim[2], victim[3], {v_rung}, affected | {v_rung}) != victim[4]
    print(f"\nNEGATIVE CONTROL: treating {victim[0]}/{v_rung} {victim[2]} as a firing rung "
          f"{'MOVES' if moved else 'does NOT move'} its verdict "
          f"({victim[4]!r} -> {classify(victim[2], victim[3], {v_rung}, affected | {v_rung})!r})")
    if not moved:
        print("CONTROL FAILED: the verdict does not depend on the affected set")
        return 1

    bad = [r for r in rows if r[4] in (FLIP_REACHABLE, AMBIGUOUS)]
    print(f"\nfindings this flag could have caused: {len(bad)}")
    for r in bad:
        print(f"  {r[0]}/{r[1]} {r[2]}: {r[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
