#!/usr/bin/env python3
"""Audit the exemption reasons a fresh p300c record left behind, and prove the other cards held.

Three things a re-record does to reasons, all of which have bitten this baseline before:

  IT LEAVES TODOs. A lever that is dark at a rung the previous entry had no reason for gets
  `TODO`, and a TODO is a red cell for the arm. Those need a judgement written by a human.

  IT CARRIES TEXT THE NEW NUMBERS CONTRADICT. `_size_ladder_fill_reasons` regenerates the
  evidence half of a reason only when the head is one of `SIZE_LADDER_EVIDENCE_HEADS`. An older
  reason opening "declines every call" matches nothing, so it travels verbatim onto an entry that
  now reads zero declines -- exactly the defect that function's own comment forbids. Listing
  those is not enough, because a static reason can also be RIGHT. So every integer in a carried
  reason is checked against the counts the fresh cell actually holds (served, declined, their
  sum, and each reject tally), and only the ones that quote a number the entry no longer says
  are reported as needing a hand. That is the 09-16 pass's own lesson -- a clause claiming
  "identically at all six rungs" that the new numbers contradict -- turned into a check.

  IT CAN TOUCH ANOTHER CARD. A record pass writes one card's entry, and p150a and tt-galaxy-wh
  must come out byte-identical. Asserted here rather than eyeballed, the same check
  `perf/sizeladder_p300c/reasons_boltz2.py` carried.

Usage:  reasons.py [--ref origin/main] [--card p300c] <model> [<model> ...]
"""
import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rg = _load("rg", ROOT / "scripts" / "release_gate.py")
HEADS = tuple(rg.SIZE_LADDER_EVIDENCE_HEADS)


#: The claims a reason makes ABOUT THIS CELL, as opposed to the line numbers, shapes and
#: sibling-lever counts its judgement half legitimately cites. Matching every integer in the
#: text flags all 29 of nesso1's reasons, including the ones that are right: the REBLOCK_PERMUTE
#: text names `L1_N_MIN`/`L1_N_MAX` of 288/352, source lines 338-339 and :387, a 110-core
#: measurement and REBLOCK_PERMUTE_GATED's own 192, none of which this counter holds or should.
#: So only the phrasings that assert this cell's own counts are checked.
AT_RUNG = re.compile(r"\b(\d+) calls at this rung\b")
DECLINES_ALL = re.compile(r"\bdeclines all (\d+)\b")
ZERO_HERE = re.compile(r"\b0 served AND 0 declined at this rung\b")
ZERO_EVERY = re.compile(r"\b0 (?:served|L1 blocks) AND 0 (?:declined|L1 refusals) at every rung\b")


def contradicted(reason, cell, levers, flag):
    """What this reason asserts about its own cell that the fresh entry does not say.

    `at every rung` is checked across EVERY rung of the entry, not just this one: a clause that
    claims a lever is uniformly dark and is wrong at one rung is precisely the 09-16 pass's
    defect ("identically at all six rungs" contradicted by the new numbers), and checking it
    only where it sits cannot see that.
    """
    served, declined = cell.get("served") or 0, cell.get("declined") or 0
    out = []
    for n in AT_RUNG.findall(reason):
        if int(n) not in {served, declined, served + declined} | set(
                (cell.get("rejects") or {}).values()):
            out.append(f"claims {n} calls at this rung; cell has served={served} "
                       f"declined={declined}")
    for n in DECLINES_ALL.findall(reason):
        if int(n) != declined:
            out.append(f"claims it declines all {n}; cell declined={declined}")
    if ZERO_HERE.search(reason) and (served or declined):
        out.append(f"claims 0 served AND 0 declined at this rung; cell has "
                   f"served={served} declined={declined}")
    if ZERO_EVERY.search(reason):
        busy = [r for r in levers
                if (levers[r].get(flag, {}).get("served") or 0)
                or (levers[r].get(flag, {}).get("declined") or 0)]
        if busy:
            out.append(f"claims 0 at EVERY rung; rungs {sorted(busy, key=int)} are not 0")
    return out


def self_test():
    """The checker must be able to FAIL. Without this, "0 stale" is indistinguishable from a
    function that returns nothing (`negative-control-must-break-what-check-reads`)."""
    cell = {"served": 0, "declined": 7684, "rejects": {"window_BufferType.L1": 3842}}
    levers = {"256": {"F": {"served": 0, "declined": 0}},
              "512": {"F": {"served": 5, "declined": 0}}}
    cases = [
        ("the window declining is the window working: 3842 calls at this rung ask for L1", cell, 0),
        ("the window declining is the window working: 9999 calls at this rung ask for L1", cell, 1),
        ("declines all 7684 on the band", cell, 0),
        ("declines all 560 on the band", cell, 1),
        ("no call site: 0 served AND 0 declined at this rung", cell, 1),
        ("no call site: 0 served AND 0 declined at every rung", cell, 1),
    ]
    print("SELF-TEST")
    ok = True
    for text, c, want in cases:
        got = len(contradicted(text, c, levers, "F"))
        flag = "ok" if got == want else "MISMATCH"
        ok &= got == want
        print(f"  want {want} got {got}  {flag}  {text[:62]}")
    # The "at every rung" arm must also read the OTHER rungs, not just the one it sits on.
    quiet = {"256": {"F": {"served": 0, "declined": 0}},
             "512": {"F": {"served": 0, "declined": 0}}}
    n = len(contradicted("no call site: 0 served AND 0 declined at every rung",
                         {"served": 0, "declined": 0}, quiet, "F"))
    print(f"  want 0 got {n}  {'ok' if n == 0 else 'MISMATCH'}  same clause, every rung really 0")
    ok &= n == 0
    assert ok, "the reason checker does not respond to a reason it should reject"
    print("  PASS: the checker fires on a contradicted claim and stays quiet on a true one\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--card", default="p300c")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    bad = 0
    for model in args.models:
        p = f"docs/size_ladder_baseline.d/{model}.json"
        new = json.loads((ROOT / p).read_text())
        old_raw = subprocess.run(["git", "show", f"{args.ref}:{p}"], cwd=ROOT,
                                 capture_output=True, text=True).stdout
        old = json.loads(old_raw) if old_raw.strip() else {}
        print("=" * 90)
        print(model)

        for card in sorted(new.get("cards", {})):
            if card == args.card:
                continue
            same = json.dumps(new["cards"][card], sort_keys=True) == \
                json.dumps((old.get("cards") or {}).get(card), sort_keys=True)
            print(f"  other card {card:<15} {'byte-identical' if same else 'CHANGED -- SCOPE LEAK'}")
            bad += 0 if same else 1

        levers = ((new.get("cards") or {}).get(args.card, {})
                  .get("models", {}).get(model, {}).get("levers") or {})
        todo, carried, stale = [], [], []
        for rung in sorted(levers, key=int):
            for flag, e in sorted(levers[rung].items()):
                r = e.get("reason")
                if not r:
                    continue
                if r.startswith("TODO"):
                    todo.append((rung, flag, r))
                elif not r.startswith(HEADS):
                    carried.append((rung, flag, r))
                    wrong = contradicted(r, e, levers, flag)
                    if wrong:
                        stale.append((rung, flag, r, wrong))
        print(f"  {len(todo)} TODO reason(s) -- each is a red cell until a judgement is written")
        for rung, flag, r in todo:
            print(f"    TODO    {rung:>5} {flag:<30} {r[:80]}")
        print(f"  {len(carried)} carried reason(s) the gate cannot regenerate the numbers of, "
              f"{len(stale)} of which assert a count this entry no longer holds")
        for rung, flag, r, nums in stale:
            print(f"    STALE   {rung:>5} {flag:<30}")
            for n in nums:
                print(f"            {n}")
        bad += len(todo) + len(stale)
    print("=" * 90)
    print(f"{bad} item(s) need a hand: TODOs to write, static heads to re-open, scope leaks to undo")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
