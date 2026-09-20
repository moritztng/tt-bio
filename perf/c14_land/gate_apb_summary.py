#!/usr/bin/env python3
"""Read the APB gate: per-arm verdict beside per-arm proof the flag actually ran.

A gate ledger says PASS or FAIL. It does not say whether the arm executed the flag under test, and
this row has one measured case where that distinction decided a merge: TT_BIO_MM_SHORT_M_BW passed
the l1-budget arm across three core grids while making the Boltz-2 structure grid-dependent,
because that arm folds protenix-v2 and protenix-v2 never calls it. So this prints the two together
and refuses to let a green stand on its own.

The interesting column is COVERAGE, not the verdict. `[0, 0]` on esmc-300m is correct and expected
-- an embedding model has no AttentionPairBias to re-lane. `[0, 0]` on a structure model that was
supposed to exercise the flag is the finding. This tool does not guess which is which; it prints
the counts so the reader can see exactly which models a green gate covered for this flag, which is
the question a shared-code default flip actually turns on.

    python3 perf/c14_land/gate_apb_summary.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Arms this instrument CANNOT see, with the reason, so a zero from them is never read as a
#: measured zero. `scripts/lever_census.py` generates its own `sitecustomize.py` and overwrites
#: PYTHONPATH for the fold it shells out (`lever_census.py:621`). `sitecustomize` is a singleton
#: hook -- Python imports the first one it finds -- so the inner tool silently shadows this one and
#: no counter is written by any process of that arm. Two instruments that both ride `sitecustomize`
#: cannot nest.
SITECUSTOMIZE_BLIND = {"rfd3-fusion"}
LEDGER = HERE / "gate_apb_ledger.txt"
FIRING = HERE / "gate_apb_firing"


def firing_for(arm: str):
    """(served, declined, processes, flag_seen) summed over every process of that arm."""
    served = declined = procs = 0
    flag = None
    saw_counter = False
    for f in sorted(glob.glob(str(FIRING / (arm + ".*")))):
        try:
            r = json.load(open(f))
        except Exception:                                             # noqa: BLE001
            continue
        procs += 1
        flag = flag or r.get("env_flags", {}).get("TT_BIO_APB_CONCAT_HEADS")
        c = r.get("counter")
        if c:
            saw_counter = True
            served += c[0]
            declined += c[1]
    return served, declined, procs, flag, saw_counter


def rc_of(status: str):
    """The exit code carried by a ledger status field, or None if it carries none."""
    m = re.search(r"rc=(-?\d+)", status)
    return int(m.group(1)) if m else None


def main() -> int:
    if not LEDGER.exists():
        print("no ledger yet")
        return 1
    # An arm can appear more than once: a re-run after a device-lease collision or a box
    # problem writes a second line. Keep the LAST attempt per arm, because that is the one that
    # stands, and parse `rc=N` out of the status rather than comparing the whole field -- a line
    # carrying a re-run note would otherwise never equal "rc=0" and would be counted as a red.
    # That is the same shape as reading a refused argument as an arm: the ledger is a record, and
    # a reader that only matches its happy-path format quietly miscounts the record.
    latest = {}
    for line in LEDGER.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ts, arm, status = parts[0], parts[1], parts[2]
        latest[arm] = (ts, arm, status)

    rows = []
    for ts, arm, status in latest.values():
        served, declined, procs, flag, saw = firing_for(arm)
        rows.append((ts, arm, status, served, declined, procs, flag, saw))
    rows.sort(key=lambda r: r[0])

    print(f"{'arm':<18}{'rc':<5}{'served':>9}{'declined':>10}{'procs':>7}  flag  coverage")
    covered, uncovered, blind_green = [], [], []
    for ts, arm, status, served, declined, procs, flag, saw in rows:
        if served > 0:
            cov = "EXECUTES the flag"
            covered.append(arm)
        elif not saw:
            cov = ("no counter recorded -- sitecustomize collision, see KNOWN_BLIND below"
                   if arm in SITECUSTOMIZE_BLIND
                   else "no counter recorded (tt_bio never loaded in any process)")
            uncovered.append(arm)
        else:
            cov = "reached no AttentionPairBias site"
            uncovered.append(arm)
        if rc_of(status) == 0 and served == 0:
            blind_green.append(arm)
        print(f"{arm:<18}{str(rc_of(status)):<5}{served:>9}{declined:>10}{procs:>7}  "
              f"{str(flag):<5} {cov}")

    print()
    print(f"arms recorded            : {len(rows)}")
    print(f"green                    : {sum(1 for r in rows if rc_of(r[2]) == 0)}")
    reds = [r[1] for r in rows if rc_of(r[2]) not in (0, None)]
    if reds:
        print(f"RED                      : {len(reds)}  {reds}")
    notes = [(r[1], r[2]) for r in rows if r[2].strip() != f"rc={rc_of(r[2])}"]
    for arm, status in notes:
        print(f"  note on {arm}: {status}")
    print(f"EXECUTED the flag        : {len(covered)}  {covered}")
    print(f"did NOT execute the flag : {len(uncovered)}  {uncovered}")
    hidden = sorted(a for a in uncovered if a in SITECUSTOMIZE_BLIND)
    if hidden:
        print()
        print(f"KNOWN_BLIND: {hidden} -- scripts/lever_census.py puts its OWN generated "
              "sitecustomize on PYTHONPATH for the fold it shells out, and sitecustomize is a "
              "singleton hook, so this instrument is shadowed there. Those arms' zeros are NOT "
              "measured zeros. (The same model's plain `rfd3` arm did record [0, 0] with the "
              "instrument attached, so RFD3 not reaching AttentionPairBias is known from there.)")
    if blind_green:
        print()
        print("READ THIS BEFORE QUOTING THE GATE. These arms are green and never ran the flag, so "
              "they are evidence that the tree is healthy, NOT that this flag is safe on them:")
        print(f"  {blind_green}")
        print("For a model with no AttentionPairBias that is correct and expected. For one that "
              "has it, the flag did not reach the site and the arm is not coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
