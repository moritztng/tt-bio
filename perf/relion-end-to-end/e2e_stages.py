#!/usr/bin/env python3
"""Turn a refinement's log into the by-stage wall-clock split the deliverable asks for.

Two instruments, one binary, and they measure different things:

  RELION's own Timer (-DTIMING) gives the OUTER split -- expectation_1/2/6, maximization,
  writeOutput, "flatten solvent". One table is printed per ITERATION and the timer is reset each
  time, so the repeated columns below are iterations, not ranks, and they are wall seconds that sum
  to the iteration. This is the table that prices the pipeline floor. Note that the LAST iteration
  of a converged auto-refine never prints one: RELION exits after the convergence message, so a
  4-column Timer table can describe a 5-iteration run.

  TTPROF (the CTIC/CTOC bodies) gives the INNER split of the accelerated E-step -- oneParticle and
  the coarse pass, the fine pass, storeWeightedSums, getFourierTransformsAndCtfs beneath it. One
  table is printed per FOLLOWER at process exit, cumulative over the whole run, so the repeated
  columns are ranks, not iterations. The values are a SUM OVER THREADS, so they exceed the elapsed
  wall by roughly the thread count and only their shares of `oneParticle` mean anything.

So the repeated columns mean opposite things in the two tables. Mixing the two currencies, or reading
one table's columns as the other's, is the mistake this script exists to prevent.

Usage: e2e_stages.py <log> [<log> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# " expectation_6                      : 130.911 sec (130911472 microsec/operation)"
TIMER = re.compile(r"^\s*([A-Za-z_][\w /.-]*?)\s*:\s*([0-9.]+)\s*sec\s*\(")
# "TTPROF getAllSquaredDifferencesCoarse            660.0600      1104    90.40%"
TTPROF = re.compile(r"^TTPROF\s+(\S.*?)\s{2,}([0-9.]+)\s+(\d+)\s+([0-9.]+)%")
OUTER_KEYS = ("expectation", "expectation_1", "expectation_1a", "expectation_2", "expectation_6",
              "maximization", "writeOutput", "flatten solvent", "iterate")


def parse(path: Path):
    timer, ttprof = {}, {}
    for line in path.read_text(errors="ignore").splitlines():
        m = TTPROF.match(line)
        if m:
            lab, sec, n, pct = m.group(1).strip(), float(m.group(2)), int(m.group(3)), float(m.group(4))
            # Several ranks print their own table. Keep them separately keyed so a per-rank
            # disagreement is visible rather than averaged away.
            ttprof.setdefault(lab, []).append({"cpu_s": sec, "calls": n, "pct_onePart": pct})
            continue
        m = TIMER.match(line)
        if m:
            lab, sec = m.group(1).strip(), float(m.group(2))
            timer.setdefault(lab, []).append(sec)
    return timer, ttprof


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = {}
    for a in argv[1:]:
        p = Path(a)
        timer, ttprof = parse(p)
        niter = max((len(v) for v in timer.values()), default=0)
        nrank = max((len(v) for v in ttprof.values()), default=0)
        nexp = p.read_text(errors="ignore").count("Expectation iteration")
        out[p.name] = {"timer_sec": timer, "ttprof": ttprof,
                       "n_timer_tables": niter, "n_ttprof_tables": nrank,
                       "n_expectation_iterations": nexp,
                       "untimed_final_iteration": bool(nexp > niter)}
        print(f"\n########## {p.name}")
        print(f"{nexp} Expectation iterations, {niter} Timer tables"
              + ("  <-- the final converged iteration prints none, so it is NOT in the Timer sums"
                 if nexp > niter else ""))
        print(f"--- RELION Timer: {niter} table(s), ONE PER ITERATION, wall seconds ---")
        if not timer:
            print("  (no Timer table: the binary was built without -DTIMING)")
        for k in OUTER_KEYS:
            if k in timer:
                v = timer[k]
                print(f"  {k:22s} " + "  ".join(f"{x:9.3f}" for x in v))
        extra = [k for k in timer if k not in OUTER_KEYS]
        if extra:
            print(f"  ({len(extra)} further Timer tags in the json)")
        print(f"--- TTPROF: {nrank} table(s), ONE PER FOLLOWER RANK, cumulative CPU-seconds "
              "summed over threads; only the % column is comparable ---")
        if not ttprof:
            print("  (no TTPROF table: TT_RELION_PROFILE was unset or the binary lacks the instrument)")
        for k, v in sorted(ttprof.items(), key=lambda kv: -kv[1][0]["cpu_s"])[:12]:
            pcts = "/".join(f"{r['pct_onePart']:.2f}" for r in v)
            secs = "/".join(f"{r['cpu_s']:.1f}" for r in v)
            calls = "/".join(str(r["calls"]) for r in v)
            print(f"  {k:42s} {secs:>22s} s  {pcts:>14s} %  n={calls}")
    dest = Path(argv[1]).parent / "e2e_stages.json"
    dest.write_text(json.dumps(out, indent=1))
    print("\nwrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
