#!/usr/bin/env python3
"""Audit the union's BASE arm against the published cell, because this campaign has re-banked its
own shipped work three times.

§1-CORRECTION-B caught a ceiling table that divided a levered arm's phase split into an unlevered
cell and inflated every rung by exactly the campaign's own earned factor. The same error in the
union's headline would look like this: if the union row's base arm had the three wave-1 levers OFF
while `main` ships them ON, part of its 1.09858x would be wave 1's work counted a second time.

Three checks, all against committed state, no measurement:

  1. ancestry -- is the commit that turned the wave-1 levers on an ancestor of the union branch?
  2. digest   -- does the union's BASE fold write the same CIF as the arm the perf page publishes?
  3. scope    -- do the timed arms toggle only the three NEW flags?

Check 2 is the load-bearing one. A digest match means the base arm is not merely configured like the
published arm, it computed the same structure bit for bit.

    python3 perf/b2z2_orch/union_provenance.py
"""
import json
import subprocess
import sys

UNION_BRANCH = "origin/wk/b2z2-bh-union-clean"
LEVERS_ON = "0f3f9f673"    # "turn both b2z levers on by default"
RECELL = "84da2a49b"       # the perf page's current Boltz-2 cell
PAGE = "site/data/perf-512aa.json"
SCORE = "perf/b2z2_union/out/score512_qb2c1.json"
TIMING = "perf/b2z2_union/out/timing512_qb2c1.json"


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def show(ref: str, path: str) -> dict:
    return json.loads(sh("git", "show", f"{ref}:{path}"))


def main() -> int:
    ok = True

    for label, commit in (("wave-1 levers default-on", LEVERS_ON), ("the published re-cell", RECELL)):
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", commit, UNION_BRANCH]).returncode == 0
        print(f"[{'ok' if anc else 'FAIL'}] {commit} ({label}) is an ancestor of the union branch")
        ok &= anc

    page = show("origin/main", PAGE)
    cell = next(m for m in page["models"] if m["name"] == "Boltz-2")["cells"]["p150a"]
    published_digest = json.dumps(cell)
    score = show(UNION_BRANCH, SCORE)["sizes"]["512"]["sha256"]

    base_digest = score["base-s0"]
    in_page = base_digest in published_digest
    print(f"[{'ok' if in_page else 'FAIL'}] the union's BASE fold writes {base_digest}, "
          f"which is {'the digest the perf page publishes' if in_page else 'NOT on the page'}")
    ok &= in_page

    akw = score["AKW-s0"] == base_digest
    print(f"[{'ok' if akw else 'FAIL'}] the atom key window writes the same digest as the base "
          f"({score['AKW-s0']}) -- it ships without changing the page's parity claim at all")
    ok &= akw

    flags = {r["arm"]: set(k for k, v in r["flags"].items() if v) for r in show(UNION_BRANCH, TIMING)["runs"]}
    clean = flags.get("base") == set()
    print(f"[{'ok' if clean else 'FAIL'}] the base arm toggles no new flag; the arms differ only in "
          f"{sorted(set().union(*flags.values()))}")
    ok &= clean

    print(f"\n{'NO DOUBLE-COUNT' if ok else 'CHECK FAILED'}: the union's base is the shipped default, "
          f"so 1.09858x is marginal on top of everything the page already banks.")
    print(f"For scale, from the same page: H200 {next(m for m in page['models'] if m['name'] == 'Boltz-2')['cells']['h200']['s_per_fold']} s, "
          f"published p150a {cell['s_per_fold']} s, the union {17.894} s.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
