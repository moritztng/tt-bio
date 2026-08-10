#!/usr/bin/env python3
"""Per-model completeness census of the N=512 rung, with the POWER_MIN guard.

Why this gate exists. abag_xm_deepn_analysis.py builds its adjacent-rung gain CIs -- the
stop-rule comparator, and the number this whole campaign exists to produce -- only over
rungs carrying at least POWER_MIN=50 targets:

    ns_powered = [n for n in ns if pts[n]["n_targets"] >= POWER_MIN]
    for lo, hi in zip(ns_powered, ns_powered[1:]): ...

A 512 rung with 49 complete targets is therefore NOT silently degraded, it is silently
ABSENT: the ladder table still prints an N=512 row, and the gain table simply has no
"256->512" entry. Verified against the real main() on synthetic pools: 49 complete targets
emits ['64->128','128->256'], 50 emits ['64->128','128->256','256->512'].

A target counts only with all 8 chunks (512/64) present, because the analysis applies the
same rule -- a 7-of-8 pool is a 448-sample oracle mislabelled N=512, so it is dropped
outright rather than reported short.

Run before the harvest and again before the analysis. Non-zero exit means at least one
model would lose its 256->512 gain CI.

    python3 n512_power_census.py ~/mthuening/p31 ~/mthuening/p32
"""
import collections
import json
import pathlib
import sys

POWER_MIN = 50          # abag_xm_deepn_analysis.py deep-block, keep in sync
CHUNKS = frozenset(range(8))


def scan(dirs):
    """(model, target) -> set of chunk indices present at rung 512."""
    have = collections.defaultdict(set)
    for d in dirs:
        base = pathlib.Path(d).expanduser()
        for name, ok in (("results.jsonl", True), ("reused_chunks.jsonl", False)):
            f = base / name
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("rung") != 512:
                    continue
                if ok and r.get("rc") != 0:
                    continue
                have[(r["model"], r["target"])].add(r.get("chunk"))
    return have


def main(argv):
    dirs = argv[1:] or ["~/mthuening/p31"]
    have = scan(dirs)
    if not have:
        print("no rung-512 records found in " + " ".join(dirs))
        return 2

    complete = collections.Counter()
    partial = collections.defaultdict(list)
    seen = collections.Counter()
    for (model, target), chunks in have.items():
        seen[model] += 1
        got = chunks & CHUNKS
        if len(got) == 8:
            complete[model] += 1
        else:
            partial[model].append((target, sorted(CHUNKS - got)))

    print("windows: " + " ".join(dirs))
    print()
    print("%-16s %8s %10s %8s %7s  %s"
          % ("model", "targets", "complete", "partial", "margin", "verdict"))
    failed = []
    for model in sorted(seen):
        n = complete[model]
        margin = n - POWER_MIN
        verdict = "OK" if n >= POWER_MIN else "BELOW POWER_MIN -- no 256->512 gain CI"
        if n < POWER_MIN:
            failed.append(model)
        print("%-16s %8d %10d %8d %+7d  %s"
              % (model, seen[model], n, len(partial[model]), margin, verdict))

    print()
    print("residual worklist (targets one or more chunks short, the cheapest path to margin):")
    for model in sorted(partial):
        rows = sorted(partial[model], key=lambda r: (len(r[1]), r[0]))
        near = sum(1 for _t, m in rows if len(m) == 1)
        print("  %-16s %3d partial, %d of them a single chunk short" % (model, len(rows), near))
        for target, missing in rows[:12]:
            print("      %-6s missing %s" % (target, ",".join(str(c) for c in missing)))
        if len(rows) > 12:
            print("      ... %d more" % (len(rows) - 12))

    if failed:
        print()
        print("GATE FAIL: " + ", ".join(failed)
              + " would produce no 256->512 gain CI. Fold the single-chunk-short targets"
                " above first: each one moves a model a full target closer to POWER_MIN.")
        return 1
    print()
    print("GATE PASS: every model clears POWER_MIN=%d complete 512 targets." % POWER_MIN)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
