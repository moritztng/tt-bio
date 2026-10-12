#!/usr/bin/env python3
"""Re-run a recorded measurement and check it still reads what it recorded.

A file under perf/ with a `RECORDED-AT:` line (tests/test_recorded_claims.py) goes stale the moment
the code that feeds it moves, which on main is every few minutes. Whether its numbers still
reproduce needs a card, so the release gate asks it here, as the `recorded:<name>` leg, and the
release train re-stamps RECORDED-AT in the release commit when the leg passes.

The artifact says how, in these lines:
    RERUN: <argv>      run in the tree, `PY` is this interpreter; one line per run, in order
    ROW: <regex>       a line of the output that matters; its groups, joined by a space, are the row
    EXPECT: <rows>     one per RERUN, in order: that run's rows joined by " | "
    ARCH: <arch>       optional, the architectures the runs fit (gate_fanout.py reads it)
Rows name verdicts and fit (ok/FAIL per output, parameter counts, DRAM peak, losses to the digits
that held across records), never timings, so a quiet chip and a busy one agree.

    python3 scripts/rerun_record.py perf/abb3_port/model_gate_qb1c3.txt

Exit 0 when the rows match, 1 with the difference otherwise.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path


def spec(text: str) -> dict:
    """The RERUN / ROW / EXPECT / ARCH lines of an artifact."""
    got = {k: [] for k in ("RERUN", "ROW", "EXPECT", "ARCH")}
    for m in re.finditer(r"^(RERUN|ROW|EXPECT|ARCH):[ \t]*(.*?)\s*$", text, re.M):
        got[m[1]].append(m[2])
    got["ARCH"] = [a for line in got["ARCH"] for a in line.split()]
    return got


def rows(output: str, patterns: list) -> list:
    """Every output line matching a ROW pattern, as its groups joined by a space."""
    res = [re.compile(p) for p in patterns]
    out = []
    for line in output.splitlines():
        for r in res:
            m = r.search(line)
            if m:
                out.append(" ".join(g for g in m.groups() if g is not None) or m[0].strip())
                break
    return out


def main(argv: list) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    s = spec(Path(argv[0]).read_text())
    if not s["RERUN"] or not s["EXPECT"]:
        print(f"{argv[0]}: no RERUN or EXPECT lines")
        return 2
    got = []
    for cmd in s["RERUN"]:
        args = [sys.executable if a == "PY" else a for a in shlex.split(cmd)]
        print(f"$ {cmd}", flush=True)
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(p.stdout, f"rc={p.returncode}", sep="", flush=True)
        got.append(" | ".join(rows(p.stdout + f"\nrc={p.returncode}\n", s["ROW"])))
    if got == s["EXPECT"]:
        print(f"REPRODUCES: {len(got)} runs as recorded")
        return 0
    print("DIFFERS from the record (expected, then got, per run):")
    for i in range(max(len(got), len(s["EXPECT"]))):
        e = s["EXPECT"][i] if i < len(s["EXPECT"]) else "-"
        g = got[i] if i < len(got) else "-"
        print(f"{'  ' if e == g else '! '}{e}\n{'  ' if e == g else '! '}{g}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
