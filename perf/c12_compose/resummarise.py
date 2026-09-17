#!/usr/bin/env python3
"""Recompute the summary from a fold_compose.py JSON's `runs` list.

A session that is cut short still has every fold it completed on disk: the harness dumps after each
one. This rebuilds the summary from those, so a truncated session is a smaller result rather than no
result. It imports the SAME summarise() the harness uses, so a number recomputed here cannot drift
from a number the harness printed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_compose import summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    d = json.loads(a.json.read_text())
    arms = d["env"]["protocol"]["arms"]
    warm = [r for r in d["runs"] if not r["cold"]]
    have = {x for x in {r["arm"] for r in warm}}
    missing = [x for x in dict.fromkeys(arms) if x not in have]
    if missing:
        print(f"  arms with no warm fold yet: {missing}")
        arms = [x for x in arms if x in have]
    if "base" not in have:
        print("  no warm base fold: nothing to base a ratio on")
        return 1

    class A:
        mhz = d["env"].get("clock_forced_mhz", 1350)
    s = summarise(d["runs"], arms, A)
    print(json.dumps(s, indent=1))
    if a.write:
        d["summary_recomputed"] = s
        a.json.write_text(json.dumps(d, indent=1))
        print(f"  written into {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
