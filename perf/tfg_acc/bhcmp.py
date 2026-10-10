"""BH vs WH on the examples at one seed: per target and condition, samples, mean DockQ and the top-ranked sample's DockQ.

usage: bhcmp.py BH_SCORES.jsonl WH_SCORES.jsonl [SEED]   (score.py output for each architecture)
"""
import json
import sys


def load(p):
    return [json.loads(line) for line in open(p)]


def cell(rows, t, c, seed):
    s = [r for r in rows if r["target"] == t and r["cond"] == c and r["seed"] == seed]
    if not s:
        return None
    top = max(s, key=lambda r: r["ranking_score"])
    return len(s), round(sum(r["dockq"] for r in s) / len(s), 3), top["dockq"]


def main():
    bh, wh = load(sys.argv[1]), load(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 101
    for t in sorted({r["target"] for r in bh}):
        for c in ("unconstrained", "contact", "pocket"):
            print(t, c, "BH", cell(bh, t, c, seed), "WH", cell(wh, t, c, seed))


if __name__ == "__main__":
    main()
