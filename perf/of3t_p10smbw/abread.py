#!/usr/bin/env python3
"""One arm's held-out numbers, and which MODE each bimodal target fell into.

7ohe takes ~10.8 or ~18.9 and 7vus ~2.48 or ~5.64, with nothing between (`of3t-p10trainout`'s
ten-run table). The midpoint of each pair is the only threshold that needs choosing, and a value
landing near it would be news in itself, so the distance to the midpoint is printed rather than
hidden inside a boolean.

    abread.py <arm.json> <variant>          one arm
    abread.py --table <dir>                 every arm in a directory, and the two rates
"""
import glob
import json
import os
import sys

#: (fine, blown) from of3t-p10trainout's ten device-native runs. 7kud and 7fb8 never move.
MODES = {"7ohe": (10.8, 18.9), "7vus": (2.48, 5.64)}


def mode(name, v):
    lo, hi = MODES[name]
    mid = (lo + hi) / 2.0
    return ("blown" if v > mid else "fine"), abs(v - mid)


def read(path):
    d = json.load(open(path))
    per = {t["pdb_id"]: t["loss"] for t in (d.get("eval_after") or {}).get("targets", [])}
    return {
        "ok": d.get("ok"),
        "wall_s": d.get("wall_s"),
        "aiclk_line": d.get("aiclk_line"),
        "before": (d.get("eval_before") or {}).get("mean_loss"),
        "after": (d.get("eval_after") or {}).get("mean_loss"),
        "targets": per,
        "modes": {k: mode(k, v)[0] for k, v in per.items() if k in MODES},
    }


def one(path, variant):
    r = read(path)
    print(f"{variant} ok={r['ok']} wall={r['wall_s']}s")
    print(f"  before {r['before']}  after {r['after']}")
    for k in sorted(r["targets"]):
        v = r["targets"][k]
        if k in MODES:
            m, gap = mode(k, v)
            print(f"  {k} {v:.6f}  {m}  |v-mid| {gap:.3f}")
        else:
            print(f"  {k} {v:.6f}")


def table(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "arm_*.json"))):
        vp = p[:-5] + ".variant"
        variant = open(vp).read().strip() if os.path.exists(vp) else "?"
        try:
            rows.append((os.path.basename(p)[4:-5], variant, read(p)))
        except Exception as e:
            print(f"{os.path.basename(p)}: unreadable, {e}")
    hdr = ["7ohe", "7vus", "7kud", "7fb8"]
    print("| run | arm | before | after | " + " | ".join(hdr) + " | modes |")
    print("|---|---|---|---|" + "---|" * (len(hdr) + 1))
    for tag, variant, r in rows:
        cells = []
        for k in hdr:
            v = r["targets"].get(k)
            cells.append("-" if v is None else f"{v:.6f}")
        mm = " ".join(f"{k}:{r['modes'][k]}" for k in sorted(r["modes"]))
        bf = "-" if r["before"] is None else f"{r['before']:.6f}"
        af = "-" if r["after"] is None else f"{r['after']:.6f}"
        print(f"| {tag} | {variant} | {bf} | {af} | " + " | ".join(cells) + f" | {mm} |")

    print()
    for arm in ("fix", "base"):
        rs = [r for _, v, r in rows if v == arm]
        n = len(rs)
        if not n:
            continue
        for k in MODES:
            have = [r for r in rs if k in r["modes"]]
            blown = sum(1 for r in have if r["modes"][k] == "blown")
            print(f"{arm}: {k} blown {blown} of {len(have)}")
        both = sum(1 for r in rs if r["modes"] and all(m == "blown" for m in r["modes"].values()))
        none = sum(1 for r in rs if r["modes"] and all(m == "fine" for m in r["modes"].values()))
        means = [r["after"] for r in rs if r["after"] is not None]
        print(f"{arm}: n={n} blown-both {both} fine-both {none} mixed {n - both - none}"
              + (f" mean-of-means {sum(means)/len(means):.6f}" if means else ""))
        befores = sorted({round(r["before"], 12) for r in rs if r["before"] is not None})
        print(f"{arm}: eval_before distinct values {befores}")


if __name__ == "__main__":
    if sys.argv[1] == "--table":
        table(sys.argv[2])
    else:
        one(sys.argv[1], sys.argv[2])
