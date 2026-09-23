#!/usr/bin/env python3
"""The other place an inference fold could pay for the gate: importing the engine.

`gate_call_cost.py` covers the per-call cost. A fold also pays module import once, and the gate
moved a function out of `tt_bio/tenstorrent.py` into `tt_bio/autograd.py`, so import is the only
other inference-path scope reachable without a card. It should be unchanged: the training
imports were already inside the function body before the move, so nothing left or entered module
scope on the inference path.

This is a WEAK bound and says so. Subprocess start dominates, the A/A floor lands near a quarter
of a second, and all it can rule out is a regression larger than that floor. It is reported
because the alternative is asserting the scope is free without looking at it.

  usage: import_cost.py --base-tree <checkout of 6d7f32dc0> --tree <this worktree> --python <py>
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def once(py, tree, env_base):
    t = time.perf_counter()
    r = subprocess.run([py, "-c", "import tt_bio.tenstorrent"], cwd=tree,
                       env=dict(env_base, PYTHONPATH=tree), capture_output=True, text=True)
    d = time.perf_counter() - t
    if r.returncode:
        raise SystemExit("import failed in %s:\n%s" % (tree, r.stderr[-1200:]))
    return d


def q95(xs):
    s = sorted(abs(x) for x in xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-tree", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    base, tree = Path(a.base_tree), Path(a.tree)
    bt, gt = (base / "tt_bio/tenstorrent.py"), (tree / "tt_bio/tenstorrent.py")
    if not bt.is_file() or not gt.is_file():
        raise SystemExit("REFUSING: one of the trees has no tt_bio/tenstorrent.py")
    if "host_softmax_hook" in bt.read_text():
        raise SystemExit("REFUSING: --base-tree already has the gate, so this would be an A/A "
                         "wearing an A/B's label. Use a checkout of 6d7f32dc0.")
    if "host_softmax_hook" not in gt.read_text():
        raise SystemExit("REFUSING: --tree does not have the gate")

    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
           "TT_METAL_HOME": str(Path.home() / "tt-metal"), "TT_VISIBLE_DEVICES": ""}
    legs = {"AA_base_1": base, "AA_base_2": base, "AA_gated_1": tree, "AA_gated_2": tree}
    s = {k: [] for k in legs}
    order = list(legs)
    for i in range(a.rounds):
        for k in (order if i % 2 == 0 else order[::-1]):
            s[k].append(once(a.python, str(legs[k]), env))

    aa_b = [x - y for x, y in zip(s["AA_base_1"], s["AA_base_2"])]
    aa_g = [x - y for x, y in zip(s["AA_gated_1"], s["AA_gated_2"])]
    ab = [(g1 + g2) / 2 - (b1 + b2) / 2 for g1, g2, b1, b2 in
          zip(s["AA_gated_1"], s["AA_gated_2"], s["AA_base_1"], s["AA_base_2"])]
    floor, med = max(q95(aa_b), q95(aa_g)), statistics.median(ab)
    readable = abs(med) > floor
    rep = {"base_median_s": round(statistics.median(s["AA_base_1"] + s["AA_base_2"]), 4),
           "gated_median_s": round(statistics.median(s["AA_gated_1"] + s["AA_gated_2"]), 4),
           "AB_median_s": round(med, 4), "AA_floor_s": round(floor, 4),
           "readable_above_the_floor": readable, "slower": readable and med > 0,
           "rounds": a.rounds,
           "bound": ("weak by construction: subprocess start dominates, so this rules out a "
                     "regression larger than the %.4f s floor and nothing finer" % floor),
           "verdict": ("REGRESSION: import is %.4f s slower, above the floor" % med
                       if readable and med > 0 else
                       "not slower: A/B %.4f s is inside the %.4f s A/A floor" % (med, floor))}
    print(json.dumps(rep, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0 if not (readable and med > 0) else 3


if __name__ == "__main__":
    sys.exit(main())
