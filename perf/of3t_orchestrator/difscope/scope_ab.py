#!/usr/bin/env python3
"""What the softmax-backward repair does to the WHOLE diffusion scope, not just its worst leaf.

CPU only. No card. No new run. Reads the two per-tensor dumps `of3t-lnaffine` pushed for its
matched A/B (`origin/wk/of3t-lnaffine`, 523 tensors on each arm, byte-equal forward).

WHY
---
The row reported **333x** on one leaf, `attention_pair_bias.layer_norm_a.layer_norm_s.weight`. That
is the right headline for the question it was asked. It leaves two scope-level questions open that
D30, D58 and D56 all turn on:

  1. what happens to the diffusion scope's error as a whole, and
  2. whether the error stays concentrated after the repair.

Both are three lines off artifacts already on the branch. The answers are 198.95x and no.

THE CAVEAT THAT MUST TRAVEL WITH THE HEADLINE
---------------------------------------------
The mass-weighted reading improves 14.10x and the MEDIAN barely moves -- 0.7316 to 0.7055 -- with
**523 of 523 tensors still over the 5.0e-02 per-tensor bar on both arms**. The repair fixes the
CONCENTRATION, not the broad floor. Reporting the 198.95x without that is the A23 error in the
other direction: a mass-weighted statistic hiding a per-tensor one.
"""
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

BRANCH = "origin/wk/of3t-lnaffine"
ARMS = {"CTRL": "perf/of3t_lnaffine/device_gradient_ctrl_all_per_tensor.json",
        "RENORM": "perf/of3t_lnaffine/device_gradient_renorm_all_per_tensor.json"}
OUT = Path(__file__).with_name("DIFFUSION_SCOPE_AB.json")


def leaf_of(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[3:]) if len(parts) > 3 and "blocks" in parts else name


def load(path):
    r = subprocess.run(["git", "show", f"{BRANCH}:{path}"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"REFUSING: cannot read {BRANCH}:{path} -- fetch first")
        sys.exit(2)
    return json.loads(r.stdout)["per_tensor"]


def main() -> int:
    out = {}
    for arm, path in ARMS.items():
        rows = load(path)
        em = sum(r["ref_norm"] ** 2 * r["rel_l2"] ** 2 for r in rows)
        rm = sum(r["ref_norm"] ** 2 for r in rows)
        by = collections.Counter()
        for r in rows:
            by[leaf_of(r["tensor"])] += r["ref_norm"] ** 2 * r["rel_l2"] ** 2
        rels = sorted(r["rel_l2"] for r in rows)
        out[arm] = {
            "n": len(rows), "error_mass": em, "reference_mass": rm,
            "mass_weighted_rel": math.sqrt(em / rm),
            "median_rel": rels[len(rels) // 2],
            "over_per_tensor_bar": sum(1 for r in rows if r["rel_l2"] > 5e-2),
            "top_leaves_pct_of_error": {l: 100 * v / em for l, v in by.most_common(5)},
        }

    c, n = out["CTRL"], out["RENORM"]
    # The comparison is only sound if the two arms score the same reference. Asserted, not assumed.
    if c["n"] != n["n"] or abs(c["reference_mass"] - n["reference_mass"]) / c["reference_mass"] > 1e-12:
        print(f"REFUSING: the arms do not score the same reference "
              f"({c['n']} vs {n['n']} tensors, reference mass {c['reference_mass']!r} vs "
              f"{n['reference_mass']!r}) -- a ratio between them would not be a measurement of the "
              f"lever, which is the trap of3t-lnaffine avoided against the pass-155 artifact")
        return 1

    out["comparison"] = {
        "error_mass_ratio": c["error_mass"] / n["error_mass"],
        "mass_weighted_rel_ratio": c["mass_weighted_rel"] / n["mass_weighted_rel"],
        "reference_mass_relative_difference": 0.0,
        "caveat": "the median barely moves and 523 of 523 tensors are over the 5.0e-02 per-tensor "
                  "bar on BOTH arms: the repair fixes the concentration, not the broad floor",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    for arm in ("CTRL", "RENORM"):
        a = out[arm]
        print(f"{arm:7s} error_mass {a['error_mass']:.6e}  mass-weighted rel "
              f"{a['mass_weighted_rel']:.6e}  median {a['median_rel']:.6f}  "
              f"over bar {a['over_per_tensor_bar']}/{a['n']}")
    print(f"\nerror mass        {out['comparison']['error_mass_ratio']:.2f}x")
    print(f"mass-weighted rel {out['comparison']['mass_weighted_rel_ratio']:.2f}x better")
    print("\nconcentration, share of each arm's own error mass:")
    for arm in ("CTRL", "RENORM"):
        print(f"  {arm}")
        for l, v in out[arm]["top_leaves_pct_of_error"].items():
            print(f"    {v:7.3f} %  {l[:62]}")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
