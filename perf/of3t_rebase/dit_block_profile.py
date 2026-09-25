#!/usr/bin/env python3
"""Is the DiT's backward defect ONE block or spread across the stack? (amendment 27(2))

The trunk was resolved by a per-block ladder: capture each block's boundary, drive one block
with its own captured cotangent, read `norm_ratio` and `cos` per track. Doing that for the DiT
needs a new capture and a new arm. But the question amendment 27 actually asks first -- "if the
DiT shows the same 'one block is different' shape, we will know in one capture; if it is
spread, that is a different defect and worth knowing early" -- can be answered from the
diffusion-scope gradient that already runs in about two minutes, by grouping its per-tensor rows
by block.

That is not the same measurement as the trunk ladder and must not be quoted as one. Here every
block is differentiated as part of one whole-module backward, so a block's error includes
whatever arrived from the blocks above it; the trunk arms isolate each block behind its own
captured cotangent. This profile can say "one block or spread" and "which blocks carry the
mass". It cannot say "the defect is GENERATED at block N" -- only an isolated arm can.

Mass-weighted throughout, because a median over tensors weights a 6e-17 layer-norm bias the
same as a 1.7e-01 projection.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
from pathlib import Path

BAR = 5.0e-02
A14 = 1e-12
BLOCK = re.compile(r"diffusion_transformer\.blocks\.(\d+)\.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="perf/of3t_rebase/device_gradient_043pt.json")
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_rebase/dit_block_profile.json"))
    a = ap.parse_args()

    d = json.loads(Path(a.report).read_text())
    pt = d.get("per_tensor")
    if not pt:
        raise SystemExit(f"{a.report} carries no per_tensor rows -- re-run device_gradient.py "
                         f"with --dump-per-tensor")

    rows = [r for r in pt if r["ref_norm"] >= A14]
    total_sq = sum(r["ref_norm"] ** 2 for r in rows)

    by_block: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        m = BLOCK.search(r["tensor"])
        by_block[f"dit_block_{int(m.group(1)):02d}" if m else "OUTSIDE_dit_blocks"].append(r)

    out = []
    for k in sorted(by_block):
        g = by_block[k]
        sq = sum(r["ref_norm"] ** 2 for r in g)
        nr = [r["norm_ratio"] for r in g if r.get("norm_ratio") is not None]
        cs = [r["cos"] for r in g if r.get("cos") is not None]
        passing = sum(r["ref_norm"] ** 2 for r in g if r["rel_l2"] <= BAR)
        out.append({
            "group": k, "n": len(g),
            "median_rel": st.median(r["rel_l2"] for r in g),
            "norm_ratio_median": st.median(nr) if nr else None,
            "cos_median": st.median(cs) if cs else None,
            "mass_share": sq / total_sq if total_sq else None,
            "passing_mass_share_within_group": passing / sq if sq else None,
        })

    hdr = ("%-22s %4s %-11s %-9s %-10s %-9s %-9s"
           % ("group", "n", "median rel", "r med", "cos med", "mass %", "pass %"))
    print(hdr); print("-" * len(hdr))
    for r in sorted(out, key=lambda r: -(r["mass_share"] or 0)):
        print("%-22s %4d %-11.4e %-9.4f %-10.6f %-9.3f %-9.3f"
              % (r["group"], r["n"], r["median_rel"], r["norm_ratio_median"] or float("nan"),
                 r["cos_median"] or float("nan"), 100 * (r["mass_share"] or 0),
                 100 * (r["passing_mass_share_within_group"] or 0)))

    blocks = [r for r in out if r["group"].startswith("dit_block_")]
    if blocks:
        meds = [r["median_rel"] for r in blocks]
        worst = max(blocks, key=lambda r: r["median_rel"])
        print("\n%d DiT blocks: median rel spans %.4e .. %.4e (%.2fx), worst is %s"
              % (len(blocks), min(meds), max(meds), max(meds) / min(meds), worst["group"]))
        print("SHAPE: %s" % ("ONE BLOCK -- the worst is >3x the next"
                             if sorted(meds)[-1] > 3 * sorted(meds)[-2]
                             else "SPREAD -- no single block dominates"))
    a.out.write_text(json.dumps({"bar": BAR, "groups": out}, indent=1) + "\n")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
