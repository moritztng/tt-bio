#!/usr/bin/env python3
"""Where does block 47's factor enter -- at the block boundary, or inside it?

Amendment 16(a) sets the test: "if tri_mul, tri_att, transitions and attn_pair_bias all read
1.18, it enters at the block boundary; if only some do, it enters inside." This answers it from
the identifiability fields the arms already carry (`norm_ratio`, `cos`), so it costs no card.

Grouped by SUB-MODULE rather than by leaf op, because the question is about where in the block's
dataflow the factor lives, and by the reference's own gradient mass, because a median over
tensors weights a 6e-17 layer-norm bias the same as a 1.7e-01 projection.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

A14 = 1e-12


def submodule(key: str) -> str:
    p = key.split(".")
    return f"{p[0]}.{p[1]}" if p[0] == "pair_stack" else p[0]


def track(sub: str) -> str:
    return "pair" if sub.startswith("pair_stack") else "single"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[
        "perf/of3t_rebase/instrument_a_bundle_A16_block0_tbshipped.json",
        "perf/of3t_rebase/instrument_a_bundle_A16_block23_tbshipped.json",
        "perf/of3t_rebase/instrument_a_bundle_A16_block47_tbshipped.json",
        "perf/of3t_rebase/instrument_a_bundle_A16_block47_tboff.json"])
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_rebase/factor_by_submodule.json"))
    a = ap.parse_args()

    rep = {"instrument": "block 47's factor, resolved by sub-module and by track",
           "question": "amendment 16(a): all sub-modules at 1.18 means it enters at the block "
                       "boundary; only some means it enters inside",
           "a14_floor": A14, "arms": {}}
    for f in a.arms:
        d = json.loads(Path(f).read_text())
        tag = Path(f).stem.replace("instrument_a_bundle_", "")
        rows = [r for r in d["per_parameter"]
                if r["ref_norm"] >= A14 and r["norm_ratio"] is not None]
        g = collections.defaultdict(list)
        for r in rows:
            g[submodule(r["key"])].append(r)
        subs = {}
        for name, v in sorted(g.items()):
            rr = [x["norm_ratio"] for x in v]
            subs[name] = {
                "track": track(name), "n": len(v),
                "norm_ratio_median": st.median(rr),
                "norm_ratio_min": min(rr), "norm_ratio_max": max(rr),
                "cos_median": st.median([x["cos"] for x in v]),
                "mass_share_of_arm": sum(x["ref_norm"] ** 2 for x in v)
                / sum(x["ref_norm"] ** 2 for x in rows),
            }
        by_track = {}
        for tk in ("pair", "single"):
            v = [x for name, vv in g.items() if track(name) == tk for x in vv]
            if v:
                rr = [x["norm_ratio"] for x in v]
                by_track[tk] = {"n": len(v), "norm_ratio_median": st.median(rr),
                                "cos_median": st.median([x["cos"] for x in v])}
        rep["arms"][tag] = {"by_submodule": subs, "by_track": by_track}
        print(f"== {tag}")
        for name, s in sorted(subs.items(), key=lambda kv: -kv[1]["norm_ratio_median"]):
            print(f"   {name:26s} {s['track']:6s} n={s['n']:2d}  r {s['norm_ratio_median']:7.3f} "
                  f"[{s['norm_ratio_min']:.3f}, {s['norm_ratio_max']:.3f}]  "
                  f"cos {s['cos_median']:7.4f}  mass {100*s['mass_share_of_arm']:5.1f} %")
        print(f"   {'BY TRACK':26s}        "
              + "  ".join(f"{k} r {v['norm_ratio_median']:.3f} (n={v['n']})"
                          for k, v in by_track.items()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print("\n->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
