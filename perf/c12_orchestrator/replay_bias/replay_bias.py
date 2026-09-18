#!/usr/bin/env python3
"""Two uncontrolled biases in the c10-fold-census replay, pushing opposite ways.

The census priced 8 op classes by re-running each key STANDALONE, and Axis B of C12 is sized from
those per-key seconds. Two properties of that replay bias the price, in opposite directions, and
the census's own aggregate sanity check (10.5368 s = 70.8 % of the 14.881 s fold, "so this replay
does not overprice the fold in aggregate") cannot see either one, because a sum landing under the
whole is not a bound on its bias when 29.2 % of the fold is unaccounted for.

BIAS 1, OVERPRICE: `replay.py` allocates every operand and every output with
`ttnn.DRAM_MEMORY_CONFIG`. There is no L1 path in it at all. The fold runs many of the same ops
L1-resident, so the replay pays DRAM passes the fold does not. Measured here as replay bytes
against the fold's counted DRAM bytes: 2.23x on `multiply_`, 1.83x on `linear`, 1.47x on `matmul`,
1.32x on `add_`. The control is `add` at 1.03x and `multiply` at 1.00x -- classes the fold really
does run through DRAM, where the two figures agree, which is what says the comparison is sound.

BIAS 2, UNDERPRICE: for `linear` and `matmul` the census's PUBLISHED per-class seconds are the
`@grid110` arm -- every call handed a 110-core grid. `linear` 4.6613 s / 42.92 us per call here
reproduces the published 4.6604 s / 42.9 us; `matmul` 1.4814 s reproduces the published 1.479 s.
The bare arm is `linear` 10.7316 s / 98.81 us per call, 2.30x slower. The fold does not pass a
core_grid at these sites: C12's own refuted list records that only 5 live sites accept one, 98.9 %
of that traffic is a single AdaLN pair, and adding it across all five measured -0.1559 s in the
fold against a 4.908 s replay gap.

So `linear`'s 4.6604 s is a grid-110 price on DRAM operands, and the fold runs neither. The two
biases partly cancel, which is how the aggregate came out plausible. A 0.3 s lever cannot be sized
against a number built that way -- only a profiled fold can settle it.

Run from a tt-bio checkout with the artifacts reachable by `git show`:
    python3 perf/c12_orchestrator/replay_bias/replay_bias.py
"""
import json
import subprocess
from collections import defaultdict

CENSUS = "origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/replay.json"
OPCENSUS = "origin/main:perf/roof_launch/op_census_512.json"
MAP = {"linear": ["ttnn.linear"], "matmul": ["ttnn.matmul"],
       "multiply_": ["ttnn.multiply_"], "add_": ["ttnn.add_"],
       "multiply": ["ttnn.multiply"], "add": ["ttnn.add"],
       "layer_norm_w": ["ttnn.layer_norm"]}
PUBLISHED = {"linear": 4.6604, "matmul": 1.479, "multiply_": 1.751,
             "layer_norm_w": 1.357, "add_": 0.847, "add": 0.140, "multiply": 0.126}


def show(ref):
    return json.loads(subprocess.run(["git", "show", ref], capture_output=True,
                                     check=True, text=True).stdout)


def collect(rows, arm, grid):
    sel = [r for r in rows
           if r["arm"] == arm and str(r["label"]).endswith("@grid110") == grid]
    calls = sum(r["calls"] or 0 for r in sel)
    secs = sum((r.get("s_per_call_qualified") or 0) * (r["calls"] or 0) for r in sel)
    B = sum((r["min_bytes_per_call"] or 0) * (r["calls"] or 0) for r in sel)
    return len(sel), calls, secs, B


def main():
    d, oc = show(CENSUS), show(OPCENSUS)["by_op"]
    rows = [r for r in d["rows"] if not r["arm"].startswith("roof")]
    arms = sorted({r["arm"] for r in rows})
    out = []
    print(f"{'class':14}{'calls':>9}{'bare_s':>9}{'grid_s':>9}{'pub_s':>8}"
          f"{'arm_pub':>9}{'replayTB':>10}{'foldTB':>9}{'byte_x':>8}")
    for arm in arms:
        nb, cb, sb, Bb = collect(rows, arm, False)
        ng, cg, sg, Bg = collect(rows, arm, True)
        if arm == "layer_norm":       # counted with layer_norm_w in op_census_512.json
            continue
        if arm == "layer_norm_w":
            nl, cl, sl, Bl = collect(rows, "layer_norm", False)
            cb, sb, Bb = cb + cl, sb + sl, Bb + Bl
        fold = sum(oc[n]["B"] for n in MAP.get(arm, []) if n in oc)
        pub = PUBLISHED.get(arm)
        # which arm does the published figure reproduce?
        which = "-"
        if pub is not None:
            which = "grid110" if (sg and abs(sg - pub) < abs(sb - pub)) else "bare"
        bx = Bb / fold if fold else float("nan")
        print(f"{arm:14}{cb:9,.0f}{sb:9.4f}{(sg if cg else 0):9.4f}"
              f"{(pub if pub else 0):8.3f}{which:>9}{Bb/1e12:10.4f}{fold/1e12:9.4f}{bx:8.2f}")
        out.append({"class": arm, "calls": cb, "bare_s": sb, "grid110_s": sg if cg else None,
                    "published_s": pub, "published_arm": which,
                    "replay_B": Bb, "fold_dram_B": fold, "byte_ratio": bx})
    p = __file__.rsplit("/", 1)[0] + "/replay_bias.json"
    json.dump({"census": CENSUS, "op_census": OPCENSUS,
               "bias1": "replay allocates all operands+outputs DRAM_MEMORY_CONFIG; fold runs many "
                        "of these L1-resident -> replay overprices",
               "bias2": "published linear/matmul seconds are the @grid110 arm; the fold does not "
                        "pass a core_grid at these sites -> published underprices a bare fold site",
               "rows": out}, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
