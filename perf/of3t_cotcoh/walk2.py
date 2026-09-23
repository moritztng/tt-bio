#!/usr/bin/env python3
"""of3t-cotcoh pass 2: the absolute by-block curves on all 48 blocks, the entry-vs-accumulation
test the amendment set, and the two levers priced against the arm that changes nothing.

Three things the amendment asked for and one this row owes:

  * both ABSOLUTE curves over all 48 blocks, ours and upstream's own bf16 autocast, each against
    the same published float64, rather than the twelve-block ratio in
    `perf/of3t_orchestrator/blockcurve/ENTRY_OR_ACCUMULATION.json`;
  * the log2-per-block slope of each, and the mean `over_upstream` at the entry blocks against
    the exit blocks, which is the statistic the prior is stated in;
  * the levers: `promote_first` and `cot_fp32` against `none`, at trunk scope and per tensor,
    with the `none` arm's reproduction of the banked 0.9349175217825587 read FIRST.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess


def slope_log2(xs, ys):
    """Least squares slope of log2(y) against x, over the points where y > 0."""
    p = [(x, math.log2(y)) for x, y in zip(xs, ys) if y and y > 0]
    n = len(p)
    if n < 3:
        return None
    mx = sum(x for x, _ in p) / n
    my = sum(y for _, y in p) / n
    den = sum((x - mx) ** 2 for x, _ in p)
    return sum((x - mx) * (y - my) for x, y in p) / den if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--none", required=True)
    ap.add_argument("--promote", default="")
    ap.add_argument("--cotfp32", default="")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--none-pertensor", default="")
    ap.add_argument("--promote-pertensor", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    N = json.load(open(a.none))
    U = json.load(open(a.upstream))
    ours = N["TRUNK"]["by_block"]
    ups = U["by_block"]
    blocks = sorted(int(b) for b in ours if b in ups)

    rows = []
    for b in blocks:
        o = ours[str(b)]["rel_l2_vs_float64"]
        u = ups[str(b)]["rel_l2_vs_float64"]
        rows.append({"block": b, "ours_rel_l2_vs_float64": o,
                     "upstream_bf16_rel_l2_vs_float64": u,
                     "over_upstream": (o / u) if (o and u) else None,
                     "ours_err_sq": ours[str(b)]["err_sq"],
                     "upstream_err_sq": ups[str(b)]["err_sq"],
                     "ref_sq": ours[str(b)]["ref_sq"], "n": ours[str(b)]["n"]})

    # depth is how far the backward has descended: block 47 is where it enters.
    depth = [47 - r["block"] for r in rows]
    so = slope_log2(depth, [r["ours_rel_l2_vs_float64"] for r in rows])
    su = slope_log2(depth, [r["upstream_bf16_rel_l2_vs_float64"] for r in rows])
    entry = [r for r in rows if r["block"] >= 39]
    exit_ = [r for r in rows if r["block"] <= 11]
    me = sum(r["over_upstream"] for r in entry) / len(entry)
    mx = sum(r["over_upstream"] for r in exit_) / len(exit_)

    lev = {"none": {"trunk_vs_float64": N["TRUNK"]["trunk_mass_weighted_rel_l2_vs_float64"],
                    "banked": N["TRUNK"]["banked_none_arm_value"],
                    "reproduces_banked_bit_identically": N["TRUNK"]["reproduces_banked"],
                    "rel_difference_vs_banked": N["TRUNK"]["rel_difference_vs_banked"],
                    "multiple_over_upstream_bf16": N["TRUNK"]["multiple_over_upstream_bf16"]}}
    for tag, path in (("promote_first", a.promote), ("cot_fp32", a.cotfp32)):
        if not path:
            continue
        try:
            P = json.load(open(path))
        except Exception as e:
            lev[tag] = {"ARM_DID_NOT_PRODUCE_A_REPORT": f"{type(e).__name__}: {e}"}
            continue
        t = P.get("TRUNK", {}).get("trunk_mass_weighted_rel_l2_vs_float64")
        base = N["TRUNK"]["trunk_mass_weighted_rel_l2_vs_float64"]
        lev[tag] = {"trunk_vs_float64": t,
                    "delta_vs_none": (t - base) if t else None,
                    "ratio_vs_none": (t / base) if t else None,
                    "bit_identical_to_none": (t == base) if t else None,
                    "multiple_over_upstream_bf16": P.get("TRUNK", {}).get(
                        "multiple_over_upstream_bf16"),
                    "lever_counters": P.get("D240", {}).get("lever_counters"),
                    "exit": P.get("provenance", {}).get("exit")}

    # per-tensor bit identity, the statement a single pooled number cannot make
    if a.none_pertensor and a.promote_pertensor:
        try:
            pn = json.load(open(a.none_pertensor))["per_tensor"]
            pp = json.load(open(a.promote_pertensor))["per_tensor"]
            n = moved = 0
            worst = 0.0
            wk = None
            for k, v in pn.items():
                if k not in pp:
                    continue
                n += 1
                d = abs(v["err_sq"] - pp[k]["err_sq"])
                if d != 0.0:
                    moved += 1
                rel = d / v["err_sq"] if v["err_sq"] > 0 else 0.0
                if rel > worst:
                    worst, wk = rel, k
            lev["promote_first"]["per_tensor"] = {
                "compared": n, "moved": moved, "worst_relative_move": worst, "worst_at": wk,
                "verdict": "bit-identical" if moved == 0 else "NOT bit-identical"}
        except Exception as e:
            lev["promote_first"]["per_tensor"] = {"ERROR": f"{type(e).__name__}: {e}"}

    out = {"what": "the absolute by-block curves on all 48 blocks, the entry-vs-accumulation "
                   "test, and the two levers against the arm that changes nothing",
           "host": socket.gethostname(), "device_involved": False,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "reference": "the published float64 grads_f64_043.pt for BOTH curves",
           "n_blocks": len(rows),
           "ours_trunk_vs_float64": N["TRUNK"]["trunk_mass_weighted_rel_l2_vs_float64"],
           "upstream_bf16_trunk_vs_float64": U["trunk_mass_weighted_rel_l2_vs_float64"],
           "upstream_reproduces_banked_0_3147698293887927":
               U["trunk_mass_weighted_rel_l2_vs_float64"] == U[
                   "banked_upstream_bf16_trunk_vs_float64"],
           "slope_log2_per_block_of_descent": {"ours": so, "upstream_bf16": su,
                                               "our_excess": (so - su) if
                                               (so is not None and su is not None) else None},
           "mean_over_upstream": {"entry_blocks_39_to_47": me, "exit_blocks_0_to_11": mx,
                                  "ratio_exit_over_entry": mx / me if me else None},
           "growth_47_to_4": {
               "ours": (ours["4"]["rel_l2_vs_float64"] / ours["47"]["rel_l2_vs_float64"]),
               "upstream_bf16": (ups["4"]["rel_l2_vs_float64"] /
                                 ups["47"]["rel_l2_vs_float64"])},
           "D240": N.get("D240"),
           "levers": lev,
           "curve": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print("ours trunk  %.16f   upstream bf16 trunk %.16f   multiple %.6f"
          % (out["ours_trunk_vs_float64"], out["upstream_bf16_trunk_vs_float64"],
             out["ours_trunk_vs_float64"] / out["upstream_bf16_trunk_vs_float64"]))
    print("slope log2/block  ours %+.4f  upstream %+.4f  excess %+.4f"
          % (so, su, so - su))
    print("mean over_upstream  entry 39-47 %.4f   exit 0-11 %.4f" % (me, mx))
    print("growth 47->4  ours %.4f  upstream %.4f"
          % (out["growth_47_to_4"]["ours"], out["growth_47_to_4"]["upstream_bf16"]))
    print(json.dumps({k: v for k, v in lev.items()}, indent=1)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
