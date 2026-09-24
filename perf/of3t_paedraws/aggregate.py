#!/usr/bin/env python3
"""of3t-paedraws: the A44 artifacts from draw 0 (CF384, banked) and draws 1-5.

    aggregate.py            -> SCORE_PD384.json, DRAWS_PD384.json, and SECTIONS_PD384.json once six draws exist

A draw counts only if its device step ran at its seed, drew the same sigma as its float64
reference, and every side replayed the rollout with 0 mismatches. Per-draw sections come from
perf/of3t_orchestrator/sections/section_ratio.py, run unedited through runpy; the mean is taken
over draws on each side, then divided.
"""
import json
import runpy
import statistics
import sys
import tempfile
from pathlib import Path

W = Path(__file__).resolve().parents[2]
P, S = W / "perf/of3t_paedraws", Path("/home/ttuser/of3t_paedraws")
SEEDS = {0: 20260922, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
SECTION_RATIO = W / "perf/of3t_orchestrator/sections/section_ratio.py"


def draw_zero():
    d = json.load(open(W / "perf/of3t_confpfe/SCORE_CF384.json"))
    d["arms"]["PD384"] = d["arms"].pop("CF384")
    json.dump(d, open(P / "SCORE_PD384.json", "w"), indent=1)
    return (P / "SCORE_PD384.json", "PD384", W / "perf/of3t_confpfe/DEV_CF384.json",
            Path("/home/ttuser/of3t_confpfe/ref384c"))


def sides(k):
    if k == 0:
        return draw_zero()
    return (P / f"SCORE_PD384_s{k}.json", f"PD384_s{k}", P / f"DEV_PD384_s{k}.json", S / f"ref_s{k}")


def sections_of(score, arm):
    with tempfile.NamedTemporaryFile(suffix=".json") as t:
        argv, sys.argv = sys.argv, [str(SECTION_RATIO), str(score), arm, t.name]
        try:
            g = runpy.run_path(str(SECTION_RATIO), run_name="section_ratio")
        finally:
            sys.argv = argv
    return g["sections"], g["LIMIT"]


def main():
    draws, per_sections, limit = [], [], None
    for k, seed in SEEDS.items():
        score, arm, dev, ref = sides(k)
        if not score.exists():
            print(f"draw {k}: no score yet")
            continue
        d, dv = json.load(open(score)), json.load(open(dev))
        rf = {m: json.load(open(ref / m / f"REF_{m.upper()}.json")) for m in ("f64", "bf16")}
        checks = {
            "device_seed": dv["seed"] == seed,
            "ref_seed": all(r["draw_seed"] == seed for r in rf.values()),
            "sigma_matches_f64": dv["of3t_denoise"]["denoise_sigma"] == rf["f64"]["denoise"]["sigma"],
            "device_replay_mismatch_0": dv["fullstep64"]["draws"]["mismatch_count"] == 0,
            "ref_replay_mismatch_0": all(r["draws"]["mismatch_count"] == 0 for r in rf.values()),
            "tt_bio_clean": dv["provenance"]["git_dirty"] == "",
        }
        if not all(checks.values()):
            raise SystemExit(f"draw {k} fails its identity checks: {checks}")
        a, b = d["arms"][arm], d["arms"]["BF16"]
        five = {
            "unread_n": d["scored"]["unread_n"],
            "placed_but_carried_nothing_n": a["placed_but_carried_nothing"]["n"],
            "multi_placed": a["multi_placed"],
            "global_rel": a["global"]["rel"], "bf16_global_rel": b["global"]["rel"],
            "mass_at_or_better_than_bf16": a["mass_at_or_better_than_bf16"],
        }
        five["holds"] = (five["unread_n"] == 0 and five["placed_but_carried_nothing_n"] == 0
                         and five["multi_placed"] == 0 and five["global_rel"] <= five["bf16_global_rel"]
                         and five["mass_at_or_better_than_bf16"] >= 0.95)
        secs, limit = sections_of(score, arm)
        per_sections.append(secs)
        aiclk = dv.get("aiclk_during", {})
        draws.append({"k": k, "seed": seed, "sigma": dv["of3t_denoise"]["denoise_sigma"],
                      "score": str(score.relative_to(W)), "arm": arm, "checks": checks,
                      "five": five, "reference_loss_f64": rf["f64"]["loss"], "device_loss": dv["loss"],
                      "rel": {n: {"ours": secs[n]["rel"], "bf16": secs[n]["bf16_rel"]}
                              for n in ("aux_heads.pae", "aux_heads.pde")}
                             | {"global": {"ours": five["global_rel"], "bf16": five["bf16_global_rel"]}},
                      "host": dv["provenance"]["host"], "card": dv["provenance"]["card"],
                      "board_class": dv["provenance"]["board_class"],
                      "git_commit": dv["provenance"]["git_commit"], "aiclk_during": aiclk})
    n = len(draws)
    json.dump({"arm": "PD384", "protocol": "A44", "draws": n,
               "five_hold_every_draw": n > 0 and all(x["five"]["holds"] for x in draws),
               "per_draw": draws}, open(P / "DRAWS_PD384.json", "w"), indent=1)
    print(f"DRAWS_PD384.json: {n} draws, five hold on every draw: "
          f"{all(x['five']['holds'] for x in draws)}")
    if n < 6:
        print(f"only {n} of 6 draws: SECTIONS_PD384.json not written (A44 needs six)")
        return 1
    zero = per_sections[0]
    sections = {}
    for name in sorted(zero):
        mo = statistics.fmean(s[name]["rel"] for s in per_sections)
        mb = statistics.fmean(s[name]["bf16_rel"] for s in per_sections)
        sections[name] = {"mean_rel_ours": mo, "mean_rel_bf16": mb, "ratio": mo / mb,
                          "within_3x": mo / mb <= limit, "single_draw0_ratio": zero[name]["ratio"],
                          "per_draw_ratio": [s[name]["ratio"] for s in per_sections]}
    json.dump({"arm": "PD384", "protocol": "A44", "draws": n, "limit": limit,
               "grouping": str(SECTION_RATIO.relative_to(W)), "sections": sections},
              open(P / "SECTIONS_PD384.json", "w"), indent=1)
    for name, s in sections.items():
        print(f"{name:45s} mean {s['ratio']:.3f}x  draw0 {s['single_draw0_ratio']:.3f}x"
              + ("" if s["within_3x"] else "  PAST 3x"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
