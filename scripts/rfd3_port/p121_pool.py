"""p121 -- pool the two A/B runs against the A/A control, because neither run got a clean round.

Two independent 200-step A/B runs under benchlock produced 7 scored rounds and the harness voided
every one of them on its 3.0 load bar: sibling workers run untimed jobs that do not take the lock,
so the box never went quiet inside a 17-minute window. benchlock locks the MEASUREMENT, not the
box (`benchlock-host-scoped-no-fairness-starves-waiters`).

That voids the harness's own median, and it is quoted as void. What it does not void is the
comparison against the A/A control, and that is what this computes. p107 measures the FRACTION
(off - on) / off precisely because co-tenancy inflates both arms roughly multiplicatively and the
fraction survives it; the load bar is belt-and-braces on top. So: is the A/B's fraction
distribution separated from the A/A's, and by how much.

The bias runs the safe way. The on arm does ~2 ms/step of host bitmap work in `plan()` that the off
arm does not, about 3.6 s per design, so CPU contention stretches the ON arm more than the off arm.
Load contamination therefore UNDER-states this prize; it cannot manufacture one.
"""
import json
import pathlib
import statistics
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p121/pooled.json")
AA = pathlib.Path("perf/p117/aa.json")
AB = [pathlib.Path("perf/p117/ab_clean.json"), pathlib.Path("perf/p117/ab_clean2.json")]
BASELINE = 94.087        # p107's CLEAN_BASELINE, card 1, for scaling a fraction to s/design
PAGE_CELL = 91.443       # the published b=8 page cell; the denominator for any outward ratio
H200_B8 = 12.974         # H200 b=8 amortised; the bar is 4 x this = 51.896


def fracs(path, want_arm_live):
    """Per-round (off - on) / off from a p107 artifact. Warm round excluded, as p107 excludes it."""
    d = json.loads(path.read_text())
    rows = [r for r in d["rows"] if not r["warm"]]
    out = []
    for rnd in sorted({r["round"] for r in rows}):
        got = {r["arm"]: r for r in rows if r["round"] == rnd}
        if len(got) != 2:
            continue
        assert all(r["arm_verified"] and r["digest_ok"] for r in got.values()), (path, rnd)
        # the point of the pool: confirm the arm was live (or dense, for A/A) in every fold used
        on_live = got["on"]["blocked"] > 0 and got["on"]["shipped"] == 0
        assert on_live is want_arm_live, (path, rnd, got["on"]["blocked"])
        o, n = got["off"]["s_per_design"], got["on"]["s_per_design"]
        out.append(dict(src=path.name, round=rnd, off=o, on=n, raw=round(o - n, 3),
                        frac=round((o - n) / o, 6),
                        load_max=max(got["off"]["load_max"], got["on"]["load_max"]),
                        load_clean=all(r["load_clean"] for r in got.values())))
    return out


aa = fracs(AA, want_arm_live=False)
ab = [r for p in AB for r in fracs(p, want_arm_live=True)]
assert aa and ab

af = [r["frac"] for r in aa]
bf = [r["frac"] for r in ab]
med = statistics.median(bf)
res = {
    "aa_control": {
        "n_rounds": len(aa), "rounds": aa,
        "frac_median": round(statistics.median(af), 6),
        "frac_min": min(af), "frac_max": max(af),
        "s_per_design_median": round(statistics.median(af) * BASELINE, 3),
        "s_per_design_min": round(min(af) * BASELINE, 3),
        "s_per_design_max": round(max(af) * BASELINE, 3),
        "all_rounds_load_clean": all(r["load_clean"] for r in aa),
    },
    "ab": {
        "n_rounds": len(ab), "n_runs": len(AB), "rounds": ab,
        "frac_median": round(med, 6),
        "frac_min": min(bf), "frac_max": max(bf),
        "frac_iqr": [round(statistics.quantiles(bf, n=4)[0], 6),
                     round(statistics.quantiles(bf, n=4)[2], 6)],
        "s_per_design_median": round(med * BASELINE, 3),
        "s_per_design_min": round(min(bf) * BASELINE, 3),
        "s_per_design_max": round(max(bf) * BASELINE, 3),
        "spread_pct": round((max(bf) - min(bf)) / med * 100, 2),
        "all_rounds_load_clean": all(r["load_clean"] for r in ab),
        "harness_verdict": "NO CLEAN ROUND -- every scored round voided on the 3.0 load bar",
    },
    "separation": {
        "median_ratio": round(med / statistics.median(af), 1),
        "ab_min_over_aa_max": round(min(bf) / max(af), 1),
        "distributions_overlap": max(af) >= min(bf),
    },
    "landing": {
        "page_cell_s_per_design": PAGE_CELL,
        "page_cell_ratio": round(PAGE_CELL / H200_B8, 3),
        "enabled_on_a_fitting_target": round(PAGE_CELL - med * BASELINE, 3),
        "enabled_ratio": round((PAGE_CELL - med * BASELINE) / H200_B8, 3),
        "bar_4x": round(4 * H200_B8, 3),
        "residual_above_bar": round(PAGE_CELL - med * BASELINE - 4 * H200_B8, 3),
        "note": ("The lever stays default-OFF, so the SHIPPED number is unchanged at 91.443 / "
                 "7.048x. 'enabled_*' is what it reaches with RFD3_BLOCK_SPARSE=1 on a target "
                 "whose padded atom axis is a multiple of 1216, which is 2.45%% of atom counts."),
    },
    "cost_models_for_comparison": {"p3_k3_gross": 4.191, "p3_k3_net": 3.787, "p3_k2_net": 3.552,
                                   "p2_per_call_ratio_projection": 28.506},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "ab"}, indent=2))
print("\nA/B per-round fracs:", [r["frac"] for r in ab])
print("A/B median %.3f s/design (n=%d, %d runs), range %.3f-%.3f, ALL LOAD-VOID"
      % (res["ab"]["s_per_design_median"], len(ab), len(AB),
         res["ab"]["s_per_design_min"], res["ab"]["s_per_design_max"]))
print("wrote", OUT)
