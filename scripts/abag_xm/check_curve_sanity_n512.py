#!/usr/bin/env python3
"""Read-only sanity gate on the deep-N analysis JSON, for the 512 rung.

Answers the three questions that are easy to get wrong by eye off the console output of
`abag_xm_deepn_analysis.py --deep`:

1. A missing `256->512` pair is usually not a defect. `ns_powered` filters the adjacency
   spine to rungs with >= POWER_MIN (50) targets, so the pair does not exist until rung 512
   carries 50. This prints rung 512's `n_targets` next to the verdict so the two are never
   confused.

2. The console prints `within_fold_oracle_curve`, whose target set VARIES with m by
   construction: `subsample_oracle_curve` skips m > len(pool), so E[oracle] at m=512 averages
   only the targets with full 512 pools while m=256 averages every target. That curve can sit
   LOWER at m=512 than at m=256 forever, at complete labeling, and it is not evidence that
   more samples hurt or that labels are missing. Only a drop with a CONSTANT target count is a
   defect. `within_fold_nt` is printed alongside every point so the distinction is visible.

3. `within_fold_common` is the monotone-interpretable curve: a fixed target set at fixed depth
   D. For a fixed pool E[max over an m-subset] is monotone non-decreasing in m by
   construction, so a decrease there beyond Monte-Carlo noise (B=200 draws) IS a real defect.
   That is the check worth gating on, and it is not printed by the analysis at all.

Exits 1 on a real violation, 0 otherwise. INFO lines are not failures.

    python3 scripts/abag_xm/check_curve_sanity_n512.py ~/abag_xm/deepn/analysis_curves_n512.json
"""
import argparse, json, sys
from pathlib import Path

POWER_MIN = 50           # must match abag_xm_deepn_analysis.py
MONO_TOL = 1e-3          # MC noise on E[max] over B=200 draws averaged across >=20 targets
                         # is ~5e-4; a target-set bug moves it ~0.09 (pass 34 saw 0.6175 ->
                         # 0.5255). The threshold sits in the gap, not near either end.
EXPECT_DEPTH = 512
PAIR = "256->512"
PREV_PAIR = "128->256"


def _mkeys(d):
    """The analysis writes m/N keys as JSON strings; sort them numerically."""
    return sorted(((int(k), v) for k, v in d.items()), key=lambda kv: kv[0])


def check_model(model, rep, fails, infos):
    print(f"\n=== {model} ===")
    rungs = rep.get(model) or {}
    nt512 = (rungs.get(str(EXPECT_DEPTH)) or {}).get("n_targets")
    print(f"rung {EXPECT_DEPTH}: n_targets={nt512}  (POWER_MIN={POWER_MIN})")

    gains = rep.get(model + "__pairwise_gain_ci") or {}
    if PAIR in gains:
        g = gains[PAIR]
        o, u = g["gain_ci"]["oracle"], g["gain_ci"]["user"]
        print(f"{PAIR}: nt={g['common_targets']} degenerate={g['degenerate']}")
        print(f"  oracle gain {o[1]:+.4f} [{o[0]:+.4f},{o[2]:+.4f}]")
        print(f"  user   gain {u[1]:+.4f} [{u[0]:+.4f},{u[2]:+.4f}]")
        print(f"  gap change (oracle-user, midpoints) {o[1] - u[1]:+.4f}")
        if g["degenerate"]:
            fails.append(f"{model}: {PAIR} is degenerate (common_targets "
                         f"{g['common_targets']} < 8) -- its CI cannot clear the stop rule")
        if PREV_PAIR in gains:
            po, pu = gains[PREV_PAIR]["gain_ci"]["oracle"], gains[PREV_PAIR]["gain_ci"]["user"]
            print(f"  vs {PREV_PAIR}: oracle {po[1]:+.4f}  gap {po[1] - pu[1]:+.4f}")
        # The cost headline is only quotable in its per-target form. Rung 512's panel tops
        # out near 137 against rung 256's 153, so the two rungs never carry the same target
        # count and an extensive (summed) denominator is never comparable between the pairs.
        cph = g.get("cost_h_per_target")
        lo_same = g.get("cost_h_per_target_lo_same_panel")
        if cph is None:
            fails.append(f"{model}: {PAIR} carries no cost_h_per_target -- some target in the "
                         f"paired set has no measured wall time, so the cost headline has no "
                         f"basis. Do not quote a marginal cost for this model.")
        else:
            print(f"  cost basis: {cph:.2f} card-h/target at 512"
                  + (f", {lo_same:.2f} at 256 over the same {g['common_targets']} targets"
                     f" ({cph / lo_same:.2f}x)" if lo_same else ""))
            prev_cph = (gains.get(PREV_PAIR) or {}).get("cost_h_per_target")
            if prev_cph:
                print(f"  vs {PREV_PAIR}: {prev_cph:.2f} card-h/target")
                infos.append(f"{model}: the two pairs' costs are per-target and so comparable in "
                             f"units, but they are measured on different target sets "
                             f"({gains[PREV_PAIR]['common_targets']} vs {g['common_targets']}). "
                             f"For a panel-clean read of what the second 256 samples cost, use "
                             f"cost_h_per_target_lo_same_panel, not the {PREV_PAIR} pair.")
    elif nt512 is not None and nt512 < POWER_MIN:
        infos.append(f"{model}: {PAIR} absent because rung {EXPECT_DEPTH} has {nt512} targets "
                     f"< POWER_MIN {POWER_MIN}. Label latency, not a defect.")
    else:
        fails.append(f"{model}: {PAIR} absent although rung {EXPECT_DEPTH} carries "
                     f"{nt512} >= POWER_MIN targets")

    deep = rep.get(model + "__deep")
    if not deep:
        infos.append(f"{model}: no __deep block (analysis was not run with --deep)")
        return

    raw = deep.get("within_fold_oracle_curve") or {}
    nt = deep.get("within_fold_nt") or {}
    print("raw within-fold curve (target set varies with m):")
    prev = None
    for m, v in _mkeys(raw):
        n = nt.get(str(m), nt.get(m))
        mark = ""
        if prev is not None:
            pm, pv, pn = prev
            if v < pv - MONO_TOL:
                if n is not None and pn is not None and n < pn:
                    mark = f"  <- drop, EXPECTED: target set shrank {pn}->{n}"
                else:
                    mark = "  <- DROP at constant target count"
                    fails.append(f"{model}: raw within-fold curve drops {pv:.4f}->{v:.4f} "
                                 f"from m={pm} to m={m} with target count unchanged at {n}")
        print(f"  m={m:<4} E[oracle]={v:.4f}  nt={n}{mark}")
        prev = (m, v, n)

    common = deep.get("within_fold_common")
    if not common:
        fails.append(f"{model}: within_fold_common is absent -- no grid depth reached 20 targets")
        return
    print(f"common-set curve: depth={common['depth']} n_targets={common['n_targets']}")
    if common["depth"] != EXPECT_DEPTH:
        infos.append(f"{model}: common-set depth is {common['depth']}, not {EXPECT_DEPTH} -- "
                     f"fewer than 20 targets hold a full {EXPECT_DEPTH} pool yet")
    prev = None
    for m, v in _mkeys(common["curve"]):
        mark = ""
        if prev is not None and v < prev[1] - MONO_TOL:
            mark = "  <- NON-MONOTONE"
            fails.append(f"{model}: common-set curve is non-monotone, {prev[1]:.4f} at "
                         f"m={prev[0]} -> {v:.4f} at m={m} (delta {v - prev[1]:+.5f}, "
                         f"tolerance {MONO_TOL}); E[max over an m-subset] cannot decrease "
                         f"for a fixed target set")
        print(f"  m={m:<4} E[oracle]={v:.4f}{mark}")
        prev = (m, v)

    print(f"cross-check: within_fold_nt[{EXPECT_DEPTH}]="
          f"{nt.get(str(EXPECT_DEPTH), nt.get(EXPECT_DEPTH))} must equal this model's "
          f"`checked` from verify_n512_nesting.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("curves", type=Path, help="analysis_curves_n512.json")
    a = ap.parse_args()
    rep = json.loads(a.curves.read_text())
    models = sorted(k for k in rep if not k.startswith("n16_") and "__" not in k)
    if not models:
        print(f"no model blocks in {a.curves}", file=sys.stderr)
        return 1
    fails, infos = [], []
    for m in models:
        check_model(m, rep, fails, infos)
    print()
    for i in infos:
        print(f"INFO: {i}")
    for f in fails:
        print(f"FAIL: {f}")
    print(f"\n{'GATE PASS' if not fails else 'GATE FAIL'}: {len(models)} model(s), "
          f"{len(fails)} failure(s), {len(infos)} info(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
