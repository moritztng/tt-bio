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
        gp = g["gain_ci"].get("gap")
        if gp is None:
            fails.append(f"{model}: {PAIR} carries no `gap` metric -- the report predates "
                         f"pass 37. Re-run the analysis; the gap CI cannot be recovered by "
                         f"subtracting the oracle and user intervals, they are paired.")
        else:
            print(f"  gap    grow {gp[1]:+.4f} [{gp[0]:+.4f},{gp[2]:+.4f}]"
                  f"   <- the campaign's headline quantity")
        if g["degenerate"]:
            fails.append(f"{model}: {PAIR} is degenerate (common_targets "
                         f"{g['common_targets']} < 8) -- its CI cannot clear the stop rule")
        if PREV_PAIR in gains:
            pg = gains[PREV_PAIR]["gain_ci"]
            print(f"  vs {PREV_PAIR}: oracle {pg['oracle'][1]:+.4f}"
                  + (f"  gap {pg['gap'][1]:+.4f}" if "gap" in pg else ""))

        # STOP RULE. The p27 convention is gain(lo->hi) against the seed-noise floor at
        # the LOW rung (boltz2 +0.0228 vs 0.0132 and protenix-v2 +0.0382 vs 0.0212 are
        # both floor[128]). For 256->512 that comparator is floor[256], which needs a
        # 512-deep pool and so exists for the first time in this campaign.
        #
        # Read it off within_fold_common, NOT seed_noise_floor_med. The latter computes
        # each m on whichever targets are deep enough for it, so floor[128] is the whole
        # panel while floor[256] is only the targets that reached 512 -- different sets,
        # and the two already swapped order once as labelling filled in (pass 35 had
        # floor[256] > floor[128], pass 36 and 37 the other way). within_fold_common
        # holds every m on one fixed panel. At and below rung 128 the two agree to within
        # a few percent because nearly every target reaches 256, which is why the p27
        # verdicts stand unchanged under either.
        deep = rep.get(model + "__deep") or {}
        common = deep.get("within_fold_common") or {}
        floors = common.get("floor_med") or {}
        lo_rung = PAIR.split("->")[0]
        floor = floors.get(lo_rung)
        if floor is None:
            fails.append(f"{model}: no panel-matched floor at m={lo_rung} in "
                         f"within_fold_common -- the stop rule has no comparator.")
        elif common.get("n_targets") != g["common_targets"]:
            fails.append(f"{model}: the floor panel ({common.get('n_targets')} targets at "
                         f"depth {common.get('depth')}) is not the gain panel "
                         f"({g['common_targets']}). Comparing them is a panel confound; "
                         f"recompute the floor on the pair's own target set before "
                         f"calling the stop rule. Suspect a per-model rung exclusion.")
        else:
            verdict = "ABOVE floor, still climbing" if o[1] > floor else "BELOW floor"
            strict = "yes" if o[0] > floor else "no"
            print(f"  stop rule: oracle gain {o[1]:+.4f} vs floor[{lo_rung}] {floor:.4f} "
                  f"on the same {common['n_targets']} targets -> {verdict} "
                  f"(CI lower bound also above floor: {strict})")
        # The cost headline is only quotable in its per-target form. Rung 512's panel tops
        # out near 137 against rung 256's 153, so the two rungs never carry the same target
        # count and an extensive (summed) denominator is never comparable between the pairs.
        cph, rate = g.get("cost_h_per_target"), g.get("s_per_sample_hi")
        rate_lo = g.get("s_per_sample_lo_same_panel")
        if cph is None:
            fails.append(f"{model}: {PAIR} carries no cost_h_per_target -- some target in the "
                         f"paired set has no measured wall time or no billed sample count, so "
                         f"the cost headline has no basis. Do not quote a marginal cost here.")
        else:
            print(f"  cost basis: {g['step_samples']} samples x {rate:.1f} s = "
                  f"{cph:.2f} card-h/target"
                  + (f"; a sample costs {rate_lo:.1f} s at 256 over the same "
                     f"{g['common_targets']} targets ({rate / rate_lo:.2f}x)" if rate_lo else ""))
            prev = gains.get(PREV_PAIR) or {}
            if prev.get("marginal_oracle_per_1000cs"):
                print(f"  vs {PREV_PAIR}: {prev['marginal_oracle_per_1000cs']:+.6f}/1000 card-s "
                      f"per target at {prev['s_per_sample_hi']:.1f} s/sample")
            # A doubling that keeps buying the same absolute gain must halve its
            # cost-efficiency, since it pays twice the samples for it. That is the
            # expected shape, not a finding; say so, so nobody reports it as one.
            if rate_lo and abs(rate / rate_lo - 1) > 0.15:
                infos.append(f"{model}: a sample costs {rate / rate_lo:.2f}x more at 512 than at "
                             f"256 on the same targets. That is a fold-rate change, not a "
                             f"property of the doubling; check the window before quoting it.")
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
