"""Emit the README's numeric block straight from analysis.json, so no number is retyped."""
from __future__ import annotations
import json, sys
from pathlib import Path


def f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def main(path):
    a = json.loads(Path(path).read_text())
    print(f"Reduced verdict: **{a['verdict']}**\n")
    for size, t in a["targets"].items():
        print(f"### {size} aa\n")
        print("| clock | folds | median | min | max | stdev | adjacent \\|delta\\| | mean board W | host CPU s | elapsed Mcycles |")
        print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for clk, c in t["cells"].items():
            if not c.get("accepted_folds"):
                print(f"| {clk} MHz | 0 | - | - | - | - | - | - | - | - |"); continue
            print(f"| {clk} MHz | {c['accepted_folds']} | {f(c['median_s'],4)} s | {f(c['min_s'],4)} s | "
                  f"{f(c['max_s'],4)} s | {f(c['sample_stdev_s'],4)} s | {f(c['adjacent_abs_delta_median_s'],4)} s | "
                  f"{f(c['mean_W'],1)} | {f(c['host_cpu_s_median'],3)} | {f(c['elapsed_device_clock_equivalent_Mcycles'],1)} |")
        print()
        for name, key in [("all accepted folds", "fit_all_folds"), ("cell medians", "fit_cell_medians")]:
            fit = t.get(key)
            if not fit: continue
            se = f"+-{f(fit['se_F_s'],4)}" if fit.get("se_F_s") is not None else "(no SE, see note)"
            print(f"- fit on {name} (n={fit['n']}, clocks {fit['clocks_MHz']}): "
                  f"**F = {f(fit['F_s'],4)} s {se}**, C = {f(fit['C_Mcycles'],1)} Mcycles"
                  + (f" +-{f(fit['se_C_Mcycles'],1)}" if fit.get("se_C_Mcycles") is not None else "")
                  + f", max |residual| {f(fit['max_abs_residual_s'],4)} s, rms {f(fit['rms_residual_s'],4)} s")
        tc = t.get("two_clock_endpoints")
        if tc:
            print(f"- closed form on the {tc['clocks_MHz'][0]}/{tc['clocks_MHz'][1]} MHz endpoints alone: "
                  f"**F = {f(tc['F_s'],4)} s +-{f(tc['se_F_s'],4)}**, C = {f(tc['C_Mcycles'],1)} Mcycles "
                  f"+-{f(tc['se_C_Mcycles'],1)}; the endpoint arms enter F with gains "
                  f"{f(tc['dF_dt_gain'][0],3)} and {f(tc['dF_dt_gain'][1],3)}")
        mc = t.get("model_check")
        if mc:
            print(f"- inverse-clock model: A/A timing floor {f(mc['aa_timing_floor_s'],4)} s, "
                  f"cell-median max |residual| {f(mc['cell_median_max_abs_residual_s'],4)} s, "
                  f"per-fold rms {f(mc['rms_residual_s'],4)} s -> "
                  f"**{'the exact two-parameter form holds, F is point-identified' if mc['exact_inverse_clock_form_holds'] else 'the exact two-parameter form is REFUTED, F is an interval'}**")
            print(f"- three estimators of F: "
                  + ", ".join(f"{k} {f(v['F_s'],4)} s +-{f(v['se_s'],4)}" for k, v in mc['estimators_F_s'].items())
                  + f" -> F bounded to {f(mc['F_bound_s'][0],4)} - {f(mc['F_bound_s'][1],4)} s")
        st = t["structure"]
        print(f"- structure: {st['distinct_cif_sha256']} distinct CIF over {t['accepted_folds']} accepted folds, "
              f"max pairwise domain RMSD {st['max_pair_domain_A']:.2e} A against the {st['bar_A']} A bar "
              f"({'byte-identical in every arm' if st['all_accepted_byte_exact'] else 'NOT byte identical'})")
        d = t.get("demand")
        if d:
            print(f"- demand for {d['target_s']} s at {d['at_MHz']} MHz (now {f(d['seconds_now'],4)} s):")
            u = d["F_untouched"]
            if d["seconds_now"] <= d["target_s"]:
                print(f"  - this size already folds in {f(d['seconds_now'],4)} s, under the {d['target_s']} s target, "
                      f"so the figures below are headroom rather than a demand")
            if "impossible" in u:
                print(f"  - F untouched: {u['impossible']}")
            else:
                print(f"  - F untouched: {f(u['cycles_allowed_Mcycles'],1)} Mcycles allowed, a cut of "
                      f"{f(u['cycle_cut_Mcycles'],1)} Mcycles, **{f(u['cycle_cut_pct'],1)} %** of the work term")
            c2 = d["F_cut_to_1s"]
            print(f"  - F cut to {c2['assumed_F_s']} s: {f(c2['cycles_allowed_Mcycles'],1)} Mcycles allowed, a cut of "
                  f"{f(c2['cycle_cut_Mcycles'],1)} Mcycles, **{f(c2['cycle_cut_pct'],1)} %**")
            print(f"  - cutting F to {c2['assumed_F_s']} s and touching no cycle at all: "
                  f"{f(d['F_alone_at_current_cycles_s'],4)} s")
        print()
    x = a.get("cross_size")
    if x:
        print("### across the two sizes\n")
        hi, lo = x["sizes"]
        print(f"- F({hi} aa) = {f(x['F_s'][hi],4)} s, F({lo} aa) = {f(x['F_s'][lo],4)} s, "
              f"difference {f(x['F_difference_s'],4)} s +-{f(x['F_difference_se_s'],4)}: "
              f"**F is {'size independent' if x['F_size_independent_within_2se'] else 'size DEPENDENT'} "
              f"within 2 standard errors**")
        print(f"- work term {f(x['C_Mcycles'][hi],1)} Mcycles at {hi} aa against {f(x['C_Mcycles'][lo],1)} at {lo} aa: "
              f"a work ratio of **{f(x['work_cycle_ratio'],3)}**")

        # what the row was opened to settle: host problem, kernel problem, or both
        d = a["targets"][hi]["demand"]
        u, c1 = d["F_untouched"], d["F_cut_to_1s"]
        print(f"""
## What it settles

At {hi} aa the {d['target_s']} s target needs a {f(u['cycle_cut_pct'],1)} % cut of the \
{f(x['C_Mcycles'][hi],1)} Mcycle work term with F left alone, or {f(c1['cycle_cut_pct'],1)} % with F \
brought to {c1['assumed_F_s']} s. Bringing F to {c1['assumed_F_s']} s while deleting no device cycle at \
all lands at {f(d['F_alone_at_current_cycles_s'],4)} s, so F alone does not get there.

The work ratio is {f(x['work_cycle_ratio'],3)}, below the 1.6 the campaign's low branch assumed and \
well below the 2.5 at which cutting F would have been sufficient on its own. So the answer is \
neither: the {d['target_s']} s target is not a host problem and not a kernel problem, it needs both \
levers, and they are comparable in size. F is also size dependent rather than a host constant, \
{f(x['F_s'][hi],4)} s at {hi} aa against {f(x['F_s'][lo],4)} s at {lo} aa, which is what \
featurization and CIF writing scaling with the target look like.

F is not shown to be removable host CPU work. Host CPU inside the timed window tracks the clock \
instead of staying flat, which is a thread waiting on the device, so part of F may be device time \
in a clock domain AICLK does not drive. Splitting F into host, dispatch and non-AICLK device time \
is not in this row.

The {lo} aa fixture already folds in {f(a['targets'][lo]['demand']['seconds_now'],4)} s and is not \
the constraint.""")


if __name__ == "__main__":
    main(sys.argv[1])
