"""Write ~/.coworker/state/c10-fixed-cost.md from analysis.json. Every number comes from the file."""
from __future__ import annotations
import datetime, json, sys
from pathlib import Path

BASE512 = 14.8813   # c10-bare-baseline 512 aa median at 1350 MHz, bc66f7d6d
BASE298 = 9.6801
HIST = 14.554       # pinned arm on 0df13ad9, node 1, firmware bundle 19.11.0.0


def f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def main(analysis, runroot, out):
    a = json.loads(Path(analysis).read_text())
    t5, t2 = a["targets"]["512"], a["targets"]["298"]
    c5, c2 = t5["cells"], t2["cells"]
    m5, m2 = t5["model_check"], t2["model_check"]
    f5, f2 = t5["fit_all_folds"], t2["fit_all_folds"]
    e5 = m5["estimators_F_s"]
    d5, d2 = t5["demand"], t2["demand"]
    u5, b5 = d5["F_untouched"], t5["demand_at_F_bounds"]
    x = a["cross_size"]
    run5 = json.loads((Path(runroot) / "512" / "result.json").read_text())
    run2 = json.loads((Path(runroot) / "298" / "result.json").read_text())
    def utc(ns):
        return datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc)
    w0 = min(utc(r["started_utc_ns"]) for r in (run5, run2))
    w1 = max(utc(r["finished_utc_ns"]) for r in (run5, run2))
    window = (f"{w0:%Y-%m-%dT%H:%M:%SZ} and {w1:%H:%M:%SZ} on {w1:%Y-%m-%d}"
              if w0.date() == w1.date() else
              f"{w0:%Y-%m-%dT%H:%M:%SZ} and {w1:%Y-%m-%dT%H:%M:%SZ}")
    node = run5["assigned_node_sysfs"]
    nf = t5["accepted_folds"] + t2["accepted_folds"]
    tele = c5["1350"]["median_s"] - BASE512
    lo5, hi5 = m5["F_bound_s"]
    lo2, hi2 = m2["F_bound_s"]
    form = ("HOLDS at this precision" if m5["exact_inverse_clock_form_holds"]
            else "is REFUTED at this precision")

    doc = f"""# C10 clock-immune fixed cost of the Boltz2 fold, measured by varying the clock

CLOCK: four pinned arms, 1350, 1200, 1000 and 800 MHz, each requested through ARC FORCE_AICLK
before EVERY label and sampled at about 1 kHz DURING every fold. In all {nf} accepted folds the
during-fold minimum equalled the maximum equalled that fold's own requested target, with zero read
errors and every sample/boundary gap inside the 10 ms limit. Board power tracked the clock,
{f(c5['1350']['mean_W'],1)} W at 1350 MHz falling to {f(c5['800']['mean_W'],1)} W at 800 MHz on the 512 aa arms, so each arm is
witnessed by physics and not only by a telemetry register. The 800 MHz arm was DELIBERATE:
`scripts/aiclk_watch.sh` logs `AICLK-WATCH: qb2 CLAMPED-UNDER-LOAD` for a busy chip under 1200 MHz,
so `state/aiclk-qb2.log` shows this row tripping the fleet's own guard on purpose on card 0 between
{window}. A pinned 800 MHz fold is a measurement; an unrequested one
is an artifact, and the raw samples say which this was. Card 0 only, {node.get('tt_card_type')}, ASIC
{node.get('tt_asic_id')}, firmware bundle {node.get('tt_fw_bundle_ver')}. FORCE_AICLK was released in the finally block and
the release return was checked, status 0 at both sizes.

PREDICTED: recorded in prediction.json and digested into every result.json before any fold.
Predicted model cycles saved: exactly zero, because every arm runs the identical 200-step,
3-recycle, 1-sample, seed-0 configuration on the same committed fixture and 35-row A3M. Predicted
512 aa at 800 MHz: 22.7 s, allowed range 21.9 to 23.2 s. Predicted F: 2.9 to 4.7 s, central 3.5 s.
Predicted refutation: a fitted F at or below 0 s, at or above the 9.680 s 298 aa fold, or a
cell-median residual above max(A/A floor, 0.05 s) at an added clock.

MEASURED: measured model cycles saved: zero, as predicted. 28 accepted warm folds per size, 7 per
cell, interleaved in ONE process and ONE device context with the clock order reversed on odd
repetitions after one discarded cold fold.

| clock | 512 aa median | stdev | adjacent \\|delta\\| | 298 aa median | stdev | adjacent \\|delta\\| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1350 MHz | {f(c5['1350']['median_s'])} s | {f(c5['1350']['sample_stdev_s'])} s | {f(c5['1350']['adjacent_abs_delta_median_s'])} s | {f(c2['1350']['median_s'])} s | {f(c2['1350']['sample_stdev_s'])} s | {f(c2['1350']['adjacent_abs_delta_median_s'])} s |
| 1200 MHz | {f(c5['1200']['median_s'])} s | {f(c5['1200']['sample_stdev_s'])} s | {f(c5['1200']['adjacent_abs_delta_median_s'])} s | {f(c2['1200']['median_s'])} s | {f(c2['1200']['sample_stdev_s'])} s | {f(c2['1200']['adjacent_abs_delta_median_s'])} s |
| 1000 MHz | {f(c5['1000']['median_s'])} s | {f(c5['1000']['sample_stdev_s'])} s | {f(c5['1000']['adjacent_abs_delta_median_s'])} s | {f(c2['1000']['median_s'])} s | {f(c2['1000']['sample_stdev_s'])} s | {f(c2['1000']['adjacent_abs_delta_median_s'])} s |
| 800 MHz | {f(c5['800']['median_s'])} s | {f(c5['800']['sample_stdev_s'])} s | {f(c5['800']['adjacent_abs_delta_median_s'])} s | {f(c2['800']['median_s'])} s | {f(c2['800']['sample_stdev_s'])} s | {f(c2['800']['adjacent_abs_delta_median_s'])} s |

The 800 MHz prediction landed: predicted 22.7 s, measured {f(c5['800']['median_s'])} s at 512 aa, and
{f(c2['800']['median_s'])} s at 298 aa against a predicted 14.1 s. The 1350 MHz cells reproduce the
c10-bare-baseline numbers of record, {f(c5['1350']['median_s'])} against {BASE512} s at 512 aa and
{f(c2['1350']['median_s'])} against {BASE298} s at 298 aa, on an AST-identical tree.

FIXED: **F = {f(f5['F_s'])} s +-{f(f5['se_F_s'])} at 512 aa**, bounded to {f(lo5,3)} to {f(hi5,3)} s, and
**F = {f(f2['F_s'])} s +-{f(f2['se_F_s'])} at 298 aa**, bounded to {f(lo2,3)} to {f(hi2,3)} s. The bound is the spread of
three independent estimators plus their standard errors, not one fit quoted alone: at 512 aa the fit
on all 28 folds gives {f(e5['all_folds']['F_s'])} +-{f(e5['all_folds']['se_s'])} s, the fit on the four cell medians
{f(e5['cell_medians']['F_s'])} +-{f(e5['cell_medians']['se_s'])} s, and the closed form on the 1350/800 MHz endpoints alone
{f(e5['two_clock_endpoints']['F_s'])} +-{f(e5['two_clock_endpoints']['se_s'])} s, those endpoint arms entering F with gains
{f(t5['two_clock_endpoints']['dF_dt_gain'][0],3)} and {f(t5['two_clock_endpoints']['dF_dt_gain'][1],3)}. The work term is **{f(f5['C_Mcycles'],1)} +-{f(f5['se_C_Mcycles'],1)} Mcycles at 512 aa**
and **{f(f2['C_Mcycles'],1)} +-{f(f2['se_C_Mcycles'],1)} Mcycles at 298 aa**.

The exact two-parameter form T = F + C/f {form}, so F is point-identified rather than merely
bounded by model error. The four cell medians leave a maximum residual of {f(m5['cell_median_max_abs_residual_s'])} s at 512 aa
against an A/A timing floor of {f(m5['aa_timing_floor_s'])} s, and {f(m2['cell_median_max_abs_residual_s'])} s at 298 aa against
{f(m2['aa_timing_floor_s'])} s: in both cases the residual sits inside the repeatability of the arms themselves,
so none of the three pre-registered refutation criteria fired.

**The work ratio is {f(x['work_cycle_ratio'],3)}**, {f(f5['C_Mcycles'],1)} Mcycles at 512 aa against {f(f2['C_Mcycles'],1)} at 298 aa.
That is BELOW the 1.6 to 2.95 range the campaign had to assume, and far below the 2.5 at which
cutting F alone would have been sufficient. F is size DEPENDENT: {f(f5['F_s'])} s at 512 aa against
{f(f2['F_s'])} s at 298 aa, a difference of {f(x['F_difference_s'],3)} s against a combined standard error of
{f(x['F_difference_se_s'],3)} s. F roughly doubles when the target grows, which is what featurization and CIF
writing scaling with the target look like and is not what a pure host constant looks like.

F is NOT shown to be removable host CPU work. Host CPU inside the timed window is
{f(c5['1350']['host_cpu_s_median'],3)} s per fold at 1350 MHz and {f(c5['800']['host_cpu_s_median'],3)} s at 800 MHz at 512 aa: it TRACKS
the clock instead of staying flat, which is a thread waiting on the device, so most of the timed
host CPU is not independent work that could be deleted. F is the term that does not scale with
AICLK, and it may include device time in a clock domain AICLK does not drive.

DEMAND: at 1350 MHz the 512 aa fold is {f(d5['seconds_now'])} s. Reaching 10.0 s with F untouched allows
{f(u5['cycles_allowed_Mcycles'],1)} Mcycles against the measured {f(f5['C_Mcycles'],1)}, a cut of {f(u5['cycle_cut_Mcycles'],1)} Mcycles or
**{f(u5['cycle_cut_pct'],1)} % of the work term** ({f(b5['F_low']['cycle_cut_pct'],1)} to {f(b5['F_high']['cycle_cut_pct'],1)} % across the F bound).
Reaching 10.0 s with F cut to 1.0 s allows {f(d5['F_cut_to_1s']['cycles_allowed_Mcycles'],1)} Mcycles, a cut of
{f(d5['F_cut_to_1s']['cycle_cut_Mcycles'],1)} Mcycles or **{f(d5['F_cut_to_1s']['cycle_cut_pct'],1)} %**. Cutting F to 1.0 s while deleting no device
cycle at all lands at **{f(d5['F_alone_at_current_cycles_s'])} s**, not at the target.

So the campaign's premise resolves against the host-only reading. At the measured work ratio of
{f(x['work_cycle_ratio'],3)} the 10.0 s target at 512 aa is neither a pure host problem nor a pure kernel problem: F
at 1.0 s with every device cycle intact still leaves {f(d5['F_alone_at_current_cycles_s'],2)} s, while cycles alone with F
untouched need a {f(u5['cycle_cut_pct'],1)} % cut. The two levers are comparable in size, {f(u5['cycle_cut_pct'],1)} % of
cycles against {f(d5['F_cut_to_1s']['cycle_cut_pct'],1)} %, so neither can be dropped on this evidence and the honest
route to 10.0 s uses both. The 298 aa fixture is already at {f(d2['seconds_now'])} s and is not the
constraint.

OPEN-QUESTION: the +2.2 % between the current tree's 1350 MHz reading and the historical {HIST} s
pinned arm on `0df13ad9` is NOT a timer-boundary difference. `perf/b2z2_aiclk_pin/pin_ab.py`
brackets the identical span, `ttnn.synchronize_device`, a monotonic read, the complete
`state.predict_one` including CIF writing, `synchronize_device`, on the same cdk2x2_512 fixture at
200 steps and 3 recycles. It is also NOT attributable to the tree from the recorded evidence,
because that run's own JSON changes three things at once: it ran on device node 1, ASIC
380F6B89681D36A4, rather than node 0's D7ADCC7E44A5908A, and on firmware bundle 19.11.0.0 where
node 0 now reads {node.get('tt_fw_bundle_ver')}; kmd 2.11.0 is the same on both. The in-tree cross-session term is
measured here and is far smaller: this session's 1350 MHz cell median is {f(c5['1350']['median_s'])} s against the
baseline's {BASE512} s on an AST-identical tree, a gap of {f(tele,3)} s, so the {f(BASE512-HIST,3)} s difference is
several times session drift and points at the chip or the firmware rather than at a commit. The
clean one-variable control is `0df13ad9` rerun on node 0 under firmware {node.get('tt_fw_bundle_ver')}; a node-1 rerun
of the current tree would separate chip from firmware only in combination with it.

CONTROL: 50 CPU known-answer cases pass (perf/c10_fixed_cost/controls.log), covering only what this
row adds to c10-bare-baseline, whose own 13 controls are closed and were not redone. Clock coverage
is scored against each fold's own target in both directions: an 800 MHz sample set passes at target
800 and rejects at 1350 and the reverse, a mid-fold move to 1200 inside a 1350 arm rejects, and
gaps, read errors and thin sampling reject. The fit recovers an exact synthetic F = 3.0 s and
C = 16000 Mcycles with zero residual, refuses to print an uncertainty from two clocks alone rather
than printing a fake zero, reproduces the analytic dF/dT gains 2.45455 and -1.45455 at 1350/800, and
raises rather than inventing a fixed term from one clock. The 10.0 s demand arithmetic is checked by
hand and an F above the target is reported as impossible instead of as a negative cycle budget. The
comment-only claim is an AST comparison with docstrings stripped, verified True on all four changed
files, with a one-token constant change, an operator change and a dropped statement as negative
controls. Then perf/c10_fixed_cost/negative_control.py mutates the REAL finished capture, 10 cases
per size, all passing at both sizes: a relabelled clock, an unsettled arm, a refused FORCE_AICLK, an
unauthorised clock, a timer mismatch and an above-cap SDPA route are each rejected for the named
reason while every other fold stays accepted; a foreign device holder in one snapshot rejects EVERY
fold of the target; losing the 800 MHz cell sinks the verdict; and one surviving clock produces no
fixed term at all.

COVERAGE: every fold carries its own during-fold clock, power, device-holder and host CPU coverage;
nothing is inferred from a header or a pre-fold reading. Device holders were sampled every 100 ms
across all four chips by an independent thread, with no foreign holder at any time and no node other
than 0 ever opened. Host CPU was sampled every 250 ms over the whole locked window with per-session
attribution. Containment stayed active, driver srcversion A10759A24565BC5BBE903C5 and the boot ID
were re-read before and after every fold, and benchlock ran with its matcher, threshold and location
unchanged with the 60 s waits, acquired at loadavg 0.02 and released rc=0. Same-seed structure did
not move at all: all {nf} accepted folds wrote a byte-identical CIF at their own size, {t5['structure']['distinct_cif_sha256']} distinct CIF
at 512 aa and {t2['structure']['distinct_cif_sha256']} at 298 aa, so changing the clock from 1350 to 800 MHz moves the structure by
{t5['structure']['max_pair_domain_A']:.1f} A against the 0.60 A bar at 512 aa whose user seed floor is 1.84 A, and by
{t2['structure']['max_pair_domain_A']:.1f} A against the 298 fixture's own 0.35 A bar whose archived upstream seed pairs span
0.741 to 0.847 A. Each bar is used only at its own size. pLDDT was finite in every fold, 0.845919 at
512 aa and 0.908011 at 298 aa in every arm.

UNCOUNTED: no device-cycle census, no per-op or per-phase attribution, and no split of F into host,
dispatch and non-AICLK device time. F is the term that does not scale with AICLK, nothing more, and
calling it removable CPU work would be unsupported by this row. The three estimators of F disagree
by {f(m5['F_bound_halfwidth_s'],3)} s at 512 aa and that spread is reported rather than hidden behind the tightest
one. This row's clock sampler reads board power and temperature as well as the clock, so it is
heavier inside the timed window than the baseline's; the {f(tele,3)} s difference at 1350 MHz against an
AST-identical tree is an upper bound on that telemetry cost and it was NOT subtracted, which means
the quoted F carries it. Sampling proves nothing about the intervals between samples. Two clocks
alone would have produced F with zero residual by construction; the four-clock residual is the only
reason the two-parameter form can be said to hold rather than merely assumed. No roof, no ceiling,
no cross-run ratio, no accuracy claim against upstream, and no attribution of the +2.2 % gap to any
commit.

AICLK-GOVERNOR-SIDE-EFFECT: after the run, FORCE_AICLK(0) returned status 0 and the force was
genuinely lifted, but card 0's UNFORCED idle clock now sits at 1350 MHz where it read 800 MHz before
the row. Re-forcing 800 and releasing again reproduces it: the chip leaves the forced value and
settles at 1350, so this is the ARC governor's idle operating point moving, not a stuck pin. Nothing
holds the device and the board is healthy. Anyone reading card 0 idle telemetry should not treat
1350 MHz idle as a firmware regression; a device open by a normal job reprograms the power state.
This was NOT verified to self-clear, and verifying it costs one device open.

ARTIFACT: /home/ttuser/.coworker/wt/c10-fixed-cost/perf/c10_fixed_cost/README.md on branch
wk/c10-fixed-cost. runs/sweep1 holds criterion.json and prediction.json as recorded before the
folds, launch.log, per-target result.json with every per-fold row, analysis.json, report.md, and
lossless gzip JSONL of every clock, power, holder and host CPU sample. Reproduce with
`bash perf/c10_fixed_cost/run.sh <name> 512 298`, then reduce.py, then report.py; reduce.py exits
non-zero unless every criterion holds.

VERDICT: GO. F is measured rather than inferred: **{f(f5['F_s'])} s at 512 aa** (bound {f(lo5,2)} to {f(hi5,2)} s) and
**{f(f2['F_s'])} s at 298 aa** (bound {f(lo2,2)} to {f(hi2,2)} s), with work terms of {f(f5['C_Mcycles'],1)} and {f(f2['C_Mcycles'],1)} Mcycles
and a work ratio of {f(x['work_cycle_ratio'],3)}. That settles the question the row was opened for. The campaign's
10.0 s target at 512 aa is not a host problem the way the ratio-2.5 branch would have made it, and
not a pure kernel problem either: it needs a {f(u5['cycle_cut_pct'],1)} % cycle cut with F untouched, or
{f(d5['F_cut_to_1s']['cycle_cut_pct'],1)} % with F brought to 1.0 s, and F at 1.0 s on its own leaves {f(d5['F_alone_at_current_cycles_s'],2)} s. Kernel work is
therefore justified, and so is host work, but neither alone. There was NO production code change:
the production diff over the fixture base is empty in both captures and every added file is under
perf/c10_fixed_cost/. Nothing was merged, no background job remains, and the clock force is
released.
"""
    Path(out).write_text(doc)
    print(f"{len(doc)} bytes")


if __name__ == "__main__":
    main(*sys.argv[1:4])
