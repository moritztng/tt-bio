#!/usr/bin/env python3
"""PHASE 3 datasheet driver: regenerate DATASHEET sections 4-8 from data, mechanically.

One command, no hand transcription (this lineage has a recorded fabrication incident;
numbers in the datasheet must come from the analysis output, never from memory):

    python3 scripts/abag_xm_deepn_datasheet.py

Runs `abag_xm_deepn_analysis.py --deep` for all four models into the canonical
deepn/analysis_curves.json, then rewrites deepn/DATASHEET.md sections 4-8 (prose +
tables, both owned by this script) between the `## 4.` and `## 9.` headers.
Sections 1-3 and 9 are hand prose and stay untouched. Idempotent; safe on partial
data (every table carries its coverage; verdicts are marked partial).

Stop-rule verdict (pre-registered): walk the campaign ladder (N=16 EXCLUDED from the
ladder; overlay rungs 200/500/1000 are reported in section 6 but are not ladder rungs),
normalize each adjacent pair's oracle gain per doubling, compare against the seed-noise
floor at the lo rung's size. Two consecutive below-floor pairs = knee; N* = the lo rung
of the first of the two. Degenerate pairs (<8 common targets) never clear. "No knee by
N=<max>" is a complete answer.

N=16 flavor: with DEEPN_N16_ARK=1 the analysis restates the rung with the ARK-interface
scorer for the models in N16_ARK_OK (report key `n16_ark_models`); this driver keys its
cross-flavor flags and prose off that list. opendde-abag's N=16 stays the global_dockq
parquet rung (its p2 structures were not retained).
"""
import json, os, subprocess, sys
from pathlib import Path

BASE = Path.home() / "abag_xm" / "deepn"
DATASHEET = BASE / "DATASHEET.md"
CURVES = BASE / "analysis_curves.json"
COSTFIT = BASE / "galaxy" / "costfit_n256.json"
ANALYSIS = Path(__file__).with_name("abag_xm_deepn_analysis.py")
MODELS = ("boltz2", "opendde-abag", "protenix-v2", "esmfold2")
THR = (0.23, 0.49, 0.80)
THR_KEY = {t: str(t).replace(".", "") for t in THR}
LADDER = (50, 64, 256, 512, 1024)  # N=16 excluded from the stop-rule ladder (section 9)

SEC5_PROSE = """\
Stop a model's ladder when the oracle-gain-per-doubling drops below the measured
seed-noise floor (median |oracle(A)\u2212oracle(B)| over disjoint n+n splits of the
largest pool per target, B=200) at two consecutive ladder rungs. N* = the last rung
before the knee. The gain per adjacent rung pair and its CI come from the pairwise
paired bootstrap (section 6); the comparator floor is the lo rung's size (the
difference's noise is dominated by the smaller sample). Pairs flagged `degenerate`
(<8 common targets) never clear the rule. "No knee by N=1024" is a complete answer."""

# Campaign decision 2026-08-05 (Moritz, Telegram): the ladder is capped at N=256 for
# every model -- N=512/1024 will never run. When DEEPN_CAP_DECISION=256 is set, a model
# with no measured knee gets the honest verdict "N* = 256 by decision" (NOT a measured
# knee; the curve state stays visible in its table) and the mid-drain "partial
# coverage" hedge is dropped. A measured knee below the cap still reports as measured.
CAP_DECISION = int(os.environ.get("DEEPN_CAP_DECISION", "0"))

SEC6_PROSE = """\
B=20000 paired bootstrap over targets (seed 20260802, one resample vector shared
across rungs and models). Basis: pairwise adjacent-rung gains over the targets
present at BOTH rungs -- the all-rungs intersection degenerates to zero once sparse
overlay rungs join, so it is never the CI basis. Metrics: oracle mean, user mean,
and the three threshold fractions."""


def n16_note(rep):
    """The N=16 flavor sentence, keyed off the analysis's restatement record."""
    ark = set(rep.get("n16_ark_models", []))
    if not ark:
        return ("The N=16 row is global_dockq-flavored (section 9) -- do not read "
                "depth effects across it.")
    rest = [m for m in MODELS if m not in ark]
    if not rest:
        return ("The N=16 row is the ARK-interface restatement for all four models "
                "(full panel, same flavor as the qb1 arms; section 9).")
    return ("The N=16 row is the ARK-interface restatement for "
            + "/".join(sorted(ark)) + " (full panel, same flavor as the qb1 arms); "
            + "for " + "/".join(rest) + " it stays the global_dockq parquet rung "
            "(section 9) -- do not read depth effects across its 16<->50 step.")

SEC7_PROSE = """\
Saturation attributed to (a) targets where the oracle never reaches the threshold at
any N (unsolvable-at-depth) vs (b) within-target diminishing returns. Metrics:
`solvable_at_top(thr)` = count of targets whose top-N oracle >= thr, and the
common-set within-fold oracle curve: E[oracle] of a uniform m-sample subset (B=200
draws) over the FIXED set of targets whose largest labeled pool has >= D samples,
against the seed-noise floor at the same m. The fixed set is what keeps the curve
monotone-interpretable -- a per-m varying set inverts because the deepest pools are
overlay-heavy hard targets."""


def run_analysis():
    subprocess.run([sys.executable, str(ANALYSIS), "--deep", "--out", str(CURVES)],
                   check=True)
    return json.loads(CURVES.read_text())


def fmt_ci(q):
    return f"{q[1]:+.4f} [{q[0]:+.4f},{q[2]:+.4f}]"


def sec4(rep):
    out = ["## 4. Saturation curves (oracle / user mean DockQ; fractions above 0.23 / 0.49 / 0.80)",
           "",
           "User = the sample the model's own confidence selector would return",
           "(top-plddt for esmfold2, top-`confidence_score` otherwise; `rank_index` excluded).",
           f"Regenerate: `python3 scripts/abag_xm_deepn_datasheet.py` (data: `{CURVES.name}`).",
           "",
           "Rows with different target counts are **not** raw-comparable across N; paired",
           "comparisons always run on common target sets (section 6).",
           n16_note(rep),
           ""]
    for m in MODELS:
        pts = rep.get(m)
        if not pts:
            continue
        out += [f"### {m}", "",
                "| N | targets | oracle | user | o>=0.23 | u>=0.23 | o>=0.49 | "
                "u>=0.49 | o>=0.80 | u>=0.80 | card-h |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in sorted(pts, key=int):
            p = pts[n]
            row = [f"{n}", f"{p['n_targets']}", f"{p['oracle_mean']:.4f}",
                   f"{p['user_mean']:.4f}"]
            for t in THR:
                row.append(f"{p['oracle_ge_' + THR_KEY[t]]:.3f}")
                row.append(f"{p['user_ge_' + THR_KEY[t]]:.3f}")
            row.append(f"{p['card_h']:.1f}")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def stop_verdict(model, rep):
    """Walk the ladder; two consecutive below-floor pairs = knee."""
    gains = rep.get(model + "__pairwise_gain_ci", {})
    floors = rep.get(model + "__deep", {}).get("seed_noise_floor_med", {})
    rows, below, nstar = [], 0, None
    ladder = [n for n in LADDER
              if any(g.startswith(f"{n}->") or g.endswith(f"->{n}") for g in gains)]
    for lo, hi in zip(ladder, ladder[1:]):
        g = gains.get(f"{lo}->{hi}")
        if not g:
            continue
        ci = g["gain_ci"]["oracle"]
        per_dbl = [q / g["doublings"] for q in ci]
        fl = floors.get(str(lo))
        fl_s = f"{fl:.4f}" if fl is not None else "n/a"
        if g["degenerate"]:
            state = "degenerate (nt<8) -- excluded"
        elif fl is None:
            state = "no floor yet"
            below = 0
        elif per_dbl[1] < fl:
            state = "BELOW floor"
            below += 1
        else:
            state = "above floor"
            below = 0
        rows.append(f"| {lo}->{hi} | {g['common_targets']} | {fmt_ci(ci)} | "
                    f"{per_dbl[1]:+.4f} | {fl_s} | {state} |")
        if below == 2 and nstar is None:
            nstar = lo
    if nstar is not None:
        verdict = f"**N* = {nstar}** (knee: two consecutive below-floor doublings)"
    elif CAP_DECISION and ladder and ladder[-1] >= CAP_DECISION:
        verdict = (f"**N* = {CAP_DECISION} by decision** (ladder capped 2026-08-05; "
                   f"no measured knee -- the gains above are the curve's true final state)")
    elif len(ladder) > 1:
        verdict = f"**no knee by N={ladder[-1]}**"
    else:
        verdict = None
    return verdict, rows


def sec5(rep):
    out = ["## 5. Stop rule and N*", "", SEC5_PROSE, ""]
    for m in MODELS:
        verdict, rows = stop_verdict(m, rep)
        if not rows:
            continue
        out += [f"### {m}", "",
                "| pair | nt | oracle gain [95 pct CI] | per doubling | floor(lo) | verdict |",
                "|---|---|---|---|---|---|"] + rows + [""]
        if verdict:
            if not CAP_DECISION:
                verdict += " *(partial coverage -- verdict not final)*"
            out.append(verdict)
            out.append("")
    return "\n".join(out)


def sec6(rep):
    ark16 = set(rep.get("n16_ark_models", []))
    out = ["## 6. Paired bootstrap confidence intervals", "", SEC6_PROSE, "",
           n16_note(rep) + " Gain chains crossing into or out of a global_dockq "
           "N=16 rung mix flavors and are flagged, never quoted as depth effects.", ""]
    for m in MODELS:
        gains = rep.get(m + "__pairwise_gain_ci", {})
        if not gains:
            continue
        out += [f"### {m}", "",
                "| pair | nt | doublings | oracle gain | user gain | flags |",
                "|---|---|---|---|---|---|"]
        for pair, d in gains.items():
            flags = []
            if d["degenerate"]:
                flags.append("degenerate")
            if (pair.startswith("16->") or pair.endswith("->16")) and m not in ark16:
                flags.append("cross-flavor")
            out.append(f"| {pair} | {d['common_targets']} | {d['doublings']:.2f} | "
                       f"{fmt_ci(d['gain_ci']['oracle'])} | {fmt_ci(d['gain_ci']['user'])} | "
                       f"{', '.join(flags)} |")
        out.append("")
    return "\n".join(out)


def sec7(rep):
    out = ["## 7. Exhaustion decomposition", "", SEC7_PROSE, ""]
    for m in MODELS:
        ds = rep.get(m + "__deep")
        if not ds:
            continue
        sol = ds["solvable_at_top"]
        out += [f"### {m}", "",
                f"Solvable at top rung (N={ds['top_rung']}, {ds['n_targets']} targets "
                "with pools): "
                + ", ".join(f">= {t}: {sol.get(str(t), 0)}" for t in THR) + ".", ""]
        wc = ds.get("within_fold_common")
        if wc:
            out += [f"Common-set within-fold curve (fixed set of {wc['n_targets']} "
                    f"targets with pools >= {wc['depth']} samples):", "",
                    "| m | E[oracle] | seed-noise floor |",
                    "|---|---|---|"]
            for k in sorted(wc["curve"], key=int):
                f = wc["floor_med"].get(str(k))
                out.append(f"| {k} | {wc['curve'][k]:.4f} | "
                           + (f"{f:.4f}" if f is not None else "n/a") + " |")
            out.append("")
    return "\n".join(out)


def sec8(rep):
    out = ["## 8. Cost per curve", ""]
    if COSTFIT.exists():
        cf = json.loads(COSTFIT.read_text())
        parts = []
        for m, d in cf["models"].items():
            parts.append(f"{m} {d['chunk_median_s']} s/chunk (p90 {d['chunk_p90_s']} s) "
                         f"-> N=256 rung {d['n256_rung_card_h']} card-h")
        out += ["Galaxy spine **[MEASURED, `galaxy/costfit_n256.json`]**: per-chunk "
                "(64-sample) fleet-folded medians from "
                f"{cf['fleet_folded_ok_chunks']} uncontended chunks "
                f"({cf['reused_ok_chunks_excluded']} skip-and-link reused chunks "
                "excluded -- their seconds were paid in earlier windows): "
                + "; ".join(parts) + f". N=256 rung total {cf['n256_total_card_h']} "
                "card-h; N=512 marginal identical by design (chunks 4-7 fold fresh, "
                "chunks 0-3 skip-and-link at zero marginal cost). Lower bound: the "
                "uncontended basis is early-finisher (small-target) biased; final "
                "refit on the drained rung.", ""]
    out += ["Measured card-hours per rung (fleet records; reused skip-and-link chunks "
            "excluded):", "",
            "| model | rung | card-h |", "|---|---|---|---|"]
    for m in MODELS:
        for n in sorted(rep.get(m, {}), key=int):
            ch = rep[m][n]["card_h"]
            if ch:
                out.append(f"| {m} | {n} | {ch:.1f} |")
    out.append("")
    return "\n".join(out)


def main():
    rep = run_analysis()
    doc = DATASHEET.read_text()
    i4 = doc.index("## 4.")
    i9 = doc.index("## 9.")
    body = "\n\n".join([sec4(rep), sec5(rep), sec6(rep), sec7(rep), sec8(rep)])
    DATASHEET.write_text(doc[:i4] + body + "\n" + doc[i9:])
    print(f"wrote {DATASHEET} (sections 4-8 regenerated from {CURVES.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
