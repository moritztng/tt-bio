"""BC4's quality grader: does a TT arm's design outcome differ from stock BindCraft 2 on the GPU?

BindCraft 2's product is designs, not tensors, so a speed lever is graded on what a campaign produces, not on a
digest (memory bit-exactness-not-required-accuracy-bar-is). This reads BindCraft 2's own campaign output for two
arms and returns a verdict, plus the numbers behind it.

The unit of analysis is a trajectory. Arms pair on `(campaign_seed, trajectory_number)`: campaign.py draws every
trajectory from `jax.random.fold_in(PRNGKey(campaign_seed), trajectory_number)`, independent of what ran before it,
so the same pair names the same draw on either engine. Where pairing holds the test is a sign-flip permutation test
on the paired differences; where it does not it is a two-sample permutation test with Cohen's d beside it. Counts
(accepted designs, completed trajectories) get a two-sided Fisher exact test.

A verdict is only as good as its floor, so the A/A control is not optional: two stock GPU runs at different seeds
must come back indistinguishable, and the spread between them is what every later A/B is read against. If the test
calls two stock runs different, the test is wrong and not the engine.

    python bc2_quality.py reference --arm A1=DIR --arm A2=DIR --target pdl1 --out state/bc4/QUALITY-REF.json
    python bc2_quality.py compare --ref state/bc4/QUALITY-REF.json --arm tt=DIR --out state/bc4/QUALITY.json

Layout under an arm directory, as BindCraft 2 writes it and state/bc4/QUALITY-REF-REQUEST.json lands it:
    <arm>_s<seed>/campaign_s<seed>/1_Trajectories/!_Trajectories.csv   terminated stage, Timing, hash
    <arm>_s<seed>/campaign_s<seed>/1_Trajectories/<design>/<design>_losses.csv   one row per iteration
    <arm>_s<seed>/campaign_s<seed>/2_Refolded/!_Refolded.csv           outcome, i_pTM, pLDDT, Binder_RMSD
    <arm>_s<seed>/campaign_s<seed>/3_Ranked/!_Ranked.csv               accepted designs, absent if none
"""
from __future__ import annotations

import csv
import glob
import math
import os
import random
import re
import statistics as st
from collections import Counter
from dataclasses import dataclass, field
from math import comb
from statistics import NormalDist

# The charter's four metrics, plus the two guardrails a lever must not move quietly.
CONTINUOUS = ("i_ptm", "plddt", "rmsd", "traj_iptm", "traj_plddt", "final_loss")
COUNTED = ("accepted", "completed")
ITERATIONS = "iterations"
MPNN = "mpnn_sequences"
# The charter's "No cheating" clause bars doing less of the model's own work: "never fewer iterations, stages,
# recycles, models or MPNN sequences". These are not quality metrics, they are work guards, and a lever that
# quietly moves one is cheating however good its designs look. Recycles and models are config and never reach the
# CSVs; stages come out as the terminated distribution; iterations and MPNN sequence count are per trajectory.
WORK = (ITERATIONS, MPNN)

ALPHA = 0.05
# The smallest change in a quality metric the campaign would act on. Half an SD: below that, a lever that moved
# i_pTM by that much would not change which binders get ordered, so the harness is not required to resolve it.
CARE_ABOUT_SD = 0.5
POWER = 0.80
SPREAD_MIN_N = 5  # below this a spread comparison cannot separate collapse from one unlucky design
# Every test grade() can fail the verdict on, in one Holm family: the mean/count tests, plus sequence diversity
# and one spread test per continuous metric. A single real effect has to clear alpha/FAMILY_SIZE to be the first
# Holm rejection, so this is what the detectable effect size is quoted at, not the bare alpha.
FAMILY_SIZE = len(CONTINUOUS + COUNTED + WORK) + 1 + len(CONTINUOUS)
PERM_N = 20000  # permutation resamples; tests lower it, nothing else should


# ---------------------------------------------------------------------------
# statistics. These live here and nowhere else; bc2_grade_read.py imports them.
# ---------------------------------------------------------------------------

def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2x2 [[a, b], [c, d]]; p of every table with the same margins at least as unlikely."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not n or not r1 or not c1 or r1 == n or c1 == n:
        return 1.0

    def pr(x):
        return comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)

    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def perm_mean_diff(x: list[float], y: list[float], n: int | None = None, seed: int = 0) -> float:
    """Two-sample permutation test on the difference of means."""
    if len(x) < 2 or len(y) < 2:
        return 1.0
    n = PERM_N if n is None else n
    rng = random.Random(seed)
    obs = abs(st.mean(x) - st.mean(y))
    pool = list(x) + list(y)
    k = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(x)]) - st.mean(pool[len(x):])) >= obs - 1e-12:
            k += 1
    return (k + 1) / (n + 1)


def paired_perm_mean_diff(diffs: list[float], n: int | None = None, seed: int = 0) -> float:
    """Sign-flip permutation test on paired differences: under the null a difference is as likely either way."""
    diffs = [d for d in diffs if d == d]
    if len(diffs) < 2:
        return 1.0
    n = PERM_N if n is None else n
    rng = random.Random(seed)
    obs = abs(st.mean(diffs))
    k = 0
    for _ in range(n):
        if abs(st.mean([d if rng.getrandbits(1) else -d for d in diffs])) >= obs - 1e-12:
            k += 1
    return (k + 1) / (n + 1)


def cohen_d(x: list[float], y: list[float]) -> float:
    """Unpaired effect size, pooled SD."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx, sy = st.stdev(x), st.stdev(y)
    pooled = math.sqrt(((len(x) - 1) * sx * sx + (len(y) - 1) * sy * sy) / (len(x) + len(y) - 2))
    return (st.mean(x) - st.mean(y)) / pooled if pooled else float("nan")


def cohen_dz(diffs: list[float]) -> float:
    """Paired effect size: mean difference over the SD of the differences."""
    diffs = [d for d in diffs if d == d]
    if len(diffs) < 2:
        return float("nan")
    s = st.stdev(diffs)
    return st.mean(diffs) / s if s else float("nan")


def mde_sd_units(n: int, paired: bool, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest effect, in SD units, that n per arm can detect at this alpha and power.

    The normal approximation. A permutation test's power tracks the corresponding t-test to within a few percent at
    n = 30, and test_bc2_quality.py checks that by simulation rather than taking it on trust.
    """
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z / math.sqrt(n) if paired else z * math.sqrt(2.0 / n)


def n_for_sd_units(effect: float, paired: bool, alpha: float = ALPHA, power: float = POWER) -> float:
    """Inverse of mde_sd_units: how many per arm it takes to detect `effect` SD units. The brief asks the harness
    to say what n WOULD work when n=30 does not, so the answer has to be computed, not described."""
    if not effect or effect != effect or effect <= 0:
        return float("inf")
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return (z / effect) ** 2 if paired else 2 * (z / effect) ** 2


def n_for_proportions(p0: float, p1: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Trajectories per arm needed to tell an acceptance rate of p1 from p0. Rare events need very large n."""
    if not 0 <= p0 <= 1 or not 0 <= p1 <= 1 or p0 == p1:
        return float("inf")
    pbar = (p0 + p1) / 2
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return 2 * pbar * (1 - pbar) * (z / (p1 - p0)) ** 2


# ---------------------------------------------------------------------------
# reading BindCraft 2's campaign output
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    seed: int
    number: int
    design: str
    hash: str
    length: int | None
    terminated: str          # "" when the trajectory completed every stage
    iterations: int          # rows of its losses.csv
    autotuned: str           # non-empty means autotune moved a setting: these are not stock trajectories
    traj_iptm: float
    traj_plddt: float
    final_loss: float
    candidates: list[dict] = field(default_factory=list)
    # Per-iteration loss traces, column -> one value per iteration. The distribution tests never read these;
    # trace_agreement() does, to compare the SAME seed on two engines iteration by iteration.
    traces: dict[str, list[float]] = field(default_factory=dict)

    @property
    def completed(self) -> int:
        return int(not self.terminated)

    @property
    def accepted(self) -> int:
        return sum(1 for c in self.candidates if (c.get("outcome") or "").lower() == "accepted")

    @property
    def key(self) -> tuple[int, int]:
        return (self.seed, self.number)


def _f(row: dict, name: str) -> float:
    v = (row or {}).get(name)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _best(cands: list[dict], name: str, lower_is_better: bool = False) -> float:
    """The candidate a user would take: best validated score of this trajectory, or NaN if it produced none."""
    vals = [_f(c, name) for c in cands]
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan")
    return min(vals) if lower_is_better else max(vals)


def _traces(rows: list[dict]) -> dict[str, list[float]]:
    """The numeric columns of a losses.csv, column -> one value per iteration. `phase` and `round` are not losses."""
    out: dict[str, list[float]] = {}
    for name in (rows[0] if rows else {}):
        if name in ("phase", "round"):
            continue
        vals = [_f(r, name) for r in rows]
        if any(v == v for v in vals):
            out[name] = vals
    return out


def read_campaign(project: str, seed: int) -> list[Trajectory]:
    """One campaign_s<seed>/ directory into its trajectories."""
    traj_csv = os.path.join(project, "1_Trajectories", "!_Trajectories.csv")
    if not os.path.exists(traj_csv):
        return []
    by_hash: dict[str, list[dict]] = {}
    for table in ("2_Refolded/!_Refolded.csv", "3_Ranked/!_Ranked.csv"):
        path = os.path.join(project, *table.split("/"))
        if not os.path.exists(path):
            continue
        ranked = table.startswith("3_")
        with open(path) as fh:
            for row in csv.DictReader(fh):
                # 3_Ranked carries no `outcome` column: a design is there because it was accepted.
                by_hash.setdefault(row.get("hash", ""), []).append(
                    dict(row, outcome="accepted") if ranked else row)
    # A candidate appears in both tables when it was accepted; the Ranked copy wins, so dedupe on design name.
    for h, rows in by_hash.items():
        seen: dict[str, dict] = {}
        for row in rows:
            name = row.get("design", "")
            if name not in seen or (row.get("outcome") == "accepted"):
                seen[name] = row
        by_hash[h] = list(seen.values())

    out = []
    with open(traj_csv) as fh:
        for row in csv.DictReader(fh):
            design = row.get("design", "")
            losses = os.path.join(project, "1_Trajectories", design, f"{design}_losses.csv")
            rows = []
            if os.path.exists(losses):
                with open(losses) as lh:
                    rows = list(csv.DictReader(lh))
            cands = by_hash.get(row.get("hash", ""), [])
            # The loss the trajectory finished on, summed over the campaign's own loss columns. The last rows are
            # the mutate phase, which records only i_pTM and pTM, so this reads back to the last row that carries
            # losses at all.
            final = float("nan")
            for row_ in reversed(rows):
                cols = [c for c in row_ if "." in c and not c.endswith(("iptm", "ptm"))]
                vals = [v for v in (_f(row_, c) for c in cols) if v == v]
                if vals:
                    final = sum(vals)
                    break
            out.append(Trajectory(
                seed=seed, number=int(row.get("trajectory") or 0), design=design, hash=row.get("hash", ""),
                length=int(row["length"]) if (row.get("length") or "").isdigit() else None,
                terminated=(row.get("terminated") or "").strip(),
                iterations=len(rows), autotuned=(row.get("autotuned") or "").strip(),
                traj_iptm=_f(row, "i_pTM"), traj_plddt=_f(row, "pLDDT"), final_loss=final, candidates=cands,
                traces=_traces(rows)))
    return out


def read_arm(root: str) -> list[Trajectory]:
    """Every campaign under an arm directory: <root>/<arm>_s<seed>/campaign_s<seed>/, or a bare campaign_s<seed>/."""
    trajectories: list[Trajectory] = []
    seen: set[str] = set()
    for project in sorted(glob.glob(os.path.join(root, "**", "campaign_s*"), recursive=True)):
        if not os.path.isdir(project) or project in seen:
            continue
        seen.add(project)
        hit = re.search(r"campaign_s(\d+)$", project)
        if hit:
            trajectories.extend(read_campaign(project, int(hit.group(1))))
    return trajectories


def metric_values(trajectories: list[Trajectory]) -> dict[str, list[tuple[tuple[int, int], float]]]:
    """Per metric, the (pair key, value) list. NaN means the trajectory produced no such value."""
    vals: dict[str, list[tuple[tuple[int, int], float]]] = {m: [] for m in CONTINUOUS + COUNTED + WORK}
    for t in trajectories:
        vals["i_ptm"].append((t.key, _best(t.candidates, "i_pTM")))
        vals["plddt"].append((t.key, _best(t.candidates, "pLDDT")))
        vals["rmsd"].append((t.key, _best(t.candidates, "Binder_RMSD", lower_is_better=True)))
        vals["traj_iptm"].append((t.key, t.traj_iptm))
        vals["traj_plddt"].append((t.key, t.traj_plddt))
        vals["final_loss"].append((t.key, t.final_loss))
        vals["accepted"].append((t.key, float(t.accepted)))
        vals["completed"].append((t.key, float(t.completed)))
        vals[ITERATIONS].append((t.key, float(t.iterations)))
        # NaN where the trajectory never reached refolding, so early terminations do not read as a
        # cut in MPNN sequences; the terminated distribution is what catches those.
        vals[MPNN].append((t.key, float(len(t.candidates)) if t.candidates else float('nan')))
    return vals


def describe(values: list[float]) -> dict:
    vals = [v for v in values if v == v]
    if not vals:
        return {"n": 0, "n_nan": len(values)}
    return {"n": len(vals), "n_nan": len(values) - len(vals), "mean": st.mean(vals),
            "sd": st.stdev(vals) if len(vals) > 1 else 0.0, "median": st.median(vals),
            "min": min(vals), "max": max(vals)}


# ---------------------------------------------------------------------------
# comparing two arms
# ---------------------------------------------------------------------------

def compare_metric(name: str, a: list[tuple], b: list[tuple], seed: int = 0) -> dict:
    """One metric, one arm against another. Paired on (campaign_seed, trajectory_number) where both arms have it.

    A key pairs only when it occurs once on each side. `dict(b)` over a side that repeats a key keeps the last
    occurrence and drops the rest in silence, which is how a pooled reference turns into a paired test against one
    arbitrarily chosen arm: pooling five B200 arms that all ran campaign seed 101 left final_loss, iterations and
    accepted pairing on 3 of 15 reference trajectories and reporting it as a paired result. Repeated keys fall back
    to the two-sample test, which uses every value, and `n_dropped_to_collisions` says it happened.
    """
    ca, cb = Counter(k for k, _ in a), Counter(k for k, _ in b)
    da, db = dict(a), dict(b)
    shared = sorted(k for k in set(da) & set(db) if ca[k] == 1 and cb[k] == 1)
    diffs = [da[k] - db[k] for k in shared if da[k] == da[k] and db[k] == db[k]]
    xa = [v for _, v in a if v == v]
    xb = [v for _, v in b if v == v]
    collisions = sum(n - 1 for n in ca.values() if n > 1) + sum(n - 1 for n in cb.values() if n > 1)
    res = {"metric": name, "arm": describe([v for _, v in a]), "ref": describe([v for _, v in b]),
           "n_paired": len(diffs), "n_dropped_to_collisions": collisions}

    if name in COUNTED:
        # A count per trajectory: compare totals over trajectories with a Fisher exact test on the 2x2.
        ka, na = int(sum(xa)), len(xa)
        kb, nb = int(sum(xb)), len(xb)
        res.update(test="fisher_exact_two_sided", p=fisher_two_sided(ka, na - ka, kb, nb - kb),
                   rate_arm=ka / na if na else float("nan"), rate_ref=kb / nb if nb else float("nan"),
                   k=[ka, kb], n=[na, nb])
        p0 = kb / nb if nb else 0.0
        # What this n can actually see: the rate a 30-trajectory arm could tell apart from the reference's.
        res["n_for_half_or_double"] = min(n_for_proportions(p0, p0 / 2), n_for_proportions(p0, min(1.0, p0 * 2)))
        return res

    # Pairing is the stronger design per observation, but only if it keeps the observations. Where the reference
    # landed partial, pairing covers a few trajectories while the two-sample test still uses every value: with a
    # 4-trajectory overlap against 10 vs 14 values, the paired MDE is 1.898 SD and the unpaired 1.698 SD, so the
    # paired branch would discard data to take the weaker test. Pick whichever detects less. This looks only at
    # the sample sizes, never at the values, so it cannot bend the false-positive rate.
    can_pair, can_unpair = len(diffs) >= 2, len(xa) >= 2 and len(xb) >= 2
    if can_pair and can_unpair:
        can_pair = mde_sd_units(len(diffs), paired=True) <= mde_sd_units(min(len(xa), len(xb)), paired=False)
        res["test_choice"] = ("paired: it covers enough of the data" if can_pair else
                              "two-sample: pairing covered too little of the data to be worth it")
    if can_pair:
        res.update(test="paired_sign_flip_permutation", paired=True, p=paired_perm_mean_diff(diffs, seed=seed),
                   effect=cohen_dz(diffs), effect_kind="cohen_dz", mean_diff=st.mean(diffs),
                   sd_diff=st.stdev(diffs), exact_match=sum(1 for d in diffs if abs(d) < 1e-12) / len(diffs),
                   mde_sd_units=mde_sd_units(len(diffs), paired=True),
                   mde_metric_units=mde_sd_units(len(diffs), paired=True) * st.stdev(diffs))
    elif can_unpair:
        pooled = math.sqrt((st.variance(xa) + st.variance(xb)) / 2)
        n = min(len(xa), len(xb))
        res.update(test="two_sample_permutation", paired=False, p=perm_mean_diff(xa, xb, seed=seed),
                   effect=cohen_d(xa, xb), effect_kind="cohen_d", mean_diff=st.mean(xa) - st.mean(xb),
                   mde_sd_units=mde_sd_units(n, paired=False),
                   mde_metric_units=mde_sd_units(n, paired=False) * pooled)
    else:
        res.update(test="none", p=float("nan"), note="too few values to test")
    return res


def holm(results: list[dict], alpha: float = ALPHA) -> list[dict]:
    """Holm-Bonferroni across the metrics. Six tests on one arm would otherwise fail one in four arms by chance."""
    testable = [r for r in results if isinstance(r.get("p"), float) and r["p"] == r["p"]]
    for rank, r in enumerate(sorted(testable, key=lambda r: r["p"])):
        r["alpha_holm"] = alpha / (len(testable) - rank)
        r["significant"] = r["p"] < r["alpha_holm"]
    for r in results:
        r.setdefault("significant", False)
    return results


def _diversity_values(trajectories: list[Trajectory]) -> list[tuple[tuple[int, int], float]]:
    """Per trajectory, the fraction of its own candidate sequences that are distinct. NaN if it produced none."""
    out = []
    for t in trajectories:
        seqs = [c.get("Binder_Sequence", "") for c in t.candidates if c.get("Binder_Sequence")]
        out.append((t.key, len(set(seqs)) / len(seqs) if seqs else float("nan")))
    return out


def collapse_tests(trajectories: list[Trajectory], ref: list[Trajectory]) -> list[dict]:
    """The p-bearing half of the collapse check: sequence diversity, then spread per metric.

    Returned un-corrected so the caller can put them in ONE Holm family with the mean and count tests. They are a
    separate family only when collapse_check is called on its own.
    """
    # Diversity is compared per trajectory, not over the pooled candidate list. Candidates are nested inside a
    # trajectory: they are MPNN's sequences for one backbone, so they are not independent draws. In all six real
    # campaigns of 2026-10-19 every duplicate was within a trajectory and none across one (TT 5 within, 0 across),
    # so pooling 58 correlated candidates into one Fisher table counts them as 58 independent observations and
    # reports a significance the data does not carry. The trajectory is this grader's unit of analysis everywhere
    # else; it is the unit here too, which also gets the test pairing and an effect size.
    tests = [compare_metric("sequence_diversity", _diversity_values(trajectories), _diversity_values(ref))]
    # Narrowed spread is the other face of collapse, and it is tested the same way: a ratio of standard deviations
    # is not evidence at these n. Stock's own i_pTM sd on 3 trajectories was 0.396, carried almost entirely by one
    # bad design, so a fixed "under a fifth of the reference" rule flagged the stock attn-xla arm too.
    # Brown-Forsythe instead: permute labels on the absolute deviation from each side's own median.
    va, vr = metric_values(trajectories), metric_values(ref)
    for m in CONTINUOUS:
        xa = [v for _, v in va[m] if v == v]
        xr = [v for _, v in vr[m] if v == v]
        r = {"metric": f"spread:{m}", "p": float("nan")}
        tests.append(r)
        if len(xa) < SPREAD_MIN_N or len(xr) < SPREAD_MIN_N:
            continue
        da = [abs(v - st.median(xa)) for v in xa]
        dr = [abs(v - st.median(xr)) for v in xr]
        r["sd"], r["ref_sd"] = describe(xa)["sd"], describe(xr)["sd"]
        if st.mean(da) >= st.mean(dr):
            continue  # the arm is at least as spread out as stock; collapse is the one-sided question
        r["p"] = perm_mean_diff(da, dr, seed=0)
    return tests


def collapse_check(trajectories: list[Trajectory], ref: list[Trajectory] | None = None,
                   tests: list[dict] | None = None) -> dict:
    """Has the arm stopped exploring? Distinct binder sequences, and per-metric spread against the reference's.

    Sequence diversity is graded against the reference, not against a number picked in advance. A bare threshold
    measures sample size rather than collapse: MPNN re-emits a sequence now and then, and on a handful of
    candidates one coincidental duplicate costs more of a fixed budget than the same duplication RATE does on
    hundreds. Measured on the real B200 campaigns of 2026-10-19, a `distinct < 0.9` rule called four of six stock
    arms collapsed, every one of them on a single repeated sequence (`attn-stock` 4/5 distinct, `attn-xla` 13/15),
    and would have failed the TT arm for duplicating at exactly the stock rate. Only catastrophic collapse is
    absolute now; everything short of it is a Fisher test against the reference's own duplicate rate.
    """
    seqs = [c.get("Binder_Sequence", "") for t in trajectories for c in t.candidates if c.get("Binder_Sequence")]
    distinct = len(set(seqs)) / len(seqs) if seqs else float("nan")
    top = max(Counter(seqs).values()) if seqs else 0
    out = {"candidates": len(seqs), "distinct_sequence_fraction": distinct,
           "top_sequence_share": top / len(seqs) if seqs else float("nan"),
           "collapsed": False, "reasons": []}
    # Catastrophic collapse, and only that, is judged without a reference: one sequence taking half the candidates,
    # or half the candidates being repeats. Neither can be reached by a stray duplicate or two.
    if len(seqs) >= 4 and top / len(seqs) >= 0.5:
        out["collapsed"] = True
        out["reasons"].append(f"one sequence is {top} of {len(seqs)} candidates")
    if len(seqs) >= 4 and distinct < 0.5:
        out["collapsed"] = True
        out["reasons"].append(f"only {distinct:.0%} of {len(seqs)} candidate sequences are distinct")
    if ref:
        rseqs = [c.get("Binder_Sequence", "") for t in ref for c in t.candidates if c.get("Binder_Sequence")]
        out["ref_distinct_sequence_fraction"] = len(set(rseqs)) / len(rseqs) if rseqs else float("nan")
        # Holm'd here only when the caller did not already do it in a wider family. grade() does, so that the
        # verdict is one family of tests rather than two each controlled at alpha: measured on synthetic A/A draws
        # at n=30, two independent families put the verdict's own false-positive rate near 10% against a nominal 5%.
        spread = holm(collapse_tests(trajectories, ref)) if tests is None else tests
        out["spread"] = spread
        diversity = spread[0]
        if diversity.get("significant") and (diversity["arm"].get("mean") or 0) < (diversity["ref"].get("mean") or 0):
            out["collapsed"] = True
            out["reasons"].append(
                f"sequence diversity {diversity['arm']['mean']:.0%} of candidates distinct per trajectory against "
                f"the reference's {diversity['ref']['mean']:.0%} ({diversity['test']} p={diversity['p']:.3g})")
        for r in spread[1:]:
            if r.get("significant"):
                out["collapsed"] = True
                out["reasons"].append(
                    f"{r['metric']} {r['sd']:.4g} against the reference's {r['ref_sd']:.4g} "
                    f"(Brown-Forsythe permutation p={r['p']:.3g})")
    return out


def nan_check(trajectories: list[Trajectory]) -> dict:
    """A NaN anywhere in a design's own metrics is a hard fail; a trajectory that terminated early has none, which
    is normal, so only completed trajectories are read."""
    bad = []
    for t in trajectories:
        if t.completed and (t.traj_iptm != t.traj_iptm or t.traj_plddt != t.traj_plddt):
            bad.append(f"{t.design}: completed but i_pTM/pLDDT missing")
        for c in t.candidates:
            for col in ("i_pTM", "pLDDT", "Binder_RMSD"):
                v = _f(c, col)
                if v != v and (c.get(col) or "").strip() not in ("", None):
                    bad.append(f"{c.get('design')}: {col}={c.get(col)!r}")
                if v == v and math.isinf(v):
                    bad.append(f"{c.get('design')}: {col} is infinite")
    return {"ok": not bad, "offenders": bad[:20], "n_offenders": len(bad)}


def autotune_check(trajectories: list[Trajectory]) -> dict:
    """Any trajectory autotune touched is not a stock trajectory and cannot be compared like one.

    The reference is run as 10-trajectory campaigns precisely so this never fires (BindCraft 2 reviews every 10 and
    the review first fires on trajectory 11). If it fires, the run was built wrong, not the engine.
    """
    tuned = [t.design for t in trajectories if t.autotuned]
    return {"ok": not tuned, "n_autotuned": len(tuned), "offenders": tuned[:10]}


def _resolution(values: list[float]) -> float:
    """The quantum BindCraft 2 actually recorded these at, read off the values themselves.

    The losses.csv is written rounded, so a trace comparison cannot resolve anything finer than this and must say
    so rather than quoting a deviation the file could not have carried.
    """
    vals = [v for v in values if v == v]
    for places in range(0, 13):
        q = 10.0 ** -places
        if all(abs(v - round(v, places)) < 1e-12 for v in vals):
            return q
    return 0.0


def _pairs(a: list[Trajectory], b: list[Trajectory]) -> list[tuple[Trajectory, Trajectory]]:
    """Trajectories sharing a (seed, number) key, where the key occurs exactly once on each side."""
    ca, cb = Counter(t.key for t in a), Counter(t.key for t in b)
    ia = {t.key: t for t in a if ca[t.key] == 1}
    ib = {t.key: t for t in b if cb[t.key] == 1}
    return [(ia[k], ib[k]) for k in sorted(set(ia) & set(ib))]


def trace_agreement(a: list[Trajectory], b: list[Trajectory], label_a: str, label_b: str) -> dict:
    """Compare the SAME (seed, trajectory) on two engines, iteration by iteration.

    This is not one of the verdict's tests and it is not a distribution test. It answers a narrower question that
    n=30 is not needed for: given the same seed and the same sha, does the other engine start from the same place?

    Read ITERATION 1 and nothing else as evidence of arithmetic. A design trajectory is gradient descent, so a
    difference of one ulp at iteration 1 is amplified by every iteration after it; two engines agreeing at
    iteration 1 and parting by iteration 50 is what correct arithmetic looks like here, not a defect. Late
    divergence is reported because it is interesting, never because it is graded.

    The ceiling on what this can see is each column's quantum: BindCraft 2 rounds the losses it writes, so a
    deviation below that quantum was never in the file.

    ALWAYS read this against a floor from two runs of the REFERENCE engine, never from the engine under test.
    Measured 2026-10-19 on the campaigns on this host: TT reproduces its own seed exactly (0 quanta, 3 pairs)
    and the B200 does not (20/70/90 quanta on hPDL1.compactness, 3 pairs, same sha and same per-trajectory
    settings). A floor taken on the deterministic side reads 0 and turns the reference engine's own scatter into
    a finding about the engine under test, which is how this function first got read.
    """
    paired = _pairs(a, b)
    # Each loss column is written at its own quantum, so a deviation is only meaningful in units of that column's
    # quantum. Measure it once over every value either side put in the column, not per trajectory.
    quantum: dict[str, float] = {}
    for ta, tb in paired:
        for c in set(ta.traces) & set(tb.traces):
            quantum.setdefault(c, []).extend(ta.traces[c] + tb.traces[c])
    quantum = {c: _resolution(v) for c, v in quantum.items()}

    per: list[dict] = []
    by_column: dict[str, float] = {c: 0.0 for c in quantum}
    for ta, tb in paired:
        cols = sorted(set(ta.traces) & set(tb.traces))
        first, first_col = 0.0, None
        worst, worst_col = 0.0, None
        for c in cols:
            xa, xb, q = ta.traces[c], tb.traces[c], quantum[c] or 1.0
            if xa and xb and xa[0] == xa[0] and xb[0] == xb[0]:
                d = abs(xa[0] - xb[0]) / q
                by_column[c] = max(by_column[c], d)
                if d > first:
                    first, first_col = d, c
            n = min(len(xa), len(xb))
            d = max((abs(xa[i] - xb[i]) / q for i in range(n)
                     if xa[i] == xa[i] and xb[i] == xb[i]), default=0.0)
            if d > worst:
                worst, worst_col = d, c
        per.append({
            "key": list(ta.key), "design_a": ta.design, "design_b": tb.design,
            "design_name_matches": ta.hash == tb.hash,
            "iterations": [ta.iterations, tb.iterations],
            "columns_compared": len(cols),
            "iteration_1_max_dev_quanta": round(first, 3), "iteration_1_worst_column": first_col,
            "whole_trace_max_dev_quanta": round(worst, 3), "whole_trace_worst_column": worst_col,
        })
    firsts = [p["iteration_1_max_dev_quanta"] for p in per if p["columns_compared"]]
    return {
        "a": label_a, "b": label_b,
        "n_paired": len(paired), "n_compared": len(firsts),
        "unit": "quanta: one unit is the smallest difference the column's own rounding could record",
        "quantum_per_column": quantum,
        "resolution_note":
            "BindCraft 2 writes these losses rounded. Nothing finer than a column's quantum was ever in the file, "
            "so 0 means 'equal at the recorded resolution', never 'bit-exact'.",
        "design_name_note":
            "`design_name_matches` is NOT evidence of agreement. BC2 hashes the design REQUEST, not the produced "
            "design: design_identity.py:EXCLUDED_SETTING_NAMES drops attention_backend, subbatch_size, use_cueq, "
            "gpu_ids, auto_multi_gpu, design_workers, workers_per_gpu and max_trajectories from the name. On the "
            "2026-10-19 B200-vs-TT data every name matched while the traces differed.",
        "iteration_1_max_dev_quanta": max(firsts) if firsts else None,
        "iteration_1_by_column": {c: round(v, 3) for c, v in sorted(by_column.items())},
        "grades_nothing": "Reported, not graded. The verdict comes from grade(); this is evidence about "
                          "arithmetic, available at n=1 where a distribution test needs n=30.",
        "per_trajectory": per,
    }


def grade(arm: list[Trajectory], ref: list[Trajectory], label: str, seed: int = 0) -> dict:
    va, vr = metric_values(arm), metric_values(ref)
    means = [compare_metric(m, va[m], vr[m], seed=seed) for m in CONTINUOUS + COUNTED + WORK]
    # ONE Holm family over every test that can fail the verdict. Holm'ing the mean tests and the collapse tests
    # separately leaves two families each controlled at alpha, and the verdict is their OR: on synthetic A/A draws
    # at n=30 that put the verdict's false-positive rate near 10% against a nominal 5%, and the charter's A/A
    # control is a statement about the VERDICT, not about any one test inside it.
    ctests = collapse_tests(arm, ref)
    holm(means + ctests)
    results, collapse = means, collapse_check(arm, ref, tests=ctests)
    nan, tune = nan_check(arm), autotune_check(arm)
    iters = next(r for r in results if r["metric"] == ITERATIONS)
    moved = [r["metric"] for r in results if r.get("significant")]
    # An arm that produced nothing testable is INCONCLUSIVE, never PASS. Every test returning NaN means no test
    # ran, and reporting that as "indistinguishable from stock" reads absence of evidence as evidence of absence:
    # a lever whose run crashed and wrote no trajectories graded PASS, which is the single most dangerous verdict
    # this grader could emit. The campaign gate's n_per_arm >= 30 also catches it, but a verdict has to be honest
    # on its own, since any later reader sees the verdict and not the n beside it.
    # A test only counts as having run if BOTH sides put values into it. Fisher still returns p=1.0 when either
    # margin is zero, so "has a p" is not the same as "was tested".
    # The reference side matters as much as the arm side, and guarding only the arm left a real hole: the A/A
    # control grades arm A1 against arm A2, so a session whose A2 campaigns all crashed produced a *PASS* A/A on
    # an empty reference. The campaign gate requires aa_control.verdict == PASS, so with A1 past 30 that would
    # have passed the gate with no A/A control at all. Found by dry-running bc4_quality_session.sh, 2026-10-19.
    testable = [r for r in results + ctests
                if isinstance(r.get("p"), float) and r["p"] == r["p"]
                and (r.get("arm") or {}).get("n", 0) >= 2 and (r.get("ref") or {}).get("n", 0) >= 2]
    reasons = []
    if not testable:
        return {
            "arm": label, "verdict": "INCONCLUSIVE",
            "reasons": [f"no test could run: the arm has {len(arm)} trajectories and the reference {len(ref)}"],
            "n_per_arm": [len(arm), len(ref)], "test": "none ran",
            "metrics": {r["metric"]: r for r in results},
            "iteration_distribution_unchanged": None,
            "nan": nan, "collapse": collapse, "autotune": tune,
            "equivalence_note": "Nothing was compared, so nothing is ruled out in either direction.",
        }
    if moved:
        reasons.append("metrics moved: " + ", ".join(moved))
    if not nan["ok"]:
        reasons.append(f"{nan['n_offenders']} NaN or infinite design metrics")
    if collapse["collapsed"]:
        reasons.append("collapse: " + "; ".join(collapse["reasons"]))
    if not tune["ok"]:
        reasons.append(f"{tune['n_autotuned']} trajectories were autotuned, so they are not stock")
    return {
        "arm": label, "verdict": "PASS" if not reasons else "FAIL", "reasons": reasons,
        "n_per_arm": [len(arm), len(ref)],
        "test": ("paired sign-flip permutation on (campaign_seed, trajectory_number) where arms pair, two-sample "
                 "permutation otherwise, Fisher exact two-sided on counts, Holm-Bonferroni across metrics at "
                 f"alpha={ALPHA}"),
        "metrics": {r["metric"]: r for r in results},
        "iteration_distribution_unchanged": not iters.get("significant", False),
        # The charter's work guards, reported on their own so a reader does not have to pick them out of metrics:
        # a lever that cuts them is cheating however good its designs look. terminated_distribution carries the
        # stages; recycles and models are config and never reach the CSVs, so they are not gradable from output.
        "work_unchanged": {m: not next(r for r in results if r["metric"] == m).get("significant", False)
                           for m in WORK},
        "terminated_distribution": {k: [sum(1 for t in a if (t.terminated or "completed") == k) for a in (arm, ref)]
                                    for k in sorted({(t.terminated or "completed") for t in list(arm) + list(ref)})},
        "nan": nan, "collapse": collapse, "autotune": tune,
        "equivalence_note": (
            "A non-significant result is not proof of equality. Each metric carries mde_metric_units: the smallest "
            "effect this n could have detected at 80% power. Effects smaller than that are not ruled out."),
    }


# ---------------------------------------------------------------------------
# the reference distribution, and the A/A control that gives it a floor
# ---------------------------------------------------------------------------

#  The gate (workstreams/_bc4_quality_check.py) requires exactly these four per target.
GATE_METRICS = ("acceptance_rate", "i_ptm", "plddt", "rmsd")


def acceptance(trajectories: list[Trajectory]) -> dict:
    """Accepted designs per trajectory, and the candidate pass rate underneath it."""
    n = len(trajectories)
    acc = sum(t.accepted for t in trajectories)
    cands = [c for t in trajectories for c in t.candidates]
    passed = sum(1 for c in cands if (c.get("outcome") or "").lower() == "accepted")
    return {"n_trajectories": n, "accepted_designs": acc,
            "rate_per_trajectory": acc / n if n else float("nan"),
            "candidates": len(cands), "candidates_accepted": passed,
            "candidate_pass_rate": passed / len(cands) if cands else float("nan")}


def target_summary(trajectories: list[Trajectory], paths: list[str]) -> dict:
    vals = metric_values(trajectories)
    metrics = {m: describe([v for _, v in vals[m]]) for m in CONTINUOUS + (ITERATIONS,)}
    metrics["acceptance_rate"] = acceptance(trajectories)
    term: dict[str, int] = {}
    for t in trajectories:
        term[t.terminated or "completed"] = term.get(t.terminated or "completed", 0) + 1
    return {"n": len(trajectories), "metrics": metrics, "paths": sorted(paths),
            "terminated": term, "iterations": describe([float(t.iterations) for t in trajectories]),
            "autotune": autotune_check(trajectories), "nan": nan_check(trajectories),
            "collapse": collapse_check(trajectories)}


def power_statement(ref_by_target: dict[str, list[Trajectory]]) -> dict:
    """What n = 30 can and cannot see, in the metrics' own units. The brief requires this be stated, not implied.

    Two MDEs per metric, because the verdict is not one test. `paired`/`unpaired` are the bare per-test numbers.
    `paired_holm`/`unpaired_holm` are what a metric has to move to fail the VERDICT, which needs it to clear
    alpha/FAMILY_SIZE as the first Holm rejection: at n=30 and the 17 tests the family holds today that is
    0.697 SD paired rather than 0.511, and 0.985 unpaired rather than 0.723. Quoting only the bare number would overstate the harness by about 36% and let a
    lever ship on a "no significant difference" the test never had the power to find.
    """
    a_holm = ALPHA / FAMILY_SIZE
    out = {
        "n_per_arm": 30,
        "alpha": ALPHA, "power": POWER, "family_size": FAMILY_SIZE, "alpha_holm_strictest": a_holm,
        "mde_sd_units": {"paired": mde_sd_units(30, paired=True), "unpaired": mde_sd_units(30, paired=False),
                         "paired_holm": mde_sd_units(30, True, alpha=a_holm),
                         "unpaired_holm": mde_sd_units(30, False, alpha=a_holm)},
        "mde_note": ("paired/unpaired are per-test. paired_holm/unpaired_holm are what it takes to fail the "
                     f"verdict, which puts all {FAMILY_SIZE} tests in one Holm family. Holm is adaptive, so a "
                     "metric that moves alongside others is judged somewhere between the two; the _holm number "
                     "is the honest bound for a single metric moving on its own."),
        "detectable_in_metric_units": {}, "what_n_would_see_an_effect_we_care_about": {},
    }
    for target, trajectories in ref_by_target.items():
        vals = metric_values(trajectories)
        per = {}
        for m in CONTINUOUS:
            d = describe([v for _, v in vals[m]])
            if d.get("n", 0) > 1 and d.get("sd"):
                per[m] = {"ref_sd": d["sd"], "paired": mde_sd_units(30, True) * d["sd"],
                          "unpaired": mde_sd_units(30, False) * d["sd"],
                          "paired_holm": mde_sd_units(30, True, alpha=a_holm) * d["sd"],
                          "unpaired_holm": mde_sd_units(30, False, alpha=a_holm) * d["sd"]}
        acc = acceptance(trajectories)
        p0 = acc["rate_per_trajectory"]
        # Acceptance is a count, so its power depends entirely on the rate: common acceptance is testable at n=30,
        # rare acceptance is not testable at any n we could pay for. Derive which regime this target is in.
        needed = n_for_proportions(p0, p0 / 2) if p0 and p0 == p0 else float("inf")
        powered = needed <= 30
        per["acceptance_rate"] = {
            "ref_rate_per_trajectory": p0,
            "n_needed_to_detect_a_halving": needed,
            "verdict": "powered at n=30" if powered else "underpowered at n=30",
            "note": (("Acceptance is common enough here that n=30 can test a halving of the rate directly."
                      if powered else
                      "Acceptance is too rare for any affordable n to test: detecting a halving would take "
                      f"{needed:.0f} trajectories per arm. It is read only as a guardrail that the arm has not "
                      "stopped accepting entirely.")
                     + " The grade rests on i_pTM, pLDDT, binder RMSD and the loss traces, which are continuous "
                       "and do have power at n=30."),
        }
        # What n WOULD see an effect worth caring about. Two sizes, because a trajectory is not an observation:
        # 8 of the 14 banked B200 trajectories terminated before they refolded and carry no i_pTM at all, so the
        # trajectories to BUY are the values needed divided by the measured yield.
        per_care = {}
        for m in CONTINUOUS:
            vals = [v for _, v in metric_values(trajectories)[m] if v == v]
            yield_rate = len(vals) / len(trajectories) if trajectories else 0.0
            n_vals = n_for_sd_units(CARE_ABOUT_SD, paired=True, alpha=a_holm)
            per_care[m] = {
                "values_needed": n_vals,
                "measured_yield_per_trajectory": yield_rate,
                "trajectories_needed": n_vals / yield_rate if yield_rate else float("inf"),
                "n30_is_enough": bool(yield_rate and 30 * yield_rate >= n_vals),
            }
        out["what_n_would_see_an_effect_we_care_about"][target] = {
            "care_about_sd_units": CARE_ABOUT_SD,
            "per_metric": per_care,
            "note": (f"Calling a {CARE_ABOUT_SD} SD change at the verdict's corrected alpha takes "
                     f"{n_for_sd_units(CARE_ABOUT_SD, True, a_holm):.0f} paired values, so n=30 trajectories per "
                     "arm cannot do it for ANY metric, not even one every trajectory carries. "
                     "`trajectories_needed` is the honest ask per metric, and it is larger than that floor "
                     "wherever trajectories terminate before they refold. Where `n30_is_enough` is false, a "
                     "verdict of 'no significant difference' at n=30 means the test could not see the effect, "
                     "not that the effect is absent; n=30 is a guardrail against gross regressions "
                     f"({mde_sd_units(30, True, a_holm):.2f} SD and up), not a fine-grained equivalence claim."),
        }
        out["detectable_in_metric_units"][target] = per
    return out


def campaign_cost(arm_dir: str, rates: dict | None, concurrency: int = 1) -> dict:
    """What one campaign's trajectories cost, from its own `runs.jsonl`.

    A reference that does not carry its price makes the next top-up a planning exercise again: somebody has to
    re-derive how much a trajectory costs before they can decide whether to buy the campaigns this one could not
    afford. The runner already writes `traj_s` per trajectory and the GPU it ran on, so the price is a reading,
    not an estimate.

    Two things the obvious version gets wrong, and this one does not:

    * **One rate does not price a mixed reference.** The 2026-10-19 reference pools 14 trajectories from a
      $6.251/h B200 with 20 from a $0.54/h RTX 4090. Multiplying every campaign by one hourly rate prices
      whichever card it was not. `rates` maps a substring of the recorded `gpu` string to its own $/h, and a
      campaign whose card matches nothing is left unpriced rather than guessed.
    * **`traj_s` is wall clock, and wall clock is shared.** With `concurrency` campaigns on one billed card the
      card bills once for all of them, so the cost of a trajectory is its wall time divided by the concurrency
      it ran at. The B200 campaigns ran three to a card, which is why their 262 s of wall per trajectory is the
      same thing as the 85 s delivered rate quoted elsewhere.
    """
    import json as _json
    path = os.path.join(arm_dir, "runs.jsonl")
    if not os.path.isfile(path):
        return {}
    gpu, spans = None, []
    with open(path) as fh:
        for line in fh:
            try:
                row = _json.loads(line)
            except ValueError:
                continue
            if row.get("ev") == "start":
                gpu = row.get("gpu")
            elif row.get("ev") == "rep" and isinstance(row.get("traj_s"), (int, float)):
                spans.append(float(row["traj_s"]))
    if not spans:
        return {"gpu": gpu, "n": 0}
    s_per = sum(spans) / len(spans)
    out = {"gpu": gpu, "n": len(spans), "concurrency": concurrency,
           "wall_s_per_trajectory": round(s_per, 1),
           "delivered_s_per_trajectory": round(s_per / max(concurrency, 1), 1)}
    dph = next((v for k, v in (rates or {}).items() if k and gpu and k in gpu), None)
    if dph:
        out["dph_usd_per_hour"] = dph
        out["usd_per_trajectory"] = round(s_per / max(concurrency, 1) * dph / 3600.0, 4)
    elif rates:
        out["unpriced_why"] = f"no hourly rate given for {gpu!r}"
    return out


def merge_requests(requests: list[dict]) -> dict:
    """Union of several contracts, for a reference whose targets were bought in separate sessions.

    The two-target reference the charter asks for cannot come from one rental: pdl1 and il2r were bought on
    different boxes on different days. Rather than hand-maintain a third request file that drifts from the two
    that were actually run, the union is taken here, keyed on run id so a request listed twice cannot double a
    campaign (which would repeat its (campaign_seed, trajectory_number) keys and break pairing).
    """
    out = dict(requests[0])
    for key in ("runs", "banked_runs", "unbought"):
        seen, merged = set(), []
        for req in requests:
            for item in req.get(key) or []:
                k = item.get("id") or item.get("what")
                if k in seen:
                    continue
                seen.add(k)
                merged.append(item)
        if merged:
            out[key] = merged
    out["dph_usd_per_hour"] = {k: v for req in requests for k, v in (req.get("dph_usd_per_hour") or {}).items()}
    out["merged_from"] = [req.get("schema") for req in requests]
    return out


def build_reference(request: dict, root, seed: int = 0, provisional: bool = False) -> dict:
    """Read every campaign the request asked for, grade the A/A control, and state the power.

    `banked_runs` are campaigns that already landed and must NOT be run again: the 2026-10-19 B200 capped
    mid-campaign and left 14 usable pdl1 trajectories at seeds 101-103. They are read into the reference exactly
    like the rest, but they are kept out of `runs` so the runner cannot repeat them. Re-running one would repeat
    its (campaign_seed, trajectory_number) keys, which then cannot pair.
    """
    rates = request.get("dph_usd_per_hour") or {}
    concurrency = int(request.get("concurrency") or 1)
    by_target: dict[str, dict[str, list[Trajectory]]] = {}
    paths: dict[str, list[str]] = {}
    per_campaign: dict[str, dict] = {}
    missing = []
    # Several roots, because a two-target reference is bought in more than one rental and each session pulls to
    # its own directory. First root that holds the campaign wins.
    roots = [root] if isinstance(root, str) else list(root)
    for run in list(request.get("runs", [])) + list(request.get("banked_runs", [])):
        leaf = os.path.join(run["target"], f"{run['arm']}_s{run['campaign_seed']}")
        arm_dir = next((os.path.join(r, leaf) for r in roots if os.path.isdir(os.path.join(r, leaf))), None)
        if arm_dir is None:
            missing.append({"id": run["id"], "tier": run["tier"],
                            "expected_at": [os.path.join(r, leaf) for r in roots]})
            continue
        got = read_arm(arm_dir)
        if len(got) != run["trajectories"]:
            missing.append({"id": run["id"], "tier": run["tier"], "expected_at": arm_dir,
                            "note": f"{len(got)} trajectories, the request asked for {run['trajectories']}"})
        by_target.setdefault(run["target"], {}).setdefault(run["arm"], []).extend(got)
        paths.setdefault(run["target"], []).append(arm_dir)
        # A banked campaign may have run at a different concurrency than this session, and the two cannot
        # share one number: the B200 ran three campaigns to a card, this one runs one.
        priced = campaign_cost(arm_dir, rates, int(run.get("concurrency") or concurrency))
        if priced:
            per_campaign[run["id"]] = priced

    targets = {t: target_summary([x for arm in arms.values() for x in arm], paths.get(t, []))
               for t, arms in by_target.items()}

    # The A/A control: two stock runs at different seeds, on the target that has two arms.
    #
    # Which two arms is a judgement, not an alphabetical accident. Taking the first two in sorted order is fine
    # while every arm came off the same box, and wrong the moment one did not: the 2026-10-19 cheap-card session
    # left pdl1 holding A1 (14 trajectories from the B200) beside C1 and C2 (10 each from an RTX 4090), where
    # sorted order picks A1 vs C1 -- a pair that differs in GPU model as well as in seed, so a difference between
    # them would not mean what an A/A control is supposed to mean. C1 vs C2 is the real control. So the request
    # may name the pair, and `aa_control_arms` is honoured on whichever target carries both.
    want = [a for a in (request.get("aa_control_arms") or []) if a]
    aa = {"verdict": "ABSENT", "reason": "no target has two stock arms to compare"}
    for target, arms in sorted(by_target.items()):
        if want:
            if not all(a in arms for a in want[:2]):
                continue
            pair = [(a, arms[a]) for a in want[:2]]
        elif len(arms) < 2:
            continue
        else:
            pair = sorted(arms.items())[:2]
        (n1, a1), (n2, a2) = pair
        g = grade(a1, a2, f"{target}:{n1}_vs_{n2}", seed=seed)
        aa = {"verdict": g["verdict"], "target": target, "arms": [n1, n2], "n_per_arm": g["n_per_arm"],
              "reasons": g["reasons"], "detail": g["metrics"],
              "spread": {m: {"mean_diff": g["metrics"][m].get("mean_diff"),
                             "sd_diff": g["metrics"][m].get("sd_diff"),
                             "p": g["metrics"][m].get("p"),
                             "ref_sd": g["metrics"][m]["ref"].get("sd")}
                         for m in CONTINUOUS + (ITERATIONS,) if m in g["metrics"]},
              "note": ("This spread is the floor every later A/B is read against. If this verdict is FAIL, the test "
                       "is wrong and not the engine: two stock GPU runs must read as indistinguishable.")}
        break
    if want and aa["verdict"] == "ABSENT":
        # Say which arms were asked for, because "no target has two stock arms" reads like nobody tried.
        found = {t: sorted(a) for t, a in by_target.items()}
        aa["reason"] = f"the request names {want[:2]} as the A/A pair and no target carries both; arms found: {found}"

    # What the reference cost, and what the campaigns it could not afford would cost to buy now. The unbought
    # items live in the request because they are a contract, not a derivation; the price is measured.
    #
    # Two things this deliberately does NOT do, both of which overpriced the 2026-10-19 top-up by ~5x:
    #  * It does not take the MEDIAN campaign rate. This reference pools a $6.251/h B200 with a $0.4469/h card,
    #    and nobody buying a top-up would choose the dearer one. The planning price is the cheapest card that
    #    has actually delivered trajectories here, not the middle of what we happened to run.
    #  * It does not price trajectories alone. A top-up is a FRESH rental, so it pays setup once (clone, pip,
    #    weights, first JAX import) and the bandwidth of pulling results back, on top of per-trajectory compute.
    #    `topup_overhead` carries both, measured, from the request.
    cost = {}
    if per_campaign:
        priced = [c for c in per_campaign.values() if c.get("usd_per_trajectory")]
        cost = {"dph_usd_per_hour": [c["usd_per_trajectory"] for c in priced], "per_campaign": per_campaign}
        if priced:
            best = min(priced, key=lambda c: c["usd_per_trajectory"])
            usd = best["usd_per_trajectory"]
            ov = request.get("topup_overhead") or {}
            fixed = round(ov.get("setup_s", 0) / 3600.0 * best.get("dph_usd_per_hour", 0.0)
                          + ov.get("bandwidth_usd", 0.0), 4)
            cost["usd_per_trajectory"] = round(usd, 4)
            cost["usd_per_trajectory_basis"] = (
                f"cheapest card that delivered trajectories here: {best.get('gpu')} at "
                f"${best.get('dph_usd_per_hour')}/h, {best.get('delivered_s_per_trajectory')} s/trajectory")
            cost["topup_fixed_usd"] = fixed
            cost["unbought"] = [dict(item, usd=round(item["trajectories"] * usd + fixed, 2),
                                     usd_basis=f"{item['trajectories']} x ${usd:.4f} + ${fixed:.4f} rental overhead")
                                for item in request.get("unbought", []) if item.get("trajectories")]
            cost["note"] = ("Measured from each campaign's own runs.jsonl against the hourly rate actually "
                            "billed, so a top-up is execution and not re-planning. Unbought items are priced as "
                            "a fresh rental of the cheapest demonstrated card: per-trajectory compute plus one "
                            "setup and the result pull.")

    return {
        "schema": "bc4-quality-ref/1",
        "targets": targets,
        "aa_control": aa,
        "cost": cost,
        "missing_runs": missing,
        **({"provisional": True,
            "provisional_why": (
                "Built with --provisional on the trajectories that happened to be on disk, at an n far below what "
                "the charter requires. It exercises the whole path (read, pair, test, Holm, A/A, power) so that at "
                "top-up time the only missing ingredient is trajectories. It is NOT the reference: "
                "_bc4_quality_check.py requires n >= 30 per target and will reject this file, which is the point.")}
           if provisional else {}),
        "test": ("Paired sign-flip permutation on (campaign_seed, trajectory_number), two-sample permutation where "
                 "arms do not pair, Fisher exact two-sided on counts, Holm-Bonferroni across metrics at "
                 f"alpha={ALPHA} over one family of {FAMILY_SIZE} tests. Power at n=30, to fail the verdict: "
                 f"{mde_sd_units(30, True, alpha=ALPHA / FAMILY_SIZE):.3f} SD paired and "
                 f"{mde_sd_units(30, False, alpha=ALPHA / FAMILY_SIZE):.3f} SD unpaired at {POWER:.0%} power "
                 f"({mde_sd_units(30, True):.3f} and {mde_sd_units(30, False):.3f} per test, before the Holm "
                 "correction the verdict applies). Acceptance rate is underpowered at any affordable n and is "
                 "read as a guardrail only; see power.detectable_in_metric_units."),
        "power": power_statement({t: [x for arm in arms.values() for x in arm] for t, arms in by_target.items()}),
        "request": request.get("schema"),
        "root": [root] if isinstance(root, str) else list(root),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write(path: str, payload: dict) -> None:
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reference", help="build the stock-GPU reference distribution and its A/A control")
    r.add_argument("--request", required=True, action="append",
                   help="state/bc4/QUALITY-REF-REQUEST.json, the contract that was run; repeatable, "
                        "one per session, and the union is taken")
    r.add_argument("--root", required=True, action="append",
                   help="where the raw campaigns landed; repeatable, searched in order")
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--provisional", action="store_true",
                   help="stamp the output as provisional: exercises the path at an n the gate will reject")

    c = sub.add_parser("compare", help="grade an arm against the reference")
    c.add_argument("--ref", required=True, help="state/bc4/QUALITY-REF.json")
    c.add_argument("--arm", required=True, help="NAME=DIR of the arm under test")
    c.add_argument("--target", default=None, help="which reference target to grade against; default every one")
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("traces", help="compare the same seed on two engines, iteration by iteration (grades nothing)")
    t.add_argument("--a", required=True, help="NAME=DIR, the reference engine")
    t.add_argument("--b", required=True, help="NAME=DIR, the engine under test")
    t.add_argument("--floor", default=None, help="NAME=DIR,NAME=DIR: two same-engine arms, for the floor")
    t.add_argument("--out", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "traces":
        def arm(spec: str) -> tuple[str, list[Trajectory]]:
            name, _, d = spec.partition("=")
            return name or d, read_arm(d)

        (na, ta), (nb, tb) = arm(a.a), arm(a.b)
        out = trace_agreement(ta, tb, na, nb)
        if a.floor:
            fa, fb = a.floor.split(",")
            (nfa, tfa), (nfb, tfb) = arm(fa), arm(fb)
            f = trace_agreement(tfa, tfb, nfa, nfb)
            out["floor"] = {k: f[k] for k in ("a", "b", "n_paired", "iteration_1_max_dev_quanta",
                                              "iteration_1_by_column")}
        if a.out:
            _write(a.out, out)
        print(f"{na} vs {nb}: {out['n_paired']} paired, iteration-1 worst "
              f"{out['iteration_1_max_dev_quanta']} quanta"
              + (f" (same-engine floor {out['floor']['iteration_1_max_dev_quanta']} over "
                 f"{out['floor']['n_paired']})" if a.floor else ""))
        for c, v in sorted(out["iteration_1_by_column"].items(), key=lambda kv: -kv[1]):
            print(f"  {c:30s} {v:g} quanta (quantum {out['quantum_per_column'][c]:g})")
        return 0

    if a.cmd == "reference":
        requests = []
        for path in a.request:
            with open(path) as fh:
                requests.append(json.load(fh))
        request = merge_requests(requests)
        ref = build_reference(request, a.root, seed=a.seed, provisional=a.provisional)
        _write(a.out, ref)
        thin = [t for t, v in ref["targets"].items() if v["n"] < 30]
        print(f"{a.out}: {len(ref['targets'])} targets, "
              + ", ".join(f"{t} n={v['n']}" for t, v in sorted(ref["targets"].items()))
              + f"; A/A {ref['aa_control']['verdict']}")
        for m in ref["missing_runs"]:
            where = m["expected_at"]
            print(f"  MISSING {m['tier']} {m['id']}: "
                  f"{m.get('note') or 'not at ' + ', '.join(where if isinstance(where, list) else [where])}")
        if thin:
            print(f"  under 30 trajectories: {', '.join(thin)}")
        return 0 if not thin and not ref["missing_runs"] and ref["aa_control"]["verdict"] == "PASS" else 1

    name, _, arm_dir = a.arm.partition("=")
    with open(a.ref) as fh:
        ref = json.load(fh)
    arm = read_arm(arm_dir)
    targets = [a.target] if a.target else sorted(ref["targets"])
    graded = {}
    for target in targets:
        ref_trajectories: list[Trajectory] = []
        for path in ref["targets"][target]["paths"]:
            ref_trajectories.extend(read_arm(path))
        graded[target] = grade(arm, ref_trajectories, f"{name}:{target}", seed=a.seed)

    # Worst wins, and anything that is not a clean PASS on every target is not a PASS overall.
    verdicts = [g["verdict"] for g in graded.values()]
    worst = ("FAIL" if "FAIL" in verdicts else
             "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts or not verdicts else "PASS")
    out = {
        "schema": "bc4-quality/1", "verdict": worst, "arm": name,
        "n_per_arm": [n for g in graded.values() for n in g["n_per_arm"]],
        "paths": [arm_dir] + [p for t in targets for p in ref["targets"][t]["paths"]],
        "test": ref["test"],
        "aa_spread": ref["aa_control"].get("spread"),
        "metrics": {t: g["metrics"] for t, g in graded.items()},
        "per_target": graded,
        "power": ref.get("power"),
    }
    _write(a.out, out)
    print(f"{a.out}: {name} {worst}; " + ", ".join(f"{t} {g['verdict']}" for t, g in sorted(graded.items())))
    for t, g in sorted(graded.items()):
        for reason in g["reasons"]:
            print(f"  {t}: {reason}")
    return 0 if worst == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
