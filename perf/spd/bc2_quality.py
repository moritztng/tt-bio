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
from dataclasses import dataclass, field
from math import comb
from statistics import NormalDist

# The charter's four metrics, plus the two guardrails a lever must not move quietly.
CONTINUOUS = ("i_ptm", "plddt", "rmsd", "traj_iptm", "traj_plddt", "final_loss")
COUNTED = ("accepted", "completed")
ITERATIONS = "iterations"

ALPHA = 0.05
POWER = 0.80


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


def perm_mean_diff(x: list[float], y: list[float], n: int = 20000, seed: int = 0) -> float:
    """Two-sample permutation test on the difference of means."""
    if len(x) < 2 or len(y) < 2:
        return 1.0
    rng = random.Random(seed)
    obs = abs(st.mean(x) - st.mean(y))
    pool = list(x) + list(y)
    k = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(x)]) - st.mean(pool[len(x):])) >= obs - 1e-12:
            k += 1
    return (k + 1) / (n + 1)


def paired_perm_mean_diff(diffs: list[float], n: int = 20000, seed: int = 0) -> float:
    """Sign-flip permutation test on paired differences: under the null a difference is as likely either way."""
    diffs = [d for d in diffs if d == d]
    if len(diffs) < 2:
        return 1.0
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
                traj_iptm=_f(row, "i_pTM"), traj_plddt=_f(row, "pLDDT"), final_loss=final, candidates=cands))
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
    vals: dict[str, list[tuple[tuple[int, int], float]]] = {m: [] for m in CONTINUOUS + COUNTED + (ITERATIONS,)}
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
    """One metric, one arm against another. Paired on (campaign_seed, trajectory_number) where both arms have it."""
    da, db = dict(a), dict(b)
    shared = sorted(set(da) & set(db))
    diffs = [da[k] - db[k] for k in shared if da[k] == da[k] and db[k] == db[k]]
    xa = [v for _, v in a if v == v]
    xb = [v for _, v in b if v == v]
    res = {"metric": name, "arm": describe([v for _, v in a]), "ref": describe([v for _, v in b]),
           "n_paired": len(diffs)}

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

    if len(diffs) >= 2:
        res.update(test="paired_sign_flip_permutation", paired=True, p=paired_perm_mean_diff(diffs, seed=seed),
                   effect=cohen_dz(diffs), effect_kind="cohen_dz", mean_diff=st.mean(diffs),
                   sd_diff=st.stdev(diffs), exact_match=sum(1 for d in diffs if abs(d) < 1e-12) / len(diffs),
                   mde_sd_units=mde_sd_units(len(diffs), paired=True),
                   mde_metric_units=mde_sd_units(len(diffs), paired=True) * st.stdev(diffs))
    elif len(xa) >= 2 and len(xb) >= 2:
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


def collapse_check(trajectories: list[Trajectory], ref: list[Trajectory] | None = None) -> dict:
    """Has the arm stopped exploring? Distinct binder sequences, and per-metric spread against the reference's."""
    seqs = [c.get("Binder_Sequence", "") for t in trajectories for c in t.candidates if c.get("Binder_Sequence")]
    distinct = len(set(seqs)) / len(seqs) if seqs else float("nan")
    out = {"candidates": len(seqs), "distinct_sequence_fraction": distinct, "collapsed": False, "reasons": []}
    if seqs and distinct < 0.9:
        out["collapsed"] = True
        out["reasons"].append(f"only {distinct:.0%} of {len(seqs)} candidate sequences are distinct")
    if ref:
        va, vr = metric_values(trajectories), metric_values(ref)
        for m in CONTINUOUS:
            sa, sr = describe([v for _, v in va[m]]), describe([v for _, v in vr[m]])
            if sa.get("n", 0) > 1 and sr.get("sd"):
                if sa["sd"] < 0.2 * sr["sd"]:
                    out["collapsed"] = True
                    out["reasons"].append(f"{m} spread {sa['sd']:.4g} is under a fifth of the reference's {sr['sd']:.4g}")
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


def grade(arm: list[Trajectory], ref: list[Trajectory], label: str, seed: int = 0) -> dict:
    va, vr = metric_values(arm), metric_values(ref)
    results = holm([compare_metric(m, va[m], vr[m], seed=seed) for m in CONTINUOUS + COUNTED + (ITERATIONS,)])
    nan, collapse, tune = nan_check(arm), collapse_check(arm, ref), autotune_check(arm)
    iters = next(r for r in results if r["metric"] == ITERATIONS)
    moved = [r["metric"] for r in results if r.get("significant")]
    reasons = []
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
        "nan": nan, "collapse": collapse, "autotune": tune,
        "equivalence_note": (
            "A non-significant result is not proof of equality. Each metric carries mde_metric_units: the smallest "
            "effect this n could have detected at 80% power. Effects smaller than that are not ruled out."),
    }
