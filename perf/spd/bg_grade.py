"""SPD accuracy harness for BoltzGen: grade a lever arm against the exact arm on same-seed designs.

    python perf/spd/bg_grade.py --base BASE_S0 --floor BASE_S1 --test ARM_S0 --mode normal|fast [--out report.md]

Each directory is one `tt-bio design <fixture> --model boltzgen --num_designs N --seed S --out_dir DIR` run through
the full pipeline (design, inverse folding, refold, analysis, filter). `--seed` reseeds every step, so design k of
two runs at the same seed is a pair: base vs test at seed 0 measures the lever, base seed 0 vs base seed 1 is the
FLOOR (what a re-seed moves). Read from each run's `aggregate_metrics_analyze.csv` (BoltzGen's own refold metrics)
and `ca_coords_sequences.pkl.gz` (design CA trace and sequence).

Per pair: |delta| of every shared numeric metric, design CA RMSD after a Kabsch fit, sequence identity.
Verdict, against state/spd/CHARTER.md's bars:
  normal  every design present in both; median paired CA RMSD <= max(0.60 A, floor median); the paired mean of each
          confidence metric (KEY) has a 95 % bootstrap interval that reaches 0 or lies on the better side
  fast    every design present; mean drop of each KEY confidence metric <= 0.03; filter pass count at most one
          design below base
"""
import argparse, gzip, json, pickle, random, re, statistics, sys
from pathlib import Path

import numpy as np
import pandas as pd

# (column, higher is better); only the ones the run actually wrote enter the verdict
KEY = [("design_to_target_iptm", True), ("design_ptm", True), ("design_iptm", True), ("iptm", True), ("ptm", True),
       ("complex_plddt", True), ("design_plddt", True), ("filter_rmsd", False), ("bb_rmsd", False)]
BAR_A, CONF_FAST = 0.60, 0.03


def load(run):
    run = Path(run)
    m = sorted(run.rglob("aggregate_metrics_analyze.csv"))
    p = sorted(run.rglob("ca_coords_sequences.pkl.gz"))
    if not m or not p:
        sys.exit(f"{run}: no aggregate_metrics_analyze.csv / ca_coords_sequences.pkl.gz (pipeline incomplete?)")
    df = pd.read_csv(m[0])
    with gzip.open(p[0]) as f:
        ca = pickle.load(f)
    # design ids are <target>, <target>_1, <target>_2, ...: the first carries no suffix
    idx = lambda s: int(m.group(1)) if (m := re.search(r"_(\d+)$", str(s))) else 0
    df["k"] = df["id"].map(idx)
    ca = {idx(r.id): (np.array(json.loads(r.ca_coords), float).reshape(-1, 3), r.sequence) for r in ca.itertuples()}
    passed = sum(1 for _ in run.rglob("final_ranked_designs/final_*_designs/*.cif"))
    return df.set_index("k"), ca, passed


def kabsch_rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def pairs(a, b):
    (da, ca_a, _), (db, ca_b, _) = a, b
    ks = sorted(set(da.index) & set(db.index) & set(ca_a) & set(ca_b))
    out = []
    for k in ks:
        (P, sa), (Q, sb) = ca_a[k], ca_b[k]
        r = kabsch_rmsd(P, Q) if P.shape == Q.shape else float("nan")
        ident = sum(x == y for x, y in zip(sa, sb)) / max(len(sa), 1)
        out.append((k, r, ident))
    return ks, out


def boot(v, n=20000, seed=0):
    rng = random.Random(seed)
    m = sorted(statistics.fmean(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--floor", required=True, type=Path, help="base arm at another seed")
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["normal", "fast"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    base, flo, test = load(a.base), load(a.floor), load(a.test)
    L = [f"# BoltzGen grade: {a.test.name} vs {a.base.name}, {a.mode} mode", ""]
    fail = []
    n_base = len(base[0])
    kf, fl = pairs(base, flo)
    kt, pt = pairs(base, test)
    if len(kt) < n_base or len(test[0]) < n_base:
        fail.append(f"designs paired {len(kt)} of {n_base}")
    keys = [(c, hi) for c, hi in KEY if c in base[0] and c in test[0]]
    num = [c for c in base[0].columns
           if all(c in r[0].columns and pd.api.types.is_numeric_dtype(r[0][c]) and not pd.api.types.is_bool_dtype(r[0][c])
                  for r in (base, flo, test))]

    L += ["| metric | floor median abs | paired mean (test - base) | 95 % CI | better |", "|---|---|---|---|---|"]
    for c in num:
        f = [abs(flo[0].at[k, c] - base[0].at[k, c]) for k in kf]
        d = [float(test[0].at[k, c] - base[0].at[k, c]) for k in kt]
        d = [x for x in d if np.isfinite(x)]
        if not d:
            continue
        lo, hi = boot(d)
        tag = next((("higher" if h else "lower") for kc, h in keys if kc == c), "")
        L.append(f"| {c} | {statistics.median(f):.4f} | {statistics.fmean(d):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {tag} |")
        if c in dict(keys):
            higher = dict(keys)[c]
            if a.mode == "normal" and (lo > 0 if not higher else hi < 0):
                fail.append(f"{c} interval [{lo:+.4f}, {hi:+.4f}] lies on the worse side")
            drop = -statistics.fmean(d) if higher else statistics.fmean(d)
            if a.mode == "fast" and c not in ("filter_rmsd", "bb_rmsd") and drop > CONF_FAST:
                fail.append(f"{c} mean drop {drop:.4f} > {CONF_FAST}")

    fr = [r for _, r, _ in fl if np.isfinite(r)]
    tr = [r for _, r, _ in pt if np.isfinite(r)]
    fmed = statistics.median(fr) if fr else float("nan")
    tmed = statistics.median(tr) if tr else float("nan")
    L += ["", f"Design CA RMSD, same seed test vs base: median {tmed:.3f} A, max {max(tr, default=float('nan')):.3f} A "
          f"(n={len(tr)}); floor (base seed vs seed) median {fmed:.3f} A; kill bar {BAR_A} A.",
          f"Sequence identity test vs base: mean {statistics.fmean([i for *_, i in pt]):.3f}; "
          f"floor {statistics.fmean([i for *_, i in fl]):.3f}.",
          f"Filter pass: base {base[2]}, test {test[2]}, floor run {flo[2]}.",
          "Per design: " + ", ".join(f"{k}: {r:.2f} A / id {i:.2f}" for k, r, i in pt)]
    if a.mode == "normal" and not tmed <= max(BAR_A, fmed):
        fail.append(f"median CA RMSD {tmed:.3f} A > max({BAR_A}, floor {fmed:.3f})")
    if a.mode == "fast" and test[2] < base[2] - 1:
        fail.append(f"filter pass {test[2]} vs base {base[2]}")
    v = "PASS" if not fail else "FAIL: " + "; ".join(fail)
    L += ["", f"VERDICT {a.mode}: {v}"]
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        a.out.write_text(txt)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
