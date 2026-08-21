#!/usr/bin/env python3
"""Does the host-concat branch sit closer to the OpenDDE torch reference than the device branch?

Both arms ran the SAME 5 device seeds against the SAME 5 committed fp32 reference seeds
(`opendde/prot/nomsa_10cycle_200step_1sample_fp32_prod`, aurekaresearch/OpenDDE a0d5134, torch
CPU fp32, 117 res). A (dev_i, ref_j) cell is therefore comparable between arms.

The 25 cells are NOT 25 independent observations: they share 5 device folds and 5 reference
folds, so a standard error over 25 cells is too narrow by roughly sqrt(5). The primary test
collapses the reference axis first (mean over ref seeds -> one number per device seed) and pairs
the 5 device seeds, df=4. The naive 25-cell interval is printed alongside, labelled, so the two
cannot be confused -- the `abag-scaling-pooled-ci-and-closed-details-chart-bug` failure mode.
"""
import itertools, json, statistics as st, sys
from pathlib import Path

ROOT = Path("/home/ttuser/.coworker/wt/opendde-size-generality-l1-work-split-p4")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from boltz2_fast_parity import compare_structure

TID = "prot_no_msa"
REF = ROOT / "docs/implementation-parity-data/ref-fixtures/opendde/prot/nomsa_10cycle_200step_1sample_fp32_prod"
ref_dirs = [REF / f"seed{i}" for i in range(5)]
arms = {a: [Path(f"/tmp/fpg_{a}/opendde-prot-prod/seed{i}/opendde_results_prot_no_msa")
            for i in range(5)] for a in ("hostcat", "devcat")}
KEYS = ("kabsch_rmsd", "1-lddt", "1-tm_score", "1-coord_pcc")
T_CRIT = {4: 2.776, 24: 2.064}          # two-sided 95%


def m(a, b):
    s = compare_structure(a / "structures" / f"{TID}.cif", b / "structures" / f"{TID}.cif")
    return {"kabsch_rmsd": s["kabsch_rmsd"], "1-lddt": 1 - s["lddt"],
            "1-tm_score": 1 - s["tm_score"], "1-coord_pcc": 1 - s["coord_pcc"]}


cross = {a: {(i, j): m(d[i], ref_dirs[j]) for i in range(5) for j in range(5)}
         for a, d in arms.items()}
selfsp = {a: [m(d[i], d[j]) for i, j in itertools.combinations(range(5), 2)]
          for a, d in arms.items()}
refsp = [m(ref_dirs[i], ref_dirs[j]) for i, j in itertools.combinations(range(5), 2)]


def ci(vals, df):
    md, se = st.mean(vals), st.stdev(vals) / len(vals) ** 0.5
    return md, md - T_CRIT[df] * se, md + T_CRIT[df] * se


out = {"target": TID, "ref_seeds": 5, "dev_seeds": 5, "metrics": {}}
print(f"{'metric':13s} {'hostX':>7s} {'devX':>7s} | per-dev-seed paired delta (n=5, df=4)"
      f"      | naive n=25          | {'refR':>6s} {'hostD':>6s} {'devD':>6s}")
for k in KEYS:
    # collapse the reference axis: one distance-to-reference per device seed
    per_seed = {a: [st.mean([cross[a][(i, j)][k] for j in range(5)]) for i in range(5)]
                for a in arms}
    dseed = [d - h for h, d in zip(per_seed["hostcat"], per_seed["devcat"])]
    md, lo, hi = ci(dseed, 4)
    cells = [cross["devcat"][p][k] - cross["hostcat"][p][k] for p in cross["hostcat"]]
    md25, lo25, hi25 = ci(cells, 24)
    R = st.mean([x[k] for x in refsp])
    Dh, Dd = (st.mean([x[k] for x in selfsp[a]]) for a in ("hostcat", "devcat"))
    sig = "EXCLUDES 0" if lo > 0 or hi < 0 else "spans 0"
    print(f"{k:13s} {st.mean(per_seed['hostcat']):7.4f} {st.mean(per_seed['devcat']):7.4f} | "
          f"{md:+.4f} [{lo:+.4f},{hi:+.4f}] {sig:10s} | {md25:+.4f} "
          f"[{lo25:+.4f},{hi25:+.4f}] | {R:6.4f} {Dh:6.4f} {Dd:6.4f}")
    out["metrics"][k] = {
        "host_X": st.mean(per_seed["hostcat"]), "dev_X": st.mean(per_seed["devcat"]),
        "per_dev_seed_delta": {"mean": md, "ci95": [lo, hi], "n": 5, "df": 4,
                               "excludes_zero": lo > 0 or hi < 0},
        "naive_25cell_delta": {"mean": md25, "ci95": [lo25, hi25], "n": 25,
                               "note": "over-narrow, cells share 5 dev and 5 ref folds"},
        "ref_spread_R": R, "host_self_D": Dh, "dev_self_D": Dd,
        "delta_over_ref_spread": md / R if R else None,
        "host_per_seed": per_seed["hostcat"], "dev_per_seed": per_seed["devcat"]}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[1])
