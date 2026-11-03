"""Grade one `panel.py` arm against the base arm, case by case.

    python perf/mol1/fair/panel_grade.py BASE_DIR ARM_DIR [--out grade.json] [--cases a,b,...]

Per case: the seed floor is the base arm's seed 42 against its seed 66 (every sample pair, median); the arm's
deviation is each seed's samples against the base samples of the same seed, paired by the least-total-RMSD
assignment (grade_ab.matched), all-atom RMSD and lDDT. PASS (one stack, never a sum of levers): panel median
same-seed RMSD <= 0.60 A, every case's median same-seed RMSD below its own floor, and every case's same-seed lDDT
median >= 0.99 (within 0.01 of main).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from perf.mol1.fair.grade_ab import BAR_A, cif_xyz, matched  # noqa: E402
from perf.mol1_pairmix.grade_ab import lddt, rmsd  # noqa: E402

LDDT_TOL = 0.01


def folds(d):
    """input -> seed -> (samples, line) for the clean folds of one panel.py out dir."""
    out = {}
    for line in open(Path(d) / "runs.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "fold" or r.get("err") or not r.get("finite"):
            continue
        cifs = sorted(Path(r["struct_dir"]).rglob("*.cif"))
        out.setdefault(r["input"], {})[r["seed"]] = ([cif_xyz(f) for f in cifs], r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--out")
    ap.add_argument("--cases", help="comma-separated inputs that decide; default every input both arms folded")
    a = ap.parse_args()
    b, x = folds(a.base), folds(a.arm)
    med = lambda v: round(statistics.median(v), 3)  # noqa: E731
    cases = {}
    for inp in sorted(set(b) & set(x)):
        seeds = sorted(set(b[inp]) & set(x[inp]))
        if not seeds:
            continue
        dev = [(rmsd(p, q), lddt(p, q)) for s in seeds for p, q in matched(b[inp][s][0], x[inp][s][0])]
        fl = []
        if len(b[inp]) >= 2:
            s1, s2 = sorted(b[inp])[:2]
            fl = [(rmsd(p, q), lddt(p, q)) for p in b[inp][s1][0] for q in b[inp][s2][0]]
        c = {"tokens": b[inp][seeds[0]][1]["tokens"], "seeds": seeds,
             "same_seed_rmsd_A": med([d[0] for d in dev]), "same_seed_rmsd_max_A": round(max(d[0] for d in dev), 3),
             "same_seed_lddt": med([d[1] for d in dev]),
             "floor_rmsd_A": med([f[0] for f in fl]) if fl else None,
             "floor_lddt": med([f[1] for f in fl]) if fl else None,
             "digest_equal": all(b[inp][s][1]["coords_sha256"] == x[inp][s][1]["coords_sha256"] for s in seeds),
             "plddt": {"base": [b[inp][s][1]["plddt"] for s in seeds], "arm": [x[inp][s][1]["plddt"] for s in seeds]},
             "wall_s": {"base": [b[inp][s][1]["wall_s"] for s in seeds], "arm": [x[inp][s][1]["wall_s"] for s in seeds]},
             "aiclk_min": min(min(v["min"] for v in r[1]["aiclk"].values()) for d in (b, x) for s in seeds
                              for r in [d[inp][s]] if r[1].get("aiclk"))}
        c["below_floor"] = c["floor_rmsd_A"] is not None and c["same_seed_rmsd_A"] < c["floor_rmsd_A"]
        c["lddt_ok"] = c["same_seed_lddt"] >= 1 - LDDT_TOL
        cases[inp] = c
    decide = a.cases.split(",") if a.cases else list(cases)
    dc = [cases[k] for k in decide if k in cases]
    res = {"base": a.base, "arm": a.arm, "bar_A": BAR_A, "deciding": decide, "missing": [k for k in decide if k not in cases],
           "panel_median_rmsd_A": med([c["same_seed_rmsd_A"] for c in dc]) if dc else None,
           "every_case_below_floor": all(c["below_floor"] for c in dc),
           "every_case_lddt_ok": all(c["lddt_ok"] for c in dc), "cases": cases}
    res["pass"] = bool(dc and not res["missing"] and res["panel_median_rmsd_A"] <= BAR_A
                       and res["every_case_below_floor"] and res["every_case_lddt_ok"])
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
