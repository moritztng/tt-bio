"""Fold the per-complex assembled-network grades into the dfd-parity record.

Reads every ``<id>.grade.json`` written by ``score_net_grade.py``, takes the worst
cell across all of them, and writes the matrix plus the headline numbers into
``state/dfd/rows/dfd-parity.json`` and ``state/dfd/parity/score.json``.

Usage: score_net_record.py --grades /tmp/dfd_assembled [--status done] [--verdict-file F]
"""
import argparse
import json
from pathlib import Path

STATE = Path("/home/moritz/.coworker/state/dfd")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--grades", required=True)
    a.add_argument("--status", default=None)
    a.add_argument("--verdict-file", default=None)
    a.add_argument("--branch-head", default=None)
    a = a.parse_args()

    matrix, worst_pcc, worst_rel = {}, None, None
    for f in sorted(Path(a.grades).glob("*.grade.json")):
        g = json.loads(f.read_text())
        cid = g["complex"]
        cell = {"missing_steps": g.get("missing", []), "steps": {}}
        for k, row in sorted(g["steps"].items(), key=lambda kv: int(kv[0])):
            cell["steps"][k] = {
                "n_rotatable": row["n_rotatable"],
                **{n: {"pcc": row[n]["pcc"], "max_rel_err": row[n]["max_rel_err"]}
                   for n in ("tr", "rot", "tor")},
            }
            for n in ("tr", "rot", "tor"):
                p, r = row[n]["pcc"], row[n]["max_rel_err"]
                if p is not None and (worst_pcc is None or p < worst_pcc[0]):
                    worst_pcc = (p, f"{cid} step {k} {n}")
                if worst_rel is None or r > worst_rel[0]:
                    worst_rel = (r, f"{cid} step {k} {n}")
        matrix[cid] = cell

    row = json.loads((STATE / "rows" / "dfd-parity.json").read_text())
    row["assembled_matrix"] = matrix
    row["pcc"] = {"worst": worst_pcc[0], "where": worst_pcc[1],
                  "bar": "per-output PCC >= 0.999 on every graded cell (D-dfd-orch-8)"}
    row["max_rel_err"] = {"worst": worst_rel[0], "where": worst_rel[1]}
    for m in row["modules"]:
        if m["name"] == "assembled_score_network":
            m["pcc"] = worst_pcc[0]
            m["matrix"] = {c: sorted(v["steps"]) for c, v in matrix.items()}
    if a.branch_head:
        row["branch_head"] = a.branch_head
    if a.status:
        row["status"] = a.status
    if a.verdict_file:
        row["verdict"] = Path(a.verdict_file).read_text().strip()
    (STATE / "rows" / "dfd-parity.json").write_text(json.dumps(row, indent=1) + "\n")

    score = {k: row[k] for k in ("status", "branch", "modules", "pcc", "max_rel_err",
                                 "bar", "esm2_reused", "verdict")}
    score["branch_head"] = row.get("branch_head")
    score["assembled_matrix"] = matrix
    (STATE / "parity" / "score.json").write_text(json.dumps(score, indent=1) + "\n")
    print(f"worst pcc {worst_pcc[0]} at {worst_pcc[1]}; "
          f"worst rel {worst_rel[0]:.3e} at {worst_rel[1]}; complexes {sorted(matrix)}")


if __name__ == "__main__":
    main()
