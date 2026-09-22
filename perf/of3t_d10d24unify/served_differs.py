"""Would main have served a different structure? Asked of the live folds, not of a record.

`rerank_vs_main.py` answers this offline from `of3t-rankunify`'s recorded cells. This asks it
of the folds THIS branch just ran on qb1: each published row carries its own pTM/ipTM/pLDDT, so
main's rule can be applied to those same rows and its winner compared with the one that was
written. The cell that matters is rf3/multimer, where the 4-decimal rounding main ordered on
collapses two samples that differ below 1e-4.

One bound: `results.json` publishes no `disorder` column, so main's OpenFold3 rule is
reconstructed with disorder = 0. That is exact on ubiquitin, where the RASA term measured
0.0 on every sample of every seed, and it would understate the term on a short extended
chain. Only OpenFold3 computes it at all.
"""
import json
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   pathlib.Path.home() / "of3t_d10d24_out")


def main_rule(model, row):
    ptm = float(row.get("ptm") or 0.0)
    iptm = row.get("iptm")
    if model == "rf3":
        i = ptm if iptm in (None, 0.0) else float(iptm)
        return round(0.8 * i + 0.2 * ptm, 4)
    if model == "openfold3":
        return (0.8 * float(iptm or 0.0) + 0.2 * ptm
                + 0.5 * float(row.get("disorder") or 0.0))
    return 0.8 * float(iptm) + 0.2 * ptm if (iptm or 0.0) > 0.0 else (
        ptm if ptm > 0.0 else float(row.get("plddt") or 0.0))


def rows_of(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from rows_of(x)
    elif isinstance(obj, dict):
        if isinstance(obj.get("all_runs"), list):
            yield obj["all_runs"]
        for v in obj.values():
            yield from rows_of(v)


out = []
for d in sorted(OUT.iterdir()):
    vf = d / "verify.json"
    if not vf.is_file():
        continue
    v = json.loads(vf.read_text())
    res = json.loads(next(d.rglob("results.json")).read_text())
    runs = next(iter(rows_of(res)))
    # rows are published in the order this branch served; index 0 is what was written.
    mainbest = max(range(len(runs)), key=lambda i: main_rule(v["model"], runs[i]))
    out.append({"tag": v["tag"], "model": v["model"], "board": v["board"],
                "score_key": v["score_key"], "rows_in_rank_order": v["rows_in_rank_order"],
                "rule_mismatches": len(v["rule_mismatches"]),
                "served_rank_under_main": mainbest,
                "main_would_serve_another": mainbest != 0,
                "served_file": v["served_file"]})
    print("%-24s ordered=%s mismatches=%d  main would serve rank %d%s"
          % (out[-1]["tag"], out[-1]["rows_in_rank_order"], out[-1]["rule_mismatches"],
             mainbest, "  <- DIFFERENT STRUCTURE" if mainbest else ""))
(pathlib.Path(__file__).parent / "verify_folds.json").write_text(
    json.dumps(out, indent=1) + "\n")
