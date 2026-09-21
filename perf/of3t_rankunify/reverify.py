"""Re-check every recorded cell's rank-to-sample mapping at the tightened tolerance.

`fold_one.py` checked at 1e-3 and that let four rf3 cells through whose per-sample record was
doubled by a second ranking_score call. The check runs at fold time, so tightening it does
nothing for cells already on disk. This re-runs it from `results.json` and `samples.json`,
so no fold is repeated and no cell keeps a verdict its own check would no longer give.

    python3 perf/of3t_rankunify/reverify.py --root ~/of3t_rankunify_out
"""
import argparse
import json
import pathlib

TOL = 6e-5


def rows(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from rows(x)
    elif isinstance(obj, dict):
        if isinstance(obj.get("all_runs"), list):
            yield obj["all_runs"]
        for v in obj.values():
            yield from rows(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/of3t_rankunify_out")
    a = ap.parse_args()
    root = pathlib.Path(a.root).expanduser()
    bad, checked = [], 0
    for sp in sorted(root.rglob("samples.json")):
        cell = json.loads(sp.read_text())
        rp = next(sp.parent.rglob("results.json"), None)
        if rp is None:
            continue
        runs = next(iter(rows(json.loads(rp.read_text()))), None)
        if not runs or len(runs) != cell["n_samples"]:
            continue
        checked += 1
        by_rank = {s["rank"]: s for s in cell["samples"]}
        worst, where = 0.0, None
        for r, row in enumerate(runs):
            mine = by_rank[row.get("rank", r)]
            for key, val in (("ptm", mine["ptm"]), ("iptm", mine["iptm"] or 0.0),
                             ("plddt", mine["plddt"]), ("confidence_score", mine["score"]),
                             ("ranking_score", mine["score"])):
                if row.get(key) is None:
                    continue
                d = abs(float(row[key]) - float(val))
                if d > worst:
                    worst, where = d, (r, key)
        status = "OK " if worst <= TOL else "BAD"
        if worst > TOL:
            bad.append(cell["tag"])
        print(f"{status} {cell['tag']:30s} calls={cell['n_rank_calls']:3d} "
              f"worst={worst:.2e} at={where}")
    print(f"\n{checked} cells checked at tol {TOL:.0e}; {len(bad)} fail: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
