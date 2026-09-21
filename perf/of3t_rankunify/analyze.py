"""Price every candidate ranking rule on the recorded folds, per model, with the seed floor.

Reads every `samples.json` under the fold root. For each (model, target) and each rule, the
served sample is the argmax of that rule over the recorded per-sample scalars, so the rule's
rank-0 Ca-RMSD is the structure that rule WOULD have served -- exact, not estimated, because
a ranking rule is post-forward and cannot move a sample.

The seed floor is measured the way of3t-confhead measured it: rank-0 structure against rank-0
structure over every pair of seeds, under the rule being reported, on the structure that rule
actually serves. A difference smaller than that floor has not been shown to be a difference.

    python3 perf/of3t_rankunify/analyze.py --root ~/of3t_rankunify_out
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ca_rmsd import ca_rmsd, _load                      # noqa: E402

#: The rule each site shipped before unification, and the unified rule. Every one of them is
#: a function of the same five recorded scalars, which is what makes the sweep free.
RULES = {
    "unified": lambda s: (0.8 * (s["iptm"] or s["plddt"]) + 0.2 * s["ptm"]
                          + 0.5 * s["disorder"] - 100.0 * s["has_clash"]),
    "old_of3": lambda s: (0.8 * (s["iptm"] or s["plddt"]) + 0.2 * s["ptm"]
                          + 0.5 * s["disorder"] - 100.0 * s["has_clash"]),
    "old_rf3": lambda s: (0.8 * (s["iptm"] or s["ptm"]) + 0.2 * s["ptm"]
                          - 100.0 * s["has_clash"]),
    "old_protenix": lambda s: (0.8 * s["iptm"] + 0.2 * s["ptm"] if s["iptm"]
                               else (s["ptm"] if s["ptm"] > 0 else s["plddt"])),
    "plddt": lambda s: s["plddt"],
    "ptm": lambda s: s["ptm"],
    "random": None,
}

#: Which rule each model shipped at wk/of3t 7aed7253b, so "before" is that model's own before.
SHIPPED = {"openfold3": "old_of3", "openbind": "old_of3", "rf3": "old_rf3",
           "protenix-v1": "old_protenix", "protenix-v2": "old_protenix",
           "opendde": "old_protenix", "opendde-abag": "old_protenix"}


def coord_digest(path: pathlib.Path) -> str:
    """sha256 over the Ca coordinates, not the file. The written filename encodes the RANK, so
    a file digest can move for no reason but a rename; the coordinates cannot."""
    a = _load(str(path))
    return hashlib.sha256(a.coord.astype("float32").tobytes()).hexdigest()[:16]


def served(cell: dict, rule: str):
    fn = RULES[rule]
    if fn is None:
        return None
    return max(cell["samples"], key=fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/of3t_rankunify_out")
    ap.add_argument("--json", default=str(HERE / "analyze.json"))
    a = ap.parse_args()
    root = pathlib.Path(a.root).expanduser()

    cells = []
    for p in sorted(root.rglob("samples.json")):
        c = json.loads(p.read_text())
        c["dir"] = str(p.parent)
        cells.append(c)
    if not cells:
        print("no folds yet")
        return 1

    groups: dict[tuple[str, str], list] = {}
    for c in cells:
        groups.setdefault((c["model"], pathlib.Path(c["target"]).stem), []).append(c)

    report = {}
    for (model, target), g in sorted(groups.items()):
        g.sort(key=lambda c: c["seed"])
        base = SHIPPED.get(model, "old_of3")
        entry = {"model": model, "target": target, "seeds": [c["seed"] for c in g],
                 "shipped_rule": base, "n_samples": g[0]["n_samples"],
                 "rank_map_verified": all(c.get("rank_map_verified") for c in g),
                 "aiclk_median": statistics.median(
                     [c["aiclk"]["median"] for c in g if c["aiclk"]["median"]]),
                 "wall_s": [c["wall_s"] for c in g], "rules": {}, "reach": {}}
        for rule in RULES:
            if RULES[rule] is None:
                vals = [statistics.mean([s["rmsd_ca"] for s in c["samples"]]) for c in g]
            else:
                vals = [served(c, rule)["rmsd_ca"] for c in g]
            if any(v is None for v in vals):
                continue
            entry["rules"][rule] = {"per_seed": [round(v, 4) for v in vals],
                                    "mean": round(statistics.mean(vals), 4)}
        # REACH: does the served structure actually move, per seed, by coordinates
        moved = []
        for c in g:
            b, u = served(c, base), served(c, "unified")
            moved.append({"seed": c["seed"], "before_sample": b["sample"],
                          "after_sample": u["sample"],
                          "before_digest": coord_digest(pathlib.Path(c["dir"]).rglob(
                              b["file"]).__next__()),
                          "after_digest": coord_digest(pathlib.Path(c["dir"]).rglob(
                              u["file"]).__next__())})
        entry["reach"] = {"per_seed": moved,
                          "n_moved": sum(m["before_digest"] != m["after_digest"] for m in moved),
                          "n_seeds": len(moved)}
        # Seed floor under the unified rule: served structure against served structure.
        floors = []
        for x, y in itertools.combinations(g, 2):
            fx = next(pathlib.Path(x["dir"]).rglob(served(x, "unified")["file"]))
            fy = next(pathlib.Path(y["dir"]).rglob(served(y, "unified")["file"]))
            floors.append(ca_rmsd(str(fx), str(fy)))
        if floors:
            entry["seed_floor"] = {"n_pairs": len(floors), "mean": round(statistics.mean(floors), 4),
                                   "min": round(min(floors), 4), "max": round(max(floors), 4)}
        report[f"{model}/{target}"] = entry

    pathlib.Path(a.json).write_text(json.dumps(report, indent=2) + "\n")

    hdr = f"{'model/target':28s} {'seeds':>5s} {'shipped':>8s} {'unified':>8s} {'plddt':>7s} {'ptm':>7s} {'random':>7s} {'floor':>7s} {'moved':>6s} {'clk':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for k, e in report.items():
        r = e["rules"]
        g = lambda n: f"{r[n]['mean']:.3f}" if n in r else "   -  "
        fl = f"{e['seed_floor']['mean']:.3f}" if "seed_floor" in e else "  -  "
        print(f"{k:28s} {len(e['seeds']):5d} {g(e['shipped_rule']):>8s} {g('unified'):>8s} "
              f"{g('plddt'):>7s} {g('ptm'):>7s} {g('random'):>7s} {fl:>7s} "
              f"{e['reach']['n_moved']}/{e['reach']['n_seeds']:<4d} {e['aiclk_median']:5.0f}")
    bad = [k for k, e in report.items() if not e["rank_map_verified"]]
    if bad:
        print("\nRANK MAP UNVERIFIED (do not quote these):", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
