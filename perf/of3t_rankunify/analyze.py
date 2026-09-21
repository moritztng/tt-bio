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


def plddt_ca(path: pathlib.Path) -> float:
    """Mean pLDDT over Ca atoms, read back out of the written structure's B-factor column.

    of3t-confhead's case for pLDDT rests on Ca-atom pLDDT (+0.414 Spearman against quality)
    rather than the all-atom mean the ranking rule actually reads (+0.357). The harness
    records only the all-atom mean, but every model here writes per-atom pLDDT x 100 into the
    B-factor column, so the Ca figure costs a file read instead of a re-fold.
    """
    import numpy as np
    a = _load(str(path))
    b = getattr(a, "b_factor", None)
    return float(np.mean(b) / 100.0) if b is not None and len(b) else float("nan")

#: The rule each site shipped before unification, and the unified rule. Every one of them is
#: a function of the same five recorded scalars, which is what makes the sweep free.
RULES = {
    "unified": lambda s: (0.8 * (s["iptm"] or s["plddt"]) + 0.2 * s["ptm"]
                          + 0.5 * s["disorder"] - 100.0 * s["has_clash"]),
    "old_of3": lambda s: (0.8 * (s["iptm"] or s["plddt"]) + 0.2 * s["ptm"]
                          + 0.5 * s["disorder"] - 100.0 * s["has_clash"]),
    "old_rf3": lambda s: (0.8 * (s["iptm"] or s["ptm"]) + 0.2 * s["ptm"]
                          - 100.0 * s["has_clash"]),
    # What rf3 ACTUALLY served: worker.py sorted on summary["ranking_score"], which is
    # rounded to 4 decimals for the published summary_confidences.json, and  is
    # stable, so a pair that collides at 1e-4 was ordered by sample index.  returns the
    # first maximal element, which reproduces that tie-break exactly.
    "old_rf3_shipped": lambda s: round(0.8 * (s["iptm"] or s["ptm"]) + 0.2 * s["ptm"]
                                       - 100.0 * s["has_clash"], 4),
    "old_protenix": lambda s: (0.8 * s["iptm"] + 0.2 * s["ptm"] if s["iptm"]
                               else (s["ptm"] if s["ptm"] > 0 else s["plddt"])),
    "plddt": lambda s: s["plddt"],
    "ptm": lambda s: s["ptm"],
    "random": None,
}

#: Which rule each model shipped at wk/of3t 7aed7253b, so "before" is that model's own before.
SHIPPED = {"openfold3": "old_of3", "openbind": "old_of3", "rf3": "old_rf3_shipped",
           "protenix-v1": "old_protenix", "protenix-v2": "old_protenix",
           "opendde": "old_protenix", "opendde-abag": "old_protenix"}


def _ranks(xs):
    """Average ranks, ties shared, so a head that reports two samples identically is not
    given a spurious order."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


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
        has_rmsd = all(s["rmsd_ca"] is not None for c in g for s in c["samples"])
        for rule in RULES:
            if not has_rmsd:
                break
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
        # Ca-atom pLDDT, read back from the B-factor column of each written structure.
        for c in g:
            for x in c["samples"]:
                if "plddt_ca" not in x:
                    x["plddt_ca"] = plddt_ca(next(pathlib.Path(c["dir"]).rglob(x["file"])))

        # REGRET, the statistic that can actually separate two rules.
        #
        # Mean rank-0 Ca-RMSD cannot: the differences are 0.01-0.04 A against seed floors of
        # 0.3-1.3 A. But the seed floor governs comparing ACROSS seeds. Two rules are compared
        # on the SAME five samples of the SAME fold, so the comparison is PAIRED and the floor
        # does not apply to the difference. Regret is served RMSD minus the best of the five:
        # 0.0 means the rule picked the best sample available, and it is bounded by how much
        # the samples differ rather than by how much a reseed moves the structure.
        if has_rmsd:
            for rule in RULES:
                if RULES[rule] is None:
                    continue
                reg, sel = [], []
                for c in g:
                    ss = c["samples"]
                    pick = served(c, rule)
                    best = min(x["rmsd_ca"] for x in ss)
                    reg.append(pick["rmsd_ca"] - best)
                    sel.append(sorted(x["rmsd_ca"] for x in ss).index(pick["rmsd_ca"]))
                entry["rules"][rule]["regret"] = round(statistics.mean(reg), 4)
                entry["rules"][rule]["selected_rank"] = round(statistics.mean(sel), 2)
            # Paired, per seed: unified minus shipped on the same samples.
            d = [served(c, "unified")["rmsd_ca"] - served(c, base)["rmsd_ca"] for c in g]
            entry["paired_delta"] = {"per_seed": [round(x, 4) for x in d],
                                     "mean": round(statistics.mean(d), 4),
                                     "n_better": sum(x < -1e-9 for x in d),
                                     "n_worse": sum(x > 1e-9 for x in d),
                                     "n_same": sum(abs(x) <= 1e-9 for x in d)}

        # Why a rule change can be inert: if pLDDT and pTM induce the same order over a
        # model's samples, no reweighting between them can move the served structure. That is
        # a property of the confidence head, so it is measured per model rather than assumed.
        def _rho(xs, ys):
            n = len(xs)
            if n < 3:
                return None
            rx, ry = _ranks(xs), _ranks(ys)
            mx, my = statistics.mean(rx), statistics.mean(ry)
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
            return round(num / den, 3) if den else None

        rhos, n_ok = {}, 0
        for key in ("plddt_vs_ptm", "plddt_vs_quality", "plddtca_vs_quality", "ptm_vs_quality"):
            rhos[key] = []
        for c in g:
            ss = c["samples"]
            if any(x["rmsd_ca"] is None for x in ss):
                continue
            n_ok += 1
            for key, xa, xb in (("plddt_vs_ptm", "plddt", "ptm"),
                                ("plddt_vs_quality", "plddt", "rmsd_ca"),
                            ("plddtca_vs_quality", "plddt_ca", "rmsd_ca"),
                                ("ptm_vs_quality", "ptm", "rmsd_ca")):
                # Against QUALITY, not against RMSD: -rmsd so a POSITIVE rho means the
                # signal orders the samples usefully, which is of3t-confhead's convention
                # ("pLDDT orders its own samples at +0.41 to +0.46") and must stay comparable.
                ya = [(-x[xb] if xb == "rmsd_ca" else x[xb]) for x in ss]
                r = _rho([x[xa] for x in ss], ya)
                if r is not None:
                    rhos[key].append(r)
        entry["spearman"] = {k: (round(statistics.mean(v), 3) if v else None)
                             for k, v in rhos.items()}
        entry["spearman"]["n_seeds"] = n_ok

        if base == "old_rf3_shipped":
            entry["rounding_only"] = sum(
                served(c, "old_rf3_shipped")["sample"] != served(c, "old_rf3")["sample"]
                for c in g)

        # INTERFACE: on a sample that HAS an interface the unified rule must equal the
        # site's own old rule exactly. The test proves that algebraically; this proves the
        # branch is executed, on recorded scalars from a real fold, which is the part a test
        # cannot do (of3t-softmax D63: construction is not execution).
        # Against the site's RULE, not against what it served. rf3's shipped ordering rounds
        # to 4 decimals (see D-FOUND); comparing the unified rule to that rounding would
        # report a 5e-5 "interface difference" that is the rounding defect, already reported
        # separately, and not a difference in the branch.
        rule_of = {"old_rf3_shipped": "old_rf3"}.get(base, base)
        iface = {"n_samples_with_iptm": 0, "n_exact": 0, "worst_abs_diff": 0.0,
                 "served_sample_same": True, "compared_against": rule_of}
        for c in g:
            for s_ in c["samples"]:
                if not s_["iptm"]:
                    continue
                iface["n_samples_with_iptm"] += 1
                d = abs(RULES["unified"](s_) - RULES[rule_of](s_))
                iface["n_exact"] += int(RULES["unified"](s_) == RULES[rule_of](s_))
                iface["worst_abs_diff"] = max(iface["worst_abs_diff"], d)
            if all(x["iptm"] for x in c["samples"]):
                iface["served_sample_same"] &= (served(c, rule_of)["sample"]
                                                == served(c, "unified")["sample"])
        entry["interface"] = iface
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
    print()
    hdr2 = (f"{'model/target':28s} {'regret shipped':>14s} {'regret unified':>14s} "
            f"{'sel rank ship':>13s} {'sel rank unif':>13s} {'paired mean':>11s} {'b/w/s':>9s}")
    print(hdr2)
    print("-" * len(hdr2))
    for k, e in report.items():
        r, pd = e["rules"], e.get("paired_delta")
        if "regret" not in r.get("unified", {}):
            continue
        b = e["shipped_rule"]
        print(f"{k:28s} {r[b]['regret']:14.4f} {r['unified']['regret']:14.4f} "
              f"{r[b]['selected_rank']:13.2f} {r['unified']['selected_rank']:13.2f} "
              f"{pd['mean']:+11.4f} {pd['n_better']}/{pd['n_worse']}/{pd['n_same']:<5d}")
    print()
    print(f"{'model/target':28s} {'iptm samples':>13s} {'exact':>6s} {'worst diff':>11s} {'served same':>12s}")
    for k, e in report.items():
        i = e["interface"]
        if not i["n_samples_with_iptm"]:
            continue
        print(f"{k:28s} {i['n_samples_with_iptm']:13d} {i['n_exact']:6d} "
              f"{i['worst_abs_diff']:11.3e} {str(i['served_sample_same']):>12s}")
    print()
    print(f"{'model/target':28s} {'pLDDT~pTM':>10s} {'pLDDT~qual':>11s} {'pLDDTca~qual':>13s} {'pTM~qual':>9s} {'seeds':>6s}")
    for k, e in report.items():
        sp = e["spearman"]
        f = lambda v: f"{v:+.3f}" if v is not None else "   -  "
        print(f"{k:28s} {f(sp['plddt_vs_ptm']):>10s} {f(sp['plddt_vs_quality']):>11s} "
              f"{f(sp['plddtca_vs_quality']):>13s} "
              f"{f(sp['ptm_vs_quality']):>9s} {sp['n_seeds']:6d}")
    bad = [k for k, e in report.items() if not e["rank_map_verified"]]
    if bad:
        print("\nRANK MAP UNVERIFIED (do not quote these):", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
