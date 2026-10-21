"""The RFD4 design-level grade: two sets of designs in, `state/rfd4/QUALITY.json` out.

    # calibrate the seed floor first: upstream against itself, two seeds
    python3 perf/rfd4/quality.py floor --feature binder --a runs/ref_seed0 --b runs/ref_seed1

    # grade an arm against the reference
    python3 perf/rfd4/quality.py grade --feature binder --tt runs/tt --ref runs/ref

    python3 perf/rfd4/quality.py show

An arm is a directory of designs (`*_generated.cif[.zst]`), an upstream
`per_example_metrics.csv`, or a directory holding both: the CSV carries upstream's own metric
columns, the CIFs give the harness's oracle-free `geom.*` columns. Nothing is recomputed that
upstream already computed.

Grading is per feature family and per metric:

* continuous metric: KS test, bar p >= 0.05. Below the bar it still passes if the move is inside
  the seed floor -- what upstream moves by when only its seed changes -- in BOTH the median shift
  and the KS statistic. A lever smaller than the noise the reference already has is not a
  regression, and the floor sits in the JSON next to every bar.
* pass rate: one-sided Fisher exact, bar p >= 0.05, only against a DROP.
* guard: a bar on the TT arm alone (unknown residues, non-finite coordinates, design diversity).

Each run rewrites only its own feature in QUALITY.json, so this is re-runnable after every
landed lever without re-grading the rest of the campaign.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry  # noqa: E402
import metrics as mspec  # noqa: E402
import stats  # noqa: E402

QUALITY_JSON = pathlib.Path("/home/moritz/.coworker/state/rfd4/QUALITY.json")
DESIGN_GLOBS = ("*_generated.cif", "*_generated.cif.zst")


# --------------------------------------------------------------------------- loading an arm
def _find_csvs(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root] if root.suffix == ".csv" else []
    return sorted(root.rglob("per_example_metrics*.csv"))


def _find_designs(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return []
    found: list[pathlib.Path] = []
    for pattern in DESIGN_GLOBS:
        found += sorted(root.rglob(pattern))
    if not found:  # a plain directory of CIFs, not an rfd4-generate output tree
        for pattern in ("*.cif", "*.cif.zst"):
            found += sorted(p for p in root.rglob(pattern) if not _is_sidecar(p))
    return found


# Upstream's naming (docs/reference/output-format.md): the design is `_generated`, everything
# else beside it is the input copy, a refold, a chain extract or a trajectory frame.
SIDECAR = ("_input", "_refolded", "_chain", "trajectories")


def _is_sidecar(path: pathlib.Path) -> bool:
    return any(s in path.name for s in SIDECAR) or "trajectories" in path.parts


def load_arm(path: str | pathlib.Path, chain: str | None = None) -> dict:
    """One arm: per-design rows keyed by design path, upstream columns plus `geom.*`."""
    root = pathlib.Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    rows: dict[str, dict] = {}
    for csv_path in _find_csvs(root):
        with csv_path.open() as fh:
            for rec in csv.DictReader(fh):
                key = rec.get("designed_structure_path") or rec.get("example_id") or str(len(rows))
                rows.setdefault(key, {}).update({k: _num(v) for k, v in rec.items()})
    designs = _find_designs(root)
    sequences: list[str] = []
    for cif in designs:
        key = str(cif)
        geom = geometry.design_geometry(cif, chain=chain)
        if not geom:
            continue
        match = key if key in rows else _match_row(rows, cif)
        rows.setdefault(match or key, {}).update(geom)
        sequences.append(geometry.sequence_of(cif, chain=chain))
    return {
        "root": str(root),
        "rows": list(rows.values()),
        "n": len(rows),
        "n_designs_parsed": len(designs),
        "n_unique_sequences": len(set(s for s in sequences if s)) if sequences else None,
        "csvs": [str(p) for p in _find_csvs(root)],
    }


def _match_row(rows: dict[str, dict], cif: pathlib.Path) -> str | None:
    stem = cif.name.split(".")[0]
    for key in rows:
        if stem in key:
            return key
    return None


def _num(v: str):
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return v


def _column(arm: dict, name: str) -> np.ndarray:
    vals = [r[name] for r in arm["rows"] if isinstance(r.get(name), float)]
    return np.array(vals, dtype=float)


# --------------------------------------------------------------------------- grading
def _floor_for(floor: dict, feature: str, column: str) -> dict | None:
    return (floor.get(feature) or {}).get(column)


def grade_feature(
    feature: str,
    tt: dict,
    ref: dict,
    floor: dict,
    n_min: int = mspec.N_MIN,
    permutations: int = stats.DEFAULT_PERMUTATIONS,
) -> dict:
    result: dict = {"n": min(tt["n"], ref["n"]), "n_tt": tt["n"], "n_ref": ref["n"], "metrics": {}}
    notes: list[str] = []
    if tt["n"] < n_min or ref["n"] < n_min:
        notes.append(f"n below the {n_min}-design bar (tt {tt['n']}, ref {ref['n']})")

    for column, kind, needs_oracle in mspec.columns_for(feature):
        entry = _grade_column(column, kind, tt, ref, _floor_for(floor, feature, column), permutations)
        entry["oracle"] = needs_oracle
        result["metrics"][column] = entry
        if entry.get("missing"):
            notes.append(f"{column}: {'no oracle run' if needs_oracle else 'column absent on both arms'}")

    result["diversity"] = _grade_diversity(tt, ref)
    graded = [m for m in result["metrics"].values() if not m.get("missing")]
    result["n_graded_metrics"] = len(graded)
    failed = [c for c, m in result["metrics"].items() if m.get("pass") is False]
    if result["diversity"].get("pass") is False:
        failed.append("diversity")
    result["failed"] = failed
    if not graded:
        result["verdict"] = "NO_DATA"
    elif notes and any("n below" in n for n in notes):
        result["verdict"] = "INSUFFICIENT_N"
    else:
        result["verdict"] = "FAIL" if failed else "PASS"
    result["notes"] = notes
    return result


def _grade_column(column, kind, tt, ref, floor, permutations) -> dict:
    a, b = _column(tt, column), _column(ref, column)
    if kind == "guard":
        return _grade_guard(column, a)
    if a.size == 0 or b.size == 0:
        return {"kind": kind, "missing": True, "n_tt": int(a.size), "n_ref": int(b.size)}
    if kind == "rate":
        tt_pass, ref_pass = int(np.nansum(a)), int(np.nansum(b))
        p = stats.fisher_exact_greater(tt_pass, a.size, ref_pass, b.size)
        rate_tt, rate_ref = tt_pass / a.size, ref_pass / b.size
        floor_drop = (floor or {}).get("delta_rate_abs", 0.0)
        ok = bool(p >= mspec.P_BAR or rate_tt >= rate_ref or (rate_ref - rate_tt) <= floor_drop)
        return {
            "kind": "rate", "tt": round(rate_tt, 6), "ref": round(rate_ref, 6),
            "n_pass_tt": tt_pass, "n_pass_ref": ref_pass, "n_tt": int(a.size), "n_ref": int(b.size),
            "p": round(p, 6), "bar": f"fisher p >= {mspec.P_BAR} (one-sided, drop only)",
            "seed_floor": floor, "pass": ok,
        }
    ks = stats.ks_2samp(a, b, permutations=permutations)
    inside_floor = False
    if floor:
        inside_floor = bool(
            abs(ks.delta_median) <= floor.get("delta_median_abs", 0.0) + 1e-12
            and ks.d <= floor.get("d", 0.0) + 1e-12
        )
    ok = bool(ks.p >= mspec.P_BAR or inside_floor)
    out = ks.as_dict()
    out.update({
        "kind": "dist", "tt": out["median_tt"], "ref": out["median_ref"],
        "bar": f"KS p >= {mspec.P_BAR}, or inside the seed floor",
        "seed_floor": floor, "inside_seed_floor": inside_floor, "pass": ok,
    })
    return out


def _grade_guard(column: str, a: np.ndarray) -> dict:
    bar = mspec.GUARDS.get(column, {})
    if a.size == 0:
        return {"kind": "guard", "missing": True, "n_tt": 0}
    mean = float(np.nanmean(a))
    limit = bar.get("max_mean")
    ok = True if limit is None else bool(mean <= limit + 1e-12)
    return {
        "kind": "guard", "tt": round(mean, 6), "ref": None, "n_tt": int(a.size),
        "bar": f"mean <= {limit}" if limit is not None else "none", "pass": ok,
    }


def _grade_diversity(tt: dict, ref: dict) -> dict:
    u_tt, u_ref = tt.get("n_unique_sequences"), ref.get("n_unique_sequences")
    if not u_tt or not u_ref or not tt["n"] or not ref["n"]:
        return {"missing": True}
    f_tt, f_ref = u_tt / tt["n"], u_ref / ref["n"]
    ok = bool(f_ref == 0 or f_tt >= mspec.DIVERSITY_RATIO_MIN * f_ref)
    return {
        "unique_frac_tt": round(f_tt, 4), "unique_frac_ref": round(f_ref, 4),
        "bar": f"unique_frac_tt >= {mspec.DIVERSITY_RATIO_MIN} x unique_frac_ref", "pass": ok,
    }


# --------------------------------------------------------------------------- the seed floor
def calibrate_floor(feature: str, a: dict, b: dict, permutations: int = stats.DEFAULT_PERMUTATIONS) -> dict:
    """How far the metrics move when only the seed changes. Same arithmetic as the grade."""
    floor: dict = {}
    for column, kind, _ in mspec.columns_for(feature):
        x, y = _column(a, column), _column(b, column)
        if x.size == 0 or y.size == 0:
            continue
        if kind == "rate":
            floor[column] = {
                "delta_rate_abs": round(abs(np.nansum(x) / x.size - np.nansum(y) / y.size), 6),
                "n_a": int(x.size), "n_b": int(y.size),
            }
        elif kind == "dist":
            ks = stats.ks_2samp(x, y, permutations=permutations)
            floor[column] = {
                "delta_median_abs": round(abs(ks.delta_median), 6),
                "d": round(ks.d, 6), "p": round(ks.p, 6),
                "median_a": ks.median_tt, "median_b": ks.median_ref,
                "n_a": int(x.size), "n_b": int(y.size),
            }
    return floor


# --------------------------------------------------------------------------- QUALITY.json
def _empty_doc() -> dict:
    return {"verdict": "FAIL", "seed_floor": {}, "features": {}, "paths": {}, "harness": "perf/rfd4/quality.py"}


def load_doc(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return _empty_doc()


def overall_verdict(doc: dict) -> str:
    feats = doc.get("features") or {}
    if not feats:
        return "FAIL"
    return "PASS" if all(f.get("verdict") == "PASS" for f in feats.values()) else "FAIL"


def write_doc(doc: dict, path: pathlib.Path) -> None:
    doc["verdict"] = overall_verdict(doc)
    doc["updated"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=False, allow_nan=True) + "\n")


# --------------------------------------------------------------------------- CLI
def _arm_summary(arm: dict) -> dict:
    return {
        "root": arm["root"], "n": arm["n"], "n_designs_parsed": arm["n_designs_parsed"],
        "n_unique_sequences": arm["n_unique_sequences"], "csvs": arm["csvs"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="grade a TT arm against a reference arm")
    g.add_argument("--feature", required=True)
    g.add_argument("--tt", required=True)
    g.add_argument("--ref", required=True)
    g.add_argument("--chain", default=None, help="chain of interest for the geom.* columns")
    g.add_argument("--n-min", type=int, default=mspec.N_MIN)
    g.add_argument("--permutations", type=int, default=stats.DEFAULT_PERMUTATIONS)
    g.add_argument("--label", default=None, help="what the TT arm is (a lever name, a sha)")
    g.add_argument("--out", default=str(QUALITY_JSON))

    f = sub.add_parser("floor", help="calibrate the seed floor from two reference seeds")
    f.add_argument("--feature", required=True)
    f.add_argument("--a", required=True)
    f.add_argument("--b", required=True)
    f.add_argument("--chain", default=None)
    f.add_argument("--permutations", type=int, default=stats.DEFAULT_PERMUTATIONS)
    f.add_argument("--out", default=str(QUALITY_JSON))

    s = sub.add_parser("show", help="print the current verdict")
    s.add_argument("--out", default=str(QUALITY_JSON))

    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    doc = load_doc(out)

    if args.cmd == "show":
        print(json.dumps({"verdict": doc.get("verdict"),
                          "features": {k: v.get("verdict") for k, v in (doc.get("features") or {}).items()}}, indent=1))
        return 0

    if args.cmd == "floor":
        a, b = load_arm(args.a, args.chain), load_arm(args.b, args.chain)
        doc.setdefault("seed_floor", {})[args.feature] = calibrate_floor(args.feature, a, b, args.permutations)
        doc.setdefault("paths", {}).setdefault(args.feature, {})["seed_floor_arms"] = [_arm_summary(a), _arm_summary(b)]
        write_doc(doc, out)
        print(f"seed floor for {args.feature}: {len(doc['seed_floor'][args.feature])} metrics -> {out}")
        return 0

    tt, ref = load_arm(args.tt, args.chain), load_arm(args.ref, args.chain)
    res = grade_feature(args.feature, tt, ref, doc.get("seed_floor") or {}, args.n_min, args.permutations)
    res["label"] = args.label
    res["tags"] = mspec.FEATURE_TAGS.get(args.feature)
    doc.setdefault("features", {})[args.feature] = res
    doc.setdefault("paths", {}).setdefault(args.feature, {}).update({"tt": _arm_summary(tt), "ref": _arm_summary(ref)})
    write_doc(doc, out)
    print(f"{args.feature}: {res['verdict']} ({res['n_graded_metrics']} metrics graded"
          f"{', failed: ' + ', '.join(res['failed']) if res['failed'] else ''}) -> {out}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
