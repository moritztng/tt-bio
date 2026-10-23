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
            found += sorted(p for p in root.rglob(pattern) if not _is_sidecar(p.relative_to(root)))
    return found


# Upstream's naming (docs/reference/output-format.md): the design is `_generated`, everything
# else beside it is the input copy, a refold, a chain extract or a trajectory frame. tt-bio's own
# design output keeps the input template, with its unresolved designed chain, under `inputs/`.
SIDECAR = ("_input", "_refolded", "_chain", "trajectories")
SIDECAR_DIRS = ("trajectories", "inputs")


def _is_sidecar(path: pathlib.Path) -> bool:
    return any(s in path.name for s in SIDECAR) or any(d in path.parts for d in SIDECAR_DIRS)


def load_arm(path, chain: str | None = None) -> dict:
    """One arm: per-design rows keyed by design path, upstream columns plus `geom.*`.

    `path` may be several roots, because one arm's metrics are split across two output trees:
    generation writes `inference_metrics/` beside the CIFs, and a later `rfd4-evaluate` pass over
    those same CIFs writes `evaluation_metrics/` somewhere else. Both CSVs carry the same absolute
    `designed_structure_path`, so the rows merge per design. An evaluate tree also re-exports every
    input design under `evaluation_samples/`, so a design whose row already carries geometry is
    skipped: one design counts once however many trees hold a copy of it.
    """
    roots = [pathlib.Path(p) for p in ([path] if isinstance(path, (str, pathlib.Path)) else path)]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
    rows: dict[str, dict] = {}
    csvs = [p for root in roots for p in _find_csvs(root)]
    for csv_path in csvs:
        with csv_path.open() as fh:
            for rec in csv.DictReader(fh):
                key = rec.get("designed_structure_path") or rec.get("example_id") or str(len(rows))
                rows.setdefault(key, {}).update({k: _num(v) for k, v in rec.items()})
    designs = [p for root in roots for p in _find_designs(root)]
    sequences: list[str] = []
    n_designs = 0
    for cif in designs:
        key = str(cif)
        match = key if key in rows else (_match_row(rows, cif) or key)
        if "geom.n_ca" in rows.get(match, {}):
            continue  # already loaded from another root's copy of this design
        # For a complex, the geometry and the sequence must describe the BINDER, not the whole
        # assembly: a radius of gyration dominated by a fixed target barely moves whatever the
        # port does to the binder, and a sequence that includes the target makes every design
        # look alike. Upstream already decided which chain that is, per design.
        coi = chain or _chain_of_interest(rows.get(match))
        geom = geometry.design_geometry(cif, chain=coi)
        if not geom:
            continue
        rows.setdefault(match, {}).update(geom)
        sequences.append(geometry.sequence_of(cif, chain=coi))
        n_designs += 1
    return {
        "root": str(roots[0]) if len(roots) == 1 else [str(r) for r in roots],
        "rows": list(rows.values()),
        "n": len(rows),
        "n_designs_parsed": n_designs,
        "n_unique_sequences": len(set(s for s in sequences if s)) if sequences else None,
        "csvs": [str(p) for p in csvs],
    }


def _chain_of_interest(row: dict | None) -> str | None:
    """Which chain a design's geometry and sequence describe, per design.

    Upstream writes a `chain_of_interest` column, but it came back EMPTY in both real
    `rfd4-evaluate` runs this harness has seen (a monomer and a PD-L1 binder, 2026-10-21), so
    reading only that column grades a complex on the whole assembly and nothing says so.
    `designed_chain_path` is the one that is actually filled: upstream extracts the chain of
    interest to its own CIF per design. On the binder run that extract is 341 atoms with
    `label_asym_id` B, exactly the 341 atoms of chain B in the full design, which is also the
    evidence that upstream's chain id and the `label_asym_id` the geometry selects on are the
    same namespace.

    Several chains in the extract, or no extract, means no single chain of interest: the whole
    structure is then the right scope (it is, for a monomer).
    """
    row = row or {}
    value = row.get("chain_of_interest")
    if isinstance(value, str) and value.strip():
        return value.strip()
    path = row.get("designed_chain_path")
    if not isinstance(path, str) or not path:
        return None
    chain_cif = pathlib.Path(path)
    if not chain_cif.exists():
        return None
    chains = set(geometry.parse_atom_site(chain_cif).get("label_asym_id", []))
    return chains.pop() if len(chains) == 1 else None


def _match_row(rows: dict[str, dict], cif: pathlib.Path) -> str | None:
    """Match a CIF to its CSV row by filename, not by substring: `monomer_short_1_generated`
    is a substring of `monomer_short_1x_generated` and of any path containing it."""
    stem = cif.name.split(".")[0]
    for key in rows:  # a path row: compare filenames
        if pathlib.PurePath(key).name.split(".")[0] == stem:
            return key
    for key in rows:  # an example_id row: upstream suffixes the id with a content hash
        if key == stem or stem.startswith(key + "_") or key.startswith(stem + "_"):
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
    """Finite values of one column. An empty CSV cell parses as NaN, which is not a measurement:
    a column that is NaN on every design is the same as a column upstream never wrote."""
    vals = [r[name] for r in arm["rows"] if isinstance(r.get(name), float) and math.isfinite(r[name])]
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
    allow_no_oracle: bool = False,
    allow_no_floor: bool = False,
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
    missing_oracle = [c for c, m in result["metrics"].items() if m.get("missing") and m.get("oracle")]
    result["missing_oracle_metrics"] = missing_oracle
    failed = [c for c, m in result["metrics"].items() if m.get("pass") is False]
    if result["diversity"].get("pass") is False:
        failed.append("diversity")
    result["failed"] = failed
    # The floor is what says a move is smaller than the reference's own seed noise. Without one
    # measured for THIS family, a p-value has nothing to be read against and the charter's
    # discipline (quote the deviation beside the floor) cannot be met.
    result["seed_floor_source"] = (floor.get(feature) or {}).get("_source")
    result["seed_floor_kind"] = (floor.get(feature) or {}).get("_kind")
    if not (floor.get(feature) or {}):
        notes.append("no seed floor calibrated for this family")

    # A pre-oracle filter the designs cannot satisfy fails every design, and upstream then skips
    # every downstream metric. The arm looks like one graded column at a rate of 0 rather than
    # like the empty CSV it is, so say what happened instead of grading the wreckage.
    all_filtered = [name for name, arm in (("tt", tt), ("ref", ref))
                    if (col := _column(arm, "pre_oracle_pass")).size and not col.any()]
    if all_filtered:
        notes.append(f"every design was rejected by a pre-oracle filter on the {', '.join(all_filtered)} "
                     f"arm, so upstream computed no metric for it")

    if not graded or all_filtered:
        result["verdict"] = "NO_DATA"
    elif failed:
        result["verdict"] = "FAIL"
    elif any("n below" in n for n in notes):
        result["verdict"] = "INSUFFICIENT_N"
    elif missing_oracle and not allow_no_oracle:
        # The pass criteria and every refold RMSD come from the oracle. A family graded on its
        # oracle-free columns alone has not been graded, and must not read as a PASS.
        result["verdict"] = "ORACLE_MISSING"
    elif not (floor.get(feature) or {}) and not allow_no_floor:
        result["verdict"] = "NO_FLOOR"
    else:
        result["verdict"] = "PASS"
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
    hi, lo = bar.get("max_mean"), bar.get("min_mean")
    ok = ((hi is None or mean <= hi + 1e-12) and (lo is None or mean >= lo - 1e-12))
    bars = [f"mean <= {hi}"] * (hi is not None) + [f"mean >= {lo}"] * (lo is not None)
    return {
        "kind": "guard", "tt": round(mean, 6), "ref": None, "n_tt": int(a.size),
        "bar": " and ".join(bars) if bars else "none", "pass": bool(ok),
    }


def _grade_diversity(tt: dict, ref: dict) -> dict:
    """Unique sequences out of n, tested against the reference arm, not against a constant.

    The unique count is a count, so it gets the same one-sided Fisher exact as a pass rate:
    small p means the TT arm really did lose diversity. A ratio bar was tried first and was
    wrong -- unique_frac quantizes in steps of 1/n, so at n=8 a single duplicate moves it by
    0.125 and upstream against itself (6/8 vs 4/8) failed a 0.8x bar. Fisher reads that as
    p=0.61, which is the honest answer. The one absolute bar left is literal mode collapse:
    every design the same sequence is a broken port at any n.
    """
    u_tt, u_ref = tt.get("n_unique_sequences"), ref.get("n_unique_sequences")
    if not u_tt or not u_ref or not tt["n"] or not ref["n"]:
        return {"missing": True}
    f_tt, f_ref = u_tt / tt["n"], u_ref / ref["n"]
    p = stats.fisher_exact_greater(u_tt, tt["n"], u_ref, ref["n"])
    collapsed = tt["n"] >= 2 and u_tt < 2
    return {
        "unique_tt": int(u_tt), "unique_ref": int(u_ref), "n_tt": tt["n"], "n_ref": ref["n"],
        "unique_frac_tt": round(f_tt, 4), "unique_frac_ref": round(f_ref, 4),
        "p": round(p, 6), "collapsed": collapsed,
        "bar": f"Fisher exact p >= {mspec.P_BAR} against a drop, and more than one unique sequence",
        "pass": bool(p >= mspec.P_BAR and not collapsed),
    }


# --------------------------------------------------------------------------- the seed floor
FLOOR_KINDS = ("two-seed", "resample")


def calibrate_floor(feature: str, a: dict, b: dict, permutations: int = stats.DEFAULT_PERMUTATIONS,
                    source: str | None = None, kind: str = "two-seed") -> dict:
    """How far the metrics move when only the seed changes. Same arithmetic as the grade.

    `source` names the arms the floor was measured on and is carried into every grade that reads
    it: a floor measured on a reduced CPU recipe does not license a bar on a released-recipe run.

    `kind` says what the two arms actually differ by, because the two are not equally strong.
    `two-seed` is the charter's floor: two runs of the same recipe at different seeds. `resample`
    is two disjoint halves of ONE arm, which measures the sampling noise of a fixed configuration
    and nothing about the seed itself. A resample floor is a real floor for a distributional test
    and it is not the charter's, so it is recorded structurally rather than left to the prose in
    `_source`, and every grade that reads it repeats the kind.
    """
    if kind not in FLOOR_KINDS:
        raise ValueError(f"floor kind {kind!r} is not one of {FLOOR_KINDS}")
    floor: dict = {"_kind": kind,
                   "_source": source or f"{a['root']} vs {b['root']}",
                   "_measured": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "_n": [a["n"], b["n"]]}
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
    """PASS only when every family passed, and otherwise the reason -- `FAIL` is reserved for a
    metric that actually moved. A gate reading this must treat anything but PASS as not passing;
    calling an ungraded campaign FAIL instead makes a missing oracle look like a regression."""
    feats = doc.get("features") or {}
    if not feats:
        return "NOT_GRADED"
    verdicts = {f.get("verdict") for f in feats.values()}
    if verdicts == {"PASS"}:
        return "PASS"
    for v in ("FAIL", "NO_DATA", "NO_FLOOR", "ORACLE_MISSING", "INSUFFICIENT_N"):
        if v in verdicts:
            return v
    return "FAIL"


def write_doc(doc: dict, path: pathlib.Path) -> None:
    doc["verdict"] = overall_verdict(doc)
    doc["updated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=False, allow_nan=True) + "\n")


# --------------------------------------------------------------------------- CLI
def _arm_summary(arm: dict) -> dict:
    return {
        "root": arm["root"], "n": arm["n"], "n_designs_parsed": arm["n_designs_parsed"],
        "n_unique_sequences": arm["n_unique_sequences"], "csvs": arm["csvs"],
    }


ARM_HELP = ("one arm; repeat the flag when the arm's designs and its metrics sit in different "
            "trees (generation writes inference_metrics/, rfd4-evaluate writes evaluation_metrics/)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="grade a TT arm against a reference arm")
    g.add_argument("--feature", required=True)
    g.add_argument("--tt", required=True, action="append", help=ARM_HELP)
    g.add_argument("--ref", required=True, action="append", help=ARM_HELP)
    g.add_argument("--chain", default=None, help="chain of interest for the geom.* columns")
    g.add_argument("--n-min", type=int, default=mspec.N_MIN)
    g.add_argument("--permutations", type=int, default=stats.DEFAULT_PERMUTATIONS)
    g.add_argument("--allow-no-oracle", action="store_true",
                   help="grade on the oracle-free columns only; the family can then PASS without a refold")
    g.add_argument("--allow-no-floor", action="store_true",
                   help="grade without a calibrated seed floor; every bar is then the bare p-value")
    g.add_argument("--label", default=None, help="what the TT arm is (a lever name, a sha)")
    g.add_argument("--out", default=str(QUALITY_JSON))

    f = sub.add_parser("floor", help="calibrate the seed floor from two reference seeds")
    f.add_argument("--feature", required=True)
    f.add_argument("--a", required=True, action="append", help=ARM_HELP)
    f.add_argument("--b", required=True, action="append", help=ARM_HELP)
    f.add_argument("--chain", default=None)
    f.add_argument("--permutations", type=int, default=stats.DEFAULT_PERMUTATIONS)
    f.add_argument("--kind", default="two-seed", choices=list(FLOOR_KINDS),
                   help="two-seed: the charter's floor. resample: two halves of one arm, which "
                        "bounds sampling noise only and is labelled as such in QUALITY.json")
    f.add_argument("--source", default=None,
                   help="what the two seed arms are (recipe, host, session) -- every grade quotes it")
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
        doc.setdefault("seed_floor", {})[args.feature] = calibrate_floor(
            args.feature, a, b, args.permutations, args.source, args.kind)
        doc.setdefault("paths", {}).setdefault(args.feature, {})["seed_floor_arms"] = [_arm_summary(a), _arm_summary(b)]
        write_doc(doc, out)
        n_cols = sum(1 for k in doc["seed_floor"][args.feature] if not k.startswith("_"))
        print(f"seed floor for {args.feature}: {n_cols} metrics -> {out}")
        return 0

    tt, ref = load_arm(args.tt, args.chain), load_arm(args.ref, args.chain)
    res = grade_feature(args.feature, tt, ref, doc.get("seed_floor") or {}, args.n_min,
                        args.permutations, args.allow_no_oracle, args.allow_no_floor)
    res["allow_no_oracle"] = args.allow_no_oracle
    res["allow_no_floor"] = args.allow_no_floor
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
