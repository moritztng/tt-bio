#!/usr/bin/env python3
"""Which (stage, dataset) pair fires each OpenFold3 loss term.

PROTOCOL §6 / LEDGER K7: no single training stage fires every loss term, so coverage
is the union over stages with a named `(stage, dataset)` pair carrying each one.
This reads that union off their four stage configs instead of asserting it.

Their loss weights are per-dataset overrides layered on the `LossWeights` defaults in
`projects/of3_all_atom/config/dataset_config_components.py`, and they reach the model
per example as `batch["loss_weights"]` (`runner.py:448`). The defaults are read from the
vendored model rather than retyped, so this cannot drift from the tree it documents.

A term is WEIGHTED when at least one (stage, dataset) pair gives it a non-zero weight.
That is a necessary condition and not a sufficient one: PROTOCOL §6 also requires a
non-zero gradient contribution, and a term that is weighted but contributes nothing has
been "skipped with extra steps". `bond` is exactly that shape -- weighted 4.0 on ten of
the sixteen pairs and contributing zero on every corpus target -- so the two numbers are
reported separately and the second is the one a coverage claim may quote.

The second number is read off measurement records, never asserted here. Each record is a
json with `term`, `stage`, `dataset`, `target` and a non-zero gradient share, written by
the instrument that measured it; `--demonstrated` takes the files or a directory of them.
A term with no record, or a record whose share is zero, stays NOT DEMONSTRATED.

Usage:
    python scripts/of3_port/stage_loss_coverage.py --yamls <dir-of-training_yamls>
    python scripts/of3_port/stage_loss_coverage.py --yamls <a> --compare-yamls <b>
    python scripts/of3_port/stage_loss_coverage.py --yamls <a> --json out.json
    python scripts/of3_port/stage_loss_coverage.py --yamls <a> --demonstrated <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STAGES = ("initial_training", "finetune_1", "finetune_2", "finetune_3")


def base_weights() -> dict[str, float]:
    """The LossWeights defaults, read from the vendored config model."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
        LossWeights,
    )

    return LossWeights().model_dump()


def read_stage(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def stage_table(yaml_dir: Path) -> dict:
    """{stage: {"crop": int|None, "datasets": {name: {"class":, "weight":, "loss":}}}}"""
    base = base_weights()
    out: dict[str, dict] = {}
    for stage in STAGES:
        p = yaml_dir / f"{stage}.yml"
        if not p.is_file():
            continue
        cfg = read_stage(p)
        train = cfg.get("dataset_configs", {}).get("train", {}) or {}
        datasets = {}
        crops = set()
        for name, spec in train.items():
            c = spec.get("config", {}) or {}
            eff = dict(base)
            eff.update((c.get("loss", {}) or {}).get("loss_weights", {}) or {})
            budget = (
                ((c.get("crop", {}) or {}).get("token_crop", {}) or {}).get("token_budget")
            )
            if budget:
                crops.add(budget)
            datasets[name] = {
                "class": spec.get("dataset_class"),
                "sampling_weight": spec.get("weight"),
                "token_budget": budget,
                "loss": eff,
            }
        out[stage] = {"crops": sorted(crops), "datasets": datasets}
    return out


def coverage(table: dict) -> dict[str, list[str]]:
    """term -> ["stage/dataset", ...] for every pair giving it a non-zero weight."""
    carriers: dict[str, list[str]] = {}
    for term in base_weights():
        hits = [
            f"{stage}/{name}"
            for stage, s in table.items()
            for name, d in s["datasets"].items()
            if d["loss"].get(term, 0.0)
        ]
        carriers[term] = hits
    return carriers


REQUIRED_EVIDENCE_FIELDS = ("term", "stage", "dataset", "target",
                            "share_of_squared_gradient_norm", "source")


def read_evidence(paths: list[Path]) -> dict[str, dict]:
    """term -> the record demonstrating a non-zero gradient contribution.

    A record is refused rather than ignored when it is malformed, because the failure
    this guards against is a coverage table that counts a term nobody measured.
    """
    found: dict[str, dict] = {}
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for f in files:
        rec = json.loads(f.read_text())
        for r in rec if isinstance(rec, list) else [rec]:
            missing = [k for k in REQUIRED_EVIDENCE_FIELDS if k not in r]
            if missing:
                raise SystemExit(f"{f}: evidence record is missing {missing}")
            share = float(r["share_of_squared_gradient_norm"])
            if share <= 0.0:
                print(f"  note: {f.name} records {r['term']} at share {share:g} -- "
                      f"a term that fires with a zero gradient contribution is NOT covered")
                continue
            prev = found.get(r["term"])
            if prev is None or share > float(prev["share_of_squared_gradient_norm"]):
                found[r["term"]] = {**r, "record": str(f)}
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yamls", type=Path, required=True)
    ap.add_argument("--compare-yamls", type=Path, default=None,
                    help="Second training_yamls dir; reports stage configs that differ.")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--demonstrated", type=Path, nargs="*", default=(),
                    help="measurement records, or directories of them, each naming a term "
                         "and the (stage, dataset, target) at which its gradient "
                         "contribution was measured non-zero")
    args = ap.parse_args()

    table = stage_table(args.yamls)
    base = base_weights()
    terms = list(base)

    print(f"base LossWeights defaults: " + "  ".join(f"{k}={v:g}" for k, v in base.items()))
    print()
    for stage, s in table.items():
        crops = ", ".join(str(c) for c in s["crops"]) or "none"
        print(f"{stage}  (token_budget {crops}, {len(s['datasets'])} train datasets)")
        w = max(len(n) for n in s["datasets"]) if s["datasets"] else 0
        print(f"  {'dataset'.ljust(w)}  " + "  ".join(t[:9].rjust(9) for t in terms))
        for name, d in s["datasets"].items():
            cells = []
            for t in terms:
                v = d["loss"].get(t, 0.0)
                cells.append(("-" if not v else f"{v:g}").rjust(9))
            print(f"  {name.ljust(w)}  " + "  ".join(cells))
        print()

    carriers = coverage(table)
    print("UNION COVERAGE -- the (stage, dataset) pair that fires each term")
    uncovered = []
    for t in terms:
        hits = carriers[t]
        if not hits:
            uncovered.append(t)
            print(f"  {t:26s} NOT COVERED by any stage/dataset")
        else:
            print(f"  {t:26s} {len(hits):2d} pair(s), e.g. {hits[0]}")
    print()
    print(f"{len(terms) - len(uncovered)}/{len(terms)} terms WEIGHTED by the union"
          + (f"; NOT WEIGHTED: {', '.join(uncovered)}" if uncovered else ""))

    evidence = read_evidence(list(args.demonstrated)) if args.demonstrated else {}
    if args.demonstrated:
        print()
        print("DEMONSTRATED -- the (stage, dataset, target) at which the term was measured "
              "to move the gradient")
        undemonstrated = []
        for t in terms:
            e = evidence.get(t)
            if e is None:
                undemonstrated.append(t)
                print(f"  {t:26s} NOT DEMONSTRATED")
            else:
                print(f"  {t:26s} {e['stage']}/{e['dataset']}/{e['target']}  "
                      f"share {float(e['share_of_squared_gradient_norm']):.6e}  "
                      f"({e['source']})")
        print()
        print(f"{len(terms) - len(undemonstrated)}/{len(terms)} terms DEMONSTRATED"
              + (f"; NOT DEMONSTRATED: {', '.join(undemonstrated)}"
                 if undemonstrated else ""))

    # The minimal set: greedily pick pairs until every coverable term is carried. This
    # is what a reproduction actually has to run, and it is smaller than four stages.
    need = {t for t in terms if carriers[t]}
    pairs = {f"{s}/{n}": set(t for t in terms if d["loss"].get(t, 0.0))
             for s, sv in table.items() for n, d in sv["datasets"].items()}
    chosen: list[str] = []
    while need:
        best = max(pairs, key=lambda p: len(pairs[p] & need))
        if not (pairs[best] & need):
            break
        chosen.append(best)
        need -= pairs[best]
    print(f"minimal covering set: {len(chosen)} pair(s)")
    for p in chosen:
        print(f"  {p:42s} carries {', '.join(sorted(pairs[p]))}")

    if args.compare_yamls:
        other = stage_table(args.compare_yamls)
        print(f"\ncompared against {args.compare_yamls}:")
        diffs = 0
        for stage in STAGES:
            a, b = table.get(stage), other.get(stage)
            if a is None or b is None:
                print(f"  {stage}: present in only one"); diffs += 1; continue
            if a != b:
                print(f"  {stage}: DIFFERS")
                for n in sorted(set(a["datasets"]) | set(b["datasets"])):
                    da, db = a["datasets"].get(n), b["datasets"].get(n)
                    if da != db:
                        print(f"      {n}: {'only here' if db is None else 'only there' if da is None else 'config differs'}")
                diffs += 1
        print("  identical" if not diffs else f"  {diffs} stage(s) differ")

    if args.json:
        args.json.write_text(json.dumps(
            {"base": base, "stages": table, "carriers": carriers,
             "uncovered": uncovered, "minimal_set": chosen,
             "demonstrated": evidence}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
