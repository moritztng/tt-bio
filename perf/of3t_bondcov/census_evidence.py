#!/usr/bin/env python3
"""Turn `of3t-gradients`' §6 runtime census into the same evidence records.

That row demonstrated seven of the eight terms on the frozen 5nw3 batch by a different
quantity than this one: the norm of the gradient each term seeds into the model outputs,
not a share of the squared parameter-gradient norm. Both are evidence that the term
contributes; neither is the other, so the record carries the quantity's name and the
coverage table prints it rather than adding the two together.

Every figure is read out of `coverage_census.json`. The `bond` entry is skipped here on
purpose: that row measured it at zero, and the record that carries it comes from the
target that actually has a polymer-ligand bond.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

QUANTITY = "norm of the gradient seeded into the model outputs"

# The census keys its union by the loss's own term names; `LossWeights`, which
# `stage_loss_coverage.py` iterates, spells one of them out. Mapped rather than renamed,
# because both files are other rows' and neither should move for this one.
TERM_ALIAS = {"resolved": "experimentally_resolved"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--target", default="5nw3",
                    help="the frozen batch's structure, named in the census' own bundle")
    a = ap.parse_args()

    c = json.loads(a.census.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for term, u in c["union"].items():
        if not u["covered"] or not u["carried_by"]:
            print(f"  skip {term}: the census measured it at zero")
            continue
        pair = u["carried_by"][0]
        stage, dataset = pair.split("/", 1)
        seed = c["per_stage_dataset"][pair]["terms"][term]["seed_norm"]
        rec = {
            "term": TERM_ALIAS.get(term, term),
            "stage": stage,
            "dataset": dataset,
            "target": a.target,
            "quantity": QUANTITY,
            "value": seed,
            "n_pairs_firing": u["n_pairs_firing"],
            "weight": c["per_stage_dataset"][pair]["terms"][term]["weight"],
            "term_value": c["per_stage_dataset"][pair]["terms"][term]["value"],
            "source": a.census.name,
        }
        (a.out_dir / f"{TERM_ALIAS.get(term, term)}.json").write_text(json.dumps(rec, indent=1) + "\n")
        n += 1
    print(f"wrote {n} record(s) to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
