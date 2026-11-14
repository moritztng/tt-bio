"""Which inputs can be graded seed-matched against upstream, and which cannot.

The featurizer's two ambient-stream draws decide this (state/rfd4/PAIR.md): the chain and
molecule label permutations, and the internal-coordinate frame rotation. An input is pairable
exactly when neither draw has any freedom -- one chain, one molecule, and no internal-coordinate
group. This featurizes each spec once and reports that.

    PYTHONPATH=<worktree> /home/moritz/rfd4-ref-env/venv/bin/python \
        perf/rfd4/e2e/pair_family_scan.py --out <json> <spec.yaml> ...
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default="/home/moritz/rfd4-pair-runs/tt/families")
    ap.add_argument("--out")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    from tt_bio import rfd4

    rows = {}
    for spec in a.specs:
        name = os.path.basename(spec).removesuffix(".yaml")
        try:
            design = rfd4.prepare_examples(spec, f"{a.work}/{name}")[0]
            feats, aa = rfd4.featurize(design.cif, seed=a.seed)
            groups = np.asarray(feats["condition_CoordinateGroup_1_atom"]).reshape(-1)
            n_internal = int((groups > 0).sum())
            n_chains = int(len(np.unique(aa.chain_iid)))
            n_molecules = int(len(np.unique(aa.molecule_iid)))
            rows[name] = {
                "chains": n_chains, "molecules": n_molecules,
                "internal_coord_atoms": n_internal,
                "pairable": n_chains == 1 and n_molecules == 1 and n_internal == 0,
            }
        except Exception as exc:  # a spec the scan cannot prepare is reported, not swallowed
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(name, json.dumps(rows[name]), flush=True)

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
