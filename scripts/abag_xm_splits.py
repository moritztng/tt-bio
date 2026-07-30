#!/usr/bin/env python3
"""AbAg-XM train/val/test splits: fold_interface_cluster_id-primary, deterministic.

Spec (closeout plan 1.3): connected components over targets sharing a
`fold_interface_cluster_id` -- the ARK/PXMeter-native homology unit -- then greedy
assignment of whole components to the split furthest below its 70/10/20 quota.
Simultaneous 3-way disjointness (interface + entity + CDR-H3) is infeasible at 164
targets (the union collapses 137 of them into one component), so entity/CDR-H3
overlap ships as per-target flags plus a published strict-test subset instead.

Determinism: components are ordered by size descending; ties are broken by a seeded
shuffle (RandomState 20260729) applied once to the initial component list.

    python3 scripts/abag_xm_splits.py [--out splits.parquet]

Exit 1 if any acceptance check fails: 164 rows, zero cross-split
fold_interface_cluster_id collisions, and the pinned 115/16/33 assignment.
"""
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"
SEED = 20260729
QUOTA = {"train": 0.70, "val": 0.10, "test": 0.20}
EXPECT = {"train": 115, "val": 16, "test": 33}


def _components(df):
    """Union-find over pairs of targets sharing a fold_interface_cluster_id."""
    parent = {t: t for t in df.pdb_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _, grp in df.groupby("fold_interface_cluster_id"):
        ids = list(grp.pdb_id)
        for other in ids[1:]:
            union(ids[0], other)
    comps = {}
    for t in df.pdb_id:
        comps.setdefault(find(t), []).append(t)
    return list(comps.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "implementation-parity-data"
                                           / "abag-xm-splits.parquet"))
    a = ap.parse_args()
    df = pd.read_parquet(MANIFEST)
    assert len(df) == 164, f"manifest has {len(df)} rows, expected 164"

    comps = _components(df)
    # Seeded shuffle first (stable tie-break), then sort by size descending.
    order = np.random.RandomState(SEED).permutation(len(comps))
    comps = [comps[i] for i in order]
    comps.sort(key=len, reverse=True)

    n = len(df)
    quota = {k: v * n for k, v in QUOTA.items()}
    have = {k: 0 for k in QUOTA}
    assign = {}
    for comp in comps:
        # Furthest below quota in ABSOLUTE targets (pinned: reproduces 115/16/33
        # exactly; a relative deficit off by one target sends the 4-target
        # component to val instead). Dict order train > val > test breaks ties.
        best = max(quota, key=lambda s: quota[s] - have[s])
        for t in comp:
            assign[t] = best
        have[best] += len(comp)

    df["split"] = df.pdb_id.map(assign)
    comp_id = {}
    for i, comp in enumerate(comps):
        for t in comp:
            comp_id[t] = i
    df["component_id"] = df.pdb_id.map(comp_id)

    # Cross-split flags: does this target's entity_2 / cdrh3 cluster appear in a
    # different split? (Interface clusters cannot cross by construction.)
    for col, flag in (("fold_entity_cluster_id_2", "entity2_cross_split"),
                      ("cdrh3_cluster", "cdrh3_cross_split")):
        split_by_cluster = df.groupby(col)["split"].agg(lambda s: set(s.dropna()))
        df[flag] = [len(split_by_cluster[c]) > 1 if pd.notna(c) else False
                    for c in df[col]]

    # Strict test: test targets whose entity_2 AND cdrh3 clusters appear nowhere
    # in train.
    train_e2 = set(df.loc[df.split == "train", "fold_entity_cluster_id_2"].dropna())
    train_c3 = set(df.loc[df.split == "train", "cdrh3_cluster"].dropna())
    test = df[df.split == "test"]
    strict = test[[(e not in train_e2) and (c not in train_c3)
                   for e, c in zip(test.fold_entity_cluster_id_2, test.cdrh3_cluster)]]
    df["strict_test"] = df.pdb_id.isin(set(strict.pdb_id))

    out_cols = ["pdb_id", "split", "component_id", "fold_interface_cluster_id",
                "fold_entity_cluster_id_2", "cdrh3_cluster",
                "entity2_cross_split", "cdrh3_cross_split", "strict_test"]
    df[out_cols].to_parquet(a.out, index=False)

    # ---- acceptance checks -----------------------------------------------------
    fails = []
    counts = df.split.value_counts().to_dict()
    if len(df) != 164:
        fails.append(f"{len(df)} rows != 164")
    clash = df.groupby("fold_interface_cluster_id")["split"].nunique()
    if int((clash > 1).sum()):
        fails.append(f"{int((clash > 1).sum())} interface clusters cross splits")
    if counts != EXPECT:
        fails.append(f"split sizes {counts} != {EXPECT}")
    if fails:
        for f in fails:
            print("FAIL:", f)
        sys.exit(1)

    print(f"components: {len(comps)} (largest {max(len(c) for c in comps)})")
    print(f"split: train {counts['train']} ({counts['train']/n:.1%}) / "
          f"val {counts['val']} ({counts['val']/n:.1%}) / test {counts['test']} "
          f"({counts['test']/n:.1%})")
    print(f"zero cross-split fold_interface_cluster_id collisions: verified")
    print(f"entity2_cross_split targets: {int(df.entity2_cross_split.sum())}; "
          f"cdrh3_cross_split targets: {int(df.cdrh3_cross_split.sum())}")
    print(f"strict test subset: {len(strict)} of {len(test)} test targets "
          f"(entity_2 AND cdrh3 clusters absent from train)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
