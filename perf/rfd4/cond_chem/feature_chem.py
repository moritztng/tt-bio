"""What chemistry actually reaches the model for one annotated CIF.

    python perf/rfd4/cond_chem/feature_chem.py <annotated.cif> [--seed 0] [--out json]

`chem_check.py` grades the chemistry of the written design. This grades the other end: the
features `tt_bio.rfd4.featurize` hands the network. A ligand whose bond orders arrive as zeros
is a ligand the model cannot see, and that is invisible in the output CIF, which copies its
chemistry from the input either way.

The fields are RFD4's own (`datapipes_schema_proteina.SPECS`): `element` and `charge` are
per atom37 slot, `bond_order`, `is_aromatic` and `na_ss_pair` are token-by-token, `na_ss_state`
is per token. Bonds are stored symmetrically, so a count of 58 nonzero entries is 29 bonds.
"""
from __future__ import annotations

import argparse
import json

#: Fields that carry chemistry rather than geometry or conditioning.
FIELDS = (
    "element", "charge", "bond_order", "is_aromatic", "bond_mask",
    "na_ss_state", "na_ss_state_mask", "na_ss_pair", "na_ss_pair_mask",
)


def summarize(cif, seed=0):
    import torch

    from tt_bio.rfd4 import featurize

    feats, _ = featurize(cif, seed=seed)
    out = {"cif": str(cif), "seed": seed, "tokens": int(feats["padding_mask"].shape[1]), "fields": {}}
    for k in FIELDS:
        if k not in feats:
            out["fields"][k] = None
            continue
        t = feats[k]
        nonzero = int(t.sum()) if t.dtype == torch.bool else int((t != 0).sum())
        entry = {"shape": list(t.shape), "dtype": str(t.dtype), "nonzero": nonzero}
        if t.dtype != torch.bool and nonzero:
            vals, counts = torch.unique(t[t != 0], return_counts=True)
            entry["values"] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
        out["fields"][k] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = summarize(a.cif, a.seed)
    text = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
