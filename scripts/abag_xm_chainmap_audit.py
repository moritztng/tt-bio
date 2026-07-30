#!/usr/bin/env python3
"""Full-dataset audit of shipped chain maps against real-sequence ground truth.

Root cause of the 9q1l finding (addendum A3 cross-validation):
`_build_seq_map` in abag_xm_dockq_interface.py falls back to raw CA-atom-count
proximity when the model CIF lacks _entity_poly sequences (OpenDDE/Protenix).
Heavy and light chains differ by only ~5-10 atoms, so an inflated native CA
count flips the map to the wrong chain type. Multi-copy natives hit the
fallback because DockQ's one-to-one group_chains leaves extra copies unmapped.

For every (target, gen) fold this audits the labels JSON's recorded
(native_chain1 -> model_chain1, native_chain2 -> model_chain2) assignment
against correspondence computed from REAL sequences (atom_site-derived,
abag_xm_dockq_xval.load_chains). A fold is flagged when the recorded model
chain matches the declared native chain worse than the best model chain by
> MARGIN matching-block residues. Sequences are invariant across a fold's 50
samples, so one model CIF (lowest rank) per fold suffices.

Run on qb1:  ~/abag_xm/tinyprot-env/bin/python scripts/abag_xm_chainmap_audit.py --out audit.csv
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abag_xm_dockq_xval as X  # noqa: E402

GEN_DIR = X.GEN_DIR
load_chains = X.load_chains
_correspondence = X._correspondence
# labels JSON file prefixes are the underscored GEN_DIR values
PREFIX2GEN = {v: k for k, v in GEN_DIR.items()}

GT_DIR = Path.home() / "abag_xm" / "ground_truth"
LABELS_DIR = Path.home() / "abag_xm" / "tier_a" / "labels"
MARGIN = 5


def audit_fold(target, gen, natives, native_maps):
    d = json.loads((LABELS_DIR / f"{GEN_DIR[gen]}_{target}.json").read_text())
    samples = [s for s in d["samples"] if (s.get("dockq") or {}).get("model_chain1")]
    if not samples:
        return None
    if target not in natives:
        natives[target], native_maps[target] = load_chains(GT_DIR / f"{target}.cif")
    nc, na = natives[target], native_maps[target]
    # distinct recorded assignments across the fold's samples
    assigns = {(s["dockq"]["native_chain1"], s["dockq"]["native_chain2"],
                s["dockq"]["model_chain1"], s["dockq"]["model_chain2"])
               for s in samples}
    first = samples[0]["dockq"]
    model_path = Path(first["model"])
    if not model_path.exists():
        model_path = Path(samples[0]["cif"])
    mc, ma = load_chains(model_path)
    rows = []
    for n1, n2, m1, m2 in sorted(assigns):
        for side, (nauth, mrec) in (("1", (n1, m1)), ("2", (n2, m2))):
            nlab = na.get(nauth, nauth)
            mlab = ma.get(mrec, mrec)
            if nlab not in nc or mlab not in mc:
                rows.append(dict(target=target, gen=gen, side=side,
                                 native=nauth, recorded=mrec, best="",
                                 rec_pairs=-1, best_pairs=-1,
                                 n_rec=len(assigns),
                                 note="unresolved"))
                continue
            nres = len(nc[nlab].rname)
            rec_n = len(_correspondence(nc[nlab].rname, mc[mlab].rname))
            best, best_n = mlab, rec_n
            for mk, mch in mc.items():
                n = len(_correspondence(nc[nlab].rname, mch.rname))
                if n > best_n:
                    best, best_n = mk, n
            rows.append(dict(target=target, gen=gen, side=side,
                             native=nauth, recorded=mlab, best=best,
                             rec_pairs=rec_n, best_pairs=best_n,
                             n_native_res=nres, n_rec=len(assigns),
                             note="flag" if best_n - rec_n > MARGIN else ""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-glob", default=None,
                    help="restrict to one prefix, e.g. opendde_abag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X._tiny()
    gens = [a.labels_glob] if a.labels_glob else sorted(GEN_DIR)
    natives, native_maps = {}, {}
    all_rows = []
    files = sorted(LABELS_DIR.glob("*.json"))
    for f in files:
        stem = f.stem
        prefix = next((p for p in PREFIX2GEN if stem.startswith(p + "_")), None)
        if prefix is None:
            continue
        gen = PREFIX2GEN[prefix]
        if gen not in gens:
            continue
        target = stem[len(prefix) + 1:]
        try:
            rows = audit_fold(target, gen, natives, native_maps)
        except Exception as e:
            rows = [dict(target=target, gen=gen, side="", native="",
                         recorded="", best="", rec_pairs=-2, best_pairs=-2,
                         note=f"error: {type(e).__name__}: {e}")]
        if rows:
            all_rows.extend(rows)
    out = pd.DataFrame(all_rows)
    out.to_csv(a.out, index=False)
    flagged = out[out.note == "flag"]
    print(f"audited {len(files)} label files -> {len(out)} side-rows")
    print(f"flagged side-rows: {len(flagged)} over "
          f"{flagged[['target','gen']].drop_duplicates().shape[0]} folds")
    if len(flagged):
        print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()
