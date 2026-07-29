#!/usr/bin/env python3
"""Build the release parquet tables from the per-fold label JSONs and progress.jsonl.

The release layout (Phases 6-7) is three parquet files plus coordinates. `targets.parquet`
already comes from abag_xm_build_manifest.py; the other two had no builder, which would have
been discovered at publish time.

  labels.parquet     one row per (target, generator, sample): accuracy labels, PAE-derived
                     scores, native confidences, and the per-fold provenance that makes a row
                     auditable back to the code that produced it.
  ensembles.parquet  one row per (target, generator): the condensed C(50,2) similarity matrix,
                     PSS, and basin clustering.

The condensed matrix is kept as a list column rather than exploded into 1225 rows per fold --
it is consumed as a vector (scipy.spatial.distance.squareform) and exploding it would multiply
the row count by three orders of magnitude for no gain.

Reads only; writes to --out_dir. Safe to run while generation is in flight, and worth running
early precisely so the schema is settled before there is a deadline.

    python3 scripts/abag_xm_build_release_tables.py --out_dir ~/abag_xm/release
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TIERA = Path.home() / "abag_xm" / "tier_a"
LABELS_DIR = TIERA / "labels"
PROGRESS = TIERA / "progress.jsonl"
DIR_TO_GEN = {"protenix_v2": "protenix-v2", "opendde_abag": "opendde-abag", "boltz2": "boltz2",
              "esmfold2": "esmfold2"}
PROVENANCE = ("host", "tt_bio_commit", "host_threads", "paired_msa", "mps", "n_samples",
              "wall_s", "device", "timeout_s", "recovered")


def _fold_provenance():
    """Latest ok record per (target, generator) -- the provenance for its samples."""
    out = {}
    if PROGRESS.exists():
        for line in PROGRESS.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                out[(r["target"], r["model"])] = {k: r.get(k) for k in PROVENANCE}
    return out


def _native_confidences(results_dir):
    """{rank: per-sample confidence dict} from a fold's results.json.

    Keyed by each entry's OWN `rank` field, never by list position. That is not defensive style,
    it is the bug this table nearly shipped: abag_xm_ranker_scores.py paired all_runs[k] with
    samples[k], and because labels.py sorts sample files by FILENAME (0, 1, 10, 11, ..., 2, ...)
    while all_runs is rank-ordered, 48 of every 50 rows carried another structure's numbers.
    `sample` in this table is the label record's rank, so looking confidence up by that rank is
    the only join that cannot drift.

    The card promised "native confidences" in labels.parquet from the beginning and nothing wrote
    them, so the release shipped a ranking benchmark whose ipTM baseline was not reproducible from
    the released files.
    """
    try:
        doc = json.loads((Path(results_dir) / "results.json").read_text())
    except Exception:
        return {}
    entry = doc[0] if isinstance(doc, list) else doc
    out = {}
    for i, r in enumerate(entry.get("all_runs") or []):
        if isinstance(r, dict):
            out[r.get("rank", i)] = r
    return out


def _flat(prefix, d):
    """Flatten one nested label block into prefix_key columns, skipping paths and raws."""
    if not isinstance(d, dict):
        return {}
    return {f"{prefix}_{k}": v for k, v in d.items()
            if k not in ("model", "native", "yaml", "raw", "chain_map") and not isinstance(v, (dict, list))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(Path.home() / "abag_xm" / "release"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prov = _fold_provenance()
    label_rows, ens_rows, no_conf = [], [], []
    for f in sorted(LABELS_DIR.glob("*.json")):
        stem = f.stem
        gen_dir = next((d for d in DIR_TO_GEN if stem.startswith(d + "_")), None)
        if gen_dir is None:
            print(f"  skip {f.name}: unrecognised generator prefix", file=sys.stderr)
            continue
        gen = DIR_TO_GEN[gen_dir]
        target = stem[len(gen_dir) + 1:]
        o = json.loads(f.read_text())
        p = prov.get((target, gen), {})
        conf = _native_confidences(o.get("results_dir") or "")
        if not conf:
            no_conf.append(f"{target}/{gen}")
        for s in o.get("samples", []):
            row = {"target": target, "generator": gen, "sample": s.get("rank"),
                   "source_sha256": o.get("source_sha256")}
            # Only scalars survive _flat, so boltz2's pair_chains_iptm / chains_ptm matrices are
            # deliberately NOT flattened here: they are keyed by chain INDEX and asymmetric, so
            # reducing them to "the declared pair's ipTM" needs an index-to-chain-id convention
            # and an orientation choice, and protenix-v2/opendde-abag do not emit them at all.
            # The declared-pair-aware scores that DO span all three generators are the PAE-derived
            # pae_ipsae / pae_pdockq2 columns.
            row.update(_flat("conf", {k: v for k, v in conf.get(s.get("rank"), {}).items()
                                      if k != "rank"}))
            row.update(_flat("dockq", s.get("dockq")))
            row.update(_flat("pae", s.get("pae_metrics")))
            row["epitope_jaccard"] = s.get("epitope_jaccard")
            # interface_lddt is a block, not a scalar: it wraps the score with the antigen
            # chain it was computed against and the interface size, and carries `_error`
            # instead of a score when the chain could not be identified.
            il = s.get("interface_lddt") or {}
            row["interface_lddt"] = il.get("interface_lddt") if isinstance(il, dict) else il
            row["interface_lddt_n_res"] = il.get("n_interface_residues") if isinstance(il, dict) else None
            row["interface_lddt_antigen_chain"] = il.get("antigen_chain") if isinstance(il, dict) else None
            row["interface_lddt_error"] = il.get("_error") if isinstance(il, dict) else None
            for cdr, v in ((s.get("cdr_rmsd") or {}).get("cdrs") or {}).items():
                row[f"cdr_{cdr.lower()}_rmsd"] = v
            row.update(p)
            label_rows.append(row)
        pm, bc = o.get("pairwise_matrix") or {}, o.get("basin_clust") or {}
        ens_rows.append({
            "target": target, "generator": gen,
            "n_samples": o.get("n_samples"),
            "pss": pm.get("PSS"), "interface": pm.get("interface"),
            "model_chain1": pm.get("model_chain1"), "model_chain2": pm.get("model_chain2"),
            "condensed_matrix": pm.get("matrix"),
            "basin_n_clusters": bc.get("n_clusters"),
            "basin_occupancy": bc.get("basin_occupancy"),
            "basin_labels": bc.get("labels"),
            "source_sha256": o.get("source_sha256"),
            **p,
        })

    if not label_rows:
        print("no label files found; nothing to build")
        return 1
    lab, ens = pd.DataFrame(label_rows), pd.DataFrame(ens_rows)
    lab.to_parquet(out_dir / "labels.parquet", index=False)
    ens.to_parquet(out_dir / "ensembles.parquet", index=False)
    print(f"labels.parquet    {len(lab):6d} rows x {len(lab.columns):3d} cols -> {out_dir}")
    print(f"ensembles.parquet {len(ens):6d} rows x {len(ens.columns):3d} cols")
    miss = [c for c in ("dockq_dockq", "epitope_jaccard", "interface_lddt", "cdr_h3_rmsd")
            if c in lab.columns and lab[c].isna().all()]
    if miss:
        print(f"  WARNING: entirely null in every row: {miss}")
    for c in ("dockq_dockq", "interface_lddt", "cdr_h3_rmsd", "conf_iptm"):
        if c in lab.columns:
            print(f"  {c:18s} non-null {lab[c].notna().sum():5d}/{len(lab)}")
    # A fold whose results.json is unreadable yields labels with no confidence at all, which is a
    # ranking benchmark row that cannot be ranked. Named, not silently blank.
    if no_conf:
        print(f"  WARNING: {len(no_conf)} fold(s) have NO native confidence "
              f"(results.json unreadable): {sorted(no_conf)[:6]}"
              + (" ..." if len(no_conf) > 6 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
