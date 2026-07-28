#!/usr/bin/env python3
"""AbAg-XM Phase 1: build the target manifest from ARK files + RCSB metadata.

Outputs (in docs/implementation-parity-data/):
  - abag-xm-targets.parquet      : one row per target (164), Accept bar
  - abag-xm-interfaces.parquet   : one row per interface (404), full ARK interface data
  - abag-xm-interface-clusters.parquet : one row per interface cluster (159)

Also fetches every target's mmCIF into examples/ground_truth_structures/.
9m8k/9m8l are obsolete (superseded by 25ST/25SU, released 2026-06-24); we record
the supersede and fetch the replacement mmCIF, but keep the ARK rows as-is so
the 159-cluster reproduction is exact. The interface-remapping for these two
is flagged for Moritz (see state doc OPEN GATES spirit).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

# 9m8k/9m8l are obsolete; RCSB REST/GraphQL return 404 / entry:null for them.
# The removed-page reports supersede targets (verified 2026-07-25):
#   9m8k -> 25ST  (TAS2R4 bitter receptor + Fab, released 2026-06-24)
#   9m8l -> 25SU  (same series, released 2026-06-24)
# Both replacements are still post-cutoff for all three generators.
SUPERSEDED = {"9m8k": "25st", "9m8l": "25su"}

GRAPHQL_QUERY = (
    "query($ids:[String!]!){"
    " entries(entry_ids:$ids){"
    " rcsb_id"
    " rcsb_accession_info{ initial_release_date }"
    " rcsb_entry_info{ resolution_combined }"
    " } }"
)


def gql_release_resolution(pdb_ids: list[str], retries: int = 4) -> dict:
    """Batched RCSB GraphQL fetch of release date + resolution."""
    out: dict[str, dict] = {}
    chunk = 100
    for i in range(0, len(pdb_ids), chunk):
        batch = pdb_ids[i : i + chunk]
        payload = {"query": GRAPHQL_QUERY, "variables": {"ids": batch}}
        data = None
        for attempt in range(retries):
            try:
                r = requests.post(RCSB_GRAPHQL, json=payload, timeout=60)
                r.raise_for_status()
                j = r.json()
                if "errors" in j:
                    print(f"[warn] GraphQL errors: {j['errors'][:1]}", file=sys.stderr)
                data = (j.get("data") or {}).get("entries", []) or []
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[warn] GraphQL batch {batch[:3]}... failed: {e}", file=sys.stderr)
                    data = []
                    break
                time.sleep(2 ** attempt)
        for e in data:
            if not e:
                continue
            pid = (e.get("rcsb_id") or "").lower()
            if not pid:
                continue
            rel = (e.get("rcsb_accession_info") or {}).get("initial_release_date")
            res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            res_val = res[0] if res else None
            out[pid] = {"release_date": rel, "resolution": res_val}
    return out


def fetch_cif(pdb_id: str, dest_dir: Path, retries: int = 4) -> str | None:
    """Download a mmCIF from RCSB into dest_dir/<pdb_id>.cif. Returns path or None."""
    pid = pdb_id.lower()
    dest = dest_dir / f"{pid}.cif"
    if dest.exists() and dest.stat().st_size > 1000:
        return str(dest)
    url = RCSB_CIF_URL.format(pdb_id=pid.upper())
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code == 404:
                print(f"[warn] 404 for {pid}", file=sys.stderr)
                return None
            r.raise_for_status()
            dest.write_bytes(r.content)
            return str(dest)
        except Exception as e:
            if attempt == retries - 1:
                print(f"[warn] cif fetch {pid} failed: {e}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


def parse_subset_flags(subset_str: str) -> dict:
    """Parse '[antibody-protein];[antibody_HL-protein];[peptide-interface]' into flags."""
    flags = {
        "has_HL": False,
        "has_H_only": False,   # VHH-like
        "has_scFv": False,
        "has_peptide_antigen": False,
    }
    for s in subset_str.strip("[]").split("];["):
        s = s.strip()
        if s == "antibody_HL-protein":
            flags["has_HL"] = True
        elif s == "antibody_H-protein":
            flags["has_H_only"] = True
        elif s == "antibody_scFv-protein":
            flags["has_scFv"] = True
        elif s in ("peptide-interface", "peptide-protein"):
            flags["has_peptide_antigen"] = True
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ark_dir", default=os.environ.get("ARK_DIR", "/home/ttuser/abag_xm/OpenDDE/benchmarks/2026ARK_AB"))
    ap.add_argument("--out_dir", default="docs/implementation-parity-data")
    ap.add_argument("--cif_dir", default="examples/ground_truth_structures")
    ap.add_argument("--skip_cif", action="store_true", help="skip mmCIF download")
    ap.add_argument("--limit", type=int, default=None, help="limit targets (debug)")
    args = ap.parse_args()

    ark = Path(args.ark_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir)
    cif_dir.mkdir(parents=True, exist_ok=True)

    # ---- read ARK files ----
    with open(ark / "common_targets.txt") as f:
        targets = [line.strip().lower() for line in f if line.strip()]
    with open(ark / "common_interfaces.csv") as f:
        interfaces = list(csv.DictReader(f))
    with open(ark / "common_interface_clusters.csv") as f:
        clusters = list(csv.DictReader(f))

    print(f"ARK: {len(targets)} targets, {len(interfaces)} interfaces, {len(clusters)} clusters")
    assert len(targets) == 164, f"expected 164 targets, got {len(targets)}"
    assert len(interfaces) == 404, f"expected 404 interfaces, got {len(interfaces)}"
    assert len(clusters) == 159, f"expected 159 clusters, got {len(clusters)}"

    # ---- fetch RCSB metadata for all targets + supersede replacements ----
    fetch_ids = sorted(set(targets) | set(SUPERSEDED.values()))
    if args.limit:
        fetch_ids = fetch_ids[: args.limit]
        targets = targets[: args.limit]
    print(f"fetching RCSB metadata for {len(fetch_ids)} ids...")
    meta = gql_release_resolution(fetch_ids)
    print(f"  got metadata for {len(meta)} ids")
    missing_meta = [pid for pid in fetch_ids if pid not in meta]
    if missing_meta:
        print(f"  [warn] missing metadata: {missing_meta}")

    # ---- fetch mmCIFs ----
    if not args.skip_cif:
        # For obsolete 9m8k/9m8l, fetch the supersede CIF under the replacement name.
        cif_fetch_list = []
        for pid in targets:
            if pid in SUPERSEDED:
                cif_fetch_list.append(SUPERSEDED[pid])
            else:
                cif_fetch_list.append(pid)
        print(f"fetching {len(cif_fetch_list)} mmCIFs into {cif_dir}...")
        ok = 0
        for i, pid in enumerate(cif_fetch_list):
            if i % 20 == 0:
                print(f"  {i}/{len(cif_fetch_list)}...")
            p = fetch_cif(pid, cif_dir)
            if p:
                ok += 1
        print(f"  fetched/verified {ok}/{len(cif_fetch_list)} mmCIFs")

    # ---- build interfaces table (404 rows, ARK + RCSB meta) ----
    iface_rows = []
    for row in interfaces:
        pid = row["pdb_id"].lower()
        m = meta.get(pid, {})
        sup = SUPERSEDED.get(pid)
        rec = dict(row)
        rec["release_date"] = m.get("release_date")
        rec["resolution"] = m.get("resolution")
        rec["superseded_by"] = sup
        rec["mmcif_path"] = str(cif_dir / f"{(sup or pid)}.cif")
        iface_rows.append(rec)
    iface_df = pd.DataFrame(iface_rows)

    # ---- build targets table (164 rows) ----
    by_target: dict[str, list[dict]] = {}
    for row in iface_rows:
        by_target.setdefault(row["pdb_id"].lower(), []).append(row)

    target_rows = []
    for pid in targets:
        rows = by_target.get(pid, [])
        if not rows:
            print(f"  [warn] no interface rows for {pid}")
        m = meta.get(pid, {})
        sup = SUPERSEDED.get(pid)
        # For obsolete entries, use the supersede replacement's metadata.
        if sup and not m and sup in meta:
            m = meta[sup]
        # aggregate interface info
        iface_ids = [r["id"] for r in rows]
        iface_cluster_ids = sorted(set(r["interface_cluster_id"] for r in rows))
        entity_cluster_ids = sorted(
            set(r["entity_cluster_id_1"] for r in rows) | set(r["entity_cluster_id_2"] for r in rows)
        )
        subset_strs = ";".join(sorted(set(r["subset"] for r in rows)))
        flags = parse_subset_flags(subset_strs if rows else "")
        # D11: pick the ARK interface row with the largest resolved_seq_length_2
        # among [antibody*-protein] rows (the fold target).
        ab_rows = [r for r in rows if "antibody" in r["subset"]]
        chosen = max(ab_rows, key=lambda r: int(r.get("resolved_seq_length_2") or 0)) if ab_rows else None
        rec = {
            "pdb_id": pid,
            "release_date": m.get("release_date"),
            "resolution": m.get("resolution"),
            "n_interfaces": len(rows),
            "interface_ids": iface_ids,
            "interface_cluster_ids": iface_cluster_ids,
            "entity_cluster_ids": entity_cluster_ids,
            "subset": subset_strs,
            "has_HL": flags["has_HL"],
            "has_H_only": flags["has_H_only"],
            "has_scFv": flags["has_scFv"],
            "has_peptide_antigen": flags["has_peptide_antigen"],
            # D11 fold target (the interface we score):
            "fold_interface_id": chosen["id"] if chosen else None,
            "fold_auth_chain_id_1": chosen["auth_chain_id_1"] if chosen else None,
            "fold_auth_chain_id_2": chosen["auth_chain_id_2"] if chosen else None,
            "fold_entity_id_1": int(chosen["entity_id_1"]) if chosen else None,
            "fold_entity_id_2": int(chosen["entity_id_2"]) if chosen else None,
            "fold_resolved_seq_length_1": int(chosen["resolved_seq_length_1"]) if chosen else None,
            "fold_resolved_seq_length_2": int(chosen["resolved_seq_length_2"]) if chosen else None,
            "fold_entity_cluster_id_1": chosen["entity_cluster_id_1"] if chosen else None,
            "fold_entity_cluster_id_2": chosen["entity_cluster_id_2"] if chosen else None,
            "fold_interface_cluster_id": chosen["interface_cluster_id"] if chosen else None,
            "superseded_by": sup,
            "mmcif_path": str(cif_dir / f"{(sup or pid)}.cif"),
        }
        target_rows.append(rec)
    target_df = pd.DataFrame(target_rows)

    # ---- build clusters table (159 rows) ----
    cluster_df = pd.DataFrame(clusters)

    # ---- write parquets ----
    targets_path = out / "abag-xm-targets.parquet"
    iface_path = out / "abag-xm-interfaces.parquet"
    clusters_path = out / "abag-xm-interface-clusters.parquet"
    target_df.to_parquet(targets_path, index=False)
    iface_df.to_parquet(iface_path, index=False)
    cluster_df.to_parquet(clusters_path, index=False)
    print(f"wrote {targets_path} ({len(target_df)} rows)")
    print(f"wrote {iface_path} ({len(iface_df)} rows)")
    print(f"wrote {clusters_path} ({len(cluster_df)} rows)")

    # ---- Accept checks ----
    print("\n=== ACCEPT CHECKS ===")
    print(f"164 rows: {len(target_df) == 164} (got {len(target_df)})")
    n_clusters = target_df["interface_cluster_ids"].explode().nunique()
    print(f"159 interface clusters reproduced: {n_clusters == 159} (got {n_clusters})")
    null_auth = target_df["fold_auth_chain_id_1"].isna().sum() + target_df["fold_auth_chain_id_2"].isna().sum()
    print(f"no null auth chain ids: {null_auth == 0} (got {null_auth} nulls)")
    # mmCIF presence
    if not args.skip_cif:
        missing_cif = [r["pdb_id"] for r in target_rows if not Path(r["mmcif_path"]).exists()]
        print(f"every target has a local mmCIF: {len(missing_cif) == 0} (missing: {missing_cif})")
    else:
        print("mmCIF check skipped (--skip_cif)")
    # 9m8k/9m8l resolution recorded
    sup_recorded = all(target_df.loc[target_df["pdb_id"] == pid, "superseded_by"].notna().any() for pid in SUPERSEDED)
    print(f"9m8k/9m8l supersede recorded: {sup_recorded}")


if __name__ == "__main__":
    main()
