#!/usr/bin/env python3
"""AbAg-XM Phase 1: ANARCI CDR-H3 extraction + MMseqs2 clustering.

For every target, extract all polymer chains from the local mmCIF, run ANARCI
(IMGT scheme) to identify antibody heavy chains, extract CDR-H3, then cluster
all CDR-H3 sequences with MMseqs2 at 80% identity. Adds `cdrh3_sequences` and
`cdrh3_cluster` columns to the targets manifest.

Robust to chain labeling: ANARCI itself classifies each chain as H/L/K or
non-antibody, so we do not depend on ARK's auth_chain_id assignment.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import pandas as pd
from anarci import run_anarci


def extract_chains(cif_path: Path) -> list[tuple[str, str]]:
    """Return [(auth_chain_id, one_letter_seq), ...] for all polymer chains.

    Sanitizes the sequence: strips gap chars ('-') which ANARCI's HMMER parser
    rejects, and converts unknown residues to 'X'.
    """
    if not cif_path.exists():
        return []
    st = gemmi.read_structure(str(cif_path))
    out = []
    for model in st:
        for ch in model:
            polymer = ch.get_polymer()
            seq = str(polymer.make_one_letter_sequence())
            seq = seq.replace("-", "")  # drop gaps (missing residues)
            seq = "".join(c if c in "ACDEFGHIKLMNPQRSTVWY" else "X" for c in seq)
            if len(seq) >= 50:  # skip very short fragments / peptides
                out.append((ch.name, seq))
        break  # first model only
    return out


def cdrh3_from_numbering(numbered, scheme: str = "imgt") -> str | None:
    """Extract CDR-H3 (IMGT 105-117 inclusive) from an ANARCI numbering result.

    `numbered` is a list of (position, aa) tuples. IMGT CDR-H3 spans 105..117.
    """
    if not numbered:
        return None
    cdr = []
    for pos, aa in numbered:
        # pos is like (105, ' ') for IMGT; insertions have a letter in pos[1]
        if not isinstance(pos, tuple):
            continue
        resnum = pos[0]
        if 105 <= resnum <= 117:
            if aa != "-":  # skip alignment gaps within CDR-H3
                cdr.append(aa)
    return "".join(cdr) if cdr else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/implementation-parity-data/abag-xm-targets.parquet")
    ap.add_argument("--cif_dir", default="examples/ground_truth_structures")
    ap.add_argument("--out_manifest", default=None, help="default: overwrite --manifest")
    ap.add_argument("--cluster_identity", default="0.8")
    ap.add_argument("--work_dir", default="/home/ttuser/abag_xm/cdrh3")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir)
    out_manifest = args.out_manifest or args.manifest

    df = pd.read_parquet(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    # ---- extract + ANARCI for every target ----
    # Collect all (target, chain, seq) for antibody chains, then batch ANARCI.
    all_seqs: list[tuple[str, str]] = []  # (id, seq); id = "pdbid_chainid"
    target_chains: dict[str, list[tuple[str, str]]] = {}  # pdb_id -> [(chain_id, seq)]

    print(f"extracting chains from {len(df)} mmCIFs...")
    for _, row in df.iterrows():
        pid = row["pdb_id"]
        cif = Path(row["mmcif_path"])
        chains = extract_chains(cif)
        target_chains[pid] = chains
        for ch_id, seq in chains:
            all_seqs.append((f"{pid}_{ch_id}", seq))
    print(f"  {len(all_seqs)} total polymer chains")

    # ---- batch ANARCI ----
    print(f"running ANARCI on {len(all_seqs)} sequences (IMGT)...")
    # run_anarci returns 4 lists, indexed by input sequence:
    #   r[1][i] = [(numbered_list, hit_idx, qend), ...]   per HMM hit
    #   r[2][i] = [{hit details incl chain_type, species}, ...]
    # (r[0][i] shape varies between tuple and list across versions; use r[2] for
    #  chain_type which is consistently a list of hit dicts.)
    _r0, r1, r2, _r3 = run_anarci(all_seqs, scheme="imgt", ncpu=4)

    # ---- extract CDR-H3 per heavy chain ----
    # Map: pdb_id -> [cdrh3_seq, ...]
    target_cdrh3: dict[str, list[str]] = {pid: [] for pid in df["pdb_id"]}
    n_heavy = 0
    n_unclassified = 0
    for i, (seq_id, seq) in enumerate(all_seqs):
        hits = r2[i] if r2[i] else []
        if not hits:
            n_unclassified += 1
            continue
        for j, hdet in enumerate(hits):
            if hdet.get("chain_type") != "H":
                continue
            numbered = r1[i][j][0] if r1[i] and len(r1[i]) > j else None
            if not numbered:
                continue
            cdr = cdrh3_from_numbering(numbered)
            if cdr and len(cdr) >= 3:
                pid = seq_id.split("_")[0]
                target_cdrh3[pid].append(cdr)
                n_heavy += 1
    print(f"  {n_heavy} heavy chains with CDR-H3; {n_unclassified} unclassified chains")

    # ---- write FASTA of all unique CDR-H3 ----
    fasta = work / "cdrh3_all.fasta"
    seen: dict[str, str] = {}  # seq -> first id
    with open(fasta, "w") as f:
        for pid, cdrs in target_cdrh3.items():
            for j, cdr in enumerate(cdrs):
                if cdr not in seen:
                    seen[cdr] = f"{pid}_h{j}"
                    f.write(f">{seen[cdr]}\n{cdr}\n")
    print(f"  {len(seen)} unique CDR-H3 sequences -> {fasta}")

    # ---- MMseqs2 cluster at 80% identity ----
    db = work / "cdrh3_db"
    db_cluster = work / "cdrh3_clu"
    cov = "0.8"
    mode = "easy-cluster"  # produces representative + cluster tsv
    cmd = [
        "mmseqs", "easy-cluster", str(fasta), str(db_cluster), str(work),
        "--min-seq-id", args.cluster_identity,
        "-c", cov,
        "--cov-mode", "1",
    ]
    print(f"MMseqs2: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[error] mmseqs failed:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    tsv = Path(f"{db_cluster}_cluster.tsv")
    if not tsv.exists():
        print(f"[error] no cluster tsv at {tsv}", file=sys.stderr)
        sys.exit(1)
    # tsv: representative_id \t member_id
    seq_to_cluster: dict[str, str] = {}
    with open(tsv) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            seq_to_cluster[member] = rep
    print(f"  {len(set(seq_to_cluster.values()))} CDR-H3 clusters from {len(seq_to_cluster)} members")

    # ---- map each target's CDR-H3 to cluster ----
    # A target may have multiple heavy chains; record all cluster reps and a
    # canonical cluster id (the first rep, sorted for determinism).
    target_cluster: dict[str, list[str]] = {}
    for pid, cdrs in target_cdrh3.items():
        clusters = []
        for cdr in cdrs:
            cid = seen.get(cdr)
            if cid and cid in seq_to_cluster:
                clusters.append(seq_to_cluster[cid])
        target_cluster[pid] = sorted(set(clusters))

    df["cdrh3_sequences"] = df["pdb_id"].map(lambda p: target_cdrh3.get(p, []))
    df["cdrh3_cluster"] = df["pdb_id"].map(
        lambda p: (target_cluster[p][0] if target_cluster.get(p) else None)
    )
    df["cdrh3_clusters"] = df["pdb_id"].map(lambda p: target_cluster.get(p, []))

    df.to_parquet(out_manifest, index=False)
    print(f"wrote {out_manifest} ({len(df)} rows)")

    # ---- report ----
    n_with_cdr = df["cdrh3_sequences"].map(len).gt(0).sum()
    n_clustered = df["cdrh3_cluster"].notna().sum()
    n_no_cdr = df["cdrh3_sequences"].map(len).eq(0).sum()
    print(f"\ntargets with >=1 CDR-H3: {n_with_cdr}/{len(df)}")
    print(f"targets with cdrh3_cluster: {n_clustered}/{len(df)}")
    print(f"targets with no CDR-H3 (no heavy chain found): {n_no_cdr}")
    if n_no_cdr:
        no_cdr = df[df["cdrh3_sequences"].map(len).eq(0)]["pdb_id"].tolist()
        print(f"  {no_cdr}")
    n_unique_clusters = df["cdrh3_cluster"].nunique()
    print(f"unique cdrh3_cluster: {n_unique_clusters}")


if __name__ == "__main__":
    main()
