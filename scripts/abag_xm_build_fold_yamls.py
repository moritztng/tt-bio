#!/usr/bin/env python3
"""AbAg-XM Phase 2: build the per-target fold YAMLs (D11) and prefetch MSAs offline.

For each target: extract chains from the local mmCIF, run ANARCI to classify
each chain as heavy/light/antigen, then build a YAML with the minimal Ab-Ag
unit (antigen + H, or antigen + H + L for paired-Fv). Sequences are taken from
the mmCIF (gaps stripped, unknowns -> X).

Then collects every unique protein sequence across all fold YAMLs, hashes each
(seq_hash = sha256[:16], matching tt_bio's cache key), and prefetches the
missing {seq_hash}.a3m files via compute_msa_offline against the local ColabFold
DB. This is CPU/disk only — no device needed — and runs concurrently with any
device work.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import gemmi
import pandas as pd
from anarci import run_anarci


def extract_chains(cif_path: Path) -> list[tuple[str, str]]:
    if not cif_path.exists():
        return []
    st = gemmi.read_structure(str(cif_path))
    out = []
    for model in st:
        for ch in model:
            polymer = ch.get_polymer()
            seq = str(polymer.make_one_letter_sequence())
            seq = seq.replace("-", "")
            seq = "".join(c if c in "ACDEFGHIKLMNPQRSTVWY" else "X" for c in seq)
            if len(seq) >= 10:
                out.append((ch.name, seq))
        break
    return out


def classify_chains(chains: list[tuple[str, str]]) -> dict[str, str]:
    """Return {chain_id: 'H'|'L'|'antigen'} via ANARCI.

    Uses bit_score_threshold=40 (below the default 80) so C-terminally truncated
    VHHs (e.g. 9lwc's 91-residue nanobody, bit score ~44) are still classified.
    We only use the classification to pick the antibody chain, not for precise
    numbering, so the lower threshold is safe here.
    """
    if not chains:
        return {}
    seqs = [(cid, seq) for cid, seq in chains]
    _r0, r1, r2, _r3 = run_anarci(seqs, scheme="imgt", ncpu=4,
                                  bit_score_threshold=40)
    cls: dict[str, str] = {}
    for i, (cid, _seq) in enumerate(seqs):
        hits = r2[i] if r2[i] else []
        if not hits:
            cls[cid] = "antigen"
            continue
        ct = hits[0].get("chain_type")
        if ct == "H":
            cls[cid] = "H"
        elif ct in ("L", "K"):
            cls[cid] = "L"
        else:
            cls[cid] = "antigen"
    return cls


def build_yaml(target_id: str, chains: list[tuple[str, str]],
               cls: dict[str, str], fold_ab_chain: str, fold_ag_chain: str,
               has_HL: bool) -> str:
    """Build the fold YAML. Chain ids in YAML: A=antigen, H=heavy, L=light.

    fold_ab_chain is the antibody chain of the fold interface (H or L); we find
    its partner (L if fold_ab is H, H if fold_ab is L) among the other ANARCI-
    classified antibody chains in the structure.
    """
    seq_map = dict(chains)
    lines = ["version: 1", f"# AbAg-XM fold target {target_id} (ARK interface)."]
    lines.append("sequences:")
    # antigen
    ag_seq = seq_map.get(fold_ag_chain, "")
    lines.append("  - protein:")
    lines.append("      id: A")
    lines.append(f"      sequence: {ag_seq}")
    # heavy: if fold_ab is H, use it; else find an H chain partner
    h_chain = fold_ab_chain if cls.get(fold_ab_chain) == "H" else None
    if h_chain is None:
        for cid, c in cls.items():
            if c == "H" and cid != fold_ag_chain:
                h_chain = cid
                break
    if h_chain and h_chain in seq_map:
        lines.append("  - protein:")
        lines.append("      id: H")
        lines.append(f"      sequence: {seq_map[h_chain]}")
    # light: if HL, find an L chain partner (not the antigen, not the heavy)
    if has_HL:
        l_chain = None
        for cid, c in cls.items():
            if c == "L" and cid != fold_ag_chain and cid != h_chain:
                l_chain = cid
                break
        if l_chain and l_chain in seq_map:
            lines.append("  - protein:")
            lines.append("      id: L")
            lines.append(f"      sequence: {seq_map[l_chain]}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/implementation-parity-data/abag-xm-targets.parquet")
    ap.add_argument("--cif_dir", default="examples/ground_truth_structures")
    ap.add_argument("--yaml_dir", default="examples/abag_xm")
    ap.add_argument("--msa_dir", default="/home/ttuser/abag_xm/msa_cache")
    ap.add_argument("--msa_db_path", default="/home/ttuser/.boltz/msa_db")
    ap.add_argument("--skip_msa", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=40,
                    help="sequences per colabfold_search call")
    args = ap.parse_args()

    yaml_dir = Path(args.yaml_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = Path(args.msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir)

    df = pd.read_parquet(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    # ---- build YAMLs + collect unique sequences ----
    all_seqs: dict[str, str] = {}  # seq_hash -> seq
    n_yaml = 0
    print(f"building fold YAMLs for {len(df)} targets...")
    for _, row in df.iterrows():
        pid = row["pdb_id"]
        cif = Path(row["mmcif_path"])
        chains = extract_chains(cif)
        if not chains:
            print(f"  [warn] {pid}: no chains in {cif}")
            continue
        cls = classify_chains(chains)
        fold_ab = row["fold_auth_chain_id_1"]
        fold_ag = row["fold_auth_chain_id_2"]
        # Determine which fold interface chain is antibody vs antigen.
        # If fold_ab is not H/L but fold_ag is, swap (ARK entity order is not fixed).
        if cls.get(fold_ab) not in ("H", "L") and cls.get(fold_ag) in ("H", "L"):
            fold_ab, fold_ag = fold_ag, fold_ab
        # If neither is H/L (e.g. very truncated VHH below even threshold 40),
        # fall back: pick any H chain in the structure as the antibody.
        if cls.get(fold_ab) not in ("H", "L"):
            for cid, c in cls.items():
                if c == "H":
                    fold_ab = cid
                    break
        has_HL = bool(row["has_HL"])
        yaml_text = build_yaml(pid, chains, cls, fold_ab, fold_ag, has_HL)
        ypath = yaml_dir / f"{pid}.yaml"
        ypath.write_text(yaml_text)
        n_yaml += 1
        # collect sequences (A, H, L from the YAML)
        for line in yaml_text.splitlines():
            ls = line.strip()
            if ls.startswith("sequence: "):
                seq = ls[len("sequence: "):]
                if len(seq) >= 10:
                    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
                    all_seqs[h] = seq
    print(f"  wrote {n_yaml} YAMLs to {yaml_dir}")
    print(f"  {len(all_seqs)} unique protein sequences across all targets")

    # ---- prefetch MSAs ----
    if args.skip_msa:
        print("--skip_msa: skipping MSA prefetch")
        return
    missing = {h: s for h, s in all_seqs.items()
              if not (msa_dir / f"{h}.a3m").exists() and not (msa_dir / f"{h}.csv").exists()}
    print(f"MSA cache: {len(all_seqs)} unique seqs, {len(missing)} missing")
    if not missing:
        print("  all MSAs already cached — nothing to prefetch")
        return
    # import the offline MSA generator from tt_bio
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # Prepend localcolabfold's bin to PATH so colabfold_search picks up its own
    # mmseqs (version 18, supports --prefilter-mode) instead of the apt-installed
    # system mmseqs (version 13, which lacks that flag and breaks the search).
    lc_bin = str(Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin")
    if Path(lc_bin).exists():
        os.environ["PATH"] = lc_bin + os.pathsep + os.environ.get("PATH", "")
    from tt_bio.main import compute_msa_offline  # type: ignore
    items = sorted(missing.items())
    total = len(items)
    done = 0
    for i in range(0, total, args.batch_size):
        batch = items[i : i + args.batch_size]
        seqs = {h: s for h, s in batch}
        print(f"  prefetch {done + len(batch)}/{total}: {list(seqs.keys())[:3]}...")
        compute_msa_offline(seqs, f"prefetch_{i}", msa_dir, args.msa_db_path,
                             use_env=False, pair=False)
        done += len(batch)
    # verify
    still_missing = [h for h in all_seqs if not (msa_dir / f"{h}.a3m").exists()
                     and not (msa_dir / f"{h}.csv").exists()]
    print(f"  after prefetch: {len(still_missing)} still missing")
    if still_missing:
        print(f"  [warn] missing hashes: {still_missing[:10]}")
    cached = sum(1 for h in all_seqs if (msa_dir / f"{h}.a3m").exists())
    print(f"  cached: {cached}/{len(all_seqs)} ({100*cached/len(all_seqs):.1f}%)")


if __name__ == "__main__":
    main()
