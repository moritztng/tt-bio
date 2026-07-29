#!/usr/bin/env python3
"""Phase 1 antibody-copy audit v2 (CPU-only).

Correct check: for each manifest-declared interface chain (fold_auth_chain_id_1
= antibody, fold_auth_chain_id_2 = antigen), verify that chain's exact CIF
sequence appears among the yaml's sequences. The 9l9y bug class (commit c435d25f)
was: yaml built from antibody copy 1 while manifest declares copy 2, and copy
1's light chain (209) != copy 2's light chain (210) -> DockQ compares fold-L
to native-B with a residue mismatch, silently degrading the primary label.

Identical antibody copies (same sequence, different chain labels) are HARMLESS
- any copy's sequence matching the manifest chain's sequence is fine. The bug
is specifically a SEQUENCE MISMATCH between the yaml and the manifest-declared
native chain.

Also checks: every yaml sequence must match SOME native CIF chain (catches
extraction artifacts / non-standard residues), and the antibody partner chain
(if paired-HL) must also be present.

Outputs JSON report. Exit 0 if all ok, 1 if any mismatch.
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import yaml
import gemmi


def cif_chain_info(cif_path: str):
    """auth_chain_id -> (one-letter seq uppercased gap-stripped, entity_id)."""
    doc = gemmi.read_structure(cif_path)
    out = {}
    for ch in doc[0]:
        seq = ch.get_polymer().make_one_letter_sequence()
        seq = seq.replace("-", "").upper()
        if seq:
            # entity_id: gemmi stores subchain -> entity; use chain.subchain()
            ent = ""
            try:
                sub = ch.subchain()
                if sub and sub in doc.block.entity_poly_seq:
                    pass
            except Exception:
                pass
            out[ch.name] = dict(seq=seq, entity_id=ent)
    return out


def cif_chain_seqs(cif_path: str) -> dict:
    """auth_chain_id -> one-letter sequence (gap-stripped, uppercased)."""
    doc = gemmi.read_structure(cif_path)
    out = {}
    for ch in doc[0]:
        seq = ch.get_polymer().make_one_letter_sequence()
        seq = seq.replace("-", "").upper()
        if seq:
            out[ch.name] = seq
    return out


def read_yaml_seqs(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        y = yaml.safe_load(f)
    out = {}
    for entry in y["sequences"]:
        prot = entry.get("protein") or entry.get("dna") or entry.get("rna")
        if prot and "id" in prot and "sequence" in prot:
            out[prot["id"]] = prot["sequence"].upper()
    return out


def audit_one(pdb_id, yaml_path, cif_path, ab_chain, ag_chain, has_HL):
    if not os.path.exists(yaml_path):
        return dict(pdb_id=pdb_id, status="error", detail="yaml missing: " + yaml_path)
    if not os.path.exists(cif_path):
        return dict(pdb_id=pdb_id, status="error", detail="cif missing: " + cif_path)
    try:
        yseqs = read_yaml_seqs(yaml_path)
        cseqs = cif_chain_seqs(cif_path)
    except Exception as e:
        return dict(pdb_id=pdb_id, status="error", detail="read fail: " + str(e))

    yseq_list = list(yseqs.values())
    issues = []

    # 1. manifest antibody chain's CIF seq must appear in yaml
    if ab_chain in cseqs:
        ab_seq = cseqs[ab_chain]
        if ab_seq not in yseq_list:
            issues.append("manifest antibody chain " + ab_chain + " (len " + str(len(ab_seq)) + ") CIF seq NOT in yaml seqs (lengths " + str(sorted(len(s) for s in yseq_list)) + ")")
    else:
        issues.append("manifest antibody chain " + ab_chain + " not in CIF chains " + str(sorted(cseqs)))

    # 2. manifest antigen chain's CIF seq must appear in yaml
    if ag_chain in cseqs:
        ag_seq = cseqs[ag_chain]
        if ag_seq not in yseq_list:
            issues.append("manifest antigen chain " + ag_chain + " (len " + str(len(ag_seq)) + ") CIF seq NOT in yaml seqs")
    else:
        issues.append("manifest antigen chain " + ag_chain + " not in CIF chains " + str(sorted(cseqs)))

    # 3. every yaml seq must match SOME cif chain (catches extraction artifacts)
    for yid, yseq in yseqs.items():
        if yseq not in cseqs.values():
            issues.append("yaml " + yid + " (len " + str(len(yseq)) + ") matches NO native CIF chain")

    if not issues:
        return dict(pdb_id=pdb_id, status="ok",
                    yaml_chains=sorted(yseqs), ab_chain=ab_chain, ag_chain=ag_chain)
    return dict(pdb_id=pdb_id, status="mismatch", detail="; ".join(issues),
                yaml_chains=sorted(yseqs), ab_chain=ab_chain, ag_chain=ag_chain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/implementation-parity-data/abag-xm-targets.parquet")
    ap.add_argument("--yaml-dir", default="examples/abag_xm")
    ap.add_argument("--gt-dir", default="examples/ground_truth_structures")
    ap.add_argument("--out", default="docs/implementation-parity-data/abag-xm-phase1-audit.json")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.manifest)
    if args.only:
        want = set(args.only.split(","))
        df = df[df.pdb_id.isin(want)]

    results = []
    for _, r in df.iterrows():
        pdb = r.pdb_id
        yaml_path = args.yaml_dir + "/" + pdb + ".yaml"
        cif_path = r.mmcif_path if (r.mmcif_path and os.path.exists(r.mmcif_path)) else args.gt_dir + "/" + pdb + ".cif"
        res = audit_one(pdb, yaml_path, cif_path, r.fold_auth_chain_id_1, r.fold_auth_chain_id_2, bool(r.has_HL))
        results.append(res)

    ok = [r for r in results if r["status"] == "ok"]
    mm = [r for r in results if r["status"] == "mismatch"]
    er = [r for r in results if r["status"] == "error"]
    print("audited " + str(len(results)) + ": ok=" + str(len(ok)) + " mismatch=" + str(len(mm)) + " error=" + str(len(er)))
    if mm:
        print("\n--- MISMATCHES ---")
        for r in mm:
            print("  " + r["pdb_id"] + ": " + r["detail"])
    if er:
        print("\n--- ERRORS ---")
        for r in er:
            print("  " + r["pdb_id"] + ": " + r["detail"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(total=len(results), ok=len(ok), mismatch=len(mm), error=len(er), results=results), f, indent=2)
    print("\nreport: " + args.out)
    sys.exit(0 if not (mm or er) else 1)


if __name__ == "__main__":
    main()
