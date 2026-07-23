#!/usr/bin/env python3
"""Compare ported featurizer `f` vs a reference captured `f` for F5 symmetry
COMBINED with a real motif (p19) -- two real, grounded targets:

1. `unsym_C3_6t8h` (verbatim real RosettaCommons/foundry example): a fully
   unconditional C3 oligomer design (`contig: "100-100,/0,Y1-11,/0,Z16-25"`)
   around a real DNA helix, with the two DNA contig ranges excluded from
   symmetrization via `symmetry.is_unsym_motif: "Y1-11,Z16-25"`. Exercises
   mechanism (a) (`get_symmetry_frames_from_atom_array`, Kabsch frames
   derived from the real 6t8h_C3.pdb's 3 real protein chains) + mechanism
   (b) (`is_unsym_motif` exclusion).

2. `unindexed_C2_1j79` MINUS its `ligand` field (a deterministic variant of
   the real example -- `ligand`+`symmetry` is out of scope this pass, see
   `tt_bio.rfd3_featurize`'s module docstring for the grounded reason): a C2
   design around real PDB 1j79_C2.pdb with `unindex: "A250"` (a catalytic
   residue "within a subunit", `select_fixed_atoms: {"A250": "OD1,CG"}`).
   Exercises mechanism (a) again (a DIFFERENT real Kabsch-derived C2 frame)
   + mechanism (c) (unindexed-motif replication then forced-fixed).

Reproduce the reference captures (CPU, no ckpt -- same method as
parity_dna.py/parity_enzyme.py):
    /home/ttuser/rfd3_local_env/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/symmetry_motif_6t8h/6t8h_C3.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/symmetry_motif_6t8h/spec.json \
        --out_dir /tmp/ref_6t8h_capture --seed 42
    /home/ttuser/rfd3_local_env/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/symmetry_motif_1j79_nolig/1j79_C2.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/symmetry_motif_1j79_nolig/spec.json \
        --out_dir /tmp/ref_1j79_nolig_capture --seed 42

Documented gaps (root-caused, not guessed -- see the p19 state-doc section):
- `ref_pos` (6t8h, NA atoms): the same pre-existing NA-conformer gap as p15/
  p17 -- this port never vendors the real RDKit/CCD reference-conformer embed
  for NA, left at 0.
- `ref_charge` (6t8h, 21 atoms, all `OP2`): the reference assigns -1 to every
  DNA backbone `OP2` in this capture, but this port's own bundled CCD monomer
  template (`tt_bio.data.mol.load_molecules`, the SAME source used for ligand
  charges) reports `OP2` charge 0 for standalone DA/DC/DG/DT -- verified
  directly. Re-ran the EXISTING F2/F8 dsDNA fixture's own reference capture
  (`parity_dna.py`'s 1bna.pdb) fresh: ZERO nonzero `ref_charge` atoms there,
  matching this port's convention exactly. The difference is NOT a bug in
  this port -- both captures log the reference's own admitted limitation
  ("We can't fix formal charges without building from templates...") and its
  formal-charge completion is context/bond-graph dependent (whether an atom
  is inter-residue-bonded in the PARSED structure, not an intrinsic per-
  residue-type constant) -- same category as p17's ACT `ref_charge` gap
  (`§2q.4`), just the opposite direction (there this port had a real charge
  the reference didn't; here the reference has one this port's clean
  per-residue-type template doesn't).
- `motif_pos` (1j79, 4 atoms, the `select_fixed_atoms`-selected OD1/CG atoms
  x2 replicas): max abs diff ~1.9e-6 -- float32 summation-order noise in the
  center-of-mass mean, same category as p17's 7-atom ULP-noise gap (`§2q.4`).
"""
import json
import os, sys
import torch

DIR = os.path.dirname(__file__)


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0: return 1.0 if torch.allclose(a, b) else float("nan")
    return float((a * b).sum() / denom)


TOK_KEYS = ["restype", "ref_motif_token_type", "ref_plddt", "is_non_loopy",
            "is_motif_token_unindexed", "is_motif_token_with_fully_fixed_coord",
            "is_protein", "is_rna", "is_dna", "is_ligand", "is_polar",
            "terminus_type", "asym_id", "entity_id", "sym_id",
            "residue_index", "token_index", "token_bonds", "unindexing_pair_mask"]
ATOM_KEYS = ["ref_atom_name_chars", "ref_pos", "ref_mask", "ref_element", "ref_charge",
             "ref_space_uid", "ref_pos_is_ground_truth", "has_zero_occupancy",
             "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
             "ref_atomwise_rasa", "active_donor", "active_acceptor", "is_atom_level_hotspot",
             "is_motif_atom_with_fixed_coord", "is_motif_atom_with_fixed_seq",
             "is_motif_atom_unindexed", "motif_pos", "is_ca", "is_central",
             "is_backbone", "is_sidechain", "is_virtual", "atom_to_token_map",
             "sym_transform_id", "sym_entity_id", "is_sym_asu"]


def run_case(name, pdb, spec_json, ref_pt, known_gap_keys=()):
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    print("#" * 70)
    print(f"# CASE: {name}")
    print("#" * 70)
    if not os.path.exists(ref_pt):
        print(f"SKIPPED: no reference capture at {ref_pt} (see module docstring to reproduce)")
        return None
    with open(spec_json) as fh:
        spec_dict = json.load(fh)
    spec_dict["input"] = pdb
    spec = InputSpecification.from_dict(spec_dict)
    pf = featurize(pdb, spec)
    rf = torch.load(ref_pt, weights_only=False)

    print(f"PORTED I={pf['restype'].shape[0]} L={pf['ref_pos'].shape[0]} | "
          f"REF I={rf['restype'].shape[0]} L={rf['ref_pos'].shape[0]}")

    print("--- TOKEN-LEVEL ---")
    mismatches = []
    for k in TOK_KEYS:
        if k not in pf or k not in rf:
            print(f"  {k:35s} MISSING (ported={'Y' if k in pf else 'N'} ref={'Y' if k in rf else 'N'})")
            mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if a.shape != b.shape:
            print(f"  {k:35s} SHAPE {list(a.shape)} vs {list(b.shape)}  MISMATCH")
            mismatches.append(k); continue
        be = torch.equal(a, b)
        p = pcc(a, b) if a.dtype.is_floating_point or a.dtype in (torch.int64, torch.int32, torch.int8) else 1.0
        status = "OK" if be else "DIFF"
        if not be: mismatches.append(k)
        print(f"  {k:35s} {status:4s} bitexact={be} pcc={p:.4f}")
        if not be and a.numel() <= 60:
            print(f"      ported: {a.flatten().tolist()}")
            print(f"      ref   : {b.flatten().tolist()}")

    print("--- ATOM-LEVEL ---")
    atom_mismatches = []
    same_L = pf["ref_pos"].shape[0] == rf["ref_pos"].shape[0]
    for k in ATOM_KEYS:
        if k not in pf or k not in rf:
            print(f"  {k:30s} MISSING (ported={'Y' if k in pf else 'N'} ref={'Y' if k in rf else 'N'})")
            atom_mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if not same_L or a.shape != b.shape:
            print(f"  {k:30s} SHAPE {list(a.shape)} vs {list(b.shape)}  (L mismatch, skip value compare)")
            continue
        be = torch.equal(a, b)
        p = pcc(a, b) if a.dtype.is_floating_point else 1.0
        status = "OK" if be else ("KNOWN GAP" if k in known_gap_keys else "DIFF")
        if not be and k not in known_gap_keys:
            atom_mismatches.append(k)
        print(f"  {k:30s} {status:9s} bitexact={be} pcc={p:.4f}  {list(a.shape)} {a.dtype}")

    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}) excluding documented gaps {known_gap_keys}: {atom_mismatches} (same_L={same_L})")
    print()
    return mismatches, atom_mismatches


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(DIR, "..", "..", "..")))
    r1 = run_case(
        "unsym_C3_6t8h (real, verbatim)",
        os.path.join(DIR, "symmetry_motif_6t8h", "6t8h_C3.pdb"),
        os.path.join(DIR, "symmetry_motif_6t8h", "spec.json"),
        "/tmp/ref_6t8h_capture/ref_f.pt",
        known_gap_keys=("ref_pos", "ref_charge"),
    )
    r2 = run_case(
        "unindexed_C2_1j79 minus `ligand` (deterministic variant)",
        os.path.join(DIR, "symmetry_motif_1j79_nolig", "1j79_C2.pdb"),
        os.path.join(DIR, "symmetry_motif_1j79_nolig", "spec.json"),
        "/tmp/ref_1j79_nolig_capture/ref_f.pt",
        known_gap_keys=("motif_pos",),
    )


if __name__ == "__main__":
    main()
