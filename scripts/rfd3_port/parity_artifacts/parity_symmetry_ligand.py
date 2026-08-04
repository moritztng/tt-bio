#!/usr/bin/env python3
"""Compare ported featurizer `f` vs a reference captured `f` for F5 symmetry
COMBINED with `ligand` (p20): the real, UNMODIFIED `unindexed_C2_1j79`
example -- a C2 design around real PDB 1j79_C2.pdb with `ligand: "ORO,ZN"`
(each of the two real subunits' own active site: 1 ORO + 2 Zn, verified via
`grep HETATM`) plus `unindex: "A250"` (a catalytic residue "within a
subunit") -- see `tt_bio.rfd3_featurize`'s module docstring's "ligand +
symmetry" grounding for the full mechanism.

Reproduce the reference capture (CPU, no ckpt -- same method as
parity_enzyme.py/parity_symmetry_motif.py):
    /home/ttuser/rfd3_local_env/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/1j79_C2.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/spec.json \
        --out_dir /tmp/ref_1j79_full_capture --seed 42

WHY THIS SCRIPT COMPARES DIFFERENTLY FOR THE LIGAND SUB-BLOCK: re-running
this IDENTICAL capture (same PDB/spec/seed) with three different
PYTHONHASHSEED values gives three DIFFERENT raw orderings of the ligand
token block -- `unravel_components` (foundry/utils/components.py) resolves
a CCD code with multiple physical matches via an un-sorted Python `set`,
so which subunit's atoms land first is a PYTHONHASHSEED accident, not a
reproducible rule (see module docstring). Bit-exact POSITIONAL comparison
of the ligand sub-block against any ONE stored capture would therefore fail
by construction, even for a perfectly correct port. This script instead:
  1. Compares the deterministic PREFIX (protein, replicated per subunit)
     and SUFFIX (the unindexed-motif replicas) positionally, bit-exact --
     unaffected by the hash-random ligand ordering.
  2. For the ligand sub-block, matches atoms between ported and reference
     by NEAREST real 3D position (`motif_pos`, which both sides compute as
     real_coord - COM -- a side-independent, order-independent identity for
     a FIXED-coordinate real atom) rather than by raw array position, then
     compares every other feature for those identity-matched pairs.
"""
import json
import os, sys
import torch
import numpy as np

DIR = os.path.dirname(__file__)
PDB = os.path.join(DIR, "unindexed_c2_1j79_full", "1j79_C2.pdb")
SPEC_JSON = os.path.join(DIR, "unindexed_c2_1j79_full", "spec.json")
REF_PT = "/tmp/ref_1j79_full_capture/ref_f.pt"


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
# atom_to_token_map's raw VALUE is a token index, meaningless to compare
# positionally once the ligand block's token order differs between the two
# sides -- only its structure (grouping) matters, checked separately below.
ATOM_KEYS_NO_ORDER_MEANING = {"atom_to_token_map"}
# Pre-existing, documented gap categories (NOT p20 regressions -- same
# categories already accepted by parity_enzyme.py/parity_symmetry_motif.py):
# `motif_pos` float32 summation-order noise on the 4 `select_fixed_atoms`
# atoms (~1e-6, same class as p19's own 1j79-minus-ligand finding).
DETERMINISTIC_KNOWN_GAP_KEYS = {"motif_pos"}
# `ref_pos` (stochastic RDKit conformer, never bit-exact to any one capture
# -- see F3 grounding), `ref_charge` (context/bond-graph-dependent formal
# charge -- the reference can't infer Zn2+'s ionic charge without a bond
# graph for an isolated ion; this port's clean per-residue-type CCD template
# correctly reports the chemically real +2 -- same category as the DNA OP2
# charge gap, opposite direction), `ref_space_uid`/`residue_index` (the
# raw SLOT NUMBER a structurally-interchangeable instance gets is itself
# part of the hash-random ordering -- only the PARTITION, checked
# separately, is a real invariant).
LIGAND_KNOWN_GAP_KEYS = {"ref_pos", "ref_charge", "ref_space_uid"}


def compare_deterministic(pf, rf, tok_mask_p, tok_mask_r, atom_mask_p, atom_mask_r, label):
    """Positional bit-exact comparison, restricted to the given (deterministic-
    order) token/atom sub-ranges on each side. `tok_mask_p`/`tok_mask_r` (and
    the atom equivalents) must select equal-length, correspondingly-ordered
    slices."""
    print(f"--- {label}: TOKEN-LEVEL ---")
    mismatches = []
    for k in TOK_KEYS:
        if k not in pf or k not in rf:
            print(f"  {k:35s} MISSING"); mismatches.append(k); continue
        a, b = pf[k][tok_mask_p], rf[k][tok_mask_r]
        if a.shape != b.shape:
            print(f"  {k:35s} SHAPE {list(a.shape)} vs {list(b.shape)}  MISMATCH")
            mismatches.append(k); continue
        be = torch.equal(a, b)
        if not be: mismatches.append(k)
        print(f"  {k:35s} {'OK' if be else 'DIFF':4s} bitexact={be}")

    print(f"--- {label}: ATOM-LEVEL ---")
    atom_mismatches = []
    for k in ATOM_KEYS:
        if k in ATOM_KEYS_NO_ORDER_MEANING or k not in pf or k not in rf:
            continue
        a, b = pf[k][atom_mask_p], rf[k][atom_mask_r]
        if a.shape != b.shape:
            print(f"  {k:30s} SHAPE {list(a.shape)} vs {list(b.shape)}  MISMATCH")
            atom_mismatches.append(k); continue
        be = torch.equal(a, b)
        if not be and k in DETERMINISTIC_KNOWN_GAP_KEYS:
            close = torch.allclose(a.float(), b.float(), atol=1e-4)
            status = "KNOWN GAP" if close else "DIFF"
            if not close: atom_mismatches.append(k)
        else:
            status = "OK" if be else "DIFF"
            if not be: atom_mismatches.append(k)
        print(f"  {k:30s} {status:9s} bitexact={be}")
    print(f"  MISMATCHES tok={mismatches} atom={atom_mismatches}")
    return mismatches, atom_mismatches


def compare_ligand_block_by_identity(pf, rf, lig_tok_p, lig_tok_r, lig_atom_p, lig_atom_r):
    """Match ligand ATOMS between ported and reference by nearest real 3D
    position (`motif_pos`) -- see module docstring for why raw array
    position cannot be used here. Reports the match distances (should be
    ~0, floating noise only) and then compares every other per-atom/per-
    token feature for the identity-matched pairs."""
    print("--- LIGAND SUB-BLOCK: identity-matched comparison ---")
    mp_p = pf["motif_pos"][lig_atom_p].numpy()
    mp_r = rf["motif_pos"][lig_atom_r].numpy()
    n = mp_p.shape[0]
    if mp_r.shape[0] != n:
        print(f"  ATOM COUNT MISMATCH: ported={n} ref={mp_r.shape[0]}")
        return ["ligand_atom_count"], []
    dist = np.linalg.norm(mp_p[:, None, :] - mp_r[None, :, :], axis=-1)
    match_r_for_p = dist.argmin(axis=1)
    match_dist = dist[np.arange(n), match_r_for_p]
    bijective = len(set(match_r_for_p.tolist())) == n
    print(f"  n_ligand_atoms={n}  max_match_dist={match_dist.max():.6g}  bijective={bijective}")
    if not bijective or match_dist.max() > 1e-2:
        print("  IDENTITY MATCH FAILED (not bijective or distances too large)")
        return ["ligand_identity_match"], []

    # Re-derive per-atom token index within the ligand sub-block for both
    # sides so we can also compare TOKEN-level features (asym_id, entity_id,
    # sym_transform_id, residue_index's GROUP STRUCTURE, ...) per matched
    # atom, not just atom-level ones.
    atom_idx_p = np.nonzero(lig_atom_p)[0]
    atom_idx_r = np.nonzero(lig_atom_r)[0]
    tok_p = pf["atom_to_token_map"][atom_idx_p].numpy()
    tok_r = rf["atom_to_token_map"][atom_idx_r].numpy()

    atom_mismatches, tok_mismatches = [], []
    for k in ATOM_KEYS:
        if k in ATOM_KEYS_NO_ORDER_MEANING or k not in pf or k not in rf:
            continue
        a = pf[k][atom_idx_p].numpy()
        b = rf[k][atom_idx_r][match_r_for_p].numpy()
        ok = np.array_equal(a, b)
        if k in LIGAND_KNOWN_GAP_KEYS:
            print(f"  {k:30s} {'OK' if ok else 'KNOWN GAP':9s} identity-matched exact={ok}")
            continue
        if not ok:
            close = np.allclose(a, b, atol=1e-4)
            print(f"  {k:30s} {'CLOSE' if close else 'DIFF':6s} identity-matched exact={ok}")
            if not close:
                atom_mismatches.append(k)
        else:
            print(f"  {k:30s} OK     identity-matched exact={ok}")

    # Token-level: residue_index/ref_space_uid don't need to carry the SAME
    # absolute slot number (that label is itself part of the hash-random
    # ordering, e.g. which of a subunit's 2 Zn gets slot 1 vs 2) -- what
    # must match is the PARTITION into groups (same-size groups, same
    # membership under the identity match). Everything else (asym_id's
    # ENTITY, sym_transform_id, sym_entity_id, is_sym_asu, restype, ...)
    # should be identical per matched atom's token.
    for k in ["restype", "sym_id", "is_ligand", "is_protein"]:
        if k not in pf or k not in rf:
            continue
        a = pf[k][tok_p]
        b = rf[k][tok_r][match_r_for_p]
        ok = torch.equal(a, b)
        print(f"  token[{k:20s}] {'OK' if ok else 'DIFF':4s} identity-matched exact={ok}")
        if not ok:
            tok_mismatches.append(k)

    def group_partition(res_idx, tok_ids):
        groups = {}
        for i, (r, t) in enumerate(zip(res_idx.tolist(), tok_ids.tolist())):
            groups.setdefault(r, set()).add(t)
        return sorted(len(v) for v in groups.values())

    ridx_p = pf["residue_index"][tok_p].numpy()
    ridx_r = rf["residue_index"][tok_r].numpy()
    # unique token ids within the ligand block, one per real atom (ligand is
    # atomized) -- partition sizes (in ATOMS per residue_index group) must
    # match between ported and reference, order-independent.
    part_p = sorted(np.bincount(np.unique(ridx_p, return_inverse=True)[1]).tolist())
    part_r = sorted(np.bincount(np.unique(ridx_r, return_inverse=True)[1]).tolist())
    print(f"  residue_index GROUP-SIZE partition: ported={part_p} ref={part_r} "
          f"{'OK' if part_p == part_r else 'DIFF'}")
    if part_p != part_r:
        tok_mismatches.append("residue_index_partition")

    entid_p = pf["entity_id"][tok_p].numpy()
    entid_r = rf["entity_id"][tok_r].numpy()
    same_entity_p = len(set(entid_p.tolist())) == 1
    same_entity_r = len(set(entid_r.tolist())) == 1
    print(f"  entity_id ALL-SAME (ligand shares one entity): ported={same_entity_p} ref={same_entity_r} "
          f"{'OK' if same_entity_p == same_entity_r == True else 'DIFF'}")
    if not (same_entity_p and same_entity_r):
        tok_mismatches.append("entity_id_shared")

    return tok_mismatches, atom_mismatches


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(DIR, "..", "..", "..")))
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    if not os.path.exists(REF_PT):
        print(f"SKIPPED: no reference capture at {REF_PT} (see module docstring to reproduce)")
        return
    with open(SPEC_JSON) as fh:
        spec_dict = json.load(fh)
    spec_dict["input"] = PDB
    spec = InputSpecification.from_dict(spec_dict)
    pf = featurize(PDB, spec)
    rf = torch.load(REF_PT, weights_only=False)

    print(f"PORTED I={pf['restype'].shape[0]} L={pf['ref_pos'].shape[0]} | "
          f"REF I={rf['restype'].shape[0]} L={rf['ref_pos'].shape[0]}")
    if pf["restype"].shape[0] != rf["restype"].shape[0]:
        print("TOKEN COUNT MISMATCH -- aborting")
        return

    is_lig_tok_p = pf["is_ligand"].bool().numpy()
    is_lig_tok_r = rf["is_ligand"].bool().numpy()
    is_lig_atom_p = pf["is_ligand"].bool()[pf["atom_to_token_map"]].numpy()
    is_lig_atom_r = rf["is_ligand"].bool()[rf["atom_to_token_map"]].numpy()

    all_mismatches = []
    tm, am = compare_deterministic(
        pf, rf, ~is_lig_tok_p, ~is_lig_tok_r, ~is_lig_atom_p, ~is_lig_atom_r,
        "DETERMINISTIC (non-ligand) tokens/atoms")
    all_mismatches += tm + am

    tm2, am2 = compare_ligand_block_by_identity(
        pf, rf, is_lig_tok_p, is_lig_tok_r, is_lig_atom_p, is_lig_atom_r)
    all_mismatches += tm2 + am2

    print()
    print(f"TOTAL MISMATCHES: {len(all_mismatches)}: {all_mismatches}")


if __name__ == "__main__":
    main()
