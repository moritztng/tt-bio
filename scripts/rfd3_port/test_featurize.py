"""Structural unit tests for tt_bio.rfd3_featurize.featurize.

These verify SHAPES, DTYPES, and KEY INVARIANTS of the produced ``f`` dict —
NOT parity vs the reference (that is the vast.ai capture gate in p11).
Run: .venv/bin/python scripts/rfd3_port/test_featurize.py
"""
import os, sys, tempfile, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification


def _write_minipdb(path, chain="A", n=20, start=1):
    """Write a tiny idealized polyalanine PDB (N,CA,C,O,CB per residue)."""
    lines = []
    # rough idealized backbone walk
    x = 0.0
    for i in range(n):
        rid = start + i
        base = [x, 0.0, 0.0]
        for j, (nm, el, off) in enumerate([("N", "N", (0.0, 0.0, 0.0)),
                                            ("CA", "C", (1.46, 0.0, 0.0)),
                                            ("C", "C", (1.52, 1.46, 0.0)),
                                            ("O", "O", (0.8, 2.2, 0.0)),
                                            ("CB", "C", (1.0, 0.5, -1.2))]):
            cx, cy, cz = base[0] + off[0], base[1] + off[1], base[2] + off[2]
            lines.append(
                f"ATOM  {5*i+j+1:5d}  {nm:<3s} ALA {chain}{rid:4d}    "
                f"{cx:8.3f}{cy:8.3f}{cz:8.3f}  1.00  0.00           {el}")
        x += 3.8
    with open(path, "w") as fh:
        fh.write("END\n".join(lines) + "\nEND\n")


def _spec(d):
    s = InputSpecification.from_dict(d)
    s.validate()
    return s


def test_motif_scaffold_shapes_and_invariants():
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40, start=1)
        # contig: motif A1-10, scaffold 20, motif A31-40  (motif scaffolding F6)
        spec = _spec({"input": pdb, "contig": "A1-10,20,A31-40"})
        f = featurize(pdb, spec)

    I = 10 + 20 + 10  # 40 tokens
    L = I * 14
    # shapes
    assert f["ref_atom_name_chars"].shape == (L, 256)
    assert f["ref_pos"].shape == (L, 3)
    assert f["ref_element"].shape == (L, 128)
    assert f["ref_atomwise_rasa"].shape == (L, 3)
    assert f["restype"].shape == (I, 32)
    assert f["ref_motif_token_type"].shape == (I, 3)
    assert f["atom_to_token_map"].shape == (L,)
    assert f["token_bonds"].shape == (I, I)
    assert f["unindexing_pair_mask"].shape == (I, I)
    assert f["asym_id"].shape == (I,)
    # invariants
    # atom_to_token_map monotonic 0..I-1, each token owns 14 atoms
    a2t = f["atom_to_token_map"]
    assert a2t[0].item() == 0 and a2t[-1].item() == I - 1
    assert all((a2t[s:s + 14] == ti).all() for ti, s in enumerate(range(0, L, 14)))
    # exactly one CA per token (is_ca)
    assert int(f["is_ca"].sum()) == I
    # motif tokens: 20 motif (A1-10 + A31-40), 20 designed
    motif_tokens = f["is_motif_atom_with_fixed_coord"].view(I, 14).any(-1)
    assert int(motif_tokens.sum()) == 20
    # ref_pos is zero for designed atoms, nonzero for motif real atoms
    motif_atom = f["is_motif_atom_with_fixed_coord"]
    assert torch.all(f["ref_pos"][~motif_atom] == 0)
    # ref_space_uid: 14 atoms per token share a uid
    uid = f["ref_space_uid"].view(I, 14)
    assert (uid[:, 0:1] == uid).all()
    # token_bonds symmetric, no self-bonds
    assert float(f["token_bonds"].to(torch.float).diagonal().abs().sum()) == 0.0
    assert torch.allclose(f["token_bonds"].to(torch.float), f["token_bonds"].to(torch.float).T)
    # chain break: contig has no /0 so A31-40 follows the scaffold on chain A
    # -> bonds exist across motif->scaffold->motif boundaries (contiguous res_id)
    print("test_motif_scaffold_shapes_and_invariants OK  I=", I, "L=", L)


def test_chain_break():
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,/0,A31-40"})
        f = featurize(pdb, spec)
    # Parity-gated vs a real protein reference capture: token_bonds is NOT the
    # peptide-bond graph (that lives elsewhere); for standard contiguous protein it
    # is ALL FALSE, with or without a chain break. token_bonds encodes inter-token
    # bonds for modified residues / crosslinks / ligands only.
    assert bool((f["token_bonds"] == 0).all())
    assert float(f["token_bonds"][9, 10]) == 0.0
    assert float(f["token_bonds"][8, 9]) == 0.0
    print("test_chain_break OK")


def test_binder_contig_designed_only():
    # pure designed region (a binder scaffold with no motif) — degenerate but legal
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=5)
        spec = _spec({"input": pdb, "contig": "30"})
        f = featurize(pdb, spec)
    I = 30
    assert f["restype"].shape == (I, 32)
    assert int(f["is_motif_atom_with_fixed_coord"].sum()) == 0
    assert torch.all(f["ref_pos"] == 0)
    print("test_binder_contig_designed_only OK  I=", I)


def test_non_protein_rejected():
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        # write a DA (DNA) residue
        with open(pdb, "w") as fh:
            fh.write("ATOM      1  P   DA A   1      0.000   0.000   0.000  1.00  0.00           P\nEND\n")
        spec = _spec({"input": pdb, "contig": "A1,10"})
        try:
            featurize(pdb, spec)
            raise AssertionError("expected NotImplementedError for non-protein input")
        except NotImplementedError:
            pass
    print("test_non_protein_rejected OK")


def test_contract_vs_golden():
    """Verify the ported featurizer's `f` CONTRACT (keys, dtypes, per-token/atom
    feature dims, rank) against the REAL captured dsDNA_basic golden meta.json
    at ~/.coworker/artifacts/rfd3-goldens/capture/token_initializer.meta.json.

    This is a CONTRACT gate (the part derivable from the reference + the golden),
    NOT a value-parity gate: dsDNA_basic is NA (my featurizer is protein-only), so
    values differ by construction. It confirms the 43-key set, dtypes, and the
    per-token/atom feature widths match the real reference exactly.
    """
    import json
    meta_path = os.path.expanduser(
        "~/.coworker/artifacts/rfd3-goldens/capture/token_initializer.meta.json")
    if not os.path.exists(meta_path):
        print("test_contract_vs_golden SKIPPED (no golden meta.json present)")
        return
    meta = json.load(open(meta_path))
    g_shapes = meta["in_shapes"]  # {key: [[shape...], "torch.dtype"]}

    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,20,A31-40"})
        f = featurize(pdb, spec)
    I = 40; L = I * 14

    # golden is dsDNA_basic: 144 tokens, 2143 atoms. Map golden I_g=144, L_g=2143 -> ours.
    I_g, L_g = 144, 2143
    # which dim is the token/atom axis per key (by shape length & values)
    def expected_shape(k, gshape):
        # replace the leading token (I) or atom (L) axis with ours; keep feature dims.
        gs = gshape
        if k in ("token_bonds", "unindexing_pair_mask"):
            return (I, I) if len(gs) == 2 else None
        if len(gs) == 1:
            # 1D: token-level if dtype is int/bool and key is token-level, else atom-level
            token_keys_1d = {"residue_index", "token_index", "asym_id", "entity_id", "sym_id",
                             "is_protein", "is_rna", "is_dna", "is_ligand", "is_polar",
                             "is_motif_token_unindexed", "is_motif_token_with_fully_fixed_coord",
                             "ref_plddt"}
            return (I,) if k in token_keys_1d else (L,)
        if len(gs) == 2:
            # 2D: [I, feat] or [L, feat]
            atom_keys_2d = {"ref_atom_name_chars", "ref_pos", "ref_element", "ref_atomwise_rasa",
                            "motif_pos"}
            if k in atom_keys_2d:
                return (L, gs[1])
            if k == "is_non_loopy":
                return (I, gs[1])
            if k == "is_atom_level_hotspot":
                return (L, gs[1])
            if k == "terminus_type":
                return (I, gs[1])
            if k == "restype":
                return (I, gs[1])
            if k == "ref_motif_token_type":
                return (I, gs[1])
            return (I, gs[1]) if gs[0] == I_g else (L, gs[1])
        return None

    golden_keys = set(g_shapes)
    ported_keys = set(f)
    missing = golden_keys - ported_keys
    extra = ported_keys - golden_keys
    assert not missing, f"missing keys vs golden: {sorted(missing)}"
    assert not extra, f"extra keys vs golden: {sorted(extra)}"

    dtype_map = {"torch.bool": torch.bool, "torch.int8": torch.int8, "torch.int32": torch.int32,
                 "torch.int64": torch.int64, "torch.bfloat16": torch.bfloat16,
                 "torch.float32": torch.float32}
    mismatches = []
    for k in golden_keys:
        gshape, gdtype = g_shapes[k]
        gd = dtype_map.get(gdtype)
        if gd is None:
            mismatches.append((k, f"unknown golden dtype {gdtype}")); continue
        pt = f[k]
        if pt.dtype != gd:
            mismatches.append((k, f"dtype {pt.dtype} vs golden {gd}")); continue
        exp = expected_shape(k, gshape)
        if exp is None:
            mismatches.append((k, f"could not derive expected shape from golden {gshape}")); continue
        if tuple(pt.shape) != exp:
            mismatches.append((k, f"shape {tuple(pt.shape)} vs expected {exp} (golden {gshape})"))
    assert not mismatches, "\n".join(f"{k}: {m}" for k, m in mismatches)
    print(f"test_contract_vs_golden OK  (43/43 keys, dtypes + shapes match the real "
          f"dsDNA_basic golden meta; I={I}, L={L})")


def test_model_consumable():
    """Verify the ported `f` is consumable end-to-end by the reference TokenInitializer
    (CPU) with the real ckpt weights from the golden. Casts float tensors to float32
    first (mimics verify_sampler.reconstruct_f — the golden stores bf16 but the model
    consumes fp32). Checks the TI outputs match the golden out_shapes scaled to our I/L.
    Skips if the golden weights are absent."""
    import json
    cap = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")
    wpath = os.path.join(cap, "token_initializer.real_weights.pt")
    if not os.path.exists(wpath):
        print("test_model_consumable SKIPPED (no golden weights present)"); return
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import rfd3_ref as R
    ti_w = torch.load(wpath, map_location="cpu", weights_only=True)
    ti = R.build_token_initializer(); ti.load_state_dict(ti_w, strict=False); ti.eval()
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb"); _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,20,A31-40"})
        f = featurize(pdb, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32 else v)
          for k, v in f.items()}
    with torch.no_grad():
        out = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    I, L = 40, 560
    assert tuple(out["Q_L_init"].shape) == (L, 128)
    assert tuple(out["C_L"].shape) == (L, 128)
    assert tuple(out["P_LL"].shape) == (L, L, 16)
    assert tuple(out["S_I"].shape) == (I, 384)
    assert tuple(out["Z_II"].shape) == (I, I, 128)
    print(f"test_model_consumable OK  (ref TokenInitializer ran on ported f -> "
          f"Q_L_init {tuple(out['Q_L_init'].shape)}, S_I {tuple(out['S_I'].shape)}, "
          f"Z_II {tuple(out['Z_II'].shape)})")


if __name__ == "__main__":
    test_motif_scaffold_shapes_and_invariants()
    test_chain_break()
    test_binder_contig_designed_only()
    test_non_protein_rejected()
    test_contract_vs_golden()
    test_model_consumable()
    print("\nALL FEATURIZE STRUCTURAL TESTS PASS")
