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
    assert f["ref_atom_name_chars"].shape == (L, 4, 64)
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
    assert float(f["token_bonds"].diagonal().abs().sum()) == 0.0
    assert torch.allclose(f["token_bonds"], f["token_bonds"].T)
    # chain break: contig has no /0 so A31-40 follows the scaffold on chain A
    # -> bonds exist across motif->scaffold->motif boundaries (contiguous res_id)
    print("test_motif_scaffold_shapes_and_invariants OK  I=", I, "L=", L)


def test_chain_break():
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,/0,A31-40"})
        f = featurize(pdb, spec)
    # two segments on chain A separated by a chain break -> no bond between token 9 and 10
    assert float(f["token_bonds"][9, 10]) == 0.0
    assert float(f["token_bonds"][8, 9]) == 1.0  # within first motif
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


if __name__ == "__main__":
    test_motif_scaffold_shapes_and_invariants()
    test_chain_break()
    test_binder_contig_designed_only()
    test_non_protein_rejected()
    print("\nALL FEATURIZE STRUCTURAL TESTS PASS")
