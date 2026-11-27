"""The DiffDock-L input pipeline against upstream's own graphs.

The goldens in `tests/fixtures/diffdock/` were captured from upstream
`gcorso/DiffDock` at 85c49b60 by `tests/diffdock_input_capture.py`, running
upstream's `moad_extract_receptor_structure` and `get_lig_graph_with_matching`
with the released v1.1 checkpoints' graph settings. Everything integer or
categorical has to match exactly; the geometry is graded to float tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio.diffdock.input import (
    ComplexSpec,
    build_complex_graph,
    fold_sequence_receptors,
    split_embedding_by_chain,
)
from tt_bio.diffdock.ligand import build_ligand_graph, generate_conformer, read_molecule
from tt_bio.diffdock.receptor import build_receptor_graph, knn_edge_index
from tt_bio.diffdock.structure import parse_receptor

FIXTURES = Path(__file__).parent / "fixtures" / "diffdock"
INDEX = FIXTURES / "INDEX.json"

pytestmark = pytest.mark.skipif(
    not INDEX.exists(), reason="DiffDock input fixtures are not present")


def _index() -> list[dict]:
    return json.loads(INDEX.read_text())["complexes"]


def _ids(key=None) -> list[str]:
    return [c["id"] for c in _index() if key is None or key(c)]


def _golden(cid: str, kind: str):
    return np.load(FIXTURES / f"{cid}.{kind}.npz", allow_pickle=True)


def _receptor_path(cid: str) -> Path:
    return FIXTURES / f"{cid}_protein.pdb"


def _ligand_path(cid: str) -> Path:
    return FIXTURES / f"{cid}_ligand.sdf"


def _edge_set(edge_index) -> set:
    return set(map(tuple, np.asarray(edge_index).T.tolist()))


# ------------------------------------------------------------------ receptor

@pytest.mark.parametrize("cid", _ids() or ["none"])
@pytest.mark.parametrize("kind,all_atoms", [("score", False), ("conf", True)])
def test_receptor_graph_matches_upstream(cid, kind, all_atoms):
    """Receptor nodes, features, chains, side-chain vectors and edges."""
    g = _golden(cid, kind)
    out = build_receptor_graph(
        parse_receptor(_receptor_path(cid)),
        c_alpha_max_neighbors=24, all_atoms=all_atoms, atom_max_neighbors=8)

    assert "".join(out["sequence"]) == "".join(g["rec_sequence"].tolist())
    assert np.array_equal(out["x"].numpy(), g["rec_x"])
    assert np.array_equal(out["pos"].numpy(), g["rec_pos"])
    assert np.array_equal(out["chain_ids"].numpy(), g["rec_chain_ids"])
    assert np.allclose(out["side_chain_vecs"].numpy(), g["rec_side_chain_vecs"],
                       atol=1e-4, equal_nan=True)
    assert np.array_equal(out["edge_index"].numpy(), g["rec_edge_index"])

    if all_atoms:
        atom = out["atom"]
        assert np.array_equal(atom["x"].numpy(), g["atom_x"])
        assert np.array_equal(atom["pos"].numpy(), g["atom_pos"])
        assert np.array_equal(
            atom["atom_rec_edge_index"].numpy(), g["atom_rec_edge_index"])
        # The atom graph is graded as a set of edges. Two atoms exactly
        # equidistant from a third are ordered differently by torch_cluster's
        # heap than by a stable sort; the edge set, which is all the model
        # reads (every consumer scatter-sums over edges), is identical.
        assert _edge_set(atom["edge_index"]) == _edge_set(g["atom_edge_index"])


def test_residue_order_is_file_order_not_chain_grouped():
    """A chain part written after other chains stays where the file puts it.

    ProDy numbers residues as they are written, and the receptor node order IS
    that numbering, so it propagates into every node feature, the k-NN graph and
    the per-chain ESM-2 embedding split. gemmi's reader gathers same-named chain
    parts together by default, which silently reorders exactly this case: in
    7FT9_4MB three single-residue parts of chains C and D are written after all
    four main chains, and merging moved one of them 191 residues earlier. The
    three other fixtures are all written chain-contiguously and cannot see it.
    """
    rec = parse_receptor(_receptor_path("7FT9_4MB"))
    keys = list(rec.chain_keys)
    # The three single-residue parts are last in the file and must be last here.
    assert keys[-3:] == ["LC", "RD", "TD"], keys[-5:]
    # The four main chains stay contiguous and in file order ahead of them.
    main = keys[:-3]
    assert [k for i, k in enumerate(main) if i == 0 or k != main[i - 1]] == \
        ["AA", "BB", "CC", "DD"]


def test_receptor_above_3000_residues_is_refused():
    """Upstream raises above 3000 residues; so do we, with the same message."""
    from tt_bio.diffdock.receptor import MAX_RECEPTOR_RESIDUES
    from tt_bio.diffdock.structure import Receptor

    n = MAX_RECEPTOR_RESIDUES + 1
    big = Receptor(
        sequence="A" * n,
        coords=np.zeros((n, 14, 3)),
        chain_keys=np.array(["A"] * n),
        resnames=("ALA",) * n)
    with pytest.raises(ValueError, match="too large"):
        build_receptor_graph(big)


def test_knn_tie_break_is_not_cdist():
    """The k-NN distance is the direct one, not cdist's expanded form.

    cdist computes |a|^2+|b|^2-2ab in float32, which reorders neighbours a few
    1e-5 A apart. The regression it caused was real (4 of 18,192 receptor edges
    on 7A9E_R4W), so the property is pinned rather than left to a golden.
    """
    torch.manual_seed(0)
    pos = torch.randn(400, 3) * 20
    edges = knn_edge_index(pos, 24)
    d = ((pos[edges[1]] - pos[edges[0]]) ** 2).sum(-1).reshape(-1, 24)
    assert torch.all(d[:, :-1] <= d[:, 1:] + 1e-12), "neighbours are not sorted by distance"
    assert edges.shape == (2, 400 * 24)


# --------------------------------------------------------------- the complex

@pytest.mark.parametrize("cid", _ids()[:1] or ["none"])
def test_pdb_plus_smiles_builds_graph(cid):
    """The default path: a PDB receptor and a SMILES ligand."""
    smiles = str(_golden(cid, "score")["smiles"])
    graph = build_complex_graph(
        ComplexSpec(name=cid, protein_path=str(_receptor_path(cid)),
                    ligand_description=smiles),
        seed=42)

    assert graph.name == cid
    assert graph.receptor["pos"].shape[1] == 3
    # Both halves are centred: the receptor on its own centroid, the ligand on
    # its own, which is what the sampler's randomize_position assumes.
    assert torch.allclose(graph.receptor["pos"].mean(0), torch.zeros(3), atol=1e-3)
    assert torch.allclose(graph.ligand["pos"].mean(0), torch.zeros(3), atol=1e-3)
    assert graph.original_center.shape == (1, 3)
    assert graph.ligand["mask_rotate"].shape[1] == graph.ligand["x"].shape[0]
    # The molecule kept for writing poses still carries its hydrogens; the
    # graph the model sees does not.
    assert graph.mol.GetNumAtoms() >= graph.ligand["x"].shape[0]


@pytest.mark.parametrize("cid", _ids()[:2] or ["none"])
def test_torsion_mask_matches_upstream(cid):
    """The rotatable-bond mask and the atoms each torsion moves."""
    g = _golden(cid, "score")
    mol = read_molecule(_ligand_path(cid), sanitize=True)
    mol.RemoveAllConformers()
    from rdkit.Chem import AddHs

    mol = AddHs(mol)
    generate_conformer(mol, seed=int(g["seed"]))
    lig = build_ligand_graph(mol, remove_hs=True)

    assert np.array_equal(lig["x"].numpy(), g["lig_x"])
    assert np.array_equal(lig["edge_index"].numpy(), g["lig_edge_index"])
    assert np.array_equal(lig["edge_attr"].numpy(), g["lig_edge_attr"])
    assert np.array_equal(lig["edge_mask"].numpy(), g["lig_edge_mask"])
    assert np.array_equal(lig["mask_rotate"], g["lig_mask_rotate"])


def test_mmcif_receptor_equals_pdb_receptor(tmp_path):
    """mmCIF is a tt-bio extension and must give the same receptor as the PDB."""
    import gemmi

    cid = _ids()[0]
    pdb = _receptor_path(cid)
    cif = tmp_path / f"{cid}.cif"
    structure = gemmi.read_structure(str(pdb))
    structure.setup_entities()
    cif.write_text(structure.make_mmcif_document().as_string())

    a = build_receptor_graph(parse_receptor(pdb), all_atoms=True)
    b = build_receptor_graph(parse_receptor(cif), all_atoms=True)

    assert a["sequence"] == b["sequence"]
    assert np.array_equal(a["x"].numpy(), b["x"].numpy())
    assert np.allclose(a["pos"].numpy(), b["pos"].numpy(), atol=1e-3)
    assert np.array_equal(a["chain_ids"].numpy(), b["chain_ids"].numpy())
    assert np.array_equal(a["edge_index"].numpy(), b["edge_index"].numpy())
    assert np.array_equal(a["atom"]["x"].numpy(), b["atom"]["x"].numpy())


def test_two_chain_receptor_embedding_order():
    """Per-chain ESM-2 embeddings concatenate in upstream's chain order.

    Upstream numbers chains with `np.unique` over segment+chain, which is
    alphabetical rather than file order, and concatenates the per-chain
    embeddings in that numbering. A receptor node's embedding row therefore has
    to be the one for its own residue, which is what this checks end to end.
    """
    multichain = _ids(lambda c: c["chains"] > 1)
    if not multichain:
        pytest.skip("no multi-chain fixture")
    cid = multichain[0]
    receptor = parse_receptor(_receptor_path(cid))
    sequences, chain_ids = receptor.chain_sequences()
    assert len(sequences) > 1
    assert list(chain_ids) == sorted(chain_ids), "chains are not contiguous in node order"

    # A fake embedding whose value identifies the chain and the residue, so a
    # misplaced block is visible rather than merely wrong.
    embeddings = [
        np.full((len(seq), 1280), i, dtype=np.float32) + np.arange(len(seq))[:, None] * 1e-3
        for i, seq in enumerate(sequences)]
    graph = build_receptor_graph(receptor, lm_embeddings=embeddings)

    x = graph["x"].numpy()
    assert x.shape == (receptor.n_residues, 1281)
    assert np.array_equal(x[:, 1].astype(int), chain_ids)

    # And the inverse split puts them back.
    joined = np.concatenate(embeddings, axis=0)
    back = split_embedding_by_chain(joined, sequences)
    assert [len(b) for b in back] == [len(s) for s in sequences]


def test_sequence_only_receptor_folds_then_docks(tmp_path):
    """A sequence with no structure is folded first, then parsed and docked.

    The fold itself is tt-bio's esmfold2 on the card and is not run here; the
    contract this row owns is which specs get folded, where the structure is
    written, that an existing one is reused, and that what comes back parses
    into a receptor graph.
    """
    import gemmi

    cid = _ids()[0]
    structure = gemmi.read_structure(str(_receptor_path(cid)))
    structure.setup_entities()
    source = structure.make_mmcif_document().as_string()
    calls = []

    def fake_fold(sequence, path):
        calls.append((sequence, Path(path)))
        Path(path).write_text(source)
        return Path(path)

    specs = [
        ComplexSpec(name="withstructure", protein_path=str(_receptor_path(cid)),
                    ligand_description="CCO"),
        ComplexSpec(name="seqonly", protein_sequence="MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAE",
                    ligand_description="CCO"),
    ]
    paths = fold_sequence_receptors(specs, tmp_path, fold=fake_fold)

    assert list(paths) == ["seqonly"], "only the sequence-only spec is folded"
    assert len(calls) == 1
    assert calls[0][1].name.endswith("_esmfold2.cif")
    assert Path(paths["seqonly"]).exists()

    # Second call reuses what is on disk instead of folding again.
    fold_sequence_receptors(specs, tmp_path, fold=fake_fold)
    assert len(calls) == 1

    graph = build_complex_graph(specs[1], seed=1, receptor_path=paths["seqonly"])
    assert graph.receptor["pos"].shape[0] > 0


def test_fasta_for_folding_splits_diffdock_chain_separator(tmp_path):
    """DiffDock's ':'-separated sequence becomes one FASTA record per chain."""
    from tt_bio.diffdock.folding import write_fasta

    path = write_fasta("AAAA:GGG", tmp_path / "r.fasta")
    records = [line for line in path.read_text().splitlines() if line.startswith(">")]
    assert records == [">A|protein", ">B|protein"]
    assert "AAAA" in path.read_text() and "GGG" in path.read_text()
