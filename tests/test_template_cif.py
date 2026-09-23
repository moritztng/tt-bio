"""A top-level `templates:` cif block, turned into the per-chain alignment npz.

Host-only, no card. The fixtures are upstream openfold-3's own 1a8q and 1y57 template cases:
each ships the template mmCIF and the alignment npz upstream built for it, so the cif route is
scored against the npz route on the same template.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tt_bio.protenix_template import chain_template_arrays, read_alignment_entries
from tt_bio.template_cif import align, structure_template_npz, template_chains

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "of3_upstream"
STRUCT = EX / "template_structures"


def _query(name):
    doc = yaml.safe_load((EX / f"{name}_template_on_dummymsa.yaml").read_text())
    return doc["sequences"][0]["protein"]["sequence"]


def _chains(seq, cid="A"):
    return [(cid, seq, None, "protein", None)]


def test_1a8q_cif_route_is_the_npz_route(tmp_path):
    """Same alignment, and the same template arrays the featurizer reads from it."""
    q = _query("1a8q")
    built = structure_template_npz([{"cif": str(STRUCT / "1a8q.cif"), "chain_id": "A"}],
                                   _chains(q), tmp_path, "openfold3")
    (mine,) = read_alignment_entries(built["A"])
    (up,) = read_alignment_entries(EX / "template_alignments" / "1a8q.npz")
    assert mine[1] == up[1] == "A"
    np.testing.assert_array_equal(mine[2], up[2])
    for a, b in zip(chain_template_arrays(len(q), [mine], tmp_path),
                    chain_template_arrays(len(q), [up], STRUCT)):
        np.testing.assert_array_equal(a, b)


def test_1y57_homolog_alignment_agrees_with_upstream(tmp_path):
    """1y57 is a homolog (63% identity), aligned upstream by a profile search. The pairwise
    alignment here places 434 of upstream's 444 pairs identically; the rest are end trims."""
    q = _query("1y57")
    t = template_chains(STRUCT / "1y57.cif")["A"]
    (up,) = read_alignment_entries(EX / "template_alignments" / "1y57.npz")
    common = {tuple(x) for x in align(q, t)} & {tuple(x) for x in up[2]}
    assert len(common) >= 430


def test_gapped_alignment_uses_both_sides_of_an_indel():
    """An ungapped local alignment (Boltz-2's) keeps one segment; the npz can carry both."""
    t = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    q = t[:30] + t[34:]                            # a 4-residue deletion in the query
    m = align(q, t)
    assert len(m) == len(q)
    assert (m[:30, 1] == np.arange(1, 31)).all() and (m[30:, 1] == np.arange(35, len(t) + 1)).all()


def test_template_chain_picked_by_score_and_by_name(tmp_path):
    q = _query("1a8q")
    cif = str(STRUCT / "1a8q.cif")
    auto = structure_template_npz([{"cif": cif}], _chains(q), tmp_path, "m")
    named = structure_template_npz([{"cif": cif, "chain_id": "A", "template_id": "A"}],
                                   _chains(q), tmp_path, "m")
    assert auto == named


def test_two_blocks_on_one_chain_are_ranked_in_order(tmp_path):
    q = _query("1a8q")
    blocks = [{"cif": str(STRUCT / "1a8q.cif")}, {"cif": str(STRUCT / "1y57.cif")}]
    (npz,) = structure_template_npz(blocks, _chains(q), tmp_path, "m").values()
    with np.load(npz, allow_pickle=True) as z:
        ranks = {k: z[k].item()["index"] for k in z.files}
    assert sorted(ranks.values()) == [0, 1]
    assert read_alignment_entries(npz)[0][2].shape == (274, 2)   # 1a8q first


def test_cif_is_cached_by_content_not_by_name(tmp_path):
    q = _query("1a8q")
    src = tmp_path / "in"
    src.mkdir()
    (src / "1a8q.cif").write_bytes((STRUCT / "1a8q.cif").read_bytes())
    cache = tmp_path / "cache"
    structure_template_npz([{"cif": str(src / "1a8q.cif")}], _chains(q), cache, "m")
    assert not (cache / "1a8q.cif").exists()        # never shadows the RCSB entry
    assert len(list(cache.glob("cif*.cif"))) == 1


def test_atom_site_fallback_matches_entity_sequence(tmp_path):
    """A cif without entity_poly_seq (some predictors write none) reads the same sequence
    wherever the residues are observed."""
    import biotite.structure.io.pdbx as pdbx

    f = pdbx.CIFFile.read(str(STRUCT / "1a8q.cif"))
    for cat in ("entity_poly_seq", "entity_poly", "struct_asym"):
        del f.block[cat]
    bare = tmp_path / "bare.cif"
    f.write(str(bare))
    full = template_chains(STRUCT / "1a8q.cif")["A"]
    fallback = template_chains(bare)["A"]
    assert all(a == b for a, b in zip(fallback, full) if a != "X")
    assert sum(a != "X" for a in fallback) > 250


@pytest.mark.parametrize("block, match", [
    ({"pdb": "x.pdb"}, "gives a pdb file"),
    ({"cif": str(STRUCT / "1a8q.cif"), "force": True, "threshold": 1.0}, "force"),
    ({"cif": "/nonexistent.cif"}, "does not exist"),
    ({"cif": str(STRUCT / "1a8q.cif"), "chain_id": "Z"}, "not protein chains"),
    ({"cif": str(STRUCT / "1a8q.cif"), "template_id": "Q"}, "no protein chain"),
    ({"cif": str(STRUCT / "1a8q.cif"), "chain_id": ["A"], "template_id": ["A", "B"]},
     "pair up one to one"),
])
def test_refusals(tmp_path, block, match):
    with pytest.raises(RuntimeError, match=match):
        structure_template_npz([block], _chains("MKVLAAGIVG"), tmp_path, "m")


def test_template_map_takes_the_top_level_block(tmp_path):
    from tt_bio.main import _read_bio_chains
    from tt_bio.worker import _template_map

    q = _query("1a8q")
    p = tmp_path / "in.yaml"
    p.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {q}\n"
                 f"templates:\n  - cif: {STRUCT / '1a8q.cif'}\n    chain_id: A\n")
    got = _template_map(p, "protenix-v2", _read_bio_chains(p), tmp_path)
    assert list(got) == ["A"] and got["A"].endswith(".npz")

    both = tmp_path / "both.yaml"
    both.write_text(p.read_text().replace(
        f"sequence: {q}\n",
        f"sequence: {q}\n      templates: {EX / 'template_alignments' / '1a8q.npz'}\n"))
    with pytest.raises(RuntimeError, match="give one"):
        _template_map(both, "protenix-v2", _read_bio_chains(both), tmp_path)


def test_top_level_block_is_the_templates_feature(tmp_path):
    """One feature, two spellings: a model that takes the npz takes the cif, and a model
    that refuses templates refuses both."""
    from tt_bio.capabilities import CAPABILITY, HONOURED, check_capabilities, detect
    from tt_bio.main import _read_bio_chains

    p = tmp_path / "in.yaml"
    p.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKVL\n"
                 "templates:\n  - cif: t.cif\n    chain_id: A\n")
    assert detect(p)["templates"] == "chain(s) A"
    chains = _read_bio_chains(p)
    for model, caps in CAPABILITY.items():
        if model == "nesso1":
            continue
        if caps["templates"] == HONOURED:
            check_capabilities(p, chains, model, echo=lambda m: None)
        else:
            with pytest.raises(RuntimeError, match="templates"):
                check_capabilities(p, chains, model, echo=lambda m: None)
