"""crop_cif must number each chain's label_seq from 1, or BoltzGen refuses the crop.

BoltzGen's `parse_polymer` walks `entity_poly_seq` positionally: it matches a residue on
`j == polymer[i].label_seq`, where `j` is the 1-based index into `full_sequence`. So a chain
whose `label_seq` does not start at 1 makes position and number disagree, and the parser
asserts on the first residue it believes it matched.

At crop offset 0 the two agree by luck, which is why every fixture this repo has shipped
parses and why the offset axis had never actually run: measured 2026-09-24, a 512-residue
crop at offset 100 died in 7.5 s on `AssertionError` in `parse_polymer`, before the device
was opened. Offsets exist so that a size ladder can separate a size effect from a target
effect, so the axis silently not working removes the only control on that confound.

This test pins the invariant rather than the parse, deliberately: the parse needs the
BoltzGen mols artifact, and the invariant is both the thing the parser requires and the thing
that was broken. It fails on the pre-fix crop_cif at offset 100 and passes at offset 0.
"""
import pathlib

import pytest

gemmi = pytest.importorskip("gemmi")

from perf.bhdesign.ladder import crop_cif  # noqa: E402

TARGET = pathlib.Path(__file__).resolve().parents[1] / "perf/mgxaccuracy/targets/gpb_dimer_1646.cif"


@pytest.mark.parametrize("n_res", [512, 1536])
@pytest.mark.parametrize("offset", [0, 100])
def test_each_chain_numbers_label_seq_from_one(tmp_path, n_res, offset):
    dst = tmp_path / f"crop_{n_res}_{offset}.cif"
    res, atoms = crop_cif(TARGET, n_res, dst, offset)
    assert res == n_res
    assert atoms > 0

    st = gemmi.read_structure(str(dst))
    assert len(st[0]) > 0, "crop produced no chains"
    for chain in st[0]:
        seqs = [r.label_seq for r in chain]
        assert seqs == list(range(1, len(chain) + 1)), (
            f"chain {chain.name} at offset {offset} numbers label_seq "
            f"{seqs[0]}..{seqs[-1]} over {len(chain)} residues; BoltzGen indexes "
            f"entity_poly_seq positionally and will assert on the first mismatch"
        )


@pytest.mark.parametrize("n_res", [512, 1536])
def test_full_sequence_length_matches_each_chain(tmp_path, n_res):
    """The other half of the same invariant: one entity per chain, sized to that chain.

    If an entity's full_sequence were longer than its chain -- two identical chains folded
    into one entity, say -- position and number would disagree again even with label_seq
    rebased, and this is the cheap place to notice.
    """
    dst = tmp_path / f"crop_{n_res}.cif"
    crop_cif(TARGET, n_res, dst, 100)
    st = gemmi.read_structure(str(dst))
    by_name = {ch.name: len(ch) for ch in st[0]}
    for ent in st.entities:
        covered = sum(by_name[s] for s in ent.subchains if s in by_name)
        assert len(ent.full_sequence) == covered, (
            f"entity {ent.name} carries {len(ent.full_sequence)} sequence entries over "
            f"subchains {list(ent.subchains)} totalling {covered} residues"
        )
