"""A contig whose designed segment comes BEFORE the kept motif is still one chain.

A comma in an RFD3 contig is a within-chain separator and ``/0`` is the chain break
(``docs/rfd3-design.md``), so ``12-20,C33-59,18-28`` asks for one continuous protein.
It used to come back as two: the leading designed run opened the token plan with no
preceding block to take a chain from, got a fresh synthetic chain letter, and so got
its own ``asym_id`` and its own pair of termini. The model built what it was told to
build, 11-13 A apart with no peptide bond, in 10 of 10 designs on 2026-11-10, while
the same request written motif-first (``C33-59,25-35``) fused every time.

Host-only: the token plan and the `f` dict are built on CPU, no device.
"""
import numpy as np
import pytest

from tt_bio.rfd3.featurize import featurize
from tt_bio.rfd3.input import InputSpecification

ATOMS = (("N", "N", 0.0), ("CA", "C", 1.5), ("C", "C", 2.4), ("O", "O", 3.0), ("CB", "C", 1.9))


@pytest.fixture(scope="module")
def motif(tmp_path_factory):
    """27 residues of chain C, numbered 33-59 like 1ZAA's second zinc finger."""
    lines, n = [], 0
    for i, rid in enumerate(range(33, 60)):
        for name, element, dx in ATOMS:
            n += 1
            x, z = i * 3.8 + dx, 1.3 if name == "CB" else 0.0
            lines.append(f"ATOM  {n:5d}  {name:<3s} ALA C{rid:4d}    "
                         f"{x:8.3f}{0.0:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
    p = tmp_path_factory.mktemp("rfd3") / "motif.pdb"
    p.write_text("\n".join(lines + ["END"]) + "\n")
    return str(p)


def _chains(path, contig):
    f = featurize(path, InputSpecification(contig=contig))
    term = f["terminus_type"].numpy()
    return (sorted(set(f["asym_id"].tolist())), f["residue_index"].tolist(),
            list(np.flatnonzero(term[:, 1])), list(np.flatnonzero(term[:, 0])))


@pytest.mark.parametrize("contig,length", [
    ("12-20,C33-59,18-28", 16 + 27 + 23),   # the reported scaffolding contig
    ("8-12,C33-59,8-12", 10 + 27 + 10),     # the same shape, shorter
    ("30,C33-34", 30 + 2),                  # designed run longer than the motif's start
    ("C33-59,25-35", 27 + 30),              # motif first: already worked, must not move
])
def test_one_chain_one_pair_of_termini(motif, contig, length):
    asym, residue_index, n_term, c_term = _chains(motif, contig)
    assert asym == [0], f"{contig} split into {len(asym)} chains"
    assert residue_index == list(range(length))
    assert n_term == [0] and c_term == [length - 1]


def test_a_chain_break_still_starts_a_new_chain(motif):
    # `/0` is the one thing that does split a contig, and a run that keeps
    # nothing still gets its own synthetic chain.
    asym, _, n_term, c_term = _chains(motif, "C33-59,/0,20")
    assert asym == [0, 1]
    assert list(n_term) == [0, 27] and list(c_term) == [26, 46]
