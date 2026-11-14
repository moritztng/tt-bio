"""Write the antibody_cdr condition spec from the Fab complex already in the repo.

    python perf/rfd4/cond/make_antibody_spec.py [--out perf/rfd4/cond/specs/antibody.yaml]

Upstream ships no paired VH/VL design input. Its antibody surface on the design side is the VHH
example, and `antibody` on the scoring side is a set of metrics. So this spec is written rather
than copied, from `examples/ground_truth_structures/1ahw.cif`: Fab 5G9 (chain A light, chain B
heavy) bound to the tissue factor extracellular domain (chain C), label_seq_id numbered 1-214,
1-214 and 4-211, no insertion codes.

What it asks for is the antibody task: hold the antigen and both frameworks, redesign the six
CDRs. The six loops are found by matching Kabat-definition substrings against the chains' own
sequences, so the ranges are derived from the structure, not typed in; each is replaced by an
expandable segment that spans the native length either way, which is what lets CDR length vary
per design. The Fv is cropped after FR4 and the antigen to the residues near it, because the
constant domains are tokens the model would pay for and nothing conditions on.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STRUCTURE = ROOT / "examples" / "ground_truth_structures" / "1ahw.cif"

#: Kabat CDRs of 5G9, as substrings of the chains' own sequences (1ahw, examples/1ahw_abag.yaml).
CDRS = {
    "A": [("L1", "KASQDIRKYLN"), ("L2", "YATSLAD"), ("L3", "LQHGESPYT")],
    "B": [("H1", "DYYMH"), ("H2", "LIDPENGNTIYDPKFQG"), ("H3", "DNSYYFDY")],
}
#: Framework 4 runs to the end of the variable domain; 11 residues past the last CDR.
FR4 = 11
#: Antigen residues within this many Angstrom of the Fv are kept.
EPITOPE_A = 12.0
#: How far a designed loop may depart from the native length.
LENGTH_SLACK = 1


def chains(path: Path):
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    array = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    return array[struc.filter_amino_acids(array)]


def sequence(array, chain: str):
    """``(one-letter sequence, res_ids)`` for one chain, ordered by residue."""
    import biotite.structure as struc
    import numpy as np
    from biotite.sequence import ProteinSequence

    ca = array[(array.chain_id == chain) & (array.atom_name == "CA")]
    order = np.argsort(ca.res_id)
    ca = ca[order]
    letters = "".join(ProteinSequence.convert_letter_3to1(r) for r in ca.res_name)
    return letters, ca.res_id


def loops(array, chain: str):
    """``[(name, first_res_id, last_res_id)]`` for the chain's CDRs."""
    seq, res_ids = sequence(array, chain)
    out = []
    for name, motif in CDRS[chain]:
        i = seq.find(motif)
        if i < 0:
            raise SystemExit(f"{chain}: {name} motif {motif} is not in the chain's sequence")
        out.append((name, int(res_ids[i]), int(res_ids[i + len(motif) - 1])))
    return out


def contig_for(array, chain: str) -> tuple[str, list[tuple[int, int]]]:
    """The chain's contig, and the framework ranges that stay preserved."""
    seq, res_ids = sequence(array, chain)
    cdrs = loops(array, chain)
    first, last = int(res_ids[0]), int(cdrs[-1][2]) + FR4
    tokens, framework, cursor = [], [], first
    for name, lo, hi in cdrs:
        tokens.append(f"{chain}{cursor}-{lo - 1}")
        framework.append((cursor, lo - 1))
        length = hi - lo + 1
        tokens.append(f"{name}:{length - LENGTH_SLACK}-{length + LENGTH_SLACK}")
        cursor = hi + 1
    tokens.append(f"{chain}{cursor}-{last}")
    framework.append((cursor, last))
    return ", ".join(tokens), framework


def epitope(array, antigen: str, fv_chains: tuple[str, ...], fv_last: dict[str, int]):
    """The antigen's residue span near the Fv, as one preserved range."""
    import numpy as np

    ag = array[array.chain_id == antigen]
    near = np.zeros(len(ag), dtype=bool)
    for chain in fv_chains:
        fv = array[(array.chain_id == chain) & (array.res_id <= fv_last[chain])]
        d = np.linalg.norm(ag.coord[:, None, :] - fv.coord[None, :, :], axis=-1)
        near |= d.min(axis=1) <= EPITOPE_A
    contact = ag.res_id[near]
    if not len(contact):
        raise SystemExit("no antigen residue is near the Fv; check the chain ids")
    return int(contact.min()), int(contact.max())


def build() -> str:
    array = chains(STRUCTURE)
    parts, framework, fv_last = [], {}, {}
    for chain in ("A", "B"):
        contig, ranges = contig_for(array, chain)
        parts.append(contig)
        framework[chain] = ranges
        fv_last[chain] = ranges[-1][1]
    lo, hi = epitope(array, "C", ("A", "B"), fv_last)
    parts.append(f"C{lo}-{hi}")
    contig = ", /0, ".join(parts)

    pinned = ["sel('C')"] + [f"sel('{c}/*/[{a}-{b}]')" for c in ("A", "B") for a, b in framework[c]]
    lines = [
        "# Written by perf/rfd4/cond/make_antibody_spec.py; edit that, not this.",
        "fab5g9_tissuefactor_cdr:",
        f"  input: {Path('../../../..') / STRUCTURE.relative_to(ROOT)}",
        f'  contig: "{contig}"',
        "  conditions:",
        "    C_CRDGRP: {1: [{select: \"sel('C/*/*/[N,CA,C,O]')\"}]}",
        "    C_SEQ: {true: [" + ", ".join("{select: \"%s\"}" % s for s in pinned) + "]}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "specs" / "antibody.yaml"))
    a = ap.parse_args()
    text = build()
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
