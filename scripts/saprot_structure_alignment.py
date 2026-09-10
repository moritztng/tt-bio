#!/usr/bin/env python3
"""Does `tt-bio saprot --structure` put each 3Di token on the residue it was computed for?

Foldseek reports 3Di only for residues the structure resolves, so a deposited structure
with an unresolved loop returns a shorter string than the FASTA sequence. Reconciling the
two by length alone shifts every token after the gap, and the run reports success.
`saprot._map_3di` matches the structure's own amino-acid sequence instead, which foldseek
returns in the same call, and fills the unresolved positions with `#`.

The negative control is the old length-based reconciliation, recomputed here: on the same
input it has to fail the same check, or the check is not measuring the shift.

Needs FOLDSEEK_BIN (or foldseek on PATH). No device.
"""
import sys
from difflib import SequenceMatcher
from pathlib import Path

import gemmi

from tt_bio.saprot import _map_3di, foldseek_3di, load_sequences_with_structure

SRC = "examples/ground_truth_structures/prot.cif"
GAP = (41, 60)  # residues removed from the structure, kept in the FASTA


def make_gapped(out: Path) -> Path:
    """The same structure with residues GAP[0]..GAP[1] unresolved, as a PDB."""
    st = gemmi.read_structure(SRC)
    st.setup_entities()
    full = out / "full.pdb"
    st.write_pdb(str(full))
    lines = []
    for line in full.read_text().splitlines(True):
        if line.startswith(("ATOM", "HETATM")) and GAP[0] <= int(line[22:26]) <= GAP[1]:
            continue
        lines.append(line)
    p = out / "gapped.pdb"
    p.write_text("".join(lines))
    return p


def blocks(aa: str, struct_aa: str):
    """(sequence index, structure index) pairs for every residue the structure resolves."""
    out = []
    for i, j, n in SequenceMatcher(a=aa, b=struct_aa, autojunk=False).get_matching_blocks():
        out += [(i + k, j + k) for k in range(n)]
    return out


def score(fed: str, aa: str, struct_aa: str, struct_3di: str):
    """(tokens on the wrong residue, resolved-but-unmasked positions)."""
    pairs = blocks(aa, struct_aa)
    wrong = sum(1 for i, j in pairs if fed[i] != struct_3di[j])
    placed = {i for i, _ in pairs}
    unmasked = sum(1 for i in range(len(aa)) if i not in placed and fed[i] != "#")
    return wrong, unmasked


def by_length(aa: str, struct_3di: str) -> str:
    """What the reader used to do: pad or truncate the 3Di string to the AA length."""
    if len(struct_3di) >= len(aa):
        return struct_3di[: len(aa)]
    return struct_3di + "#" * (len(aa) - len(struct_3di))


def main():
    out = Path("scratch")
    out.mkdir(exist_ok=True)
    gapped = make_gapped(out)
    # Compare like with like: foldseek runs on the PDB form in both arms, so the gap is
    # the only difference.
    full = foldseek_3di(str(out / "full.pdb"))
    aa, tdi = full[next(iter(full))]
    g = foldseek_3di(str(gapped))
    g_aa, g_tdi = g[next(iter(g))]
    print("full structure   : %d residues, %d 3Di chars" % (len(aa), len(tdi)))
    print("gapped structure : %d residues, %d 3Di chars (removed %d-%d)"
          % (len(g_aa), len(g_tdi), GAP[0], GAP[1]))

    fasta = out / "gap_full.fasta"
    fasta.write_text(">T\n%s\n" % aa)
    fed = load_sequences_with_structure(fasta, str(gapped))["T"][1]
    print("what the run feeds the model: %d 3Di chars for %d residues" % (len(fed), len(aa)))

    wrong, unmasked = score(fed, aa, g_aa, g_tdi)
    print("tokens on the wrong residue: %d; unresolved positions not masked '#': %d"
          % (wrong, unmasked))
    old_wrong, old_unmasked = score(by_length(aa, g_tdi), aa, g_aa, g_tdi)
    print("NEG the old length-based reconciliation, same input: %d wrong, %d unmasked"
          % (old_wrong, old_unmasked))

    ctrl = load_sequences_with_structure(fasta, str(out / "full.pdb"))["T"][1]
    ctrl_wrong, ctrl_unmasked = score(ctrl, aa, aa, tdi)
    print("NEG ungapped structure, same FASTA: %d wrong, %d unmasked"
          % (ctrl_wrong, ctrl_unmasked))

    # For information, not a criterion: 3Di encodes local geometry, so deleting residues
    # legitimately changes their neighbours' descriptors. Those tokens differ from the
    # full structure's without any of them being on the wrong residue.
    neighbour = sum(1 for i, j in blocks(aa, g_aa) if g_tdi[j] != tdi[i])
    print("resolved residues whose 3Di the deletion itself changed: %d" % neighbour)

    # A structure of a different sequence has residues that land nowhere, so it is refused
    # rather than paired with whatever happened to line up.
    try:
        _map_3di(aa[::-1], aa, tdi, "full.pdb")
        refused = False
    except ValueError as e:
        refused = True
        print("a structure of another sequence is refused: %s" % str(e)[:130])
    print("_map_3di on an exact match returns the string unchanged: %s"
          % (_map_3di(aa, aa, tdi, "x") == tdi))

    ok = (wrong == 0 and unmasked == 0 and ctrl_wrong == 0 and ctrl_unmasked == 0
          and old_wrong > 0 and refused)
    print("RESULT: %s" % ("pass" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
