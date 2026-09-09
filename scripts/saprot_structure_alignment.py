#!/usr/bin/env python3
"""Does `tt-bio saprot --structure` put the right 3Di token on the right residue?

Foldseek reports 3Di only for residues the structure resolves. A deposited structure
with an unresolved loop therefore returns a SHORTER 3Di string than the FASTA
sequence, and `saprot._align_3di` right-pads it to length. Every residue after the
gap then carries a later residue's structural token, and the run reports success.

Foldseek also returns the structure's own amino-acid sequence, which is exactly what
would catch this, so the check costs one comparison. This script builds the gapped
case from a real structure and shows how far the assignment slips.

Needs FOLDSEEK_BIN (or foldseek on PATH). No device.
"""
import sys
from pathlib import Path

import gemmi

from tt_bio.saprot import _align_3di, foldseek_3di, load_sequences_with_structure

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
        if line.startswith(("ATOM", "HETATM")):
            if GAP[0] <= int(line[22:26]) <= GAP[1]:
                continue
        lines.append(line)
    p = out / "gapped.pdb"
    p.write_text("".join(lines))
    return p


def main():
    out = Path("scratch")
    out.mkdir(exist_ok=True)
    gapped = make_gapped(out)
    # Compare like with like: foldseek is run on the PDB form in both arms, so the
    # gap is the only difference. (The CIF and PDB forms of this structure give
    # byte-identical 3Di, checked separately.)
    full = foldseek_3di(str(out / "full.pdb"))
    chain = next(iter(full))
    aa, tdi = full[chain]
    print("full structure   : %d residues, %d 3Di chars" % (len(aa), len(tdi)))

    g = foldseek_3di(str(gapped))
    g_aa, g_tdi = g[next(iter(g))]
    print("gapped structure : %d residues, %d 3Di chars (removed %d-%d)"
          % (len(g_aa), len(g_tdi), GAP[0], GAP[1]))

    fasta = out / "gap_full.fasta"
    fasta.write_text(">T\n%s\n" % aa)
    got = load_sequences_with_structure(fasta, str(gapped))["T"]
    aligned = got[1]
    print("what the run feeds the model: %d 3Di chars for %d residues" % (len(aligned), len(aa)))

    truth = tdi  # the 3Di the full structure gives for these same residues
    wrong = [i for i in range(len(aa)) if aligned[i] != truth[i]]
    print("residues whose 3Di token is wrong: %d of %d (first at %d, last at %d)"
          % (len(wrong), len(aa), wrong[0] + 1, wrong[-1] + 1) if wrong
          else "residues whose 3Di token is wrong: 0")
    print("  truth   [38:52] = %r" % truth[38:52])
    print("  fed     [38:52] = %r" % aligned[38:52])
    print("  fasta aa[38:52] = %r" % aa[38:52])
    print("  structure's own aa (discarded by the port) = %d chars vs fasta %d"
          % (len(g_aa), len(aa)))

    # Negative control: the same call on the ungapped structure must align perfectly.
    ok = load_sequences_with_structure(fasta, str(out / "full.pdb"))["T"][1]
    bad = sum(1 for i in range(len(aa)) if ok[i] != truth[i])
    print("NEG ungapped structure, same FASTA: %d wrong tokens (must be 0)" % bad)

    # And _align_3di itself, in isolation.
    print("_align_3di('AAAA', 'dd') = %r  (pads, does not refuse)" % _align_3di("AAAA", "dd"))
    print("_align_3di('AA', 'dddd') = %r  (truncates, does not refuse)" % _align_3di("AA", "dddd"))
    return 0 if (wrong and bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
