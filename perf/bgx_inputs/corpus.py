"""Build the target files a researcher actually brings, out of BindCraft 2's own hPDL1.

Every case here is a real PDB habit: an unresolved loop, waters and a metal left in the file,
selenomethionine from a SAD phasing experiment, two side-chain conformations, an insertion code
in an antibody, mmCIF instead of PDB. The originals are BindCraft 2's shipped structures; each
variant is written next to this script so a failure can be reproduced from the file alone.
"""
import pathlib
import sys

BC2 = pathlib.Path("/home/ttuser/bcx_e2e/bc2")
SRC = BC2 / "settings/target/structures"
OUT = pathlib.Path(__file__).parent / "inputs"

ATOM = ("ATOM  ", "HETATM")


def records(path):
    return [l for l in path.read_text().splitlines() if l.startswith(ATOM)]


def resnum(line):
    return int(line[22:26])


def write(name, lines, header=()):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join([*header, *lines, "END"]) + "\n")
    return OUT / name


def main():
    pdl1 = records(SRC / "hPDL1.pdb")
    il2r = records(SRC / "hIL2R_beta_gamma.pdb")

    # 1. An unresolved loop. 60-70 simply are not in the file, which is what a crystal
    #    structure with a mobile loop looks like. Hotspot 66 lives inside it.
    write("gap.pdb", [l for l in pdl1 if not 60 <= resnum(l) <= 70])

    # 2. Waters, a zinc and an N-linked glycan left in, as downloaded.
    het = [
        "HETATM 2000 ZN    ZN A 201      20.000 -30.000 -10.000  1.00 20.00          ZN  ",
        "HETATM 2001  O   HOH A 301      10.000 -20.000  -5.000  1.00 30.00           O  ",
        "HETATM 2002  O   HOH A 302      11.000 -21.000  -6.000  1.00 30.00           O  ",
        "HETATM 2003  C1  NAG A 401      15.000 -25.000  -8.000  1.00 40.00           C  ",
        "HETATM 2004  C2  NAG A 401      16.000 -25.500  -8.500  1.00 40.00           C  ",
        "HETATM 2005  O5  NAG A 401      15.500 -26.000  -7.000  1.00 40.00           O  ",
    ]
    write("ligands.pdb", pdl1 + het)

    # 3. Selenomethionine, written as HETATM MSE with SE in place of SD, as the PDB does.
    mse = []
    for l in pdl1:
        if l[17:20] == "MET":
            l = "HETATM" + l[6:17] + "MSE" + l[20:]
            if l[12:16] == " SD ":
                l = l[:12] + "SE  " + l[16:]
                l = l[:76] + "SE" + l[78:] if len(l) > 78 else l
        mse.append(l)
    write("mse.pdb", mse)

    # 3b. Phosphoserine, the other modification habit, which BindCraft 2 does not map: only
    #     MSE is in MODIFIED_RESIDUE_PARENTS, so this has to end in a clear refusal.
    phospho = []
    for l in pdl1:
        if resnum(l) == 54:
            phospho.append("HETATM" + l[6:17] + "SEP" + l[20:])
        else:
            phospho.append(l)
    phospho.append("HETATM 2100  P   SEP A  54      20.000 -30.000 -10.000  1.00 20.00           P  ")
    write("phospho.pdb", phospho)

    # 4. Two side-chain conformations on three residues, altloc A at 0.6 and B at 0.4.
    alt = []
    for l in pdl1:
        if resnum(l) in (54, 56, 66) and l[12:16].strip() not in ("N", "CA", "C", "O"):
            alt.append(l[:16] + "A" + l[17:54] + "0.60" + l[58:])
            alt.append(l[:16] + "B" + l[17:54] + "0.40" + l[58:])
        else:
            alt.append(l)
    write("altloc.pdb", alt)

    # 5. An insertion code, the antibody numbering habit: 100, 100A, 100B.
    ins = []
    for l in pdl1:
        if resnum(l) == 100:
            ins.append(l[:26] + "A" + l[27:])
        else:
            ins.append(l)
    write("insertion.pdb", ins)

    # 6. A file whose header is not the PDB's own: a modelling program's banner.
    write("weird_header.pdb", pdl1,
          header=["TITLE     MODEL BUILT BY SOMEONE'S PIPELINE",
                  "REMARK   1 NOT A WWPDB ENTRY",
                  "COMPND    CHAIN: A;",
                  "MODEL        1"])

    # 7. The same structure renumbered from 1, which is what a modelling tool hands back.
    #    The hotspots the reader wrote against the wwPDB numbering now name other residues.
    first = min(resnum(l) for l in pdl1)
    write("renumbered.pdb", [l[:22] + f"{resnum(l) - first + 1:4d}" + l[26:] for l in pdl1])

    # 8. A residue stripped to its backbone-minus-CA, the incomplete trim preflight warns about.
    write("trimmed.pdb", [l for l in pdl1 if not (resnum(l) == 80 and l[12:16] == " CA ")])

    # 9. Chain B of the IL-2 receptor alone, to select one chain out of a complex.
    write("twochain.pdb", il2r)

    # 10. mmCIF, via biotite, of the plain target.
    import biotite.structure.io.pdb as _pdb
    import biotite.structure.io.pdbx as _pdbx
    structure = _pdb.PDBFile.read(str(SRC / "hPDL1.pdb")).get_structure(model=1)
    block = _pdbx.CIFFile()
    _pdbx.set_structure(block, structure)
    OUT.mkdir(parents=True, exist_ok=True)
    block.write(str(OUT / "hPDL1.cif"))

    for path in sorted(OUT.iterdir()):
        print(f"{path.name:20s} {path.stat().st_size:8d} B")


if __name__ == "__main__":
    sys.exit(main())
