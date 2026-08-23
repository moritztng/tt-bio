"""Generate the Nesso-1 input YAMLs, committed so the GPU and TT arms score the same thing.

Three axes, because Nesso-1's cost can scale with either side of the complex and the workload it
is built for is throughput, not latency:

  ladder/     protein length 128/256/512/768/1024 aa at ONE fixed ligand.
              The protein is the fleet's own size fixture, CDK2 (PDB 1HCL) apo 298 aa, tiled and
              truncated to N -- same construction as perf/rf3/make_inputs.py and
              perf/wh-correctness/matrix.py, so a Nesso rung at N aa lines up with the
              esmfold2 / boltz2 / protenix / opendde rungs already on the perf page. A kinase is
              also the right host for a small-molecule binder, so the fixture is not just
              convenient.
  ligands/    ONE protein length (256 aa) against a monotonic ligand-size series.
              Ligands are glycine/alanine oligomers built here rather than named drugs: the axis
              that matters is heavy-atom count (Nesso tokenises a ligand per atom), and a
              constructed series gives an exact, reproducible count instead of a SMILES quoted
              from memory. Two real drug-like ligands from the upstream README are included as
              realism anchors.
  screen/     ONE protein, 64 chemically distinct ligands of near-identical size.
              This is the shape the model is actually for: virtual screening one target against
              many compounds. Near-constant ligand size keeps the throughput number from being a
              size-variance measurement, and 64 distinct SMILES force 64 distinct conformers and
              64 distinct records (no output-exists short-circuit, no shared RDKit cache entry).

    python perf/nesso1/make_inputs.py
"""

import itertools
import pathlib

# CDK2 (PDB 1HCL) apo, 298 aa. Verbatim from perf/rf3/make_inputs.py:CDK2_298.
CDK2_298 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
            "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
            "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
            "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")

LADDER_AA = (128, 256, 512, 640, 768, 1024)
LIGAND_SWEEP_AA = 256
SCREEN_AA = 256

# The fixed ligand for the protein-length ladder: the upstream README's example SMILES
# (4-fluoro-N-(4-sulfamoylphenyl)benzamide), 22 heavy atoms. Quoted from
# github.com/recursionpharma/nesso README, not from memory.
LADDER_LIGAND = "Fc1ccc(cc1)C(=O)Nc1ccc(cc1)S(=O)(=O)N"

# The upstream tutorial's ligand (tyrosine), 13 heavy atoms. Also quoted from the README.
TUTORIAL_LIGAND = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"

RESIDUE_SMILES = {"G": "NCC(=O)", "A": "NC(C)C(=O)"}


def cdk2(n: int) -> str:
    """The fleet fixture's sequence at length n: tile the 298 aa domain, truncate."""
    return (CDK2_298 * (n // len(CDK2_298) + 1))[:n]


def peptide(code: str) -> str:
    """Linear peptide SMILES from a G/A residue code, e.g. 'GAG'.

    Heavy atoms = 4*(#G) + 5*(#A) + 1 for the C-terminal OH. Exact by construction, which is the
    whole point of building the series instead of naming compounds.
    """
    return "".join(RESIDUE_SMILES[c] for c in code) + "O"


def heavy_atoms(code: str) -> int:
    return sum(4 if c == "G" else 5 for c in code) + 1


def yaml_for(seq: str, smiles: str) -> str:
    return ("sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            "      sequence: %s\n"
            "  - ligand:\n"
            "      id: B\n"
            "      smiles: '%s'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n" % (seq, smiles))


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    root = pathlib.Path(__file__).parent / "inputs"

    # --- axis 1: protein length, fixed ligand -------------------------------------------------
    for n in LADDER_AA:
        d = root / "ladder" / ("aa%d" % n)
        write(d / ("cdk2_%d.yaml" % n), yaml_for(cdk2(n), LADDER_LIGAND))
        print("ladder  %4d aa  ligand 22 heavy" % n)

    # --- axis 2: ligand size, fixed protein ---------------------------------------------------
    seq = cdk2(LIGAND_SWEEP_AA)
    for code in ("GG", "GGGGGG", "GGGGGGGGGGGG", "GGGGGGGGGGGGGGGGGGGG"):
        d = root / "ligands" / ("hv%02d" % heavy_atoms(code))
        write(d / ("cdk2_%d_gly%d.yaml" % (LIGAND_SWEEP_AA, len(code))),
              yaml_for(seq, peptide(code)))
        print("ligand  %2d heavy  gly%d" % (heavy_atoms(code), len(code)))
    for tag, smi, hv in (("readme22", LADDER_LIGAND, 22), ("tutorial13", TUTORIAL_LIGAND, 13)):
        d = root / "ligands" / ("real_%s" % tag)
        write(d / ("cdk2_%d_%s.yaml" % (LIGAND_SWEEP_AA, tag)), yaml_for(seq, smi))
        print("ligand  %2d heavy  %s (real)" % (hv, tag))

    # --- axis 3: the screening set -------------------------------------------------------------
    seq = cdk2(SCREEN_AA)
    d = root / "screen"
    codes = ["".join(c) for c in itertools.product("GA", repeat=6)]
    for i, code in enumerate(codes):
        write(d / ("lig%02d_%s.yaml" % (i, code)), yaml_for(seq, peptide(code)))
    print("screen  %d ligands, %d-%d heavy atoms, protein %d aa"
          % (len(codes), heavy_atoms("G" * 6), heavy_atoms("A" * 6), SCREEN_AA))


if __name__ == "__main__":
    main()
