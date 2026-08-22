"""Generate the OpenBind-0 benchmark inputs, committed so the GPU and TT arms fold the same thing.

Two ladders, because OB0 has two cost axes:

  apo      a single protein chain at 128 / 256 / 512 / 768 / 1024 aa. CDK2 (PDB 1HCL) apo,
           298 aa, tiled and truncated to N -- the same construction as
           `perf/wh-correctness/matrix.py` and `perf/rf3/make_inputs.py`, so an OB0 rung at N aa
           lines up with the esmfold2 / boltz2 / protenix / opendde / rf3 rungs already on the
           perf page.
  ligand   CDK2 298 aa plus one small molecule, at three ligand sizes. OB0 is the protein-ligand
           specialisation of OpenFold3, so a protein-only ladder measures the wrong half of it.
           Ligand heavy-atom count is a second axis and gets its own column.

Every spec is emitted three ways: a neutral `.spec.json`, the OpenFold3 query set the GPU arm
folds (`.of3.json`), and the tt-bio YAML the TT arm folds (`.tt.yaml`). All three are committed
with their sha256 in `SHA256SUMS`. `perf-page-matched-batch-protocol-recurrence` has shipped four
times in this org: if the TT side folds a file whose sha256 is not in that list, the comparison is
void.

Single sequence, no MSA, no templates. MSA depth is a third axis and the paired-MSA path is host
cost the port does not own, so both arms run single-sequence to remove the confound.

    python perf/openbind/make_inputs.py
"""

import hashlib
import json
import pathlib

# CDK2 (PDB 1HCL) apo, 298 aa. Verbatim from perf/wh-correctness/matrix.py:CDK2_298.
CDK2_298 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
            "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
            "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
            "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")

SIZES = (128, 256, 512, 768, 1024)

# Tile-alignment probes, protein only. Every rung in SIZES is a multiple of 32 AND of 128, which is
# the most favourable shape family this trunk has, so a lever scored only there is scored on its
# best case (`one-size-tuning-is-a-standing-defect-class`). 547 is the token count the 512 aa + STU
# cell lands on; reached here with no ligand at all, it separates the raggedness from the ligand and
# from the +1 bucket step. 544 (17x32) and 576 (18x32) bracket it, so a cost step between 544 and
# 547 is three residues and nothing else.
TILE_SIZES = (544, 547, 576)


# Ligands by CCD code, so no SMILES is transcribed by hand and both arms resolve the same
# component from the same chemical component dictionary. Sizes below are the formula heavy-atom
# counts; the harness re-derives the count from the featurised batch and records the measured
# value, so a wrong number here shows up as a mismatch rather than propagating.
#   CFF  caffeine, C8H10N4O2                     14 heavy atoms
#   STU  staurosporine, C28H26N4O3               35 heavy atoms   (pan-kinase inhibitor)
#   NAD  nicotinamide adenine dinucleotide       44 heavy atoms
LIGANDS = (("s", "CFF", 14), ("m", "STU", 35), ("l", "NAD", 44))

# The medium ligand is repeated at 512 aa. One point per axis cannot tell an additive ligand cost
# from one that interacts with chain length, and `one-size-tuning-is-a-standing-defect-class` is
# exactly this failure.
LIGAND_LENGTHS = ((298, ("s", "m", "l")), (512, ("m",)))


def cdk2(n: int) -> str:
    """The fleet fixture's sequence at length n: tile the 298 aa domain, truncate."""
    return (CDK2_298 * (n // len(CDK2_298) + 1))[:n]


def apo_spec(n: int) -> dict:
    return {"name": "ob_apo_%d" % n, "n_residues": n,
            "chains": [{"molecule_type": "protein", "chain_id": "A", "sequence": cdk2(n)}]}


def ligand_spec(n: int, tag: str, ccd: str, heavy: int) -> dict:
    return {"name": "ob_lig_%s_%d" % (tag, n), "n_residues": n,
            "ligand_ccd": ccd, "ligand_heavy_atoms_formula": heavy,
            "chains": [{"molecule_type": "protein", "chain_id": "A", "sequence": cdk2(n)},
                       {"molecule_type": "ligand", "chain_id": "L", "ccd_codes": ccd}]}


def to_of3(spec: dict) -> dict:
    """One OpenFold3 InferenceQuerySet holding a single query.

    `use_msas` false is the single-sequence path. The GPU harness rewrites this file to repeat the
    query under N names so one process does cold + warm reps with no weight reload; the committed
    copy is the one-query canonical form.
    """
    chains = []
    for c in spec["chains"]:
        if c["molecule_type"] == "protein":
            chains.append({"molecule_type": "protein", "chain_ids": c["chain_id"],
                           "sequence": c["sequence"]})
        else:
            chains.append({"molecule_type": "ligand", "chain_ids": c["chain_id"],
                           "ccd_codes": c["ccd_codes"]})
    return {"queries": {spec["name"]: {"chains": chains, "use_msas": False,
                                       "use_paired_msas": False, "use_main_msas": False}}}


def to_tt_yaml(spec: dict) -> str:
    """tt-bio's predict YAML for the same content."""
    lines = ["version: 1", "sequences:"]
    for c in spec["chains"]:
        if c["molecule_type"] == "protein":
            lines += ["  - protein:", "      id: %s" % c["chain_id"],
                      "      sequence: %s" % c["sequence"]]
        else:
            lines += ["  - ligand:", "      id: %s" % c["chain_id"],
                      "      ccd: %s" % c["ccd_codes"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = pathlib.Path(__file__).parent / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    specs = [apo_spec(n) for n in SIZES + TILE_SIZES]
    for n, tags in LIGAND_LENGTHS:
        for tag, ccd, heavy in LIGANDS:
            if tag in tags:
                specs.append(ligand_spec(n, tag, ccd, heavy))

    written = []
    for spec in specs:
        base = out / spec["name"]
        for path, text in ((base.with_suffix(".spec.json"), json.dumps(spec, indent=2) + "\n"),
                           (base.with_suffix(".of3.json"), json.dumps(to_of3(spec), indent=2) + "\n"),
                           (base.with_suffix(".tt.yaml"), to_tt_yaml(spec))):
            path.write_text(text)
            written.append(path)
        print("%-22s %4d aa  %s" % (spec["name"], spec["n_residues"],
                                    spec.get("ligand_ccd", "-")))

    sums = out / "SHA256SUMS"
    sums.write_text("".join(
        "%s  %s\n" % (hashlib.sha256(p.read_bytes()).hexdigest(), p.name)
        for p in sorted(written)))
    print("\n%d files, sha256 in %s" % (len(written), sums))


if __name__ == "__main__":
    main()
