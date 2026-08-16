#!/usr/bin/env python3
"""Dry-run every pass-3 payload against the DEPLOYED validator, without submitting.

Runs on UF-EV-A13-GWH02 against /home/cust-team/mthuening/aiand-bio, which is the code
the live API actually runs. It calls limits.check_targets / check_design /
check_rfd3_design / check_sequences directly: no HTTP, no job, no chip, no load on
production. The point is that a payload the exec tier is about to spend an hour on is
known to reach the engine, so an authoring typo does not come back as a 400 after a
45-minute design run.

    ssh japanfold-ssh 'cd /home/cust-team/mthuening/aiand-bio && \
        ./env/bin/python /tmp/validate_payloads.py'
"""
import sys

from tt_bio.platform import limits

P64 = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTEKPEVIDAS"
LYS = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCS"
       "ALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")
DNA1, DNA2 = "AATTGTGAGCGGATAACAATT", "AATTGTTATCCGCTCACAATT"
TAR = "GGCAGAUCUGAGCCUGGGAGCUCUCUGGC"

BG = {
    "des_bg_protein": f"""entities:
  - protein:
      id: B
      sequence: 110..130
  - protein:
      id: A
      msa: empty
      sequence: {LYS}
""",
    "des_bg_sm": """entities:
  - protein:
      id: B
      sequence: 80..120
  - ligand:
      id: A
      smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
""",
    "des_bg_dna": f"""entities:
  - protein:
      id: B
      sequence: 60..90
  - dna:
      id: A
      sequence: {DNA1}
  - dna:
      id: C
      sequence: {DNA2}
""",
    "des_bg_rna": f"""entities:
  - protein:
      id: B
      sequence: 60..90
  - rna:
      id: A
      sequence: {TAR}
""",
}

RFD3 = {
    "des_rfd3_binder": ("iai_protein.pdb", "A1-150,60-80"),
    "des_rfd3_scaffold": ("iai_protein.pdb", "A1-10,20,A31-40"),
    "des_rfd3_na": ("1bna_dna.pdb", "A1-12,B1-12,60-80"),
}

VARIANT_TARGET = f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"


def report(name, fn):
    try:
        fn()
        print(f"  OK      {name}")
        return 0
    except Exception as e:
        print(f"  REJECT  {name}: {type(e).__name__}: {e}")
        return 1


def main(struct_dir):
    bad = 0
    print("BoltzGen design specs (limits.check_design, engine=boltzgen):")
    for name, spec in BG.items():
        bad += report(name, lambda s=spec: limits.check_design(s, engine="boltzgen"))

    print("RFD3 design submissions (limits.check_rfd3_design):")
    for name, (fn, contig) in RFD3.items():
        try:
            text = open(f"{struct_dir}/{fn}").read()
        except OSError as e:
            print(f"  SKIP    {name}: {e}")
            bad += 1
            continue
        bad += report(f"{name} [{fn}, '{contig}']",
                      lambda t=text, c=contig: limits.check_rfd3_design(t, c))

    print("Predict targets (limits.check_targets):")
    for model in ("boltz2", "esmfold2-fast", "protenix-v2"):
        bad += report(f"variant fixture -> {model}",
                      lambda m=model: limits.check_targets([{"content": VARIANT_TARGET}], m))

    print("Embed sequences (limits.check_sequences):")
    cdk2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
            "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
            "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
            "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")
    tiled = lambda n: (cdk2 * (n // len(cdk2) + 1))[:n]
    for model, n in (("esmc-300m", 2000), ("esmc-600m", 2000), ("esmc-6b", 1968),
                     ("saprot-650m", 2000), ("saprot-1.3b", 2000)):
        recs = [(f"s{i}", tiled(n)) for i in range(50)]
        bad += report(f"50 x {n} -> {model}", lambda r=recs, m=model: limits.check_sequences(r, m))
    # One past the 6B ceiling must be refused: this one is EXPECTED to reject.
    try:
        limits.check_sequences([("s0", tiled(1969))], "esmc-6b")
        print("  UNEXPECTED  esmc-6b accepted 1969 residues (its cap is 1968)")
        bad += 1
    except Exception as e:
        print(f"  OK (expected reject)  esmc-6b 1969: {e}")

    print(f"\n{bad} payload(s) the deployed validator would refuse.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
