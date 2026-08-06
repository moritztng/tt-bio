"""Build Arm-3 specificity-frontier decoy yamls: pair each antibody with the WRONG
(non-cognate) antigen from a different, immunologically-unrelated original Stage-3
target, cyclic-shift-by-1 over 8 targets chosen to avoid any known/plausible
cross-reactivity cluster (excludes all SARS-CoV-2 spike variants as a group, all
Plasmodium/malaria CSP-family targets as a group, and the shared-antigen Envelopment
polyprotein trio) -- see docs/implementation-parity-data/abag-arm3-decoy-pairs.json
for the antigen identity used to justify each choice (from live RCSB struct.title).

Cognate side of the comparison reuses the ALREADY-FOLDED Stage-3 ipTM/DockQ values
for these same 8 targets (docs/implementation-parity-data/abag-pilot-stage3-final.json)
-- no re-fold needed for the "real pair" arm, only the decoy (never-before-folded)
pairs need new folds.

    python3 scripts/abag_decoy_build.py
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = f"{ROOT}/examples/abag_pilot"
OUT_DIR = f"{ROOT}/examples/abag_pilot_decoys"

# antigen identity (live RCSB struct.title, fetched this pass) -- kept here so the
# non-cognate pairing rationale is auditable, not just asserted.
ANTIGEN = {
    "9ck4": "L9 epitope scaffold (malaria CSP-family)",
    "9i5n": "human CD40 ligand",
    "9m72": "ASFV p15 (African swine fever virus)",
    "22ps": "human IgE-Fc",
    "9obn": "Pfs48/45 (malaria transmission-blocking antigen)",
    "9gfr": "collagen type II triple-helical peptide (THP59)",
    "9udq": "MPXV A35R (monkeypox virus)",
    "9jkr": "human GM-CSF",
}
# cyclic shift by 1: antibody_i paired with antigen_{i+1}. All 8 antigens are from
# distinct organisms/protein families (viral capsid/envelope, human cytokines/Fc,
# parasite surface antigens, structural collagen) -- no pair shares an antigen class,
# so no plausible real cross-reactivity.
ORDER = ["9ck4", "9i5n", "9m72", "22ps", "9obn", "9gfr", "9udq", "9jkr"]


def parse_chains(yaml_path):
    """Minimal extraction of {chain_id: sequence} -- these yamls are a flat,
    single-level `sequences: [{protein: {id, sequence}}, ...]` list with no nested
    structures, so a line-based parse avoids adding a yaml dependency for this
    one-off script."""
    chains = {}
    cur_id = None
    with open(yaml_path) as f:
        for line in f:
            m = re.match(r"\s*id:\s*(\S+)", line)
            if m:
                cur_id = m.group(1)
                continue
            m = re.match(r"\s*sequence:\s*(\S+)", line)
            if m and cur_id:
                chains[cur_id] = m.group(1)
                cur_id = None
    return chains


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = []
    n = len(ORDER)
    for i, ab_pdb in enumerate(ORDER):
        ag_pdb = ORDER[(i + 1) % n]
        ab_chains = parse_chains(f"{SRC_DIR}/{ab_pdb}_abag.yaml")
        ag_chains = parse_chains(f"{SRC_DIR}/{ag_pdb}_abag.yaml")
        decoy_id = f"decoy_{ab_pdb}ab_{ag_pdb}ag"
        lines = ["version: 1",
                 f"# Arm-3 specificity decoy: antibody from {ab_pdb} (real antigen: "
                 f"{ANTIGEN[ab_pdb]}) forced onto the UNRELATED antigen from {ag_pdb} "
                 f"({ANTIGEN[ag_pdb]}). Non-cognate by construction -- these two PDB "
                 "entries share no antigen class. Auto-generated for "
                 "flagship-abag-trust-validation Arm 3.",
                 "sequences:",
                 "  - protein:",
                 "      id: A",
                 f"      sequence: {ag_chains['A']}",
                 "  - protein:",
                 "      id: H",
                 f"      sequence: {ab_chains['H']}"]
        if "L" in ab_chains:
            lines += ["  - protein:", "      id: L", f"      sequence: {ab_chains['L']}"]
        out_path = f"{OUT_DIR}/{decoy_id}.yaml"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        pairs.append({"decoy_id": decoy_id, "antibody_source": ab_pdb, "antigen_source": ag_pdb,
                      "antibody_real_antigen": ANTIGEN[ab_pdb], "decoy_antigen": ANTIGEN[ag_pdb],
                      "yaml": out_path})
        print(f"wrote {out_path}")

    meta_path = f"{ROOT}/docs/implementation-parity-data/abag-arm3-decoy-pairs.json"
    with open(meta_path, "w") as f:
        json.dump({"method": "cyclic shift by 1 over 8 targets from distinct antigen "
                              "classes (viral/parasite/human-cytokine/structural), "
                              "following ipSAE's random-shuffle-across-entries decoy "
                              "convention (Dunbrack et al. bioRxiv 2025.02.10.637595) "
                              "adapted to keep antigen classes disjoint rather than "
                              "purely random, to avoid accidental real cross-reactivity.",
                   "pairs": pairs}, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
