"""Static metadata that drives the frontend: the model catalog, BoltzGen
protocols, tunable parameters (for progressive disclosure), and a handful of
ready-to-run example inputs.

Kept declarative so the UI can render forms generically and stay in sync with
the engine without hand-editing HTML.
"""

from __future__ import annotations

# --- Models offered for structure / affinity prediction (the `predict` path) ---
# Described as "improved successors of AlphaFold 3" per the launch positioning;
# we do not host AlphaFold itself.
# `caps` is the single source of truth for what each model can do; the frontend
# uses it to filter examples, disable builder controls / params, and block
# impossible inputs. Verified empirically: ESMFold folds one OR MORE protein
# chains, but silently drops ligands / nucleic acids and computes no affinity;
# ESMFold-2 Fast additionally has no MSA encoder.
#   msa · ligands · nucleic · affinity · constraints · multichain · modifications · potentials · pae
MODELS = [
    {
        "id": "boltz2",
        "name": "Boltz-2",
        "tagline": "Structure + binding affinity for proteins, nucleic acids and ligands.",
        "blurb": (
            "An improved open successor to AlphaFold 3. Folds protein / DNA / RNA / "
            "ligand complexes and predicts binding affinity. Uses an MSA."
        ),
        "needs_msa": True,
        "caps": ["msa", "ligands", "nucleic", "affinity", "constraints",
                 "multichain", "modifications", "potentials", "pae"],
    },
    {
        "id": "esmfold2",
        "name": "ESMFold-2",
        "tagline": "Protein folding, with or without an MSA.",
        "blurb": (
            "Language-model folding. No MSA required, but will use one if supplied "
            "for extra accuracy. Protein chains only — no ligands, nucleic acids, or affinity."
        ),
        "needs_msa": False,
        "caps": ["msa", "multichain", "modifications"],
    },
    {
        "id": "esmfold2-fast",
        "name": "ESMFold-2 Fast",
        "tagline": "The fastest fold — block-fp8, no MSA encoder.",
        "blurb": (
            "A lightweight ESMFold-2 variant for maximum throughput. Always folds "
            "single-sequence (no MSA); protein chains only. Accuracy typically very close to ESMFold-2."
        ),
        "needs_msa": False,
        "caps": ["multichain", "modifications"],
    },
]

# Input feature -> capability it requires. Used to detect impossible inputs.
FEATURE_CAPS = {
    "ligands": "ligands",
    "nucleic": "nucleic",
    "affinity": "affinity",
    "constraints": "constraints",
}

# --- BoltzGen design protocols (the `gen run` path) ---
PROTOCOLS = [
    {"id": "protein-anything", "name": "Protein binder", "blurb": "De-novo mini-protein binder against any target."},
    {"id": "peptide-anything", "name": "Peptide binder", "blurb": "Short peptide binder."},
    {"id": "nanobody-anything", "name": "Nanobody (VHH)", "blurb": "Single-domain antibody / nanobody binder."},
    {"id": "antibody-anything", "name": "Antibody", "blurb": "Antibody binder design."},
    {"id": "protein-small_molecule", "name": "Binder + affinity", "blurb": "Protein binder with a binding-affinity step."},
    {"id": "protein-redesign", "name": "Redesign", "blurb": "Re-design residues of an existing binder."},
]

DESIGN_STEPS = ["design", "inverse_folding", "folding", "analysis", "filtering"]

ACCELERATORS = ["tenstorrent", "gpu", "cpu"]

# --- Tunable parameters, surfaced under "Advanced settings" (progressive disclosure) ---
# Each entry: key, label, type (bool|int|float|enum|text), default, help, [choices].
# Only the controls a structural-biology user actually reaches for — no
# low-level / hardware / raw-CLI knobs (the platform abstracts those away).
PREDICT_PARAMS = [
    {"key": "use_msa_server", "label": "Generate MSA", "type": "bool", "default": True, "cap": "msa",
     "help": "Build a multiple-sequence alignment for the input. Required for Boltz-2; optional for ESMFold-2."},
    {"key": "fast", "label": "Fast mode", "type": "bool", "default": False,
     "help": "Lower precision for higher throughput — slightly less accurate."},
    {"key": "recycling_steps", "label": "Recycling steps", "type": "int", "default": 3, "help": "More can improve accuracy, at the cost of speed."},
    {"key": "sampling_steps", "label": "Sampling steps", "type": "int", "default": 200, "help": "Diffusion steps per structure."},
    {"key": "diffusion_samples", "label": "Number of predictions", "type": "int", "default": 1, "help": "How many structures to generate per target."},
    {"key": "output_format", "label": "Output format", "type": "enum", "default": "cif", "choices": ["cif", "pdb"], "help": "Structure file format."},
]

DESIGN_PARAMS = [
    {"key": "num_designs", "label": "Designs to generate", "type": "int", "default": 100,
     "help": "Total candidates generated before filtering. Production runs use ~10,000; lower is faster for a quick look."},
    {"key": "budget", "label": "Top designs to keep", "type": "int", "default": 30, "help": "How many ranked designs to report after filtering."},
    {"key": "fast", "label": "Fast mode", "type": "bool", "default": True, "help": "Lower precision for higher throughput — slightly less accurate."},
]

# --- Curated examples (also discoverable from the examples/ dir at runtime) ---
EXAMPLES = [
    {
        "id": "monomer",
        "kind": "predict",
        "name": "Single protein (monomer)",
        "builder": {"chains": [
            {"type": "protein", "id": "A",
             "sequence": "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG
""",
    },
    {
        "id": "complex",
        "kind": "predict",
        "name": "Protein complex (multimer)",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA"},
            {"type": "protein", "id": "B", "sequence": "MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA
  - protein:
      id: B
      sequence: MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA
""",
    },
    {
        "id": "affinity",
        "kind": "predict",
        "model": "boltz2",
        "requires": ["ligands", "affinity"],
        "name": "Protein–ligand binding affinity",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"},
            {"type": "ligand", "id": "B", "ligandMode": "smiles", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"}],
            "affinity": "B"},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
""",
    },
    {
        "id": "pocket",
        "kind": "predict",
        "model": "boltz2",
        "requires": ["ligands", "constraints"],
        "name": "Ligand with pocket constraint",
        "builder": {"chains": [
            {"type": "protein", "id": "A1", "sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLD"},
            {"type": "ligand", "id": "B1", "ligandMode": "ccd", "ccd": "SAH"}],
            "constraints": [{"kind": "pocket", "binder": "B1", "contacts": "A1:10, A1:12", "maxDistance": 6}]},
        "content": """version: 1
sequences:
  - protein:
      id: [A1]
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLD
  - ligand:
      id: [B1]
      ccd: SAH
constraints:
  - pocket:
      binder: B1
      contacts: [[A1, 10], [A1, 12]]
""",
    },
    {
        "id": "nanobody",
        "kind": "design",
        "protocol": "nanobody-anything",
        "name": "Nanobody against a target",
        "format": "yaml",
        "content": """entities:
  - protein:
      id: B
      sequence: 110..130
  - protein:
      id: A
      msa: empty
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHG
""",
    },
    {
        "id": "binder",
        "kind": "design",
        "protocol": "protein-anything",
        "name": "Mini-protein binder",
        "format": "yaml",
        "content": """entities:
  - protein:
      id: B
      sequence: 80..120
  - protein:
      id: A
      msa: empty
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSL
""",
    },
]


def catalog() -> dict:
    """The full catalog payload served to the frontend."""
    return {
        "models": MODELS,
        "protocols": PROTOCOLS,
        "design_steps": DESIGN_STEPS,
        "predict_params": PREDICT_PARAMS,
        "design_params": DESIGN_PARAMS,
        "examples": EXAMPLES,
    }
