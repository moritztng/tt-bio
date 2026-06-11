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
        "tagline": "Most capable — structure + binding affinity.",
        "blurb": (
            "Folds protein, DNA, RNA and ligand complexes and predicts binding "
            "affinity. Uses a multiple-sequence alignment (MSA) for best accuracy. "
            "The most capable model — choose it when in doubt, or whenever you need "
            "ligands, nucleic acids, affinity or constraints."
        ),
        "needs_msa": True,
        "caps": ["msa", "ligands", "nucleic", "affinity", "constraints",
                 "multichain", "modifications", "potentials", "pae"],
    },
    {
        "id": "esmfold2",
        "name": "ESMFold-2",
        "tagline": "Fast protein folding, MSA optional.",
        "blurb": (
            "Language-model folding — no MSA required, though it will use one if "
            "supplied for extra accuracy. Protein chains only: no ligands, nucleic "
            "acids or affinity. A quick, lightweight choice for routine protein structures."
        ),
        "needs_msa": False,
        "caps": ["msa", "multichain", "modifications"],
    },
    {
        "id": "esmfold2-fast",
        "name": "ESMFold-2 Fast",
        "tagline": "The fastest fold — no MSA encoder.",
        "blurb": (
            "ESMFold-2 tuned for maximum throughput: always single-sequence, protein "
            "chains only. Accuracy is typically very close to ESMFold-2 — the model to "
            "reach for when screening thousands of sequences at once."
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
    # --- Fold / predict: a capability ladder, all real, all small enough to be quick ---
    {
        "id": "monomer",
        "kind": "predict",
        "name": "Ubiquitin (monomer)",
        "blurb": "Human ubiquitin (76 aa) — the classic small single-domain fold. Works on every model.",
        "builder": {"chains": [
            {"type": "protein", "id": "A",
             "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
""",
    },
    {
        "id": "complex",
        "kind": "predict",
        "name": "Insulin (two-chain complex)",
        "blurb": "Human insulin — its A (21 aa) and B (30 aa) chains fold as a small two-chain complex.",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "GIVEQCCTSICSLYQLENYCN"},
            {"type": "protein", "id": "B", "sequence": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: GIVEQCCTSICSLYQLENYCN
  - protein:
      id: B
      sequence: FVNQHLCGSHLVEALYLVCGERGFFYTPKT
""",
    },
    {
        "id": "affinity",
        "kind": "predict",
        "model": "boltz2",
        "requires": ["ligands", "affinity"],
        "name": "BRD4 + JQ1 (binding affinity)",
        "blurb": "The BRD4 bromodomain (BD1) with its inhibitor JQ1 — predicts the protein–ligand binding affinity.",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "RQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE"},
            {"type": "ligand", "id": "B", "ligandMode": "smiles", "smiles": "CC1=C(SC2=C1C(=N[C@H](C3=NN=C(N32)C)CC(=O)OC(C)(C)C)C4=CC=C(C=C4)Cl)C"}],
            "affinity": "B"},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: RQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE
  - ligand:
      id: B
      smiles: 'CC1=C(SC2=C1C(=N[C@H](C3=NN=C(N32)C)CC(=O)OC(C)(C)C)C4=CC=C(C=C4)Cl)C'
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
        "name": "HIV-1 protease + darunavir (pocket)",
        "blurb": "HIV-1 protease with the drug darunavir, restrained to the catalytic Asp25 — a pocket constraint.",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"},
            {"type": "ligand", "id": "B", "ligandMode": "smiles", "smiles": "CC(C)CN(C[C@H]([C@H](CC1=CC=CC=C1)NC(=O)O[C@H]2CO[C@@H]3[C@H]2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N"}],
            "constraints": [{"kind": "pocket", "binder": "B", "contacts": "A:25", "maxDistance": 6}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF
  - ligand:
      id: B
      smiles: 'CC(C)CN(C[C@H]([C@H](CC1=CC=CC=C1)NC(=O)O[C@H]2CO[C@@H]3[C@H]2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N'
constraints:
  - pocket:
      binder: B
      contacts: [[A, 25]]
      max_distance: 6
""",
    },
    {
        "id": "dna_complex",
        "kind": "predict",
        "model": "boltz2",
        "requires": ["nucleic"],
        "name": "Homeodomain–DNA complex",
        "blurb": "The engrailed homeodomain bound to its double-stranded DNA site — a protein–DNA complex.",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "DEKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKKS"},
            {"type": "dna", "id": "B", "sequence": "GCGGTAATTACCGC"},
            {"type": "dna", "id": "C", "sequence": "GCGGTAATTACCGC"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: DEKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKKS
  - dna:
      id: B
      sequence: GCGGTAATTACCGC
  - dna:
      id: C
      sequence: GCGGTAATTACCGC
""",
    },
    # --- Design / BoltzGen: one per major protocol, each against a real, recognizable target ---
    {
        "id": "binder",
        "kind": "design",
        "protocol": "protein-anything",
        "name": "Mini-binder vs PD-L1",
        "blurb": "De-novo mini-protein binder against the PD-L1 IgV domain — a flagship immuno-oncology target.",
        "builder": {
            "binderId": "B", "targetId": "A", "lengthRange": "80..120",
            "target": "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRIT",
        },
        "content": """entities:
  - protein:
      id: B
      sequence: 80..120
  - protein:
      id: A
      msa: empty
      sequence: FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRIT
""",
    },
    {
        "id": "peptide",
        "kind": "design",
        "protocol": "peptide-anything",
        "name": "Peptide vs MDM2",
        "blurb": "Short peptide binder against MDM2's p53-binding cleft — a classic protein–protein-interaction inhibitor.",
        "builder": {
            "binderId": "B", "targetId": "A", "lengthRange": "12..25",
            "target": "SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN",
        },
        "content": """entities:
  - protein:
      id: B
      sequence: 12..25
  - protein:
      id: A
      msa: empty
      sequence: SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN
""",
    },
    {
        "id": "nanobody",
        "kind": "design",
        "protocol": "nanobody-anything",
        "name": "Nanobody vs lysozyme",
        "blurb": "Single-domain antibody (VHH) against hen egg-white lysozyme — the original model nanobody antigen.",
        "builder": {
            "binderId": "B", "targetId": "A", "lengthRange": "110..130",
            "target": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        },
        "content": """entities:
  - protein:
      id: B
      sequence: 110..130
  - protein:
      id: A
      msa: empty
      sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
""",
    },
    {
        "id": "sm_binder",
        "kind": "design",
        "protocol": "protein-small_molecule",
        "name": "Binder vs caffeine",
        "blurb": "Design a protein that binds a small molecule (caffeine) — the small-molecule-target workflow.",
        "builder": {
            "binderId": "B", "targetId": "A", "lengthRange": "80..120",
            "ligand": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "ligandMode": "smiles",
        },
        "content": """entities:
  - protein:
      id: B
      sequence: 80..120
  - ligand:
      id: A
      smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
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
