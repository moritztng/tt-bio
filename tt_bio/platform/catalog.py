"""Static metadata that drives the frontend: the model catalog, BoltzGen
protocols, tunable parameters (for progressive disclosure), and a handful of
ready-to-run example inputs.

Kept declarative so the UI can render forms generically and stay in sync with
the engine without hand-editing HTML.
"""

from __future__ import annotations

# --- Free-demo limits -------------------------------------------------------
# This platform runs as a free public demo on shared sovereign compute, so every
# input is bounded to keep one user from blocking everyone else. The limits are
# the single source of truth: enforced server-side on the *parsed* input (see
# tt_bio.platform.limits) and surfaced in the UI so users understand why.
LIMITS = {
    "max_residues": 1024,            # total protein/nucleic residues in one structure
    "max_chains_per_complex": 10,    # chains (incl. id-list copies) in one structure
    "max_complexes": 10,             # structures/targets per submission (compose + bulk)
    "max_ligands_per_complex": 10,
    "max_constraints_per_complex": 20,
    "max_designs": 10,               # binders to generate (design)
    "max_budget": 10,                # ranked designs to keep
    "max_recycling_steps": 10,
    "max_sampling_steps": 500,
    "max_diffusion_samples": 5,
    "max_content_chars": 50_000,     # raw size of one target/spec (parse-independent guard)
    # Capacity guards (shared, unauthenticated demo): bound how much work can
    # pile up so one visitor can't exhaust the queue (memory) or disk, and so no
    # single visitor can monopolize the shared fleet or spam it with requests.
    "max_active_jobs": 64,           # queued + running across ALL visitors -> 429 when full
    "max_active_jobs_per_session": 3,  # queued + running for ONE cookie session -> 429 (fairness)
    "max_submits_per_min": 12,       # submissions/min per cookie session -> 429 (anti rapid-fire)
    # Per-IP caps: the public abuse backstop (cookie sessions reset freely; the
    # real client IP from X-Forwarded-For does not). NAT-tolerant (offices share
    # an IP) yet well under the global cap so no one network monopolizes the fleet.
    "max_active_jobs_per_ip": 8,     # queued + running for ONE client IP -> 429
    "max_submits_per_min_per_ip": 40,  # submissions/min per client IP -> 429
    "max_retained_jobs": 200,        # finished jobs kept for browsing; oldest auto-evicted
    # Watchdog: a job that hangs (wedged device, stalled download, model bug)
    # would otherwise hold its devices forever and block the shared fleet. These
    # are generous ceilings — demo-bounded inputs finish well under them — so a
    # job that exceeds one is treated as stuck, killed, and its devices freed.
    "max_runtime_predict_s": 1500,   # 25 min
    "max_runtime_design_s": 2700,    # 45 min
    # Stall ceiling: a healthy run streams progress to its log continuously
    # (per-target / per-stage events). If the log stays frozen this long the
    # run is almost certainly wedged on a stuck device — fail it fast so the
    # slot turns over in minutes instead of waiting out the full runtime cap.
    "max_stall_s": 600,              # 10 min with no log progress (predict)
    "max_stall_design_s": 1200,      # 20 min for design — its central filter runs quietly
    # Embed (ESMC) is a language-model forward pass only — no diffusion, no MSA —
    # so it finishes in seconds and gets much tighter watchdog ceilings.
    "max_runtime_embed_s": 300,       # 5 min
    "max_stall_embed_s": 120,         # 2 min with no log progress
    "max_embed_sequences": 50,        # sequences per embed submission
    "max_embed_sequence_residues": 2000,  # residues per sequence (LM only, no folding: higher cap than max_residues is safe)
}
DEMO_NOTE = (
    "This is a free public demo on shared compute, so inputs are capped "
    "(e.g. {max_residues} residues per structure, {max_complexes} structures per run, "
    "{max_designs} binders per design). The full platform has no such limits."
).format(**LIMITS)

# --- Models offered for structure / affinity prediction (the `predict` path) ---
# Positioned as "open models beyond Google DeepMind's AlphaFold 3" per the launch
# messaging; we do not host AlphaFold itself.
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
        "tagline": "Most capable, with affinity.",
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
        "tagline": "Fast, MSA optional.",
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
        "tagline": "Fastest, no MSA.",
        "blurb": (
            "ESMFold-2 tuned for maximum throughput: always single-sequence, protein "
            "chains only. Accuracy is typically very close to ESMFold-2 — the model to "
            "reach for when screening thousands of sequences at once."
        ),
        "needs_msa": False,
        "caps": ["multichain", "modifications"],
    },
    {
        "id": "protenix-v2",
        "name": "Protenix-v2",
        "tagline": "Strong at antibody-antigen, no affinity.",
        "blurb": (
            "An AlphaFold3-family model (Pairformer trunk + atom diffusion) running "
            "fully on-device. Folds multi-chain complexes of proteins, nucleic acids "
            "(RNA/DNA) and ligands (CCD codes or SMILES), using a multiple-sequence "
            "alignment by default for best accuracy (turn it off to fold "
            "single-sequence). Reports PAE/PDE and per-atom pLDDT confidence. For "
            "binding affinity, use Boltz-2."
        ),
        "needs_msa": False,   # MSA optional (single-sequence works) ...
        "msa_default": True,  # ... but default it ON, since it sharpens accuracy a lot
        # Full multimodal AF3 port: real multi-chain complexes (not chimeric
        # concatenation), RNA/DNA and ligand (CCD + SMILES) featurization with
        # reference parity, and PAE/PDE + per-atom pLDDT output. Affinity prediction
        # stays Boltz-2-only.
        "caps": ["msa", "ligands", "nucleic", "multichain", "pae"],
    },
    # OpenDDE is one model family with two checkpoints: a general protein-complex
    # checkpoint and an antibody-antigen checkpoint. They share the `family` label
    # so the catalog (and /v1/models) presents them as one product line with two
    # variants, not two unrelated models. Both are protein-only (no ligands,
    # nucleic acids or affinity) and use an MSA by default.
    {
        "id": "opendde",
        "family": "opendde",
        "name": "OpenDDE — General",
        "tagline": "General complex docking.",
        "blurb": (
            "OpenDDE with its general protein-complex checkpoint. Co-folds "
            "multi-chain protein complexes on the Protenix-v2 stack plus a "
            "structural-token expander, using an MSA by default. Protein-only (no "
            "ligands, nucleic acids or affinity). For binding affinity, use "
            "Boltz-2; for ligands or DNA/RNA, use Protenix-v2 or Boltz-2."
        ),
        "needs_msa": False,
        "msa_default": True,
        "caps": ["msa", "multichain"],
    },
    {
        "id": "opendde-abag",
        "family": "opendde",
        "name": "OpenDDE — Antibody-Antigen",
        "tagline": "Antibody-antigen checkpoint.",
        "blurb": (
            "OpenDDE with its antibody-antigen checkpoint selected. Co-folds an "
            "antibody (Fab heavy/light) with its antigen on the Protenix-v2 stack, "
            "using an MSA by default. Protein-only, no affinity. Accuracy is "
            "verified to match the reference OpenDDE implementation: strong on "
            "standard antibody-antigen complexes, and it shares the reference's "
            "own limitation on some hard targets (a checkpoint characteristic, not "
            "a port defect) — so don't expect uniformly high accuracy on every "
            "antibody-antigen input."
        ),
        "needs_msa": False,
        "msa_default": True,
        "caps": ["msa", "multichain"],
    },
]

# --- Models offered for protein-language-model embeddings (the `embed` path) ---
# ESMC computes per-residue and pooled embeddings from its language-model trunk
# alone — no structure, no MSA. Useful for search, clustering, and as features
# for downstream models.
EMBED_MODELS = [
    {
        "id": "esmc-300m",
        "name": "ESMC 300M",
        "tagline": "Smallest, fastest.",
        "blurb": "The smallest ESMC variant — quickest embeddings, still a strong "
                 "general-purpose protein representation.",
    },
    {
        "id": "esmc-600m",
        "name": "ESMC 600M",
        "tagline": "Balanced default.",
        "blurb": "A larger ESMC trunk with a better representation than 300M, at "
                 "modest extra cost. The default choice.",
    },
    {
        "id": "esmc-6b",
        "name": "ESMC 6B",
        "tagline": "Largest, most accurate.",
        "blurb": "The full-size ESMC model — the strongest embeddings, at higher "
                 "compute cost per sequence.",
    },
]

# --- BoltzGen design protocols (the `gen run` path) ---
PROTOCOLS = [
    {"id": "protein-anything", "name": "Protein binder", "blurb": "De-novo mini-protein binder against any target."},
    {"id": "peptide-anything", "name": "Peptide binder", "blurb": "Short peptide binder."},
    {"id": "nanobody-anything", "name": "Nanobody (VHH)", "blurb": "Single-domain antibody / nanobody binder."},
    {"id": "antibody-anything", "name": "Antibody", "blurb": "Antibody binder design."},
    {"id": "protein-small_molecule", "name": "Binder + affinity", "blurb": "Protein binder with a binding-affinity step."},
    {"id": "protein-redesign", "name": "Redesign", "blurb": "Re-design residues of an existing binder."},
]

# Coarse pipeline stages a BoltzGen design run moves through, in order. Single
# source of truth — the job engine imports this for progress/stage detection.
DESIGN_STEPS = ["design", "inverse_folding", "folding", "analysis", "filtering"]

# --- Tunable parameters, surfaced under "Advanced settings" (progressive disclosure) ---
# Each entry: key, label, type (bool|int|float|enum|text), default, help, [choices].
# Only the controls a structural-biology user actually reaches for — no
# low-level / hardware / raw-CLI knobs (the platform abstracts those away).
PREDICT_PARAMS = [
    {"key": "use_msa_server", "label": "Generate MSA", "type": "bool", "default": True, "cap": "msa",
     "help": "Build a multiple-sequence alignment (MSA) of related sequences for the input. Required for Boltz-2 and Protenix; optional for ESMFold-2."},
    {"key": "fast", "label": "Fast mode", "type": "bool", "default": True,
     "help": "Higher throughput — may be slightly less accurate."},
    {"key": "recycling_steps", "label": "Recycling steps", "type": "int", "default": 3, "min": 1, "max": LIMITS["max_recycling_steps"], "help": "More can improve accuracy, at the cost of speed."},
    {"key": "sampling_steps", "label": "Sampling steps", "type": "int", "default": 200, "min": 10, "max": LIMITS["max_sampling_steps"], "help": "Diffusion steps per structure."},
    {"key": "diffusion_samples", "label": "Number of predictions", "type": "int", "default": 1, "min": 1, "max": LIMITS["max_diffusion_samples"], "help": "How many structures to generate per target."},
    {"key": "output_format", "label": "Output format", "type": "enum", "default": "cif", "choices": ["cif", "pdb"], "help": "Structure file format."},
]

DESIGN_PARAMS = [
    {"key": "num_designs", "label": "Designs to generate", "type": "int", "default": 10,
     "min": 1, "max": LIMITS["max_designs"],
     "help": "Binders to generate before filtering."},
    {"key": "budget", "label": "Top designs to keep", "type": "int", "default": 10,
     "min": 1, "max": LIMITS["max_budget"],
     "help": "How many ranked designs to report after filtering."},
    {"key": "fast", "label": "Fast mode", "type": "bool", "default": True, "help": "Higher throughput — may be slightly less accurate."},
]

EMBED_PARAMS = [
    {"key": "pool", "label": "Pooling", "type": "enum", "default": "mean",
     "choices": ["mean", "max", "cls"],
     "help": "How per-residue embeddings are combined into one fixed-size vector per sequence."},
    {"key": "format", "label": "Output format", "type": "enum", "default": "npz",
     "choices": ["npz", "parquet"],
     "help": "npz: per-residue + pooled embeddings, one file per sequence. parquet: pooled vectors only, one table."},
    {"key": "fast", "label": "Fast mode", "type": "bool", "default": False,
     "help": "Higher throughput — may be slightly less accurate."},
]

# --- Curated examples (also discoverable from the examples/ dir at runtime) ---
EXAMPLES = [
    # --- Fold / predict: a capability ladder, all real, all small enough to be quick ---
    {
        "id": "monomer",
        "kind": "predict",
        "name": "Single protein",
        "blurb": "Fold one protein chain — here Aequorea victoria GFP (238 aa), the iconic β-barrel. Works on every model.",
        "builder": {"chains": [
            {"type": "protein", "id": "A",
             "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
""",
    },
    {
        "id": "complex",
        "kind": "predict",
        "name": "Protein complex",
        "blurb": "Fold a multi-chain complex — here the textbook barnase (110 aa) · barstar (90 aa) enzyme–inhibitor pair.",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"},
            {"type": "protein", "id": "B", "sequence": "MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
  - protein:
      id: B
      sequence: MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS
""",
    },
    {
        "id": "affinity",
        "kind": "predict",
        "model": "boltz2",
        "requires": ["ligands", "affinity"],
        "name": "Protein–ligand affinity",
        "blurb": "Predict protein–ligand binding affinity — here the BRD4 bromodomain with its inhibitor JQ1.",
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
        "name": "Ligand in a pocket",
        "blurb": "Steer a ligand into a binding pocket — here the drug darunavir at HIV-1 protease's catalytic Asp25.",
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
        "name": "Protein–DNA complex",
        "blurb": "Fold a protein bound to DNA — here the p53 tumour-suppressor DNA-binding domain on its response element.",
        "builder": {"chains": [
            {"type": "protein", "id": "A", "sequence": "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNT"},
            {"type": "dna", "id": "B", "sequence": "GGGCATGCCCGGGCATGCCC"},
            {"type": "dna", "id": "C", "sequence": "GGGCATGCCCGGGCATGCCC"}]},
        "content": """version: 1
sequences:
  - protein:
      id: A
      sequence: SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNT
  - dna:
      id: B
      sequence: GGGCATGCCCGGGCATGCCC
  - dna:
      id: C
      sequence: GGGCATGCCCGGGCATGCCC
""",
    },
    # --- Design / BoltzGen: one per major protocol, each against a real, recognizable target ---
    {
        "id": "binder",
        "kind": "design",
        "protocol": "protein-anything",
        "name": "Protein binder",
        "blurb": "Design a de-novo mini-protein binder — here against the PD-L1 IgV domain, a flagship immuno-oncology target.",
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
        "name": "Peptide binder",
        "blurb": "Design a short peptide binder — here against MDM2's p53-binding cleft, a classic protein–protein-interaction inhibitor.",
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
        "name": "Nanobody (VHH)",
        "blurb": "Design a single-domain antibody (VHH) — here against hen egg-white lysozyme, the original model nanobody antigen.",
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
        "name": "Small-molecule binder",
        "blurb": "Design a protein that binds a small molecule — here caffeine. The small-molecule-target workflow.",
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


# --- Presentation shape for the web UI's main model selection ---------------
# The flat :data:`MODELS` above is the single source of truth for what's
# dispatchable: limits/jobs validate against it (so both OpenDDE checkpoint ids
# stay directly submittable to /v1/predictions), and /v1/models lists it verbatim
# for API/CLI users. The web UI's *main model selection* is a different surface —
# a human picks a model card — and there OpenDDE should appear ONCE, with its two
# checkpoints reachable via a picker on that single card. This collapses entries
# that share a ``family`` (e.g. the two OpenDDE checkpoints) into one card with a
# nested ``checkpoints`` list, leaving every other model as-is. It is purely
# visual: the concrete dispatch id a card submits is one of the original flat
# ids, so no model_id is ever asked to switch checkpoints per-request (that would
# force an engine resident-model reload — see memory
# japanfold-catalog-collapse-checkpoint-variant-trap).
def _checkpoint_label(name: str) -> str:
    # "OpenDDE — General" -> "General" (split on any dash kind, take the last part).
    for sep in ("—", "–", "-", "—"):
        if sep in name:
            tail = name.rsplit(sep, 1)[-1].strip()
            if tail:
                return tail
    return name


def _presentation_models() -> list[dict]:
    """MODELS with same-``family`` entries collapsed into one card with a
    ``checkpoints`` picker. Other entries pass through unchanged."""
    out: list[dict] = []
    families: dict[str, list[dict]] = {}
    for m in MODELS:
        fam = m.get("family")
        if not fam:
            out.append(m)
            continue
        families.setdefault(fam, []).append(m)
    for fam, group in families.items():
        # The family card inherits the shared dispatch metadata (caps, needs_msa,
        # msa_default, family) from the first member; the two OpenDDE checkpoints
        # agree on all of these. The card-level name/tagline/blurb describe the
        # family; each checkpoint keeps its own verbatim blurb (incl. the accuracy
        # notes) so nothing is reworded.
        base = group[0]
        # "OpenDDE — General" -> "OpenDDE": the family name is the shared prefix
        # before the checkpoint suffix.
        card_name = base["name"].split("—", 1)[0].rstrip(" –-").strip() or base["name"]
        card = {
            "id": base["id"],
            "family": fam,
            "name": card_name,
            "tagline": "Protein-complex docking — pick a checkpoint.",
            "blurb": (
                "OpenDDE co-folds multi-chain protein complexes on the Protenix-v2 "
                "stack plus a structural-token expander, using an MSA by default. "
                "Protein-only (no ligands, nucleic acids or affinity). Pick the "
                "checkpoint that matches your input below — for binding affinity, "
                "use Boltz-2; for ligands or DNA/RNA, use Protenix-v2 or Boltz-2."
            ),
            "needs_msa": base.get("needs_msa", False),
            "msa_default": base.get("msa_default", False),
            "caps": list(base.get("caps", [])),
            "checkpoints": [
                {
                    "id": m["id"],
                    "label": _checkpoint_label(m["name"]),
                    "name": m["name"],
                    "tagline": m.get("tagline", ""),
                    "blurb": m.get("blurb", ""),
                }
                for m in group
            ],
        }
        out.append(card)
    return out


def catalog() -> dict:
    """The full catalog payload served to the frontend."""
    return {
        "models": _presentation_models(),
        "protocols": PROTOCOLS,
        "predict_params": PREDICT_PARAMS,
        "design_params": DESIGN_PARAMS,
        "embed_models": EMBED_MODELS,
        "embed_params": EMBED_PARAMS,
        "examples": EXAMPLES,
        "limits": LIMITS,
        "demo_note": DEMO_NOTE,
    }
