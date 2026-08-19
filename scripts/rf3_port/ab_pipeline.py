"""A/B the RF3 host feature pipeline across environments.

Builds the AF3 transform pipeline standalone with fixed inference kwargs, runs it
on one input and dumps every tensor. Run it under two environments and diff, to
attribute a feature difference to the interpreter, biotite, rdkit or the seeding.

    PYTHONPATH=<foundry>/models/rf3/src:<foundry>/src \
        python scripts/rf3_port/ab_pipeline.py out.pt input.json

This builds the pipeline standalone rather than through RF3InferenceEngine, so it
is for attribution only. The committed fixtures under parity_artifacts/ are the
faithful-to-inference reference; those go through the engine.
"""
import json, os, random, sys

import numpy as np
import torch

from rf3.data.pipelines import build_af3_transform_pipeline
from rf3.utils.inference import prepare_inference_inputs_from_paths

out_path = sys.argv[1]
inp = sys.argv[2]

pipeline = build_af3_transform_pipeline(
    is_inference=True,
    protein_msa_dirs=[],
    rna_msa_dirs=[],
    n_recycles=10,
    diffusion_batch_size=5,
    undesired_res_names=[],
    template_noise_scales={"atomized": 1e-5, "not_atomized": 1e-5},
    allowed_chain_types_for_conditioning=None,
    p_give_polymer_ref_conf=0.0,
    p_give_non_polymer_ref_conf=0.0,
    p_dropout_ref_conf=0.0,
    use_element_for_atom_names_of_atomized_tokens=True,
    fallback_conformer_to_input_coords=True,
    run_confidence_head=True,
)

specs = prepare_inference_inputs_from_paths(
    inputs=os.path.abspath(inp), existing_outputs_dir=None, sharding_pattern=None,
    template_selection=None, ground_truth_conformer_selection=None, add_missing_atoms=True,
)
# Seed everything the engine seeds. RDKit ETKDG conformer embedding does not read
# torch RNG, so a torch-only seed leaves feats/ref_pos non-reproducible run to run.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
out = pipeline(specs[0].to_pipeline_input())

flat, plain = {}, {}
def split(o, p=""):
    k = p.rstrip("/")
    if isinstance(o, torch.Tensor):
        flat[k] = o
    elif isinstance(o, dict):
        for kk, vv in o.items():
            split(vv, f"{p}{kk}/")
    else:
        try:
            json.dumps(o); plain[k] = o
        except Exception:
            pass
split(out)
torch.save(flat, out_path)
with open(out_path + ".plain.json", "w") as fh:
    json.dump(plain, fh, sort_keys=True)
print(f"python {sys.version.split()[0]}: {len(flat)} tensors -> {out_path}")
