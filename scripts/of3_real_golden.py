"""P3 real-distribution golden: OF3 InputEmbedderAllAtom + MSAModuleEmbedder fed a real
example (via P1's build_openfold3_features), captured to extend ~/of3_ref_out.pkl.

Mirrors scripts/protenix_ref_forward.py's method (real weights + real features, forward
hooks / direct submodule calls -- not synthetic tensors) rather than of3_golden.py's
seeded-N(0,1) approach. See docs/openfold3-port.md status log tick 3: the 48-block stack
gate needs REAL trunk-scale (s, z), not off-manifold noise that makes the reference
trunk explode.

Adds these keys to the golden pkl:
  - "input_embedder_real": {"in": batch feat dict, "out": (s_input, s, z)}
  - "pairformer_stack_real": {"in": (s, z), "out": stack(s, z)} -- the new stack gate input
  - "msa_block0_real": {"in": (m, z), "out": (m, z)} -- one MSAModuleBlock (has both the
    OPM and the PWA/transition MSA update; opm_first=True)
  - "msa_stack_real": {"in": (m, z), "out": z} -- full 4-block MSAModuleStack (returns
    z only, matching the reference model.py's own usage)

Run with the CPU reference venv, NOT the tt-bio device env:
    /tmp/of3-refvenv/bin/python scripts/of3_real_golden.py
"""
import os, sys, pickle
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.environ.get("OF3_CKPT", os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))
QUERY_JSON = os.environ.get("OF3_QUERY", os.path.join(OF3_REF, "examples/example_inference_inputs/query_ubiquitin.json"))
OUT = os.environ.get("OF3_GOLD", os.path.expanduser("~/of3_ref_out.pkl"))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def to_batch(feat):
    """Add a leading batch dim to every tensor feature; drop non-tensor entries
    (e.g. atom_array) the embedders don't consume."""
    out = {}
    for k, v in feat.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0)
    return out


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom, MSAModuleEmbedder,
    )
    from openfold3.core.model.latent.pairformer import PairFormerStack
    from openfold3.core.model.latent.msa_module import MSAModuleBlock, MSAModuleStack

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    print("featurizing real example:", QUERY_JSON)
    qs = InferenceQuerySet.from_json(QUERY_JSON)
    query = next(iter(qs.queries.values()))
    feat = build_openfold3_features(query)
    n_tokens = int(feat["token_mask"].shape[0])
    print("n_tokens:", n_tokens, "feat keys:", len(feat))
    batch = to_batch(feat)

    torch.manual_seed(0)  # MSAModuleEmbedder subsampling is stochastic
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    ie = InputEmbedderAllAtom(**C.architecture.input_embedder).eval()
    ie.load_state_dict(sub(sd, "input_embedder"), strict=True)
    with torch.no_grad():
        s_input, s_init, z_init = ie(batch=batch)
    print("input_embedder: s_input", s_input.shape, "s", s_init.shape, "z", z_init.shape,
          "s std", float(s_init.std()), "z std", float(z_init.std()))

    me = MSAModuleEmbedder(**C.architecture.msa.msa_module_embedder).eval()
    me.load_state_dict(sub(sd, "msa_module_embedder"), strict=True)
    with torch.no_grad():
        m, msa_mask = me(batch=batch, s_input=s_input)
    print("msa_module_embedder: m", m.shape, "msa_mask", msa_mask.shape)

    inter = {}
    inter["input_embedder_real"] = {
        "in": {k: v[0].clone() for k, v in batch.items()},
        "out": (s_input[0].clone(), s_init[0].clone(), z_init[0].clone()),
        "msa_out": (m[0].clone(), msa_mask[0].clone()),
    }

    PF = dict(C.architecture.pairformer)
    stack = PairFormerStack(**PF).eval()
    stack.load_state_dict(sub(sd, "pairformer_stack"), strict=True)
    single_mask = batch["token_mask"]
    pair_mask = single_mask[..., None] * single_mask[..., None, :]
    with torch.no_grad():
        ss, zs = stack(s_init.clone(), z_init.clone(), single_mask, pair_mask)
    print("pairformer_stack_real:", ss.shape, zs.shape,
          "s_out std", float(ss.std()), "z_out std", float(zs.std()))
    inter["pairformer_stack_real"] = {
        "in": (s_init[0].clone(), z_init[0].clone()),
        "out": (ss[0].clone(), zs[0].clone()),
    }

    msa_mask_b = msa_mask.to(z_init.dtype)
    pair_mask_b = pair_mask.to(z_init.dtype)

    import inspect
    MSA_PF = dict(C.architecture.msa.msa_module)
    blk0_params = set(inspect.signature(MSAModuleBlock.__init__).parameters) - {"self", "last_block"}
    blk0 = MSAModuleBlock(**{k: v for k, v in MSA_PF.items() if k in blk0_params}, last_block=False).eval()
    blk0.load_state_dict(sub(sd, "msa_module.blocks.0"), strict=True)
    with torch.no_grad():
        m0_out, z0_out = blk0(m.clone(), z_init.clone(), msa_mask=msa_mask_b, pair_mask=pair_mask_b)
    print("msa_block0_real: m_out std", float(m0_out.std()), "z_out std", float(z0_out.std()))
    inter["msa_block0_real"] = {
        "in": (m[0].clone(), z_init[0].clone()),
        "out": (m0_out[0].clone(), z0_out[0].clone()),
    }

    msa_stack = MSAModuleStack(**MSA_PF).eval()
    msa_stack.load_state_dict(sub(sd, "msa_module"), strict=True)
    with torch.no_grad():
        z_stack_out = msa_stack(m.clone(), z_init.clone(), msa_mask=msa_mask_b, pair_mask=pair_mask_b)
    print("msa_stack_real: z_out std", float(z_stack_out.std()))
    inter["msa_stack_real"] = {
        "in": (m[0].clone(), z_init[0].clone()),
        "out": z_stack_out[0].clone(),
    }

    gold = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {"intermediates": {}, "config": PF, "N": n_tokens}
    gold["intermediates"].update(inter)
    gold["N_real"] = n_tokens
    with open(OUT, "wb") as f:
        pickle.dump(gold, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
