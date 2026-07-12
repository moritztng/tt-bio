"""P3 real-distribution golden: OF3 InputEmbedderAllAtom + MSAModuleEmbedder fed a real
example (via P1's build_openfold3_features), captured to extend ~/of3_ref_out.pkl.

Mirrors scripts/protenix_ref_forward.py's method (real weights + real features, forward
hooks / direct submodule calls -- not synthetic tensors) rather than of3_golden.py's
seeded-N(0,1) approach. See docs/openfold3-port.md status log tick 3: the 48-block stack
gate needs REAL trunk-scale (s, z), not off-manifold noise that makes the reference
trunk explode.

Adds these keys to the golden pkl:
  - "input_embedder_real": {"in": batch feat dict, "out": (s_input, s, z)}
  - "input_embedder_atom_enc_real": {"in": batch, "out": (ai, ql, cl, plm)} -- the
    InputEmbedder's atom-encoder sub-outputs (per-token ai BEFORE the s_inputs concat),
    for sub-component-granularity device PCC gating.
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

    # Atom-encoder sub-outputs (ai, ql, cl, plm) -- lets the device InputEmbedder leg be
    # PCC-gated at sub-component granularity (atom encoder vs glue linears) rather than only
    # at the combined s_input/s/z. ai is the per-token (N_token, c_token=384) aggregation
    # BEFORE the cat([ai, restype, profile, deletion_mean]) -> s_inputs.
    with torch.no_grad():
        ai, ql, cl, plm = ie.atom_attn_enc(batch=batch)
    print("input_embedder atom_attn_enc: ai", ai.shape, "ql", ql.shape, "cl", cl.shape,
          "plm", plm.shape, "ai std", float(ai.std()))

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
    inter["input_embedder_atom_enc_real"] = {
        "in": {k: v[0].clone() for k, v in batch.items()},
        "out": (ai[0].clone(), ql[0].clone(), cl[0].clone(), plm[0].clone()),
    }

    PF = dict(C.architecture.pairformer)
    stack = PairFormerStack(**PF).eval()
    stack.load_state_dict(sub(sd, "pairformer_stack"), strict=True)
    single_mask = batch["token_mask"]
    pair_mask = single_mask[..., None] * single_mask[..., None, :]
    # Run block-by-block (via the stack's own _prep_blocks, so this IS the reference
    # forward) to capture the 47-block PREFIX as well as the full 48-block output. P5
    # bisect localized the entire device z_pcc collapse to the LAST block, whose z-update
    # nearly cancels the accumulated ~std-134 residual down to ~std-30 (catastrophic
    # cancellation, ~10x rounding amplification -- CPU-bf16 hits the same 0.90 wall). The
    # 47-block prefix is the honest correctness gate (device tracks it to z_pcc>=0.97);
    # the full 48 stays xfail as a documented bf16-conditioning limit, not a port bug.
    with torch.no_grad():
        blocks = stack._prep_blocks(
            s=s_init.clone(), z=z_init.clone(), single_mask=single_mask, pair_mask=pair_mask,
            chunk_size=None, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
            use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False, _mask_trans=True,
        )
        s_cur, z_cur = s_init.clone(), z_init.clone()
        s_pre = z_pre = None
        nb = len(blocks)
        for bi, b in enumerate(blocks):
            s_cur, z_cur = b(s_cur, z_cur)
            if bi == nb - 2:  # after block 46 = 47-block prefix
                s_pre, z_pre = s_cur.clone(), z_cur.clone()
        ss, zs = s_cur, z_cur
    print("pairformer_stack_real:", ss.shape, zs.shape,
          "s_out std", float(ss.std()), "z_out std", float(zs.std()),
          "| prefix47 z_out std", float(z_pre.std()))
    inter["pairformer_stack_real"] = {
        "in": (s_init[0].clone(), z_init[0].clone()),
        "out": (ss[0].clone(), zs[0].clone()),
    }
    inter["pairformer_stack_prefix47"] = {
        "in": (s_init[0].clone(), z_init[0].clone()),
        "out": (s_pre[0].clone(), z_pre[0].clone()),
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
