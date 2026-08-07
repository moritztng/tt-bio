"""P8: extend ~/of3_ref_out.pkl with the OF3 reference TRUNK forward (run_trunk, AF3
Algorithm 1 lines 1-14) so the device trunk assembly is PCC-gated at sub-component
granularity -- the new top-level cycle glue (linear_z/layer_norm_z/linear_s/
layer_norm_s) in isolation, and the assembled cycle orchestration.

OF3 ``run_trunk`` cycle body (per cycle, num_cycles = num_recycles + 1 = 4):

    z = z_init + linear_z(layer_norm_z(z))          # top-level z glue
    z = z + template_embedder(batch, z, pair_mask)  # template embedder
    m, msa_mask = msa_module_embedder(batch, s_input)
    z = msa_module(m, z, msa_mask, pair_mask)       # MSA module
    s = s_init + linear_s(layer_norm_s(s))          # top-level s glue
    s, z = pairformer_stack(s, z, ...)              # 48-block Pairformer

s/z start at zeros; s_input, s_init, z_init come from the InputEmbedderAllAtom (already
captured under ``input_embedder_real``). The top-level ``linear_z``/``layer_norm_z``/
``linear_s``/``layer_norm_s`` are SEPARATE trunk weights (not the input embedder's
linears): affine LayerNorms (eps=1e-5) + bias-free ``init="final"`` Linears.

The MSA embedder subsamples stochastically (torch global RNG, no per-call generator) and
is re-called every cycle, so each cycle's ``m`` is a fresh draw. ``torch.manual_seed(0)``
is set before the InputEmbedder (which consumes no RNG), so the per-cycle subsample
sequence is reproducible and the cycle-0 ``m`` matches the one already stored under
``input_embedder_real["msa_out"]``.

Three of the trunk's sub-components are documented device-xfail on a known open device
precision / kernel gap (see docs/openfold3-port.md P8 tick 12): the template pair_stack
(throws on device at sub-tile head_dim=16), the MSA pair_stack (z_pcc~0.75), and the
pairformer stack's final-block z (z_pcc~0.66, cancellation). So a fully-device-gated
trunk output is NOT achievable; the golden's per-cycle intermediates let the device gate
the NEW glue code in isolation (feed golden z_prev/s_prev -> device glue -> compare) and
run the assembled trunk with the xfail pair-stack z substituted from the golden (gating
the device-runnable path: glue + pairformer s-track).

Adds key ``trunk_real``:
  num_cycles: int
  s_input:  [N, c_s_input=449]
  s_init:   [N, c_s=384]
  z_init:   [N, N, c_z=128]
  token_mask: [N]
  cycles: [ {                          # one dict per cycle (0 .. num_cycles-1)
      z_prev:          [N, N, c_z]      # z at cycle start (prev pairformer z; zeros @ c0)
      z_after_zglue:   [N, N, c_z]      # z_init + linear_z(layer_norm_z(z_prev))
      z_after_template:[N, N, c_z]      # + template_embedder
      m:               [N_seq, N, c_m]  # msa_module_embedder output (this cycle's subsample)
      z_after_msa:     [N, N, c_z]      # + msa_module
      s_prev:          [N, c_s]         # s at cycle start (prev pairformer s; zeros @ c0)
      s_after_sglue:   [N, c_s]         # s_init + linear_s(layer_norm_s(s_prev))
  }, ... ]
  s_trunk: [N, c_s]                     # final trunk single
  z_trunk: [N, N, c_z]                  # final trunk pair

Run with the CPU reference venv, NOT the tt-bio device env:
    /tmp/of3-venv/bin/python scripts/of3_trunk_golden.py
"""
import os, sys, pickle, collections.abc
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
QUERY_JSON = os.environ.get(
    "OF3_QUERY",
    os.path.join(OF3_REF, "examples/example_inference_inputs/query_ubiquitin.json"),
)
NUM_CYCLES = int(os.environ.get("OF3_TRUNK_CYCLES", "4"))  # = num_recycles(3) + 1


def _strip(o):
    if isinstance(o, torch.Tensor):
        return o
    if (isinstance(o, (dict, collections.abc.Mapping))
            or (hasattr(o, "items") and callable(getattr(o, "items")) and hasattr(o, "__getitem__"))):
        return {k: _strip(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_strip(v) for v in o)
    return o


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def to_batch(feat):
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
    from openfold3.core.model.latent.msa_module import MSAModuleStack
    from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
    from openfold3.core.model.primitives import LayerNorm, Linear

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    print("featurizing real example:", QUERY_JSON)
    qs = InferenceQuerySet.from_json(QUERY_JSON)
    query = next(iter(qs.queries.values()))
    feat = build_openfold3_features(query)
    batch = to_batch(feat)
    N = int(feat["token_mask"].shape[0])
    print("n_tokens:", N)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    # Reproduce run_trunk's RNG ordering: InputEmbedder (no RNG draw) is called first,
    # then the cycle loop's msa_module_embedder draws the per-cycle subsamples.
    torch.manual_seed(0)
    ie = InputEmbedderAllAtom(**C.architecture.input_embedder).eval()
    ie.load_state_dict(sub(sd, "input_embedder"), strict=True)
    with torch.no_grad():
        s_input, s_init, z_init = ie(batch=batch)
    print("input_embedder: s_input", tuple(s_input.shape), "s_init", tuple(s_init.shape),
          "z_init", tuple(z_init.shape))

    # Top-level trunk glue (separate trunk weights, not the input embedder's linears).
    ln_z = LayerNorm(128).eval()
    ln_z.load_state_dict({k: sd["layer_norm_z." + k] for k in ("weight", "bias")})
    lin_z = Linear(128, 128, bias=False).eval()
    lin_z.load_state_dict({"weight": sd["linear_z.weight"]})
    ln_s = LayerNorm(384).eval()
    ln_s.load_state_dict({k: sd["layer_norm_s." + k] for k in ("weight", "bias")})
    lin_s = Linear(384, 384, bias=False).eval()
    lin_s.load_state_dict({"weight": sd["linear_s.weight"]})

    te = TemplateEmbedderAllAtom(config=C.architecture.template).eval()
    te.load_state_dict(sub(sd, "template_embedder"), strict=True)

    me = MSAModuleEmbedder(**C.architecture.msa.msa_module_embedder).eval()
    me.load_state_dict(sub(sd, "msa_module_embedder"), strict=True)

    MSA_PF = dict(C.architecture.msa.msa_module)
    msa_stack = MSAModuleStack(**MSA_PF).eval()
    msa_stack.load_state_dict(sub(sd, "msa_module"), strict=True)

    PF = dict(C.architecture.pairformer)
    pairformer = PairFormerStack(**PF).eval()
    pairformer.load_state_dict(sub(sd, "pairformer_stack"), strict=True)

    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    single_mask = token_mask
    pair_mask_f = pair_mask.to(z_init.dtype)
    single_mask_f = single_mask.to(z_init.dtype)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    cycles = []
    with torch.no_grad():
        for c in range(NUM_CYCLES):
            z_prev = z.clone()
            z = z_init + lin_z(ln_z(z_prev))                         # z glue
            z_after_zglue = z.clone()
            z_template = te(batch=batch, z=z, pair_mask=pair_mask)
            z = z + z_template
            z_after_template = z.clone()
            m, msa_mask = me(batch=batch, s_input=s_input)           # this cycle's subsample
            msa_mask_f = msa_mask.to(z_init.dtype)
            z = msa_stack(m, z, msa_mask=msa_mask_f, pair_mask=pair_mask_f)
            z_after_msa = z.clone()
            s_prev = s.clone()
            s = s_init + lin_s(ln_s(s_prev))                         # s glue
            s_after_sglue = s.clone()
            s, z = pairformer(
                s=s, z=z, single_mask=single_mask_f, pair_mask=pair_mask_f,
                use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False,
                _mask_trans=True,
            )
            print(f"cycle {c}: z_prev std {float(z_prev.std()):.3f} "
                  f"z_after_zglue {float(z_after_zglue.std()):.3f} "
                  f"z_after_template {float(z_after_template.std()):.3f} "
                  f"z_after_msa {float(z_after_msa.std()):.3f} "
                  f"-> pairformer out s {float(s.std()):.3f} z {float(z.std()):.3f} "
                  f"m {tuple(m.shape)}")
            cycles.append({
                "z_prev": z_prev[0].clone(),
                "z_after_zglue": z_after_zglue[0].clone(),
                "z_after_template": z_after_template[0].clone(),
                "m": m[0].clone(),
                "z_after_msa": z_after_msa[0].clone(),
                "s_prev": s_prev[0].clone(),
                "s_after_sglue": s_after_sglue[0].clone(),
            })

    s_trunk, z_trunk = s[0].clone(), z[0].clone()
    print("trunk: s_trunk std", float(s_trunk.std()), "z_trunk std", float(z_trunk.std()),
          "num_cycles", NUM_CYCLES)

    rec = {
        "num_cycles": NUM_CYCLES,
        "s_input": s_input[0].clone(),
        "s_init": s_init[0].clone(),
        "z_init": z_init[0].clone(),
        "token_mask": token_mask[0].clone(),
        "cycles": cycles,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["trunk_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("wrote trunk_real: num_cycles", NUM_CYCLES, "N", N,
          "s_trunk", tuple(s_trunk.shape), "z_trunk", tuple(z_trunk.shape))


if __name__ == "__main__":
    main()
