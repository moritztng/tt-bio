"""P8: extend ~/of3_ref_out.pkl with the OF3 TemplateEmbedderAllAtom reference I/O so
the device template embedder (feature linears + 2-block AF2 pair stack + aggregate) is
PCC-gated at sub-component granularity.

OF3 ``TemplateEmbedderAllAtom`` (AF3 Algorithm 16) is:
  1. TemplatePairEmbedderAllAtom (feature embedder): 8 bias-free linears summed into
     ``a`` [N_templ, N, N, c_t=64] (dgram, pseudo_beta_mask, aatype_1/2, x/y/z unit
     vectors, backbone_mask), then ``t = linear_z(layer_norm_z(z))[...,None,:,:,:] + a``
     -> ``t_embed`` [N_templ, N, N, c_t].
  2. TemplatePairStack: 2 AF2 PairBlocks (tri_mul_out/in + tri_att_start/end +
     swiglu pair_transition, tri_mul_first=True) + a final stack layer_norm, run
     per-template (no cross-template interaction) -> ``t_stack`` [N_templ, N, N, c_t].
  3. Aggregate: ``z_template = linear_t(relu(sum_t(t_stack) / n_templ))`` [N, N, c_z=128].

The trunk feeds the template embedder the cycle's ``z = z_init + linear_z(layer_norm_z(z))``
(z starts at zeros). At cycle 0 this is ``z_init + linear_z(layer_norm_z(zeros))`` -- a
constant shift of z_init, reproduced here exactly from the trunk's top-level
layer_norm_z/linear_z weights (same LayerNorm/Linear classes, eps=1e-5). The mask
products (multichain/pseudo_beta/backbone_frame pair masks) are mask-derived, so the
per-template feature tensors ready to feed the device linears are captured here (same
discipline as the RefAtomFeatureEmbedder dlm/vlm/inv_sq_dists and the glue's relpos):
the device port is gated against the exact reference masks, isolating the device linear
precision from the mask logic.

Adds key ``template_embedder_real``:
  z:            [N, N, c_z=128]            (cycle-0 trunk z, the embedder input)
  pair_mask:    [N, N]
  feat: {                       # per-template, stacked [N_templ, ...] (dim 0 = template)
     distogram, pseudo_beta_pair_mask, restype_ti, restype_tj,
     unit_vec_x, unit_vec_y, unit_vec_z, backbone_frame_pair_mask
  }
  t_embed:  [N_templ, N, N, c_t=64]   (TemplatePairEmbedderAllAtom output)
  t_stack:  [N_templ, N, N, c_t=64]   (TemplatePairStack output, post final LN)
  z_template: [N, N, c_z=128]         (TemplateEmbedderAllAtom output)

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_template_embedder_golden.py
"""
import os, sys, pickle, collections.abc
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


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


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
    from openfold3.core.model.primitives import LayerNorm, Linear

    g = pickle.load(open(GOLD, "rb"))["intermediates"]["input_embedder_real"]
    s_input_ref, s_init_ref, z_init_ref = g["out"]
    b = g["in"]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in b.items()}
    z_init = z_init_ref.unsqueeze(0)  # [1, N, N, 128]
    N = z_init.shape[-2]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    # Cycle-0 trunk z = z_init + linear_z(layer_norm_z(zeros_like(z_init))). The trunk's
    # top-level layer_norm_z (128, affine) and bias-free linear_z (128->128) reproduce
    # the exact constant shift the reference feeds the template embedder at cycle 0.
    ln_z = LayerNorm(c_z := 128).eval()
    ln_z.load_state_dict({k: sd["layer_norm_z." + k] for k in ("weight", "bias")})
    lin_z = Linear(c_z, c_z, bias=False).eval()
    lin_z.load_state_dict({"weight": sd["linear_z.weight"]})
    with torch.no_grad():
        z_cycle0 = z_init + lin_z(ln_z(torch.zeros_like(z_init)))
    print("cycle-0 z:", tuple(z_cycle0.shape), "std", float(z_cycle0.std()),
          "delta-from-z_init std", float((z_cycle0 - z_init).std()))

    te = TemplateEmbedderAllAtom(config=C.architecture.template).eval()
    te.load_state_dict(sub(sd, "template_embedder"), strict=True)

    captured: dict = {}

    def embedder_hook(_mod, inp, out):
        # inp = (batch, z); out = t_embed [1, N_templ, N, N, c_t]
        captured["t_embed"] = out.detach().clone()

    def stack_hook(_mod, inp, out):
        # inp = (t, mask); out = t_stack [1, N_templ, N, N, c_t]
        captured["t_stack"] = out.detach().clone()

    te.template_pair_embedder.register_forward_hook(embedder_hook)
    te.template_pair_stack.register_forward_hook(stack_hook)

    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    with torch.no_grad():
        z_template = te(batch=batch, z=z_cycle0, pair_mask=pair_mask)
    print("template_embedder: t_embed", tuple(captured["t_embed"].shape),
          "t_stack", tuple(captured["t_stack"].shape),
          "z_template", tuple(z_template.shape),
          "z_template std", float(z_template.std()))

    # Per-template feature tensors ready to feed the device linears (mask products
    # precomputed on host; same isolation discipline as the other OF3 golden legs).
    asym = batch["asym_id"][0]                                  # [N]
    mcm = (asym[:, None] == asym[None, :])                     # [N, N]
    mcm = mcm[None, None, :, :, None]                          # [1,1,N,N,1]
    pbm = batch["template_pseudo_beta_mask"]                     # [1, N_t, N]
    pbpm = (pbm[..., None] * pbm[..., None, :])[..., None] * mcm  # [1, N_t, N, N, 1]
    bbfm = batch["template_backbone_frame_mask"]
    bfpm = (bbfm[..., None] * bbfm[..., None, :])[..., None] * mcm  # [1, N_t, N, N, 1]
    restype = batch["template_restype"].to(z_cycle0.dtype)       # [1, N_t, N, 32]
    n_tok = restype.shape[-2]
    rti = restype[..., None, :].expand(*restype.shape[:-2], -1, n_tok, -1)  # [1,N_t,N,N,32]
    rtj = restype[..., None, :, :].expand(*restype.shape[:-2], n_tok, -1, -1)
    uv = batch["template_unit_vector"].to(z_cycle0.dtype)       # [1, N_t, N, N, 3]
    ux, uy, uz = uv.unbind(dim=-1)                               # [1, N_t, N, N]

    feat = {
        "distogram": batch["template_distogram"][0],            # [N_t, N, N, 39]
        "pseudo_beta_pair_mask": pbpm[0],                        # [N_t, N, N, 1]
        "restype_ti": rti[0],                                    # [N_t, N, N, 32]
        "restype_tj": rtj[0],                                    # [N_t, N, N, 32]
        "unit_vec_x": ux[..., None][0],                          # [N_t, N, N, 1]
        "unit_vec_y": uy[..., None][0],                          # [N_t, N, N, 1]
        "unit_vec_z": uz[..., None][0],                          # [N_t, N, N, 1]
        "backbone_frame_pair_mask": bfpm[0],                     # [N_t, N, N, 1]
    }

    rec = {
        "z": z_cycle0[0].clone(),
        "pair_mask": pair_mask[0].clone(),
        "feat": feat,
        "t_embed": captured["t_embed"][0].clone(),              # [N_t, N, N, c_t]
        "t_stack": captured["t_stack"][0].clone(),              # [N_t, N, N, c_t]
        "z_template": z_template[0].clone(),                    # [N, N, c_z]
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["template_embedder_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added template_embedder_real: nt", feat["distogram"].shape[0],
          "N", N, "t_embed", tuple(rec["t_embed"].shape),
          "t_stack", tuple(rec["t_stack"].shape),
          "z_template", tuple(rec["z_template"].shape))


if __name__ == "__main__":
    main()
