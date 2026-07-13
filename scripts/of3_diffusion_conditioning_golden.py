"""P8: extend ~/of3_ref_out.pkl with the OF3 DiffusionConditioning leg (Algorithm 21)
so the device conditioning (pair/single linears + Fourier noise embedding + 2x SwiGLU
transition on each of s/z) is PCC-gated in isolation.

Reuses the already-captured real-distribution trunk tensors from the pkl as the
conditioning inputs (no full trunk re-run): ``si_input`` from ``input_embedder_real/out``,
``si_trunk``/``zij_trunk`` from ``pairformer_stack_real/out`` (the 48-block Pairformer
output -- a real trunk-scale single/pair representation, exactly what the conditioning
consumes in one recycle). ``t`` is a real noise level (``s_max`` from the noise schedule,
the initial sampling sigma), so the Fourier noise embedding is on-manifold.

The mask-derived relpos (``relpos_complex``, 139-dim) and the Fourier embedding output
(``n_emb``, 256-dim) are captured via forward hooks (on ``linear_z`` input and
``fourier_emb`` output), so the device port is gated against the exact reference artifacts
-- isolating the device linear/LN/SwiGLU precision from the relpos/Fourier host math, the
same discipline as the other OF3 golden legs.

Adds key ``diffusion_conditioning_real``:
  t:         float (noise level)
  si_input:  [N_token, 449]
  si_trunk:  [N_token, 384]
  zij_trunk: [N_token, N_token, 128]
  relpos:    [N_token, N_token, 139]   (reference relpos_complex, dc's max_relative_idx/chain)
  n_emb:     [256]                     (post-Fourier noise embedding, the linear_n input)
  token_mask:[N_token]
  si_ref:    [N_token, 384]            (conditioned single)
  zij_ref:   [N_token, N_token, 128]   (conditioned pair)

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_diffusion_conditioning_golden.py
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


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning

    inter = pickle.load(open(GOLD, "rb"))["intermediates"]
    ie = inter["input_embedder_real"]
    pf = inter["pairformer_stack_real"]
    s_input_ref, _, _ = ie["out"]
    si_trunk_ref, zij_trunk_ref = pf["out"]
    b = ie["in"]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in b.items()}
    si_input = s_input_ref.unsqueeze(0)
    si_trunk = si_trunk_ref.unsqueeze(0)
    zij_trunk = zij_trunk_ref.unsqueeze(0)
    token_mask = batch["token_mask"]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dc_cfg = dict(C.architecture.diffusion_module.diffusion_conditioning)
    dc = DiffusionConditioning(**dc_cfg).eval()
    dc.load_state_dict({k[len("diffusion_module.diffusion_conditioning."):]: v
                        for k, v in sd.items()
                        if k.startswith("diffusion_module.diffusion_conditioning.")}, strict=True)

    sigma_data = float(dc.sigma_data)
    # Real noise level: s_max from the AF3 noise schedule (the initial sampling sigma).
    t = torch.tensor(float(C.architecture.noise_schedule.s_max), dtype=torch.float32)

    captured: dict = {}

    def lnz_hook(_module, inp):
        # linear_z input = layer_norm_z(cat([zij_trunk, relpos])) -- NO, linear_z takes the
        # LN'd cat. Hook the LN-then-linear chain at linear_z input -> cat is its input.
        # Actually linear_z input IS layer_norm_z(cat). To get the raw relpos, hook
        # layer_norm_z input instead. Use a separate hook below.
        pass

    def lnz_in_hook(_module, inp):
        captured["cat"] = inp[0].detach().clone()  # [1, N, N, 267]

    def fourier_hook(_module, _inp, out):
        captured["n_emb"] = out.detach().clone()  # [1, 1, 256]

    dc.layer_norm_z.register_forward_pre_hook(lnz_in_hook)
    dc.fourier_emb.register_forward_hook(fourier_hook)

    with torch.no_grad():
        si_ref, zij_ref = dc(
            batch=batch, t=t, si_input=si_input, si_trunk=si_trunk,
            zij_trunk=zij_trunk, use_conditioning=True,
        )

    relpos = captured["cat"][0, ..., 128:].clone()   # drop zij_trunk -> 139-dim relpos
    n_emb = captured["n_emb"].reshape(-1).clone()  # [256]
    assert relpos.shape[-1] == 139, relpos.shape
    assert n_emb.shape[-1] == 256, n_emb.shape

    rec = {
        "t": float(t),
        "si_input": s_input_ref.clone(),
        "si_trunk": si_trunk_ref.clone(),
        "zij_trunk": zij_trunk_ref.clone(),
        "relpos": relpos,
        "n_emb": n_emb,
        "token_mask": token_mask[0].clone(),
        "si_ref": si_ref[0].clone(),
        "zij_ref": zij_ref[0].clone(),
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["diffusion_conditioning_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added diffusion_conditioning_real: t", rec["t"],
          "relpos", tuple(rec["relpos"].shape), "n_emb", tuple(rec["n_emb"].shape),
          "si", tuple(rec["si_ref"].shape), "zij", tuple(rec["zij_ref"].shape),
          "si std", float(si_ref.std()), "zij std", float(zij_ref.std()))


if __name__ == "__main__":
    main()
