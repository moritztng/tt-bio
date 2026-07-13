"""P8: extend ~/of3_ref_out.pkl with the OF3 token-level DiT block
(DiffusionTransformer, AF3 Algorithm 23, non-cross-attention path) so the device
DiT block + 24-block stack are PCC-gated in isolation.

Reuses the already-captured real trunk tensors from the pkl as the conditioning
inputs (``si_input`` from ``input_embedder_real/out``; ``si_trunk``/``zij_trunk``
from ``pairformer_stack_real/out``) and the real batch feature dict from
``input_embedder_real/in`` (so the atom encoder has ref_pos/ref_element/etc.).
Instantiates the full ``DiffusionModule`` (so the atom encoder runs and produces a
real on-manifold DiT input ``a = ai + linear_s(LN_s(si))``), then forward-hooks the
``diffusion_transformer`` to capture its exact ``(a, s, z, mask)`` input and ``a``
output, plus the block-0 input/output for a unit-level bisect gate. ``t`` is the
real initial sampling sigma (``s_max=160``) and ``xl_noisy = randn * t`` is a real
noisy sample at that sigma -- the exact first sampling step.

The mask (``token_mask``) is host-side; the DiT is PCC-gated against the exact
reference ``(a, s, z, mask)`` and reference ``a`` outputs, isolating the device
block precision from the atom-encoder/conditioning host math -- the same discipline
as the other OF3 golden legs.

Adds key ``diffusion_transformer_real``:
  t:           float (noise level)
  token_mask:  [N_token]
  a_in:        [N_token, 768]   DiT stack input (post atom-enc + linear_s glue)
  s:           [N_token, 384]   conditioning single (si)
  z:           [N_token, N_token, 128]  conditioning pair (zij)
  a_block0:    [N_token, 768]   output of DiT block 0
  a_stack:     [N_token, 768]   output of the full 24-block DiT stack

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_diffusion_transformer_golden.py
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
    from openfold3.core.model.structure.diffusion_module import DiffusionModule

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
    atom_mask = batch["atom_mask"]
    n_atom = int(atom_mask.shape[1])

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dm = DiffusionModule(C.architecture.diffusion_module).eval()
    dm.load_state_dict(sub(sd, "diffusion_module"), strict=True)

    t = torch.tensor(float(C.architecture.noise_schedule.s_max), dtype=torch.float32)

    # Real noisy sample at the initial sampling sigma: xl_noisy = randn * t.
    torch.manual_seed(1234)
    xl_noisy = torch.randn(1, n_atom, 3, dtype=torch.float32) * t

    captured: dict = {}

    # DiffusionTransformer.forward is invoked with all kwargs (a=, s=, z=, mask=).
    def dt_pre_hook(_module, args, kwargs):
        kw = kwargs if kwargs else {}
        captured["a_in"] = kw["a"].detach().clone()
        captured["s"] = kw["s"].detach().clone()
        captured["z"] = kw["z"].detach().clone()
        m = kw.get("mask")
        captured["mask"] = m.detach().clone() if m is not None else torch.ones_like(kw["a"][..., 0])

    def dt_post_hook(_module, _args, _kwargs, out):
        captured["a_stack"] = out.detach().clone()

    def blk0_post_hook(_module, _args, _kwargs, out):
        captured["a_block0"] = out.detach().clone()

    def make_blk_hook(i):
        def h(_module, _args, _kwargs, out):
            captured["blk"].append(out.detach().clone())
        return h

    dm.diffusion_transformer.register_forward_pre_hook(dt_pre_hook, with_kwargs=True)
    dm.diffusion_transformer.register_forward_hook(dt_post_hook, with_kwargs=True)
    dm.diffusion_transformer.blocks[0].register_forward_hook(blk0_post_hook, with_kwargs=True)
    captured["blk"] = []
    for i in range(len(dm.diffusion_transformer.blocks)):
        dm.diffusion_transformer.blocks[i].register_forward_hook(make_blk_hook(i), with_kwargs=True)

    with torch.no_grad():
        _xl_out = dm(
            batch=batch, xl_noisy=xl_noisy, token_mask=token_mask, atom_mask=atom_mask,
            t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
            use_conditioning=True,
        )

    rec = {
        "t": float(t),
        "token_mask": token_mask[0].clone(),
        "a_in": captured["a_in"][0].clone(),
        "s": captured["s"][0].clone(),
        "z": captured["z"][0].clone(),
        "a_block0": captured["a_block0"][0].clone(),
        "a_stack": captured["a_stack"][0].clone(),
        "a_traj": [t[0].clone() for t in captured["blk"]],
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["diffusion_transformer_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added diffusion_transformer_real: t", rec["t"],
          "a_in", tuple(rec["a_in"].shape), "s", tuple(rec["s"].shape),
          "z", tuple(rec["z"].shape),
          "a_in std", float(rec["a_in"].std()),
          "a_block0 std", float(rec["a_block0"].std()),
          "a_stack std", float(rec["a_stack"].std()))


if __name__ == "__main__":
    main()
