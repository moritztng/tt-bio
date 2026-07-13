"""P8: extend ~/of3_ref_out.pkl with the OF3 ``AtomAttentionDecoder`` (Algorithm 6)
golden, so the device ``OF3AtomAttentionDecoder`` is PCC-gated in isolation against
on-manifold inputs.

Runs the full reference ``DiffusionModule`` forward (real of3-p2-155k.pt weights, real
ubiquitin batch from ``input_embedder_real``; real noisy sample at the initial sampling
sigma ``s_max``) and forward-hooks ``diffusion_module.atom_attn_dec`` to capture its
exact ``(ai, ql, cl, plm)`` input and ``rl_update`` output. The mask-derived atom
windowing (``convert_single_rep_to_blocks`` -> ``get_block_indices`` /
``get_pair_atom_block_mask``) and the token->atom broadcast index
(``atom_to_token_index``) are precomputed on host here and replayed on device via
``ttnn.embedding`` -- the same isolation discipline as the P7
``input_embedder_atom_transformer_real`` golden.

Adds key ``diffusion_decoder_real``:
  ai:    [N_token, c_token]    decoder token input (= LN_a(DiT(ai)))
  ql:    [N_atom, c_atom]      encoder atom single rep (decoder atom init)
  cl:    [N_atom, c_atom]      atom conditioning (fixed across decoder blocks)
  plm:   [nb, nq, nk, c_atom_pair]  blocked atom pair (fixed)
  rl_update: [N_atom, 3]       reference decoder output (atom position update)
  atom_to_token_index: [N_atom] long   (token->atom broadcast gather index)
  atom_mask:      [N_atom] float
  key_block_idxs: [nb, nk] int64 ; invalid_mask: [nb, nk] bool
  mask_trunked:   [nb, nq, nk] float
  n_atom, n_token, nb, nq, nk, pad_right, NP

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_diffusion_decoder_golden.py
"""
import os, sys, pickle, math, collections.abc
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")

N_QUERY, N_KEY = 32, 128


def _strip(o):
    if isinstance(o, torch.Tensor):
        return o
    if (isinstance(o, (dict, collections.abc.Mapping))
            or (hasattr(o, "items") and callable(getattr(o, "items"))
                and hasattr(o, "__getitem__"))):
        return {k: _strip(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_strip(v) for v in o)
    return o


def sub(sd, prefix):
    return {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix)}


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.structure.diffusion_module import DiffusionModule
    from openfold3.core.utils.atom_attention_block_utils import (
        get_block_indices, get_pair_atom_block_mask, get_query_block_padding,
    )

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
    n_token = int(token_mask.shape[1])

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dm = DiffusionModule(C.architecture.diffusion_module).eval()
    dm.load_state_dict(sub(sd, "diffusion_module"), strict=True)

    t = torch.tensor(float(C.architecture.noise_schedule.s_max), dtype=torch.float32)
    torch.manual_seed(1234)
    xl_noisy = torch.randn(1, n_atom, 3, dtype=torch.float32) * t

    captured: dict = {}

    def dec_pre_hook(_module, args, kwargs):
        kw = kwargs if kwargs else {}
        captured["ai"] = kw["ai"].detach().clone()
        captured["ql"] = kw["ql"].detach().clone()
        captured["cl"] = kw["cl"].detach().clone()
        captured["plm"] = kw["plm"].detach().clone()

    def dec_post_hook(_module, _args, _kwargs, out):
        captured["rl_update"] = out.detach().clone()

    dm.atom_attn_dec.register_forward_pre_hook(dec_pre_hook, with_kwargs=True)
    dm.atom_attn_dec.register_forward_hook(dec_post_hook, with_kwargs=True)

    with torch.no_grad():
        _xl_out = dm(
            batch=batch, xl_noisy=xl_noisy, token_mask=token_mask, atom_mask=atom_mask,
            t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
            use_conditioning=True,
        )

    atom_mask_1d = atom_mask[0].float()
    atom_to_token_index = batch["atom_to_token_index"][0].long()
    nb = math.ceil(n_atom / N_QUERY)
    pad_right = get_query_block_padding(n_atom, N_QUERY)
    NP = nb * N_QUERY
    key_block_idxs, invalid_mask = get_block_indices(
        atom_mask=atom_mask_1d, n_query=N_QUERY, n_key=N_KEY, device=torch.device("cpu"))
    mask_trunked = get_pair_atom_block_mask(
        atom_mask=atom_mask_1d, num_blocks=nb, n_query=N_QUERY, n_key=N_KEY,
        pad_len_right_q=pad_right, key_block_idxs=key_block_idxs, invalid_mask=invalid_mask)

    rec = {
        "ai": captured["ai"][0].clone(),
        "ql": captured["ql"][0].clone(),
        "cl": captured["cl"][0].clone(),
        "plm": captured["plm"][0].clone(),
        "rl_update": captured["rl_update"][0].clone(),
        "atom_to_token_index": atom_to_token_index.clone(),
        "atom_mask": atom_mask_1d.clone(),
        "key_block_idxs": key_block_idxs.long(),
        "invalid_mask": invalid_mask,
        "mask_trunked": mask_trunked.float(),
        "n_atom": n_atom, "n_token": n_token, "nb": nb,
        "nq": N_QUERY, "nk": N_KEY, "pad_right": pad_right, "NP": NP,
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["diffusion_decoder_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added diffusion_decoder_real: n_atom", n_atom, "n_token", n_token,
          "nb", nb, "NP", NP,
          "ai", tuple(rec["ai"].shape), "ql", tuple(rec["ql"].shape),
          "plm", tuple(rec["plm"].shape),
          "rl_update", tuple(rec["rl_update"].shape),
          "rl_update std", float(rec["rl_update"].std()))


if __name__ == "__main__":
    main()
