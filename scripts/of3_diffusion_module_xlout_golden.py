"""P8: extend ~/of3_ref_out.pkl with the full ``DiffusionModule`` -> ``xl_out`` golden
plus the NoisyPositionEmbedder ``zij`` block-gather index, so the device
``OF3DiffusionModule`` post-conditioning assembly is PCC-gated end-to-end against
``xl_out``.

Re-runs the full reference ``DiffusionModule`` forward (real of3-p2-155k.pt weights,
real ubiquitin batch; identical state to ``diffusion_transformer_real`` -- same seed
1234, same ``xl_noisy = randn * s_max``, same trunk/conditioning inputs) and captures:

  * ``xl_out``    [N_atom, 3]  -- the full-module denoised-positions output (EDM-scaled);
  * ``xl_noisy``  [N_atom, 3]  -- the noisy sample (pre atom-mask, for EDM replay);
  * ``rl_noisy``  [N_atom, 3]  -- ``xl_noisy * atom_mask / sqrt(t^2 + sigma_data^2)``
                                  (captured via an ``atom_attn_enc`` pre-hook so the
                                  mask + sigma scaling match the reference exactly);
  * ``sigma_data`` float       -- EDM ``sigma_data`` from the model config;
  * ``zij_flat_idx`` [nb, nq, nk] int64 -- flat gather index
                                  ``q_token*N_token + k_token`` for the
                                  NoisyPositionEmbedder pair broadcast
                                  (``convert_pair_rep_to_blocks`` -> ``plm +=
                                  zij_blocks``); mask-derived, replayed on device via
                                  ``ttnn.embedding`` (same isolation discipline as the
                                  P7 atom-transformer key gather);
  * ``zij_mask``  [nb, nq, nk] float -- combined ``(1-invalid_key) * atom_pair_mask``
                                  for the gathered ``zij_blocks``.

The conditioned ``si``/``zij`` (diffusion outputs), raw ``si_trunk``, token/atom masks,
ref-atom-feat aux (``dlm``/``vlm``/``inv_sq_dists``), atom-windowing aux
(``key_block_idxs``/``invalid_mask``/``mask_trunked``) and the atom->token mean matrix
already live in the pkl under ``diffusion_conditioning_real`` /
``pairformer_stack_real`` / ``input_embedder_ref_atom_feat_real`` /
``input_embedder_atom_transformer_real`` and are reused as-is (the conditioning leg is
gated separately, so this gate isolates the post-conditioning assembly). The DiT
``a_in``/``a_stack`` and decoder ``rl_update`` goldens serve as intermediate bisect
checkpoints for the ``xl_out`` gate.

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_diffusion_module_xlout_golden.py
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
    sigma_data = float(C.architecture.diffusion_module.diffusion_module.sigma_data)

    t = torch.tensor(float(C.architecture.noise_schedule.s_max), dtype=torch.float32)
    torch.manual_seed(1234)
    xl_noisy = torch.randn(1, n_atom, 3, dtype=torch.float32) * t

    captured: dict = {}

    def enc_pre_hook(_module, args, kwargs):
        kw = kwargs if kwargs else {}
        captured["rl_noisy"] = kw["rl"].detach().clone()

    def npe_pre_hook(_module, args, kwargs):
        kw = kwargs if kwargs else {}
        captured["cl0"] = kw["cl"].detach().clone()
        captured["plm0"] = kw["plm"].detach().clone()
        captured["npe_si_trunk"] = kw["si_trunk"].detach().clone()
        captured["npe_zij"] = kw["zij_trunk"].detach().clone()

    def npe_post_hook(_module, _args, _kwargs, out):
        cl, plm, ql = out
        captured["npe_cl"] = cl.detach().clone()
        captured["npe_plm"] = plm.detach().clone()
        captured["npe_ql"] = ql.detach().clone()

    dm.atom_attn_enc.register_forward_pre_hook(enc_pre_hook, with_kwargs=True)
    dm.atom_attn_enc.noisy_position_embedder.register_forward_pre_hook(npe_pre_hook, with_kwargs=True)
    dm.atom_attn_enc.noisy_position_embedder.register_forward_hook(npe_post_hook, with_kwargs=True)

    with torch.no_grad():
        xl_out = dm(
            batch=batch, xl_noisy=xl_noisy, token_mask=token_mask, atom_mask=atom_mask,
            t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
            use_conditioning=True,
        )

    # NoisyPositionEmbedder pair-broadcast gather index (convert_pair_rep_to_blocks).
    atom_mask_1d = atom_mask[0].float()
    atom_to_token_index = batch["atom_to_token_index"][0].long()
    nb = math.ceil(n_atom / N_QUERY)
    pad_right = get_query_block_padding(n_atom, N_QUERY)
    key_block_idxs, invalid_mask = get_block_indices(
        atom_mask=atom_mask_1d, n_query=N_QUERY, n_key=N_KEY, device=torch.device("cpu"))
    mask_trunked = get_pair_atom_block_mask(
        atom_mask=atom_mask_1d, num_blocks=nb, n_query=N_QUERY, n_key=N_KEY,
        pad_len_right_q=pad_right, key_block_idxs=key_block_idxs, invalid_mask=invalid_mask)
    q_indices = torch.nn.functional.pad(atom_to_token_index, (0, pad_right), value=0).reshape(nb, N_QUERY).long()
    k_indices = torch.gather(atom_to_token_index.unsqueeze(0).expand(nb, n_atom),
                             1, key_block_idxs.long())  # [nb, N_KEY]
    zij_mask = ((~invalid_mask).float())[:, None, :].expand(nb, N_QUERY, N_KEY) * mask_trunked

    rec = {
        "xl_out": xl_out[0].clone(),
        "xl_noisy": xl_noisy[0].clone(),
        "rl_noisy": captured["rl_noisy"][0].clone(),
        "sigma_data": sigma_data,
        "t": float(t),
        "npe_q_indices": q_indices.clone(),   # [nb, nq] token idx per query atom (stride-agnostic)
        "npe_k_indices": k_indices.clone(),   # [nb, nk] token idx per key atom
        "zij_mask": zij_mask.float(),
        "n_atom": n_atom, "n_token": n_token, "nb": nb, "NP": nb * N_QUERY,
        # NoisyPositionEmbedder isolation (Algorithm 5 L8-12):
        "cl0": captured["cl0"][0].clone(),          # RefAtomFeatureEmbedder single out
        "plm0": captured["plm0"][0].clone(),        # RefAtomFeatureEmbedder pair out
        "npe_si_trunk": captured["npe_si_trunk"][0].clone(),
        "npe_zij": captured["npe_zij"][0].clone(),  # = zij (conditioned)
        "npe_cl": captured["npe_cl"][0].clone(),
        "npe_plm": captured["npe_plm"][0].clone(),
        "npe_ql": captured["npe_ql"][0].clone(),
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["diffusion_module_xlout_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added diffusion_module_xlout_real: n_atom", n_atom, "n_token", n_token,
          "nb", nb, "sigma_data", sigma_data, "t", float(t),
          "xl_out", tuple(rec["xl_out"].shape), "xl_out std", float(rec["xl_out"].std()),
          "rl_noisy std", float(rec["rl_noisy"].std()))


if __name__ == "__main__":
    main()
