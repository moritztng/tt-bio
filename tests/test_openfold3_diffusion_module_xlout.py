"""On-device parity for the full OpenFold3 ``DiffusionModule`` -> ``xl_out`` (P9 merge
gate, leg 1). Assembles the already-gated sub-legs (NoisyPositionEmbedder, encoder +
decoder ``OF3AtomTransformer``, ``OF3DiffusionTransformer``) with the fresh post-
conditioning wiring (encoder pair update ``linear_l``/``linear_m`` + ``pair_mlp``,
``linear_q`` atom->token mean aggregation, the ``linear_s``/``layer_norm_s`` and
``layer_norm_a`` glues, EDM output scaling) and PCC-gates ``xl_out`` against the
full-module golden.

Golden: ``~/of3_ref_out.pkl`` -- ``diffusion_module_xlout_real`` (``xl_out``,
``xl_noisy``, ``rl_noisy``, ``cl0``/``plm0``, conditioned ``npe_zij``, NPE gather aux,
``sigma_data``/``t``), ``diffusion_conditioning_real`` (conditioned ``si_ref``),
``diffusion_decoder_real`` (``atom_to_token_index``, ``atom_mask``, block-gather aux),
``input_embedder_atom_transformer_real`` (``atom_to_token_mean``), and the
``diffusion_transformer_real`` bisect checkpoints (``a_in``/``a_stack``). The
conditioned ``(si, zij)`` and ``(cl0, plm0)`` are fed from their goldens (the
conditioning and RefAtom legs are gated separately) -- the same bisect discipline the
NPE / decoder gates use -- so this gate isolates the post-conditioning assembly.
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def test_of3_diffusion_module_xlout_on_device():
    """Full device OF3DiffusionModule -> xl_out vs the reference on real ubiquitin.
    xl_out is the denoised atom-position output (EDM-scaled); gated at PCC > 0.98."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_diffusion_module import OF3DiffusionModule
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    I = pickle.load(open(_GOLD, "rb"))["intermediates"]
    g = I["diffusion_module_xlout_real"]
    cond = I["diffusion_conditioning_real"]
    dec_g = I["diffusion_decoder_real"]
    at_g = I["input_embedder_atom_transformer_real"]
    dit_g = I["diffusion_transformer_real"]

    xl_out_ref = g["xl_out"]
    xl_noisy_ref = g["xl_noisy"]
    rl_noisy_ref = g["rl_noisy"]
    cl0_ref, plm0_ref = g["cl0"], g["plm0"]
    zij_ref = g["npe_zij"]                 # conditioned pair
    si_ref = cond["si_ref"]                # conditioned single
    si_trunk_ref = g["npe_si_trunk"]      # raw trunk single (NPE input)
    sigma_data, t = g["sigma_data"], g["t"]
    q_idx, k_idx = g["npe_q_indices"], g["npe_k_indices"]
    zij_mask = g["zij_mask"]
    n_atom, n_token, nb, NP = g["n_atom"], g["n_token"], g["nb"], g["NP"]
    n_tok_pad = ((n_token + 31) // 32) * 32

    atom_to_token_index = dec_g["atom_to_token_index"]
    atom_mask = dec_g["atom_mask"]
    key_block_idxs, invalid_mask, mask_trunked = (dec_g["key_block_idxs"],
                                                  dec_g["invalid_mask"], dec_g["mask_trunked"])
    atom_to_token_mean = at_g["atom_to_token_mean"]            # [n_token, n_atom]
    token_mask = dit_g["token_mask"]                            # [n_token]

    dev = get_device()
    dm = OF3DiffusionModule(_sub(sd, "diffusion_module"), _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)

    # Conditioned si/zij -> tile-padded to n_tok_pad.
    si_trunk_t = torch.zeros(1, n_tok_pad, 384); si_trunk_t[0, :n_token] = si_trunk_ref.float()
    si_t = torch.zeros(1, n_tok_pad, 384); si_t[0, :n_token] = si_ref.float()
    zij_t = torch.zeros(1, n_tok_pad, n_tok_pad, 128)
    zij_t[0, :n_token, :n_token] = zij_ref.float()
    si_trunk_d, si_d, zij_d = ft(si_trunk_t), ft(si_t), ft(zij_t)
    # cl0 [n_atom,128] -> [1, NP, 128]; rl_noisy [n_atom,3] -> [1, NP, 3].
    cl0_t = torch.zeros(1, NP, 128); cl0_t[0, :n_atom] = cl0_ref.float()
    rl_t = torch.zeros(1, NP, 3); rl_t[0, :n_atom] = rl_noisy_ref.float()
    cl0_d, rl_d = ft(cl0_t), ft(rl_t)
    # plm0 [nb,32,128,16] -> [1, nb, 32, 128, 16].
    plm0_d = ft(plm0_ref.unsqueeze(0))
    # xl_noisy (pre-mask in golden) -> mask -> [1, n_atom, 3] for EDM.
    xl_noisy_masked = xl_noisy_ref.float() * atom_mask.float().unsqueeze(-1)
    xl_d = ft(xl_noisy_masked.unsqueeze(0))

    # atom_mask_col [1, NP, 1]; atom_mask_col_na [1, n_atom, 1].
    amc = torch.zeros(1, NP, 1); amc[0, :n_atom, 0] = atom_mask.float()
    amc_d = ft(amc)
    amc_na = torch.zeros(1, n_atom, 1); amc_na[0, :, 0] = atom_mask.float()
    amc_na_d = ft(amc_na)
    # atom_to_token_index [NP] uint32 (padded atoms -> 0).
    idx = torch.zeros(NP, dtype=torch.long); idx[:n_atom] = atom_to_token_index.long()
    idx_tt = ttnn.from_torch(idx.unsqueeze(0), layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=dev, dtype=ttnn.uint32)
    # npe_flat_idx [1, nb*32*128] uint32 = q_token*n_tok_pad + k_token (device stride).
    flat = (q_idx.unsqueeze(-1) * n_tok_pad + k_idx.unsqueeze(1)).reshape(1, nb * 32 * 128)
    flat_tt = ttnn.from_torch(flat.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev, dtype=ttnn.uint32)
    # npe_zij_mask [1, nb, 32, 128, 1].
    zij_mask_d = ft(zij_mask.unsqueeze(0).unsqueeze(-1))
    # Encoder block-gather aux (pair-update + atom-transformer).
    kidx = key_block_idxs.reshape(1, nb * 128).to(torch.int32)
    kidx_tt = ttnn.from_torch(kidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    valid = (~invalid_mask).float().reshape(1, nb, 128, 1)
    valid_d = ft(valid)
    mask_bias = (1e9 * (mask_trunked - 1)).reshape(1, nb, 1, 32, 128)
    mask_bias_d = ft(mask_bias)
    pair_mask = mask_trunked.reshape(1, nb, 32, 128, 1)
    pair_mask_d = ft(pair_mask)
    # atom_to_token_mean [1, n_token, n_atom].
    mean_d = ft(atom_to_token_mean.unsqueeze(0))
    # token_mask padded to n_tok_pad.
    tok_pad = torch.zeros(n_tok_pad, dtype=torch.float32); tok_pad[:n_token] = token_mask.float()
    tok_pad_tt = ft(tok_pad.reshape(1, n_tok_pad))
    tok_col_pad_tt = ft(tok_pad.reshape(1, n_tok_pad, 1))

    xl_d, ai_pg, ai_pd, rl_cp, plm_pu, ql_enc_cp = dm(si_trunk_d, si_d, zij_d, cl0_d, plm0_d, rl_d, xl_d,
              amc_d, amc_na_d, idx_tt, flat_tt, zij_mask_d,
              kidx_tt, valid_d, mask_bias_d, pair_mask_d,
              mean_d, tok_pad_tt, tok_col_pad_tt,
              n_atom, NP, nb, n_token, n_tok_pad, t, sigma_data,
              _return_intermediates=True)

    a_in_ref = dit_g["a_in"]; a_stack_ref = dit_g["a_stack"]; rl_ref = dec_g["rl_update"]
    plm_ref = dec_g["plm"]; ql_enc_ref = dec_g["ql"]
    ai_pg_t = torch.Tensor(ttnn.to_torch(ai_pg)).float().reshape(a_in_ref.shape)
    ai_pd_t = torch.Tensor(ttnn.to_torch(ai_pd)).float()[0,:n_token].reshape(a_stack_ref.shape)
    rl_cp_t = torch.Tensor(ttnn.to_torch(rl_cp)).float().reshape(rl_ref.shape)
    plm_pu_t = torch.Tensor(ttnn.to_torch(plm_pu)).float().reshape(plm_ref.shape)
    ql_enc_t = torch.Tensor(ttnn.to_torch(ql_enc_cp)).float().reshape(ql_enc_ref.shape)
    print(f"BISECT plm(postpairupdate) pcc={_pcc(plm_pu_t, plm_ref.float()):.5f} std={plm_pu_t.std():.4f} ref={plm_ref.float().std():.4f}")
    print(f"BISECT ql_enc(postatomtransformer) pcc={_pcc(ql_enc_t, ql_enc_ref.float()):.5f} std={ql_enc_t.std():.4f} ref={ql_enc_ref.float().std():.4f}")
    print(f"BISECT a_in(postglue) pcc={_pcc(ai_pg_t, a_in_ref.float()):.5f} std={ai_pg_t.std():.4f} ref={a_in_ref.float().std():.4f}")
    print(f"BISECT a_stack(postdit) pcc={_pcc(ai_pd_t, a_stack_ref.float()):.5f} std={ai_pd_t.std():.4f} ref={a_stack_ref.float().std():.4f}")
    print(f"BISECT rl_update pcc={_pcc(rl_cp_t, rl_ref.float()):.5f} std={rl_cp_t.std():.4f} ref={rl_ref.float().std():.4f}")
    xl = torch.Tensor(ttnn.to_torch(xl_d)).float().reshape(xl_out_ref.shape)
    pcc = _pcc(xl, xl_out_ref.float())
    print(f"\nOF3 DiffusionModule xl_out: pcc={pcc:.5f}  (std dev={xl.std():.4f} ref={xl_out_ref.float().std():.4f})")
    assert pcc > 0.98, f"xl_out_pcc={pcc:.5f} below 0.98"
