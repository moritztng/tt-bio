"""On-device parity for the OpenFold3 ``AtomAttentionDecoder`` (AF3 Algorithm 6).

The exit leg of ``DiffusionModule.forward``: ``ai`` (token-level DiT output, post
``layer_norm_a``) -> per-atom coordinate update ``rl_update`` [N_atom, 3].

Golden: ~/of3_ref_out.pkl["intermediates"]["diffusion_decoder_real"], captured by
scripts/of3_diffusion_decoder_golden.py (real of3-p2-155k.pt weights, real ubiquitin
batch; full reference ``DiffusionModule`` forward with forward-hooks on
``atom_attn_dec``). The mask-derived atom windowing (``key_block_idxs`` /
``invalid_mask`` / ``mask_trunked``) and the token->atom broadcast index
(``atom_to_token_index``) are precomputed on host in the golden and replayed on device
via ``ttnn.embedding`` -- identical isolation to the P7 atom-transformer gate.

The 3-block cross-attention reuses the gated ``OF3AtomTransformer`` (decoder weights);
this gate isolates the fresh device work: ``linear_q_in`` + token->atom broadcast,
weight-only ``layer_norm``, and ``linear_q_out`` (c_atom -> 3).
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


def test_of3_diffusion_decoder_on_device():
    """Device OF3AtomAttentionDecoder (-> rl_update) vs the reference on real ubiquitin.
    rl_update is a weighted output (linear_q_out of the LN'd atom_transformer output),
    gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_diffusion_decoder import OF3AtomAttentionDecoder
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_decoder_real"]
    ai_ref, ql_ref, cl_ref, plm_ref, rl_ref = (g["ai"], g["ql"], g["cl"],
                                               g["plm"], g["rl_update"])
    atom_mask = g["atom_mask"]
    atom_to_token_index = g["atom_to_token_index"]
    key_block_idxs, invalid_mask, mask_trunked = (g["key_block_idxs"], g["invalid_mask"],
                                                  g["mask_trunked"])
    n_atom, n_token, nb, NP = g["n_atom"], g["n_token"], g["nb"], g["NP"]

    dev = get_device()
    dec = OF3AtomAttentionDecoder(_sub(sd, "diffusion_module.atom_attn_dec"), _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)

    # ai [n_token, 768] -> [1, n_token_pad, 768] (seq padded to a multiple of 32).
    n_tok_pad = ((n_token + 31) // 32) * 32
    ai_t = torch.zeros(1, n_tok_pad, 768)
    ai_t[0, :n_token] = ai_ref.float()
    ai_d = ft(ai_t)

    # ql, cl [n_atom, 128] -> [1, NP, 128] (atom-padded with zeros).
    ql_t = torch.zeros(1, NP, 128); ql_t[0, :n_atom] = ql_ref.float()
    cl_t = torch.zeros(1, NP, 128); cl_t[0, :n_atom] = cl_ref.float()
    ql_d, cl_d = ft(ql_t), ft(cl_t)

    # plm [nb, 32, 128, 16] -> [1, nb, 32, 128, 16].
    plm_d = ft(plm_ref.unsqueeze(0))

    # atom_mask_col [1, NP, 1] (zero at padded atoms).
    amc = torch.zeros(1, NP, 1); amc[0, :n_atom, 0] = atom_mask.float()
    amc_d = ft(amc)

    # atom_to_token_index [NP] uint32 (padded atoms -> 0, zeroed by atom_mask_col).
    idx = torch.zeros(NP, dtype=torch.long); idx[:n_atom] = atom_to_token_index.long()
    idx_tt = ttnn.from_torch(idx.unsqueeze(0), layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=dev, dtype=ttnn.uint32)

    # Block-gather aux (identical to the encoder atom-transformer gate).
    valid = (~invalid_mask).float().reshape(1, nb, 128, 1)
    mask_bias = (1e9 * (mask_trunked - 1)).reshape(1, nb, 1, 32, 128)
    kidx = key_block_idxs.reshape(1, nb * 128).to(torch.int32)
    kidx_tt = ttnn.from_torch(kidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    valid_d = ft(valid)
    mask_bias_d = ft(mask_bias)

    rl_d = dec(ai_d, ql_d, cl_d, plm_d, amc_d, idx_tt, kidx_tt, valid_d, mask_bias_d,
               n_atom, NP, nb)

    rl = torch.Tensor(ttnn.to_torch(rl_d)).float().reshape(rl_ref.shape)
    rl_pcc = _pcc(rl, rl_ref.float())
    print(f"\nOF3 AtomAttentionDecoder: rl_update_pcc={rl_pcc:.5f}")
    assert rl_pcc > 0.98, f"rl_update_pcc={rl_pcc:.5f} below 0.98"
