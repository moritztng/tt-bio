"""On-device parity for the OpenFold3 ``NoisyPositionEmbedder`` (AF3 Algorithm 5 L8-12).

The entry leg of the ``DiffusionModule`` atom encoder: fuses the trunk single/pair into
the reference-conformer atom conditioning and seeds ``ql`` from the noisy coordinates.

Golden: ~/of3_ref_out.pkl["intermediates"]["diffusion_module_xlout_real"], captured by
scripts/of3_diffusion_module_xlout_golden.py (real of3-p2-155k.pt weights, real
ubiquitin batch; full reference ``DiffusionModule`` forward with forward-hooks on
``atom_attn_enc.noisy_position_embedder`` capturing ``cl0``/``plm0`` (RefAtomFeatureEmbedder
outs) + ``si_trunk``/``zij``/``rl`` in and ``cl``/``plm``/``ql`` out). The mask-derived
broadcasts (``atom_to_token_index`` single gather; ``q_indices``/``k_indices`` pair
gather) are precomputed on host and replayed on device via ``ttnn.embedding`` -- same
isolation discipline as the P7 atom-transformer gate. ``cl0``/``plm0`` are fed from the
golden so this gate isolates the NoisyPos linears/LNs/gathers from RefAtomFeatureEmbedder.
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


def test_of3_noisy_position_embedder_on_device():
    """Device OF3NoisyPositionEmbedder (-> cl, plm, ql) vs the reference on real ubiquitin.
    cl/plm/ql are weighted outputs (linear_s/linear_z/linear_r of LN'd trunk + cl0/plm0),
    gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_diffusion_module import OF3NoisyPositionEmbedder
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_module_xlout_real"]
    dec_g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_decoder_real"]
    cl0_ref, plm0_ref = g["cl0"], g["plm0"]
    si_trunk_ref, zij_ref, rl_ref = g["npe_si_trunk"], g["npe_zij"], g["rl_noisy"]
    cl_ref, plm_ref, ql_ref = g["npe_cl"], g["npe_plm"], g["npe_ql"]
    q_idx, k_idx = g["npe_q_indices"], g["npe_k_indices"]      # [nb, nq], [nb, nk]
    zij_mask = g["zij_mask"]                                   # [nb, nq, nk]
    atom_to_token_index = dec_g["atom_to_token_index"]         # [n_atom]
    atom_mask = dec_g["atom_mask"]
    n_atom, n_token, nb, NP = g["n_atom"], g["n_token"], g["nb"], g["NP"]

    dev = get_device()
    npe = OF3NoisyPositionEmbedder(
        _sub(sd, "diffusion_module.atom_attn_enc.noisy_position_embedder"), _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    n_tok_pad = ((n_token + 31) // 32) * 32

    # cl0 [n_atom,128] -> [1, NP, 128]; rl [n_atom,3] -> [1, NP, 3] (atom-padded with 0).
    cl0_t = torch.zeros(1, NP, 128); cl0_t[0, :n_atom] = cl0_ref.float()
    rl_t = torch.zeros(1, NP, 3); rl_t[0, :n_atom] = rl_ref.float()
    cl0_d, rl_d = ft(cl0_t), ft(rl_t)
    # plm0 [nb,32,128,16] -> [1, nb, 32, 128, 16].
    plm0_d = ft(plm0_ref.unsqueeze(0))
    # si_trunk [n_tok,384] -> [1, n_tok_pad, 384]; zij [n_tok,n_tok,128] -> [1, n_tok_pad, n_tok_pad, 128].
    si_t = torch.zeros(1, n_tok_pad, 384); si_t[0, :n_token] = si_trunk_ref.float()
    zij_t = torch.zeros(1, n_tok_pad, n_tok_pad, 128)
    zij_t[0, :n_token, :n_token] = zij_ref.float()
    si_d, zij_d = ft(si_t), ft(zij_t)
    # atom_mask_col [1, NP, 1].
    amc = torch.zeros(1, NP, 1); amc[0, :n_atom, 0] = atom_mask.float()
    amc_d = ft(amc)
    # atom_to_token_index [NP] uint32 (padded atoms -> 0, zeroed by atom_mask_col).
    idx = torch.zeros(NP, dtype=torch.long); idx[:n_atom] = atom_to_token_index.long()
    idx_tt = ttnn.from_torch(idx.unsqueeze(0), layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=dev, dtype=ttnn.uint32)
    # zij_flat_idx [1, nb*nq*nk] uint32 = q_token*n_tok_pad + k_token (device stride).
    flat = (q_idx.unsqueeze(-1) * n_tok_pad + k_idx.unsqueeze(1)).reshape(1, nb * 32 * 128)
    flat_tt = ttnn.from_torch(flat.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev, dtype=ttnn.uint32)
    # zij_mask [1, nb, nq, nk, 1].
    zij_mask_d = ft(zij_mask.unsqueeze(0).unsqueeze(-1))

    cl_d, plm_d, ql_d = npe(cl0_d, plm0_d, si_d, zij_d, rl_d, amc_d, idx_tt, flat_tt,
                            zij_mask_d, n_atom, NP)

    cl = torch.Tensor(ttnn.to_torch(cl_d)).float().reshape(cl_ref.shape)
    plm = torch.Tensor(ttnn.to_torch(plm_d)).float().reshape(plm_ref.shape)
    ql = torch.Tensor(ttnn.to_torch(ql_d)).float().reshape(ql_ref.shape)
    cl_pcc = _pcc(cl, cl_ref.float())
    plm_pcc = _pcc(plm, plm_ref.float())
    ql_pcc = _pcc(ql, ql_ref.float())
    print(f"\nOF3 NoisyPositionEmbedder: cl_pcc={cl_pcc:.5f} plm_pcc={plm_pcc:.5f} ql_pcc={ql_pcc:.5f}")
    assert cl_pcc > 0.98, f"cl_pcc={cl_pcc:.5f} below 0.98"
    assert plm_pcc > 0.98, f"plm_pcc={plm_pcc:.5f} below 0.98"
    assert ql_pcc > 0.98, f"ql_pcc={ql_pcc:.5f} below 0.98"
