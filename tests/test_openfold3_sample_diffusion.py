"""On-device parity for the OpenFold3 ``SampleDiffusion`` EDM sampler loop (P9 leg 2,
sub-leg). A reduced-step rollout (4 steps, 1 sample) around the gated
``OF3DiffusionConditioning`` + ``OF3DiffusionModule``, PCC-gated against the reference
rollout golden.

Golden: ``~/of3_ref_out.pkl["intermediates"]["sample_diffusion_rollout_real"]``, captured
by ``scripts/of3_sample_diffusion_golden.py`` (real of3-p2-155k.pt weights, real
ubiquitin batch; the reference ``DiffusionModule`` is run unmodified per step, the light
loop math replicated verbatim with the sample dim squeezed -- RNG draw counts match so
the trajectory is bit-exact). The per-step random artefacts (``centre_random_augmentation``
rotation/translation, the additive noise) and the per-step ``(t, c_tau)`` are replayed
from the golden; the Fourier noise embedding is computed on host (bit-exact vs the
reference). The fixed trunk / ref-atom / mask aux reuse the existing
``diffusion_module_xlout_real`` / ``diffusion_conditioning_real`` / ``diffusion_decoder_real``
/ ``input_embedder_atom_transformer_real`` goldens -- so this gate isolates the device
conditioning+DiffusionModule precision composed across the EDM loop from the random
augmentation/noise host math.

This is NOT the full ``fold()`` Kabsch merge gate (full production rollout is 200 steps
x 5 samples; ``fold()`` additionally needs the trunk + confidence heads -- see
``docs/openfold3-port.md``). It proves the EDM loop + per-step conditioning compose on
device.
"""
import os, pickle, math, pytest, torch, ttnn

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


def test_of3_sample_diffusion_rollout_on_device():
    """Device OF3SampleDiffusion 4-step rollout -> xl_final vs the reference golden.
    xl_final is the denoised atom-position sample after the EDM loop; gated at PCC > 0.98."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_sample_diffusion import OF3SampleDiffusion, fourier_noise_emb
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    I = pickle.load(open(_GOLD, "rb"))["intermediates"]
    g = I["sample_diffusion_rollout_real"]
    xlout = I["diffusion_module_xlout_real"]
    cond = I["diffusion_conditioning_real"]
    dec_g = I["diffusion_decoder_real"]
    at_g = I["input_embedder_atom_transformer_real"]

    sigma_data = g["sigma_data"]
    scfg = g["sample_diffusion_cfg"]
    step_scale = scfg["step_scale"]
    noise_schedule = g["noise_schedule"]
    xl_init_ref = g["xl_init"]                 # [n_atom, 3]
    xl_final_ref = g["xl_final"]               # [n_atom, 3]
    steps = g["steps"]
    n_steps = g["n_rollout_steps"]
    rots_list = [s["rots"] for s in steps]
    trans_list = [s["trans"] for s in steps]
    noise_list = [s["noise"] for s in steps]
    t_list = [s["t"] for s in steps]
    c_tau_list = [float(noise_schedule[i + 1]) for i in range(n_steps)]

    # Fixed trunk/conditioning/ref-atom inputs (from the existing goldens).
    si_input = cond["si_input"]                # [n_token, 449]
    si_trunk_raw = xlout["npe_si_trunk"]       # [n_token, 384]  (raw, NPE input)
    zij_trunk = cond["zij_trunk"]              # [n_token, n_token, 128]
    relpos = cond["relpos"]                    # [n_token, n_token, 139]
    token_mask = cond["token_mask"]            # [n_token]
    cl0_ref, plm0_ref = xlout["cl0"], xlout["plm0"]
    n_atom, n_token, nb, NP = xlout["n_atom"], xlout["n_token"], xlout["nb"], xlout["NP"]
    n_tok_pad = ((n_token + 31) // 32) * 32
    atom_to_token_index = dec_g["atom_to_token_index"]
    atom_mask = dec_g["atom_mask"]
    key_block_idxs, invalid_mask, mask_trunked = (dec_g["key_block_idxs"],
                                                  dec_g["invalid_mask"], dec_g["mask_trunked"])
    atom_to_token_mean = at_g["atom_to_token_mean"]
    q_idx, k_idx = xlout["npe_q_indices"], xlout["npe_k_indices"]
    zij_mask = xlout["zij_mask"]

    fourier_w = sd["diffusion_module.diffusion_conditioning.fourier_emb.w"]
    fourier_b = sd["diffusion_module.diffusion_conditioning.fourier_emb.b"]

    dev = get_device()
    sampler = OF3SampleDiffusion(_sub(sd, "diffusion_module"), _cfg(dev),
                                 fourier_w, fourier_b, sigma_data)
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)

    # Fixed conditioning inputs (unpadded n_token; conditioning pads internally).
    zij_trunk_dev = ft(zij_trunk.unsqueeze(0))
    relpos_dev = ft(relpos.unsqueeze(0))
    si_input_dev = ft(si_input.unsqueeze(0))
    si_trunk_dev = ft(si_trunk_raw.unsqueeze(0))   # raw trunk single (NPE input)
    n_tok = token_mask.shape[0]
    pair_mask = (token_mask[:, None] * token_mask[None, :]).reshape(n_tok, n_tok, 1).unsqueeze(0)
    tok_mask = token_mask.reshape(n_tok, 1).unsqueeze(0)
    pair_mask_dev, tok_mask_dev = ft(pair_mask), ft(tok_mask)

    # Fixed DiffusionModule aux (same as the xlout test).
    cl0_t = torch.zeros(1, NP, 128); cl0_t[0, :n_atom] = cl0_ref.float()
    plm0_d = ft(plm0_ref.unsqueeze(0))
    cl0_d = ft(cl0_t)
    amc = torch.zeros(1, NP, 1); amc[0, :n_atom, 0] = atom_mask.float()
    amc_d = ft(amc)
    amc_na = torch.zeros(1, n_atom, 1); amc_na[0, :, 0] = atom_mask.float()
    amc_na_d = ft(amc_na)
    idx = torch.zeros(NP, dtype=torch.long); idx[:n_atom] = atom_to_token_index.long()
    idx_tt = ttnn.from_torch(idx.unsqueeze(0), layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=dev, dtype=ttnn.uint32)
    flat = (q_idx.unsqueeze(-1) * n_tok_pad + k_idx.unsqueeze(1)).reshape(1, nb * 32 * 128)
    flat_tt = ttnn.from_torch(flat.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev, dtype=ttnn.uint32)
    zij_mask_d = ft(zij_mask.unsqueeze(0).unsqueeze(-1))
    kidx = key_block_idxs.reshape(1, nb * 128).to(torch.int32)
    kidx_tt = ttnn.from_torch(kidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    valid = (~invalid_mask).float().reshape(1, nb, 128, 1); valid_d = ft(valid)
    mask_bias = (1e9 * (mask_trunked - 1)).reshape(1, nb, 1, 32, 128); mb_d = ft(mask_bias)
    pair_mask_m = mask_trunked.reshape(1, nb, 32, 128, 1); pm_d = ft(pair_mask_m)
    mean_d = ft(atom_to_token_mean.unsqueeze(0))
    tok_pad = torch.zeros(n_tok_pad, dtype=torch.float32); tok_pad[:n_token] = token_mask.float()
    tok_pad_tt = ft(tok_pad.reshape(1, n_tok_pad))
    tok_col_pad_tt = ft(tok_pad.reshape(1, n_tok_pad, 1))

    xl_init_dev = ft(xl_init_ref.float().unsqueeze(0))

    xl_final_dev = sampler(
        xl_init_dev, si_trunk_dev, si_input_dev, zij_trunk_dev, relpos_dev,
        ft(token_mask.reshape(1, n_tok)), pair_mask_dev, tok_mask_dev, cl0_d, plm0_d,
        amc_d, amc_na_d, idx_tt, flat_tt, zij_mask_d, kidx_tt, valid_d, mb_d, pm_d,
        mean_d, tok_pad_tt, tok_col_pad_tt,
        n_atom, NP, nb, n_token, n_tok_pad,
        noise_schedule, rots_list, trans_list, noise_list, t_list, c_tau_list, step_scale)

    xl_final = torch.Tensor(ttnn.to_torch(xl_final_dev)).float().reshape(xl_final_ref.shape)
    pcc = _pcc(xl_final, xl_final_ref.float())
    print(f"\nOF3 SampleDiffusion 4-step rollout: xl_final_pcc={pcc:.5f} "
          f"(std dev={xl_final.std():.4f} ref={xl_final_ref.float().std():.4f})")
    # Per-step bisect: replay the golden xl_post_step trajectory check is implicit in the
    # final PCC; the per-step xl_denoised PCC is printable on demand.
    assert pcc > 0.98, f"xl_final_pcc={pcc:.5f} below 0.98"
