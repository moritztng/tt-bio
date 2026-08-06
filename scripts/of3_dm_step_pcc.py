"""P13/H3: per-step device-vs-reference DiffusionModule PCC along a REAL failing
device rollout (the S0 baseline configuration: fixture-hybrid s_input, fixture
template block, fixture DM aux, tt-bio 1024-row MSA draw, seed 1234, 200 steps).

The device rollout runs EXACTLY as fold() runs it (host augmentation + noise, device
DiffusionConditioning + DiffusionModule per step, host EDM update). At probe steps the
same (xl_noisy, t) is additionally fed to the REFERENCE DiffusionModule on CPU with
identical conditioning inputs (the device trunk outputs, read back to host), and we
report PCC / relative-L2 on the denoised coordinates and on the EDM delta that
actually drives the update, plus PCC on the conditioned (si, zij).

Reading the output:
  - conditioning PCC low            -> device DiffusionConditioning is the defect.
  - conditioning OK, DM PCC low     -> device atom-attention encoder/transformer/
                                       decoder numerics.
  - everything PCC-high but the
    rollout still lands ~9 A        -> per-step bias compounding; inspect delta
                                       rel-L2 scaling with 1/t.

Run with the TT-BIO DEVICE env (needs ttnn + the CPU reference package both):
  PYTHONPATH=$WT:/tmp/of3-ref TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_dm_step_pcc.py
"""
import math
import os
import pickle
import sys

import numpy as np
import torch

WT = os.environ.get(
    "WT", "/home/ttuser/.coworker/wt/tt-bio-openfold3-p13-msa-templates-e2e"
)
OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, WT)
sys.path.insert(0, OF3_REF)

from of3_ref_rollout_hybrid import build_batch, kabsch_rmsd, load_pdb_ca  # noqa: E402

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
GT = os.path.join(WT, "examples/ground_truth_structures/ubiquitin.pdb")
CA_MASK = os.path.join(WT, "tests/fixtures/of3_ubiquitin_ca_mask.npy")

SEED = 1234
N_STEPS = 200
PROBES = {0, 2, 5, 10, 25, 50, 75, 100, 125, 150, 175, 190, 195, 199}


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def rel_l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_fold import OpenFold3 as TTOpenFold3
    from tt_bio.openfold3_fold import build_dm_device_aux, create_noise_schedule
    from tt_bio.openfold3_data import make_openfold3_msa_features
    from tt_bio.openfold3_sample_diffusion import fourier_noise_emb

    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3 as RefOpenFold3
    from openfold3.core.utils.tensor_utils import tensor_tree_map

    torch.manual_seed(0)
    np.random.seed(0)

    # ---- shared inputs -----------------------------------------------------
    features, batch = build_batch()  # tt-bio featurization, batch dim added
    msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)
    I = pickle.load(open(GOLD, "rb"))["intermediates"]
    ie = I["input_embedder_real"]
    te = I["template_embedder_real"]["feat"]
    atom_enc = I["input_embedder_atom_enc_real"]
    xl = I["diffusion_module_xlout_real"]
    dec = I["diffusion_decoder_real"]
    at = I["input_embedder_atom_transformer_real"]

    ai = atom_enc["out"][0].float()
    s_input = torch.cat(
        [ai, features["restype"], features["profile"],
         features["deletion_mean"].unsqueeze(-1)], dim=-1)
    n_atom = xl["n_atom"]; n_token = xl["n_token"]; nb = xl["nb"]; NP = xl["NP"]
    dm_aux_host = dict(
        cl0=xl["cl0"], plm0=xl["plm0"], atom_mask=dec["atom_mask"],
        atom_to_token_index=dec["atom_to_token_index"],
        npe_q_indices=xl["npe_q_indices"], npe_k_indices=xl["npe_k_indices"],
        zij_mask=xl["zij_mask"], key_block_idxs=dec["key_block_idxs"],
        invalid_mask=dec["invalid_mask"], mask_trunked=dec["mask_trunked"],
        atom_to_token_mean=at["atom_to_token_mean"], nb=nb, NP=NP)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    # ---- device side: glue + trunk (fold() first half) ---------------------
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    model = TTOpenFold3(sd, ckc, num_cycles=4)

    ft = model._ft
    s_input_d = ft(s_input.unsqueeze(0))
    relpos_dev = ft(ie["relpos"].unsqueeze(0))
    token_bonds_dev = ft(features["token_bonds"].unsqueeze(0).unsqueeze(-1))
    s_init_d, z_init_d = model.input_glue(s_input_d, relpos_dev, token_bonds_dev)
    tmpl_d = {k: ft(v) for k, v in te.items()}
    msa_d = ft(msa_feat.unsqueeze(0))
    si_trunk_d, zij_trunk_d = model.trunk(s_init_d, z_init_d, tmpl_d, msa_d, s_input_d)

    si_trunk_h = ttnn.to_torch(si_trunk_d).float().reshape(n_token, -1)
    zij_trunk_h = ttnn.to_torch(zij_trunk_d).float().reshape(n_token, n_token, -1)

    token_mask = features["token_mask"]
    n_tok = int(token_mask.shape[0])
    n_tok_pad = ((n_token + 31) // 32) * 32
    pair_mask = (token_mask[:, None] * token_mask[None, :]).reshape(n_tok, n_tok, 1).unsqueeze(0)
    tok_mask = token_mask.reshape(n_tok, 1).unsqueeze(0)
    pair_mask_dev, tok_mask_dev = ft(pair_mask), ft(tok_mask)
    tm_dev = ft(token_mask.reshape(1, n_tok))

    aux = build_dm_device_aux(
        dev, ft,
        cl0=dm_aux_host["cl0"], plm0=dm_aux_host["plm0"],
        atom_mask=dm_aux_host["atom_mask"],
        atom_to_token_index=dm_aux_host["atom_to_token_index"],
        npe_q_indices=dm_aux_host["npe_q_indices"],
        npe_k_indices=dm_aux_host["npe_k_indices"],
        zij_mask=dm_aux_host["zij_mask"],
        key_block_idxs=dm_aux_host["key_block_idxs"],
        invalid_mask=dm_aux_host["invalid_mask"],
        mask_trunked=dm_aux_host["mask_trunked"],
        atom_to_token_mean=dm_aux_host["atom_to_token_mean"],
        token_mask=token_mask, n_atom=n_atom, n_token=n_token, nb=nb, NP=NP,
        n_tok_pad=n_tok_pad)

    sampler = model.sampler
    sigma_data = model.sigma_data
    atom_mask_h = features["atom_mask"].float().reshape(n_atom)

    # ---- reference side ----------------------------------------------------
    C.settings.memory.eval.use_triton_triangle_kernels = False
    C.settings.memory.eval.use_deepspeed_evo_attention = False
    C.settings.memory.eval.use_cueq_triangle_kernels = False
    ref = RefOpenFold3(config=C).eval()
    ref.load_state_dict(sd, strict=False)
    ref_dm = ref.diffusion_module
    ref_dc = ref_dm.diffusion_conditioning

    perm = batch.pop("ref_space_uid_to_perm", None)
    batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
    if perm is not None:
        batch["ref_space_uid_to_perm"] = perm
    si_input_r = s_input.unsqueeze(0).unsqueeze(0)
    si_trunk_r = si_trunk_h.unsqueeze(0).unsqueeze(0)
    zij_trunk_r = zij_trunk_h.unsqueeze(0).unsqueeze(0)
    token_mask_r = batch["token_mask"]
    atom_mask_r = batch["atom_mask"]

    # ---- rollout -----------------------------------------------------------
    noise_schedule = create_noise_schedule(N_STEPS, sigma_data=sigma_data,
                                           s_max=160.0, s_min=4e-4, p=7)
    xl_init, rots_l, trans_l, noise_l, t_l, c_tau_l = model._gen_rollout(
        noise_schedule, n_atom, SEED)

    xl_host = xl_init.clone()
    with torch.no_grad():
        for tau in range(N_STEPS):
            rots = rots_l[tau].float()
            trans = trans_l[tau].float()
            mean_xl = (xl_host * atom_mask_h[:, None]).sum(0) / atom_mask_h.sum().clamp_min(1.0)
            xl_aug = (xl_host - mean_xl) @ rots.t() + trans
            xl_aug = xl_aug * atom_mask_h[:, None]
            t = float(t_l[tau])
            xl_noisy = xl_aug + noise_l[tau].float()

            # device per-step: conditioning + DM (OF3SampleDiffusion.__call__ verbatim)
            n_emb = fourier_noise_emb(t, sigma_data, sampler.fourier_w, sampler.fourier_b)
            si_dev, zij_dev = sampler.dc(
                zij_trunk_d, relpos_dev, si_trunk_d, s_input_d,
                sampler._to_dev(n_emb.reshape(1, 1, 256)), pair_mask_dev, tok_mask_dev)
            si_pad = sampler._pad_tokens(si_dev, n_token, n_tok_pad)
            zij_pad = sampler._pad_pair(zij_dev, n_token, n_tok_pad)
            ttnn.deallocate(si_dev); ttnn.deallocate(zij_dev)
            rl_noisy = xl_noisy * atom_mask_h[:, None] / math.sqrt(t * t + sigma_data ** 2)
            rl_noisy_dev = sampler._to_dev(sampler._pad_atoms_host(rl_noisy, n_atom, NP))
            xl_noisy_masked = xl_noisy * atom_mask_h[:, None]
            xl_noisy_dev = sampler._to_dev(xl_noisy_masked.unsqueeze(0))
            xl_denoised_dev = sampler.dm(
                si_trunk_d, si_pad, zij_pad, aux["cl0_d"], aux["plm0_d"],
                rl_noisy_dev, xl_noisy_dev,
                aux["amc_d"], aux["amc_na_d"], aux["idx_tt"], aux["flat_tt"],
                aux["zij_mask_d"], aux["kidx_tt"], aux["valid_d"], aux["mb_d"],
                aux["pm_d"], aux["mean_d"], aux["tok_pad_tt"], aux["tok_col_pad_tt"],
                n_atom, NP, nb, n_token, n_tok_pad, t, sigma_data)
            xl_den_dev_h = ttnn.to_torch(xl_denoised_dev).float().reshape(n_atom, 3)
            if tau in PROBES:
                si_dev_h = ttnn.to_torch(si_pad).float().reshape(n_tok_pad, -1)[:n_token]
                zij_dev_h = ttnn.to_torch(zij_pad).float().reshape(
                    n_tok_pad, n_tok_pad, -1)[:n_token, :n_token]
            ttnn.deallocate(si_pad); ttnn.deallocate(zij_pad)
            ttnn.deallocate(rl_noisy_dev); ttnn.deallocate(xl_noisy_dev)
            ttnn.deallocate(xl_denoised_dev)

            if tau in PROBES:
                t_r = torch.tensor(t, dtype=torch.float32).reshape(1, 1)
                si_ref, zij_ref = ref_dc(
                    batch=batch, t=t_r, si_input=si_input_r, si_trunk=si_trunk_r,
                    zij_trunk=zij_trunk_r, use_conditioning=True)
                xl_den_ref = ref_dm(
                    batch=batch,
                    xl_noisy=xl_noisy.reshape(1, 1, n_atom, 3),
                    token_mask=token_mask_r, atom_mask=atom_mask_r, t=t_r,
                    si_input=si_input_r, si_trunk=si_trunk_r, zij_trunk=zij_trunk_r,
                    use_conditioning=True)
                xl_den_ref = xl_den_ref.reshape(n_atom, 3).float()
                m = atom_mask_h.bool()
                d_dev = (xl_noisy - xl_den_dev_h)[m] / t
                d_ref = (xl_noisy - xl_den_ref)[m] / t
                print(f"tau={tau:3d} t={t:9.3f} "
                      f"pcc_si={pcc(si_dev_h, si_ref.reshape(n_token, -1)):.6f} "
                      f"pcc_zij={pcc(zij_dev_h, zij_ref.reshape(n_token, n_token, -1)):.6f} "
                      f"pcc_xlden={pcc(xl_den_dev_h[m], xl_den_ref[m]):.6f} "
                      f"relL2_xlden={rel_l2(xl_den_dev_h[m], xl_den_ref[m]):.4f} "
                      f"pcc_delta={pcc(d_dev, d_ref):.6f} "
                      f"relL2_delta={rel_l2(d_dev, d_ref):.4f}", flush=True)

            delta = (xl_noisy - xl_den_dev_h) / t
            dt = float(c_tau_l[tau]) - t
            xl_host = xl_noisy + model.step_scale * dt * delta
            if tau % 50 == 0:
                print(f"  step {tau}: xl std={float(xl_host.std()):.3f}", flush=True)

    ca_mask = np.load(CA_MASK).astype(bool)
    gt_ca = load_pdb_ca(GT)
    rmsd = kabsch_rmsd(xl_host[ca_mask].double(), gt_ca)
    print(f"RESULT-H3: sample0(seed={SEED}) RMSD={rmsd:.4f} A")


if __name__ == "__main__":
    main()
