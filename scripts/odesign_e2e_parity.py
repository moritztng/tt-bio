#!/usr/bin/env python3
# ODesign end-to-end closed-loop sampler parity (pass 7, scope item 4). Feeds the
# PORTED on-device trunk's (s_inputs, s_trunk, z_trunk) into the pass-5 on-device
# closed-loop sampler (shared seed-42 draws) and compares final coords to a
# CPU-fp32 reference that feeds the CPU-fp32 reference trunk's outputs into the
# CPU-fp32 DiffusionModule sampler with the IDENTICAL draws. This is the real
# portable-complete gate: the full pipeline (trunk -> cond -> 200-step diffusion
# sampler) end-to-end, device bf16 vs CPU fp32, shared RNG draws.
#
# Per the pass-5 shared-draws lesson (memory `diffusion-port-parity-shared-draws`),
# device-vs-golden RMSD is NOT a parity metric (the golden is a CUDA Philox sample,
# unreproducible here); only device-vs-CPU-reference-with-shared-seed is. The
# run-to-run floor (CPU-ref vs golden, device vs golden) is reported separately.
#
# Modes (run separately so the device run uses TT_VISIBLE_DEVICES=0 and the CPU-ref
# run stays pure-torch):
#   device  : ported trunk + on-device closed-loop -> p7_e2e_device_coords.npy
#   cpuref  : CPU-fp32 ref trunk + CPU-fp32 DiffusionModule closed-loop -> p7_e2e_cpuref_coords.npy
#   compare : RMSD/PCC device-vs-cpuref (parity), cpuref-vs-golden, device-vs-golden (floor)
import os, sys, argparse, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN = "/home/moritz/.coworker/scratch/odesign-ref/golden"
CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
SCRATCH = "/home/moritz/.coworker/scratch/odesign-ref/ckpt"
DRAWS = os.path.join(SCRATCH, "p5_draws.pkl")               # reuse pass-5 seed-42 draws
TRUNK_IN = os.path.join(GOLDEN, "odesign_trunk_inputs.pkl")  # captured feature_data
CPU_REF_TRUNK = os.path.join(GOLDEN, "odesign_trunk_full_ref.pkl")  # CPU-fp32 trunk outputs
DEV_OUT = os.path.join(SCRATCH, "p7_e2e_device_coords.npy")
REF_OUT = os.path.join(SCRATCH, "p7_e2e_cpuref_coords.npy")


def _pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def _kabsch_rmsd(a, b):
    a = a.double(); b = b.double()
    a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
    h = a.t() @ b
    u, s, vh = torch.linalg.svd(h)
    d = torch.sign(torch.det(vh.t() @ u.t()))
    corr = torch.diag(torch.cat([torch.ones(2, dtype=s.dtype), d.unsqueeze(0)]))
    rot = u @ corr @ vh
    a2 = (a @ rot).numpy(); bn = b.numpy()
    return float(np.sqrt(((a2 - bn) ** 2).sum(-1).mean()))


def _rmsd(a, b):
    a = a.double().numpy(); b = b.double().numpy()
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def _ca_mask(pre):
    aa = pickle.load(open(os.path.join(GOLDEN, "atom_array.pkl"), "rb"))
    return torch.from_numpy(aa.atom_name == "CA").bool()


def _load_pre():
    return pickle.load(open(os.path.join(GOLDEN, "odesign_denoiser_pre.pkl"), "rb"))


def mode_device(args):
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.odesign import ODesign, ODesignTrunk
    pre = _load_pre()
    draws = pickle.load(open(DRAWS, "rb"))
    feat_fd = pickle.load(open(TRUNK_IN, "rb"))["feature_data"]
    feat = {k: feat_fd[k] for k in feat_fd}
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    trunk = ODesignTrunk(sd, ckc, dev)
    model = ODesign(sd, ckc, dev)
    print("running PORTED on-device trunk (10 cycles, 48 blocks/cycle)...", flush=True)
    import time
    t0 = time.time()
    s_trunk_dev, z_trunk_dev = trunk(feat, n_cycles=10)
    s_inputs_dev = trunk.ti.input_feature_embedder(feat)
    print("  trunk done in %.1fs  s_trunk %s  z_trunk %s"
          % (time.time() - t0, tuple(s_trunk_dev.shape), tuple(z_trunk_dev.shape)), flush=True)
    pre_dev = dict(pre)
    pre_dev["s_inputs"] = s_inputs_dev.float()
    pre_dev["s_trunk"] = s_trunk_dev.float()
    pre_dev["z_trunk"] = z_trunk_dev.float()
    print("running on-device closed-loop sampler (200 steps, shared draws)...", flush=True)
    coords = model.closed_loop_sample(pre_dev, draws, verbose=True)
    np.save(DEV_OUT, coords.detach().to("cpu", dtype=torch.float32).numpy())
    print("saved device e2e coords ->", DEV_OUT, "shape", tuple(coords.shape), flush=True)


def mode_cpuref(args):
    sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
    os.environ.setdefault("LAYERNORM_TYPE", "")
    os.environ.setdefault("DATA_ROOT_DIR", "./data")
    from src.model.modules.diffusion import DiffusionModule
    from src.api.model_interface import DiffusionInput, PairFormerOutput
    from tt_bio.odesign import (edm_step_params, add_noise_with_condition,
                                update_with_condition, centre_random_augmentation,
                                reverse_centre_random_augmentation)
    pre = _load_pre()
    draws = pickle.load(open(DRAWS, "rb"))
    ref_trunk = pickle.load(open(CPU_REF_TRUNK, "rb"))
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    dm_sd = {k[len("diffusion_module."):]: v for k, v in sd.items() if k.startswith("diffusion_module.")}
    dm = DiffusionModule(c_s=384, c_z=128, c_s_inputs=453, c_token=768,
                         atom_encoder={"n_blocks": 3, "n_heads": 4},
                         transformer={"n_blocks": 24, "n_heads": 16, "drop_path_rate": 0},
                         atom_decoder={"n_blocks": 3, "n_heads": 4})
    dm.load_state_dict(dm_sd, strict=False); dm.eval(); torch.set_grad_enabled(False)
    feat = pre["input_data"]
    input_data = DiffusionInput.from_feature_data(feat)
    # CPU-fp32 reference trunk outputs (not the golden bf16-GPU pre) -> isolates the
    # device pipeline's bf16 error from the irreducible bf16-GPU-vs-fp32 gap.
    input_embedding = PairFormerOutput(s_inputs=ref_trunk["s_inputs"].float(),
                                       s=ref_trunk["s_trunk"].float(),
                                       z=ref_trunk["z_trunk"].float())
    schedule = draws["schedule"]; cond_mask = draws["condition_mask"]
    x_gt = pre["gt_coordinate"].float()
    x_l = draws["init_noise"].clone()
    n_step = len(draws["rots"])
    print("running CPU-fp32 reference closed-loop (CPU-fp32 trunk, 200 steps)...", flush=True)
    for i in range(n_step):
        x_l_aug, trans, rot, x_center = centre_random_augmentation(
            x_l.squeeze(0), n_sample=1, rot=draws["rots"][i], trans=draws["trans"][i])
        x_l_augment = x_l_aug.clone()
        t_hat, c_tau, c_tau_last = edm_step_params(schedule, i, n_sample=1)
        x_noisy = add_noise_with_condition(x_l_aug, cond_mask, t_hat, c_tau_last=c_tau_last,
                                           noise=draws["noises"][i])
        x_update = dm(x_noisy, t_hat, input_data, input_embedding,
                       inplace_safe=False, use_conditioning=True)
        x_l = update_with_condition(x_noisy, x_update, x_gt=x_l_augment,
                                    condition_mask=cond_mask, t_hat=t_hat, c_tau=c_tau)
        x_l = reverse_centre_random_augmentation(x_l, trans, rot, x_center)
        if i % 20 == 0 or i == n_step - 1:
            print("  cpuref step %3d  t_hat=%9.4g  |x_l|=%.4f"
                  % (i, float(t_hat.max()), float(x_l.norm() / (x_l.numel() ** 0.5))), flush=True)
    np.save(REF_OUT, x_l.detach().to("cpu", dtype=torch.float32).numpy())
    print("saved cpuref e2e coords ->", REF_OUT, "shape", tuple(x_l.shape), flush=True)


def mode_compare(args):
    pre = _load_pre()
    dev = torch.from_numpy(np.load(DEV_OUT)).float().reshape(-1, 3)
    ref = torch.from_numpy(np.load(REF_OUT)).float().reshape(-1, 3)
    gold = torch.from_numpy(np.load(os.path.join(GOLDEN, "final_coords.npy"))).float().reshape(-1, 3)
    ca = _ca_mask(pre)
    print("\n======== ODesign END-TO-END closed-loop parity (pass 7) ========")
    print("N_atom=%d  N_step=200  seed=42  cond_atoms=0 (unconditional)" % dev.shape[0])
    print("device trunk: PORTED on-device (bf16, 10-cycle, z_trunk PCC 0.9857 vs CPU-ref)")
    print("ref   trunk: CPU-fp32 reference (10-cycle)")
    if ca is not None:
        print("CA-atom mask: %d CA atoms (= N_residue)" % int(ca.sum()))

    def report(name, a, b, same_frame):
        rmsd = _rmsd(a, b) if same_frame else _kabsch_rmsd(a, b)
        pcc = _pcc(a, b)
        tag = "same frame" if same_frame else "Kabsch"
        print("  %-30s all-atom RMSD %7.4f A   PCC %.5f   [%s]" % (name, rmsd, pcc, tag))
        if ca is not None:
            ca_rmsd = _rmsd(a[ca], b[ca]) if same_frame else _kabsch_rmsd(a[ca], b[ca])
            print("  %-30s CA      RMSD %7.4f A" % (name, ca_rmsd))
        return rmsd, pcc

    print("\n-- PARITY (identical seed-42 draws; device bf16 trunk vs CPU-fp32 trunk) --")
    r_dp, p_dp = report("device vs CPU-fp32-ref", dev, ref, same_frame=True)
    print("\n-- run-to-run floor (independent noise realizations; NOT a parity metric) --")
    r_rg, p_rg = report("CPU-fp32-ref vs CUDA-golden", ref, gold, same_frame=False)
    r_dg, p_dg = report("device vs CUDA-golden", dev, gold, same_frame=False)
    print("\nVERDICT: e2e parity RMSD=%.4f A PCC=%.5f ; floor(CPU-ref vs golden) RMSD=%.4f A"
          % (r_dp, p_dp, r_rg))
    with open(os.path.join(SCRATCH, "p7_e2e_compare.txt"), "w") as f:
        f.write("device_vs_cpuref_rmsd=%.6f pcc=%.6f\n" % (r_dp, p_dp))
        f.write("cpuref_vs_golden_rmsd=%.6f pcc=%.6f\n" % (r_rg, p_rg))
        f.write("device_vs_golden_rmsd=%.6f pcc=%.6f\n" % (r_dg, p_dg))
    print("saved", os.path.join(SCRATCH, "p7_e2e_compare.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["device", "cpuref", "compare"])
    args = ap.parse_args()
    if args.mode == "device":
        mode_device(args)
    elif args.mode == "cpuref":
        mode_cpuref(args)
    elif args.mode == "compare":
        mode_compare(args)


if __name__ == "__main__":
    main()
