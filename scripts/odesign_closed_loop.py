#!/usr/bin/env python3
# ODesign closed-loop sampler parity (pass 5). Runs the FULL ODesign diffusion
# sampler (Algorithm 18: seed-42 RNG draws, centre_random_augmentation + reverse,
# predictor-corrector add_noise, EDM Euler update, condition enforcement) around
# the on-device denoise_step, and compares final coords to a CPU-fp32 reference
# sampler fed the IDENTICAL noise draws. The shared-noise comparison isolates the
# on-device bf16 compounding over all 200 steps; the per-step denoiser was already
# parity-verified in pass 4.
#
# The provided golden final_coords.npy is an UNCONDITIONAL stochastic design
# sample drawn from CUDA's Philox RNG (unreproducible on this CPU-only/
# Tenstorrent box), so device-vs-golden RMSD is dominated by the stochastic
# sample spread, not port error -- reported as the run-to-run floor (per
# docs/pharma-benchmark.md methodology), not a parity metric.
#
# Modes (run separately so the device run can use TT_VISIBLE_DEVICES=0 and the
# CPU-ref run stays pure-torch):
#   gen     : generate seed-42 draws, save to scratch
#   device  : load draws, on-device closed-loop, save device_coords.npy
#   cpuref  : load draws, CPU-fp32 reference closed-loop (ODesign DiffusionModule)
#   compare : RMSD/PCC for device-vs-cpuref (parity), cpuref-vs-golden, device-vs-golden (floor)
import os, sys, argparse, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # worktree root -> tt_bio importable

GOLDEN = "/home/moritz/.coworker/scratch/odesign-ref/golden"
CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
SCRATCH = "/home/moritz/.coworker/scratch/odesign-ref/ckpt"
DRAWS = os.path.join(SCRATCH, "p5_draws.pkl")
DEV_OUT = os.path.join(SCRATCH, "p5_device_coords.npy")
REF_OUT = os.path.join(SCRATCH, "p5_cpuref_coords.npy")


def _pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def _kabsch_rmsd(a, b):
    """Kabsch-aligned RMSD between two (N,3) coord sets (a is moved onto b)."""
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
    """CA-atom mask from the golden atom_array.pkl (biotite AtomArray).
    atom_name == 'CA' -> 257 CA atoms (= N_token = N_residue). Returns (N_atom,) bool."""
    aa = pickle.load(open(os.path.join(GOLDEN, "atom_array.pkl"), "rb"))
    return torch.from_numpy(aa.atom_name == "CA").bool()


def _load_pre():
    return pickle.load(open(os.path.join(GOLDEN, "odesign_denoiser_pre.pkl"), "rb"))


def mode_gen(args):
    from tt_bio.odesign import generate_sampler_draws
    pre = _load_pre()
    N = pre["input_data"]["ref_pos"].shape[0]
    ica = pre["input_data"]["is_condition_atom"].bool()
    draws = generate_sampler_draws(N, n_step=pre["N_step"], seed=42, condition_mask=ica)
    with open(DRAWS, "wb") as f:
        pickle.dump(draws, f)
    print("drew seed=%d  N_atom=%d  N_step=%d  cond_atoms=%d  -> %s"
          % (draws["seed"], N, len(draws["rots"]), int(ica.sum()), DRAWS), flush=True)
    print("schedule[0,1,100,199,200]:", [float(draws["schedule"][i]) for i in (0, 1, 100, 199, 200)])


def mode_device(args):
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.odesign import ODesign
    pre = _load_pre()
    draws = pickle.load(open(DRAWS, "rb"))
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = ODesign.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
    print("ODesign loaded; running on-device closed-loop (200 steps)...", flush=True)
    coords = model.closed_loop_sample(pre, draws, verbose=True)
    np.save(DEV_OUT, coords.detach().to("cpu", dtype=torch.float32).numpy())
    print("saved device coords ->", DEV_OUT, "shape", tuple(coords.shape), flush=True)


def mode_cpuref(args):
    sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
    os.environ.setdefault("LAYERNORM_TYPE", "")
    from src.model.modules.diffusion import DiffusionModule
    from src.api.model_interface import DiffusionInput, PairFormerOutput
    from tt_bio.odesign import (edm_step_params, add_noise_with_condition,
                                update_with_condition, centre_random_augmentation,
                                reverse_centre_random_augmentation)
    pre = _load_pre()
    draws = pickle.load(open(DRAWS, "rb"))
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
    input_embedding = PairFormerOutput(s_inputs=pre["s_inputs"].float(),
                                       s=pre["s_trunk"].float(), z=pre["z_trunk"].float())
    schedule = draws["schedule"]; cond_mask = draws["condition_mask"]
    x_gt = pre["gt_coordinate"].float()
    x_l = draws["init_noise"].clone()
    n_step = len(draws["rots"])
    print("running CPU-fp32 reference closed-loop (200 steps)...", flush=True)
    for i in range(n_step):
        x_l_aug, trans, rot, x_center = centre_random_augmentation(
            x_l.squeeze(0), n_sample=1, rot=draws["rots"][i], trans=draws["trans"][i])
        x_l_augment = x_l_aug.clone()   # (1,N,3) -- keep N_sample dim
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
    print("saved cpuref coords ->", REF_OUT, "shape", tuple(x_l.shape), flush=True)


def mode_compare(args):
    pre = _load_pre()
    dev = torch.from_numpy(np.load(DEV_OUT)).float().reshape(-1, 3)
    ref = torch.from_numpy(np.load(REF_OUT)).float().reshape(-1, 3)
    gold = torch.from_numpy(np.load(os.path.join(GOLDEN, "final_coords.npy"))).float().reshape(-1, 3)
    ca = _ca_mask(pre)
    n_atom = dev.shape[0]
    print("\n================ ODesign closed-loop parity (pass 5) ================")
    print("N_atom=%d  N_step=200  seed=42  cond_atoms=0 (unconditional)" % n_atom)
    print("CA-atom mask: %s" % ("none (could not derive)" if ca is None else "%d atoms" % int(ca.sum())))
    if ca is not None:
        print("  (CA-RMSD uses %d CA atoms = N_residue)" % int(ca.sum()))

    def report(name, a, b, same_frame):
        rmsd = _rmsd(a, b) if same_frame else _kabsch_rmsd(a, b)
        pcc = _pcc(a, b)
        tag = "same frame" if same_frame else "Kabsch"
        print("  %-28s all-atom RMSD %7.4f A   PCC %.5f   [%s]" % (name, rmsd, pcc, tag))
        if ca is not None:
            ca_rmsd = _rmsd(a[ca], b[ca]) if same_frame else _kabsch_rmsd(a[ca], b[ca])
            print("  %-28s CA      RMSD %7.4f A" % (name, ca_rmsd))
        return rmsd, pcc

    print("\n-- parity (identical seed-42 noise realization) --")
    r_dp, p_dp = report("device vs CPU-fp32-ref", dev, ref, same_frame=True)
    print("\n-- run-to-run floor (independent noise realizations; not a parity metric) --")
    r_rg, p_rg = report("CPU-fp32-ref vs CUDA-golden", ref, gold, same_frame=False)
    r_dg, p_dg = report("device vs CUDA-golden", dev, gold, same_frame=False)
    print("\nVERDICT: device-vs-CPU-ref (parity) RMSD=%.4f A PCC=%.5f ; floor(CPU-ref vs golden) RMSD=%.4f A"
          % (r_dp, p_dp, r_rg))
    print("===================================================================\n")
    with open(os.path.join(SCRATCH, "p5_compare.txt"), "w") as f:
        f.write("device_vs_cpuref_rmsd=%.6f pcc=%.6f\n" % (r_dp, p_dp))
        f.write("cpuref_vs_golden_rmsd=%.6f pcc=%.6f\n" % (r_rg, p_rg))
        f.write("device_vs_golden_rmsd=%.6f pcc=%.6f\n" % (r_dg, p_dg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "device", "cpuref", "compare"])
    ap.parse_args()
    args = ap.parse_args()
    if args.mode == "gen":
        mode_gen(args)
    elif args.mode == "device":
        mode_device(args)
    elif args.mode == "cpuref":
        mode_cpuref(args)
    elif args.mode == "compare":
        mode_compare(args)


if __name__ == "__main__":
    main()
