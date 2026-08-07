"""P13/H2' (host rollout maths bisect): tt-bio's EXACT fold() rollout loop
(create_noise_schedule + _gen_rollout + per-step EDM update, constants and RNG order
copied from tt_bio.openfold3_fold) driving the REFERENCE DiffusionModule on CPU.

  ~1 A  -> tt-bio's host rollout maths is correct; the 7-9 A defect is the device
           per-step DiffusionModule / conditioning / decoder numerics.
  7-9 A -> the defect is in tt-bio's host rollout loop itself (RNG, schedule, EDM
           update) -- a pure host bug, no device involved.

Trunk inputs come from ~/p13_s1_trunk_msa.pt (CPU reference trunk on tt-bio's exact
MSA draw; H1 showed device trunk outputs are equally good).

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_ref_dm_ttbio_rollout.py
"""
import math
import os
import sys

import numpy as np
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
WT = os.environ.get(
    "WT", "/home/ttuser/.coworker/wt/tt-bio-openfold3-p13-msa-templates-e2e"
)
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, WT)
sys.path.insert(0, OF3_REF)

from of3_ref_rollout_hybrid import build_batch, kabsch_rmsd, load_pdb_ca  # noqa: E402

GT = os.path.join(WT, "examples/ground_truth_structures/ubiquitin.pdb")

# tt-bio fold() constants (tt_bio.openfold3_fold.OpenFold3): AF3 sample_diffusion +
# noise_schedule defaults from the OF3 model config.
GAMMA_0 = 0.8
GAMMA_MIN = 1.0
NOISE_SCALE = 1.003
STEP_SCALE = 1.5
SIGMA_DATA = 16.0
S_MAX, S_MIN, P = 160.0, 4e-4, 7
N_STEPS, N_SAMPLES, SEED = 200, 5, 1234


def create_noise_schedule(no_rollout_steps, sigma_data, s_max, s_min, p):
    t = torch.arange(0, 1 + no_rollout_steps, dtype=torch.float32) / no_rollout_steps
    return sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p


def _quat_to_rot(q):
    b, c, d, a = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
        2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b),
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d,
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def sample_rotation():
    q = torch.randn(4, dtype=torch.float32)
    q = q / torch.linalg.norm(q)
    return _quat_to_rot(q)


def gen_rollout(noise_schedule, n_atom, seed):
    """tt_bio.openfold3_fold.OpenFold3._gen_rollout verbatim (RNG order matters)."""
    torch.manual_seed(seed)
    xl_init = noise_schedule[0] * torch.randn(n_atom, 3, dtype=torch.float32)
    rots_l, trans_l, noise_l, t_l, c_tau_l = [], [], [], [], []
    for tau in range(len(noise_schedule) - 1):
        c_tau = float(noise_schedule[tau + 1])
        gamma = GAMMA_0 if c_tau > GAMMA_MIN else 0
        t = float(noise_schedule[tau]) * (gamma + 1)
        rots = sample_rotation()
        trans = 1.0 * torch.randn(3, dtype=torch.float32)
        noise = NOISE_SCALE * math.sqrt(max(t * t - float(noise_schedule[tau]) ** 2, 0.0)) \
            * torch.randn(n_atom, 3, dtype=torch.float32)
        rots_l.append(rots); trans_l.append(trans); noise_l.append(noise)
        t_l.append(t); c_tau_l.append(c_tau)
    return xl_init, rots_l, trans_l, noise_l, t_l, c_tau_l


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.core.utils.tensor_utils import tensor_tree_map

    features, batch = build_batch()
    d = torch.load(os.path.expanduser("~/p13_s1_trunk_msa.pt"),
                   map_location="cpu", weights_only=False)
    s_input, s_trunk, z_trunk = d["s_input_ref"], d["s_trunk_fixed"], d["z_trunk_fixed"]

    C.settings.memory.eval.use_triton_triangle_kernels = False
    C.settings.memory.eval.use_deepspeed_evo_attention = False
    C.settings.memory.eval.use_cueq_triangle_kernels = False

    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"),
                    map_location="cpu", weights_only=False)
    model = OpenFold3(config=C).eval()
    model.load_state_dict(sd, strict=False)
    dm = model.diffusion_module

    atom_mask = batch["atom_mask"]  # [1, n_atom]
    n_atom = int(atom_mask.shape[-1])
    atom_mask_h = atom_mask[0].float()

    # sample-dim expansion, mirroring OpenFold3.forward's eval branch
    si_input = s_input.unsqueeze(0).unsqueeze(0)
    si_trunk = s_trunk.unsqueeze(0).unsqueeze(0)
    zij_trunk = z_trunk.unsqueeze(0).unsqueeze(0)
    perm = batch.pop("ref_space_uid_to_perm", None)
    batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
    if perm is not None:
        batch["ref_space_uid_to_perm"] = perm
    token_mask = batch["token_mask"]
    atom_mask_b = batch["atom_mask"]

    noise_schedule = create_noise_schedule(N_STEPS, SIGMA_DATA, S_MAX, S_MIN, P)

    samples = []
    with torch.no_grad():
        for sample_i in range(N_SAMPLES):
            xl_init, rots_l, trans_l, noise_l, t_l, c_tau_l = gen_rollout(
                noise_schedule, n_atom, SEED + sample_i)
            xl_host = xl_init.clone()
            for tau in range(N_STEPS):
                rots = rots_l[tau].float()
                trans = trans_l[tau].float()
                mean_xl = (xl_host * atom_mask_h[:, None]).sum(0) / atom_mask_h.sum().clamp_min(1.0)
                xl_aug = (xl_host - mean_xl) @ rots.t() + trans
                xl_aug = xl_aug * atom_mask_h[:, None]
                t = float(t_l[tau])
                xl_noisy = xl_aug + noise_l[tau].float()

                xl_den = dm(
                    batch=batch,
                    xl_noisy=xl_noisy.reshape(1, 1, n_atom, 3),
                    token_mask=token_mask,
                    atom_mask=atom_mask_b,
                    t=torch.tensor(t, dtype=torch.float32).reshape(1, 1),
                    si_input=si_input,
                    si_trunk=si_trunk,
                    zij_trunk=zij_trunk,
                    use_conditioning=True,
                )
                xl_den = xl_den.reshape(n_atom, 3).float()
                delta = (xl_noisy - xl_den) / t
                dt = float(c_tau_l[tau]) - t
                xl_host = xl_noisy + STEP_SCALE * dt * delta
            samples.append(xl_host)
            print(f"sample {sample_i}: xl std={float(xl_host.std()):.3f}", flush=True)

    atom_array = features["atom_array"]
    ca_mask = torch.from_numpy(atom_array.atom_name == "CA")
    gt_ca = load_pdb_ca(GT)
    rmsds = []
    for i, xl in enumerate(samples):
        r = kabsch_rmsd(xl[ca_mask].double(), gt_ca)
        rmsds.append(r)
        print(f"sample {i}: RMSD={r:.4f} A")
    rmsds_s = sorted(rmsds)
    print(
        f"RESULT-H2prime: best={rmsds_s[0]:.4f} median={rmsds_s[len(rmsds_s)//2]:.4f} "
        f"worst={rmsds_s[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
