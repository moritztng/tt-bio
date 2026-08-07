"""P9 leg 2: extend ~/of3_ref_out.pkl with a reduced-step ``SampleDiffusion`` (AF3
Algorithm 18) rollout golden, so the device EDM sampler loop around the gated
``OF3DiffusionModule`` can be PCC-gated end-to-end against the reference trajectory.

Runs the reference ``SampleDiffusion`` loop body verbatim (same math, same RNG call
order as ``openfold3.core.model.structure.diffusion_module.SampleDiffusion.forward``)
with ``no_rollout_steps=4`` (5 schedule entries, 4 rollout steps), ``no_rollout_samples=1``,
seed 1234, on the real ubiquitin batch -- but with the sample dimension squeezed
(``xl`` is ``[1, N_atom, 3]`` not ``[1, 1, N_atom, 3]``). The reference's
``SampleDiffusion.forward`` with ``no_rollout_samples=1`` hits a shape bug in
``aggregate_atom_feat_to_tokens`` (the spurious sample dim breaks its scatter-add); the
RNG draw *count* is identical for ``[1,N,3]`` and ``[1,1,N,3]`` (torch RNG is
shape-agnostic), so squeezing the sample dim reproduces the reference trajectory exactly
while matching the ``[1, N_atom, 3]`` shape the device ``OF3DiffusionModule`` consumes
(the ``diffusion_module_xlout_real`` golden uses the same shape). The heavy reference
part -- the per-step ``DiffusionModule.forward`` (real of3-p2-155k.pt weights) -- is run
unmodified, so ``xl_denoised`` per step is the real reference output; only the light
loop math (augmentation, noise add, EDM step) is replicated from the reference source.

The full production rollout (200 steps x 5 samples = 1000 DiffusionModule forwards) is
infeasible for a CPU golden; 4 steps x 1 sample exercises every branch of the sampler
loop (augmentation, noise add, per-step conditioning with a t-dependent Fourier noise
embedding, EDM step) at tractable cost and is the right first sub-leg -- it is NOT the
full fold() Kabsch merge gate.

Every per-step random/host artifact is captured so the device replay is bit-exact:
  * the initial ``xl_init`` (``noise_schedule[0] * randn``);
  * per step ``tau``: the ``centre_random_augmentation`` rotation ``rots`` + translation
    ``trans`` (AF3 Algorithm 19), the post-augmentation ``xl_aug``, the added ``noise``,
    the DiffusionModule inputs ``(xl_noisy, t)`` and output ``xl_denoised`` (forward-hook),
    and the post-EDM-step ``xl``;
  * the noise schedule and the final ``xl``.

The device sampler replays ``rots``/``trans``/``noise`` from this golden (host), computes
the per-step Fourier noise embedding ``n = fourier_emb(0.25 * log(t / sigma_data))`` on
host (``fourier_emb.w``/``b`` are in the checkpoint), runs the gated
``OF3DiffusionConditioning`` + ``OF3DiffusionModule`` per step, and applies the EDM step
on host -- isolating the device conditioning+DiffusionModule precision from the random
augmentation/noise host math (same discipline as the other OF3 golden legs).

Adds key ``sample_diffusion_rollout_real``.

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_sample_diffusion_golden.py
"""
import os, sys, pickle, math, collections.abc
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
N_ROLLOUT_STEPS = 4
SEED = 1234


def _strip(o):
    if isinstance(o, torch.Tensor):
        return o
    if (isinstance(o, (dict, collections.abc.Mapping))
            or (hasattr(o, "items") and callable(getattr(o, "items")) and hasattr(o, "__getitem__"))):
        return {k: _strip(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_strip(v) for v in o)
    return o


def main():
    from openfold3.core.model.structure.diffusion_module import DiffusionModule, create_noise_schedule
    from openfold3.core.model.structure.augmentation import sample_rotations
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C

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

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dm = DiffusionModule(C.architecture.diffusion_module).eval()
    dm.load_state_dict({k[len("diffusion_module."):]: v for k, v in sd.items()
                        if k.startswith("diffusion_module.")}, strict=True)
    sigma_data = float(C.architecture.diffusion_module.diffusion_module.sigma_data)
    scfg = dict(C.architecture.sample_diffusion)
    gamma_0, gamma_min, noise_scale, step_scale = (scfg["gamma_0"], scfg["gamma_min"],
                                                   scfg["noise_scale"], scfg["step_scale"])

    ns_cfg = dict(C.architecture.noise_schedule)
    noise_schedule = create_noise_schedule(no_rollout_steps=N_ROLLOUT_STEPS, **ns_cfg,
                                           dtype=torch.float32, device=torch.device("cpu"))

    dm_log: list = []

    def dm_pre(_m, args, kwargs):
        kw = kwargs if kwargs else {}
        dm_log.append({"xl_noisy": kw["xl_noisy"].detach().clone(), "t": float(kw["t"].detach().clone())})

    def dm_post(_m, _args, _kwargs, out):
        dm_log[-1]["xl_denoised"] = out.detach().clone()

    dm.register_forward_pre_hook(dm_pre, with_kwargs=True)
    dm.register_forward_hook(dm_post, with_kwargs=True)

    torch.manual_seed(SEED)
    # xl init: noise_schedule[0] * randn([1, n_atom, 3]) (sample dim squeezed; RNG count
    # matches the reference's randn([1, 1, n_atom, 3])).
    xl = noise_schedule[0] * torch.randn(1, n_atom, 3, dtype=torch.float32)
    xl_init = xl[0].clone()

    steps = []
    with torch.no_grad():
        for tau, c_tau in enumerate(noise_schedule[1:]):
            # centre_random_augmentation (AF3 Algorithm 19), replicated verbatim.
            rots = sample_rotations(shape=xl.shape[:-2], dtype=xl.dtype, device=xl.device)  # [1,3,3]
            trans = 1.0 * torch.randn((*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device)  # [1,3]
            mean_xl = torch.sum(xl * atom_mask[..., None], dim=-2, keepdim=True) / torch.sum(
                atom_mask[..., None], dim=-2, keepdim=True)
            xl_aug = (xl - mean_xl) @ rots.transpose(-1, -2) + trans[..., None, :]
            xl_aug = xl_aug * atom_mask[..., None]

            gamma = gamma_0 if c_tau > gamma_min else 0
            t = noise_schedule[tau] * (gamma + 1)
            noise = noise_scale * torch.sqrt(t ** 2 - noise_schedule[tau] ** 2) * torch.randn_like(xl)
            xl_noisy = xl_aug + noise

            xl_denoised = dm(
                batch=batch, xl_noisy=xl_noisy, token_mask=token_mask, atom_mask=atom_mask,
                t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
                use_conditioning=True,
            )

            delta = (xl_noisy - xl_denoised) / t
            dt = c_tau - t
            xl = xl_noisy + step_scale * dt * delta

            d = dm_log[-1]
            steps.append({
                "rots": rots[0], "trans": trans[0],
                "xl_pre_aug": xl_aug[0].clone() - noise[0] if False else None,  # placeholder
                "xl_aug": xl_aug[0].clone(),
                "t": float(t),
                "noise": noise[0].clone(),
                "xl_noisy": d["xl_noisy"][0].clone(),
                "xl_denoised": d["xl_denoised"][0].clone(),
                "xl_post_step": xl[0].clone(),
            })

    rec = {
        "n_rollout_steps": N_ROLLOUT_STEPS,
        "sigma_data": sigma_data,
        "noise_schedule": noise_schedule.clone(),
        "sample_diffusion_cfg": scfg,
        "xl_init": xl_init,
        "steps": steps,
        "xl_final": xl[0].clone(),
        "n_atom": n_atom,
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["sample_diffusion_rollout_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added sample_diffusion_rollout_real: steps", N_ROLLOUT_STEPS,
          "n_atom", n_atom, "sigma_data", sigma_data,
          "noise_schedule", [round(float(x), 4) for x in noise_schedule],
          "xl_final std", float(xl.std()), "shape", tuple(xl[0].shape))
    for i, s in enumerate(steps):
        print(f"  step {i}: t={s['t']:.4f} xl_aug std={float(s['xl_aug'].std()):.4f} "
              f"xl_denoised std={float(s['xl_denoised'].std()):.4f} "
              f"xl_post_step std={float(s['xl_post_step'].std()):.4f}")


if __name__ == "__main__":
    main()
