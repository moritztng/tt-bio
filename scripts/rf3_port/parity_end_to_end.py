#!/usr/bin/env python3
"""Score the composed RF3 port end to end against the upstream torch reference.

Shared draws, not shared seeds: the sampler's RNG stream is recorded from the reference
run and replayed into the port, so the two rollouts see identical noise and the RMSD
between them is a property of the port rather than of two diverging random walks.
Comparing two independently-seeded rollouts is invalid and has already cost this repo a
pass on a different diffusion port.

Stages are scored separately -- trunk, distogram, coordinates, confidence -- because a
single end-to-end number cannot say which half of the model moved.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

SEL = {"template": dict(template_selection=["9dfn_A"]), "cyclic": dict(cyclic_chains=["A"])}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def rel_rms(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.std())


def rmsd(a, b):
    return float((a - b).pow(2).sum(-1).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_ref_work/rf3_latest.ckpt")
    ap.add_argument("--fixture", default="rna")
    ap.add_argument("--steps", type=int, default=8,
                    help="sampler timesteps; the default is short so the harness is "
                         "usable, --steps 200 is the shipping configuration")
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--trunk-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.featurize import featurize
    from tt_bio.rf3.host import HostInputs
    from tt_bio.rf3.sampler import Draws
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    d = REPO / "scripts/rf3_port/parity_artifacts" / args.fixture
    prev = os.getcwd()
    os.chdir(d)
    try:
        out = featurize("input.json", n_recycles=args.recycles, diffusion_batch_size=1,
                        seed=42, **SEL.get(args.fixture, {}))[0]
    finally:
        os.chdir(prev)
    f = out["feats"]
    rep_atom_idxs = out.get("ground_truth", {}).get("rep_atom_idxs")

    # --- reference, twice ---------------------------------------------------------
    # bf16 is the arm the port is scored against (the reference forward runs under
    # autocast); fp32 gives the bf16 CEILING, which is what makes a rel_rms readable.
    # 0.06 on a 48-block stack is either at the ceiling or 20x off it, and only the
    # second arm says which. Both arms need a fresh feature dict: autocast casts
    # msa_stack and the template features IN PLACE.
    net, cfg = load_reference(args.ckpt, num_steps=args.steps)

    # `AttentionPairBias.forward` and the DiT's do `A_I.to(torch.bfloat16)`
    # unconditionally -- `force_bfloat16` is hard-coded True, not read from config. So
    # toggling autocast alone does NOT give an fp32 arm: it gives a bf16 activation
    # against an fp32 weight, which raises. Clearing the flag is what actually measures
    # the ceiling; the confidence head hit the same thing one component earlier, where
    # it made the measured ceiling come out as exactly zero.
    forced = [m for m in net.modules() if getattr(m, "force_bfloat16", False)]

    class _no_force:
        def __enter__(self):
            for m in forced:
                m.force_bfloat16 = False

        def __exit__(self, *a):
            for m in forced:
                m.force_bfloat16 = True

    def ref_trunk_run(bf16):
        ff = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in f.items()}
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        from collections import deque
        from contextlib import nullcontext
        with torch.no_grad(), ctx, (nullcontext() if bf16 else _no_force()):
            ro = deque(net.trunk_forward_with_recycling(f=ff,
                                                        n_recycles=args.recycles),
                       maxlen=1).pop()
            return ({k: v.float() for k, v in ro.items()},
                    net.distogram_head(ro["Z_II"]).float())

    ref_trunk, ref_disto = ref_trunk_run(True)
    f32_trunk, f32_disto = ref_trunk_run(False)

    report = {"fixture": args.fixture, "recycles": args.recycles,
              "steps": args.steps, "force_bfloat16_modules": len(forced),
              "stages": []}

    # --- port -------------------------------------------------------------------
    dev = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    t0 = time.time()
    net_cfg = cfg
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=net_cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=net_cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=net_cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.steps,
        with_confidence="confidence_head" in net_cfg)
    report["load_s"] = round(time.time() - t0, 1)

    host = HostInputs.build(f, dev)
    report.update(atoms=host.n_atom, tokens=host.n_token,
                  atoms_padded=host.n_atom_padded)

    t0 = time.time()
    s_inputs, s, z = tt.trunk(host, args.recycles)
    report["trunk_s"] = round(time.time() - t0, 1)

    def back(x, shape):
        return torch.Tensor(ttnn.to_torch(x)).float().reshape(shape)

    def score(name, got, want, f32):
        g = got if isinstance(got, torch.Tensor) else back(got, want.shape)
        ceil = rel_rms(want, f32)
        e = rel_rms(g, want)
        report["stages"].append({
            "tensor": name, "shape": list(want.shape),
            "pcc": round(pcc(g, want), 7), "rel_rms": round(e, 6),
            "ceiling": round(ceil, 6),
            "x_ceiling": round(e / ceil, 2) if ceil else None})

    score("S_inputs", s_inputs, ref_trunk["S_inputs_I"], f32_trunk["S_inputs_I"])
    score("S_trunk", s, ref_trunk["S_I"], f32_trunk["S_I"])
    score("Z_trunk", z, ref_trunk["Z_II"], f32_trunk["Z_II"])
    score("distogram", tt.distogram_head(z), ref_disto, f32_disto)

    if not args.trunk_only:
        # ONE denoiser call, before the rollout. A rollout RMSD alone cannot separate a
        # wrong denoiser from a correct one whose small error is amplified over N steps
        # along a soft coordinate -- the relative placement of two unbonded chains is
        # exactly such a coordinate. Scoring a single call against its own bf16 ceiling
        # does separate them: at the ceiling here, a large rollout RMSD is amplification.
        gen = torch.Generator().manual_seed(0)
        sched = tt.sampler.noise_schedule()
        t_mid = sched[len(sched) // 2].reshape(1)
        x_probe = t_mid * torch.randn(1, host.n_atom, 3, generator=gen)

        def ref_denoise_at(bf16):
            ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
                   else torch.autocast("cpu", enabled=False))
            from contextlib import nullcontext
            with torch.no_grad(), ctx, (nullcontext() if bf16 else _no_force()):
                return net.diffusion_module(
                    X_noisy_L=x_probe.clone(), t=t_mid, f=f,
                    S_inputs_I=ref_trunk["S_inputs_I"], S_trunk_I=ref_trunk["S_I"],
                    Z_trunk_II=ref_trunk["Z_II"]).float()

        one_bf16, one_f32 = ref_denoise_at(True), ref_denoise_at(False)
        one_got = tt.diffusion_module(host, x_probe, t_mid, s_inputs, s, z)
        report["stages"].append({
            "tensor": "denoiser_1step", "shape": list(one_bf16.shape),
            "t": round(float(t_mid), 3),
            "pcc": round(pcc(one_got, one_bf16), 7),
            "rel_rms": round(rel_rms(one_got, one_bf16), 6),
            "ceiling": round(rel_rms(one_bf16, one_f32), 6),
            "x_ceiling": round(rel_rms(one_got, one_bf16)
                               / rel_rms(one_bf16, one_f32), 2)})

        # Record the reference's draws, then replay them into the port.
        rec = Draws()
        coord = torch.zeros(1, host.n_atom, 3)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            def ref_denoise(x_noisy, t):
                return net.diffusion_module(
                    X_noisy_L=x_noisy, t=t, f=f,
                    S_inputs_I=ref_trunk["S_inputs_I"], S_trunk_I=ref_trunk["S_I"],
                    Z_trunk_II=ref_trunk["Z_II"]).float()
            t0 = time.time()
            ref_x, rec = tt.sampler.sample(ref_denoise, coord, 1, draws=rec)
        report["ref_rollout_s"] = round(time.time() - t0, 1)

        t0 = time.time()
        got = tt.predict(f, n_recycles=args.recycles, diffusion_batch_size=1,
                         rep_atom_idxs=rep_atom_idxs, coord_to_be_noised=coord,
                         draws=Draws(rec.values))
        report["port_rollout_s"] = round(time.time() - t0, 1)
        # An RMSD is only readable next to the scale of the thing it is on. The rollout
        # starts at noise scale sigma_data*s_max ~ 2560 A and contracts towards physical
        # coordinates, so at a low --steps the structure is still enormous and an RMSD of
        # a few angstrom is a tiny relative error. `rmsd_rel` divides by the reference
        # structure's own RMS radius, which is comparable across step counts.
        radius = float(ref_x.pow(2).sum(-1).mean().sqrt())
        report["stages"].append({
            "tensor": "X_L", "shape": list(ref_x.shape),
            "rmsd_A": round(rmsd(got["X_L"], ref_x), 4),
            "ref_rms_radius_A": round(radius, 3),
            "rmsd_rel": round(rmsd(got["X_L"], ref_x) / radius, 6),
            "pcc": round(pcc(got["X_L"], ref_x), 7)})
        for k in ("plddt_logits", "pae_logits", "pde_logits", "exp_resolved_logits"):
            if k in got:
                report["stages"].append({"tensor": k, "shape": list(got[k].shape)})

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
