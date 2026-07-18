#!/usr/bin/env python3
# La-Proteina sampler loop -- per-step breakdown profiler (pass 6 perf pass).
#
# Times the pieces of one sampler step on random weights to find where the
# ~22ms/step goes: cond_factory / init_repr_factory / pair_repr_builder / trunk /
# euler. Reuses the parity harness for setup. Warm run (program cache primed).
#
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_sampler_loop_profile.py
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / "scripts"))
sys.path.insert(0, str(WORKTREE))

import la_proteina_sampler_loop_parity as parity  # noqa: E402
from tt_bio.la_proteina.sampler import _get_schedule, _get_gt, _draws_eps_for_step  # noqa: E402

B, N = parity.B, parity.N
DATA_MODES = parity.DATA_MODES
LATENT_DIMS = parity.LATENT_DIMS
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")


def main():
    mask = torch.ones(B, N, dtype=torch.bool)
    mask_f = mask.float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    NSTEPS = 5
    SEED = 1234

    dev = ttnn.open_device(device_id=0)
    try:
        arch = dev.arch()
        ck = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        dev.enable_program_cache()
        dtype = ttnn.bfloat16 if DTYPE == "bf16" else ttnn.float32

        def to_tt(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

        mask_tt = to_tt(mask_f.unsqueeze(-1))
        pair_mask_tt = to_tt(pair_mask.float().unsqueeze(-1))
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        torch.manual_seed(SEED)
        g_nn = parity.LocalLatentsTransformer(**parity.CFG).eval()
        port_sd = parity.build_port_sd(g_nn)
        fdt = os.environ.get("LA_PROTEINA_FACTORY_DTYPE", DTYPE)
        factory_dtype = ttnn.float32 if fdt == "fp32" else ttnn.bfloat16
        edt = os.environ.get("LA_PROTEINA_EULER_DTYPE", "fp32")
        euler_dtype = ttnn.float32 if edt == "fp32" else ttnn.bfloat16
        denoiser = parity.TTLaProteinaDenoiser(
            dev, ck, port_sd, parity.CFG, parity.FEAT_DIMS, dtype=dtype,
            factory_dtype=factory_dtype,
        )
        from tt_bio.la_proteina.sampler import TTEulerStep
        euler = TTEulerStep(dev, ck, dtype=dtype, math_dtype=euler_dtype)

        # build a representative batch (mid-loop: step 2, with x_sc)
        gen_holder = [torch.Generator().manual_seed(SEED + 1)]
        real_randn, patched = parity._patched_randn(gen_holder)
        torch.randn = patched
        try:
            x = {}
            for dm in DATA_MODES:
                d = LATENT_DIMS[dm]
                noise = torch.randn(B, N, d)
                noise = noise * mask_f[:, :, None]
                tile_in = ((d + 31) // 32) * 32
                if tile_in != d:
                    noise = torch.cat([noise, torch.zeros(B, N, tile_in - d)], dim=-1)
                x[dm] = to_tt(noise)
        finally:
            torch.randn = real_randn

        ts = {dm: _get_schedule(parity.SAMPLING_MODEL_ARGS[dm]["schedule"]["mode"],
                                NSTEPS, parity.SAMPLING_MODEL_ARGS[dm]["schedule"]["p"])
              for dm in DATA_MODES}
        step = 2
        t = {dm: float(ts[dm][step]) for dm in DATA_MODES}
        dt = {dm: float(ts[dm][step + 1] - ts[dm][step]) for dm in DATA_MODES}
        gt = {dm: _get_gt(ts[dm][:-1], parity.SAMPLING_MODEL_ARGS[dm]["gt"]["mode"],
                          parity.SAMPLING_MODEL_ARGS[dm]["gt"]["p"],
                          parity.SAMPLING_MODEL_ARGS[dm]["gt"]["clamp_val"])
              for dm in DATA_MODES}
        gt_step = {dm: float(gt[dm][step]) for dm in DATA_MODES}
        # x_sc = x_1_pred from a prior step (use x as a stand-in for profiling)
        x_sc = {dm: x[dm] for dm in DATA_MODES}
        batch = {"x_t": x, "t": t, "mask": mask_tt, "x_sc": x_sc}

        def sync():
            ttnn.synchronize_device(dev)

        def time_fn(fn, repeats=10):
            fn()  # warmup
            sync()
            ts_ = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn()
                sync()
                ts_.append(time.perf_counter() - t0)
            ts_.sort()
            return ts_[len(ts_) // 2] * 1e3, out

        # precompute c_pre for trunk timing
        def cond():
            return denoiser.cond_factory(batch, mask_tt, pair_mask_tt, B, N)
        def repr_seq():
            return denoiser.init_repr_factory(batch, mask_tt, pair_mask_tt, B, N)
        def pair():
            return denoiser.pair_repr_builder(batch, mask_tt, pair_mask_tt, B, N)

        c_pre = cond(); sync()
        seqs = repr_seq(); sync()
        pair_rep = pair(); sync()
        if denoiser.factory_dtype != denoiser.dtype:
            c_pre = ttnn.typecast(c_pre, denoiser.dtype)
            seqs = ttnn.typecast(seqs, denoiser.dtype)
            pair_rep = ttnn.typecast(pair_rep, denoiser.dtype)

        def trunk():
            return denoiser.trunk(seqs, pair_rep, c_pre, mask_tt, pmb_tt)
        def full_denoiser():
            return denoiser(batch, mask_tt, pair_mask_tt, pmb_tt, B, N)

        # euler step timing (per data mode)
        nn_out = denoiser(batch, mask_tt, pair_mask_tt, pmb_tt, B, N); sync()
        def euler_all():
            x_new = {}
            for dm in DATA_MODES:
                p = parity.SAMPLING_MODEL_ARGS[dm]["simulation_step_params"]
                do_draw = _draws_eps_for_step(t[dm], p["sampling_mode"],
                                              p["t_lim_ode"], p["t_lim_ode_below"])
                d = LATENT_DIMS[dm]
                tile_in = ((d + 31) // 32) * 32
                eps_host = torch.randn(B, N, d, dtype=torch.float32) if do_draw else torch.zeros(B, N, d, dtype=torch.float32)
                if tile_in != d:
                    eps_host = torch.cat([eps_host, torch.zeros(B, N, tile_in - d)], dim=-1)
                eps_tt = ttnn.from_torch(eps_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
                x_new[dm] = euler(x[dm], nn_out[dm]["v"], eps_tt, mask_tt,
                                  t=t[dm], dt=dt[dm], gt=gt_step[dm],
                                  sampling_mode=p["sampling_mode"],
                                  sc_scale_noise=p["sc_scale_noise"],
                                  sc_scale_score=p["sc_scale_score"],
                                  t_lim_ode=p["t_lim_ode"], t_lim_ode_below=p["t_lim_ode_below"],
                                  center_every_step=p["center_every_step"])
            return x_new

        print("")
        print(f"{'piece':<28} {'ms (median)':<14}")
        print("-" * 44)
        for name, fn in [
            ("cond_factory", cond),
            ("init_repr_factory (seq)", repr_seq),
            ("pair_repr_builder", pair),
            ("trunk (14 layers)", trunk),
            ("euler (both modes)", euler_all),
            ("FULL denoiser (1 step)", full_denoiser),
        ]:
            ms, _ = time_fn(fn)
            print(f"{name:<28} {ms:<14.2f}")
        print("-" * 44)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
