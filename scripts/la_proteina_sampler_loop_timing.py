#!/usr/bin/env python3
# La-Proteina sampler loop -- wall-clock timing harness (pass 6 perf pass).
#
# Measures device-side wall-clock for the full nsteps TTLaProteinaSampler loop
# (denoiser trunk + FeatureFactory/PairReprBuilder + Euler integrator) on random
# weights, B=1 N=64, bf16 trunk+factory, fp32 Euler score math, HiFi4 +
# fp32_dest_acc. Reuses the parity harness module for setup (CFG, state-dict
# builder, _patched_randn) so the timed loop is the SAME code path the parity
# harness exercises.
#
# Reports total + per-step wall-clock for a warm run (program cache primed by a
# single throwaway run). Run on card 0:
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_sampler_loop_timing.py
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
from tt_bio.la_proteina.sampler import TTLaProteinaSampler  # noqa: E402

B, N = parity.B, parity.N
DATA_MODES = parity.DATA_MODES
LATENT_DIMS = parity.LATENT_DIMS
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")


def main():
    print(f"[setup] torch={torch.__version__} dtype={DTYPE}")
    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False
    mask_f = mask.float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    NSTEPS_LIST = [int(x) for x in os.environ.get(
        "LA_PROTEINA_NSTEPS", "3,4,5,6").split(",")]
    REPEATS = int(os.environ.get("LA_PROTEINA_REPEATS", "3"))
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

        gen_holder = [None]
        real_randn, patched = parity._patched_randn(gen_holder)

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
        sampler = TTLaProteinaSampler(
            dev, ck, denoiser, DATA_MODES, parity.SAMPLING_MODEL_ARGS,
            LATENT_DIMS, dtype=dtype, math_dtype=euler_dtype)

        def device_run(nsteps):
            gen_holder[0] = torch.Generator().manual_seed(SEED + 1)
            torch.randn = patched
            try:
                x0_dev = {}
                for dm in DATA_MODES:
                    d = LATENT_DIMS[dm]
                    noise = torch.randn(B, N, d)
                    noise = noise * mask_f[:, :, None]
                    tile_in = ((d + 31) // 32) * 32
                    if tile_in != d:
                        noise = torch.cat(
                            [noise, torch.zeros(B, N, tile_in - d)], dim=-1)
                    x0_dev[dm] = to_tt(noise)
                t0 = time.perf_counter()
                p_x = sampler(x0_dev, mask_tt, pair_mask_tt, pmb_tt,
                              nsteps=nsteps, n=N, self_cond=True)
                ttnn.synchronize_device(dev)
                t1 = time.perf_counter()
                # pull a small slice to force materialization
                _ = ttnn.to_torch(p_x["bb_ca"]).float()[..., :1]
            finally:
                torch.randn = real_randn
            return t1 - t0

        # warmup (primes program cache for the largest nsteps)
        device_run(max(NSTEPS_LIST))
        ttnn.synchronize_device(dev)

        print("")
        print(f"{'nsteps':<8} {'repeats':<8} {'total_ms (median)':<20} {'per_step_ms':<14}")
        print("-" * 56)
        for nsteps in NSTEPS_LIST:
            ts = sorted([device_run(nsteps) for _ in range(REPEATS)])
            med = ts[len(ts) // 2]
            print(f"{nsteps:<8} {REPEATS:<8} {med*1e3:<20.2f} {med*1e3/nsteps:<14.2f}")
        print("-" * 56)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
