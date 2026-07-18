#!/usr/bin/env python3
# La-Proteina traced full-step sampler -- parity + timing (pass 7).
#
# Tests the pass-7 lever: capture the FULL per-step device compute (factories +
# trunk + heads + x_1 + Euler) as one trace per step, so no eager ttnn compute
# runs between replays (only sanctioned copy_host_to_device_tensor). Verifies
# (a) the pass-6 compounding-error drift is gone (PCC stable across nsteps) and
# (b) wall-clock vs the eager pass-6 loop.
#
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_traced_step_experiment.py
#
# SPDX-License-Identifier: Apache-2.0

import os, sys, time
from pathlib import Path
import torch, ttnn

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / "scripts"))
sys.path.insert(0, str(WORKTREE))

import la_proteina_sampler_loop_parity as parity
from tt_bio.la_proteina.traced_sampler import TTLaProteinaTracedSampler

B, N = parity.B, parity.N
DATA_MODES = parity.DATA_MODES
LATENT_DIMS = parity.LATENT_DIMS
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf32" if False else "bf16")


def _pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    mode = os.environ.get("LA_PROTEINA_MODE", "parity")
    print(f"[setup] torch={torch.__version__} dtype=bf16 mode={mode}")
    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False
    mask_f = mask.float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    dev = ttnn.open_device(device_id=0, trace_region_size=1 << 30)
    try:
        arch = dev.arch()
        ck = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        dev.enable_program_cache()
        dtype = ttnn.bfloat16
        fdt = os.environ.get("LA_PROTEINA_FACTORY_DTYPE", "bf16")
        factory_dtype = ttnn.float32 if fdt == "fp32" else ttnn.bfloat16
        edt = os.environ.get("LA_PROTEINA_EULER_DTYPE", "fp32")
        euler_dtype = ttnn.float32 if edt == "fp32" else ttnn.bfloat16

        def to_tt(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
        mask_tt = to_tt(mask_f.unsqueeze(-1))
        pair_mask_tt = to_tt(pair_mask.float().unsqueeze(-1))
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        gen_holder = [None]
        real_randn, patched = parity._patched_randn(gen_holder)

        def build(seed):
            torch.manual_seed(seed)
            g_nn = parity.LocalLatentsTransformer(**parity.CFG).eval()
            port_sd = parity.build_port_sd(g_nn)
            denoiser = parity.TTLaProteinaDenoiser(
                dev, ck, port_sd, parity.CFG, parity.FEAT_DIMS, dtype=dtype,
                factory_dtype=factory_dtype)
            sampler = TTLaProteinaTracedSampler(
                dev, ck, denoiser, DATA_MODES, parity.SAMPLING_MODEL_ARGS,
                LATENT_DIMS, parity.FEAT_DIMS, parity.CFG, dtype=dtype,
                math_dtype=euler_dtype, factory_dtype=factory_dtype)
            return g_nn, sampler

        if mode == "parity":
            CASES = [(1234, 3), (1234, 5), (7777, 4), (2026, 6)]
            results = []
            for seed, nsteps in CASES:
                g_nn, sampler = build(seed)
                gen_holder[0] = torch.Generator().manual_seed(seed + 1)
                torch.randn = patched
                try:
                    g_x = parity.golden_loop(g_nn, nsteps, mask, self_cond=True)
                finally:
                    torch.randn = real_randn
                gen_holder[0] = torch.Generator().manual_seed(seed + 1)
                torch.randn = patched
                try:
                    x0_dev = {}
                    for dm in DATA_MODES:
                        d = LATENT_DIMS[dm]
                        noise = torch.randn(B, N, d) * mask_f[:, :, None]
                        tile_in = ((d + 31) // 32) * 32
                        if tile_in != d:
                            noise = torch.cat([noise, torch.zeros(B, N, tile_in - d)], dim=-1)
                        x0_dev[dm] = to_tt(noise)
                    p_x = sampler(x0_dev, mask_tt, pair_mask_tt, pmb_tt,
                                  nsteps=nsteps, n=N, self_cond=True)
                finally:
                    torch.randn = real_randn
                for dm in DATA_MODES:
                    d = LATENT_DIMS[dm]
                    p = ttnn.to_torch(p_x[dm]).float()[..., :d]
                    g = g_x[dm].float()
                    results.append((f"seed={seed} n={nsteps} {dm}", _pcc(p, g)))
                sampler._release()
            print("")
            for name, pcc in results:
                tag = chr(80)+chr(65)+chr(83)+chr(83) if pcc >= 0.999 else chr(70)+chr(65)+chr(73)+chr(76)
                print(f"{name:<28} PCC={pcc:.6f}  {tag}")
            ok = all(p >= 0.999 for _, p in results)
            print("RESULT:", "PASS" if ok else "FAIL")
            sys.exit(0 if ok else 1)

        elif mode == "timing":
            SEED = 1234
            g_nn, sampler = build(SEED)
            NSTEPS = [int(x) for x in os.environ.get("LA_PROTEINA_NSTEPS", "5").split(",")]
            REPEATS = int(os.environ.get("LA_PROTEINA_REPEATS", "5"))

            def run(nsteps):
                gen_holder[0] = torch.Generator().manual_seed(SEED + 1)
                torch.randn = patched
                try:
                    x0_dev = {}
                    for dm in DATA_MODES:
                        d = LATENT_DIMS[dm]
                        noise = torch.randn(B, N, d) * mask_f[:, :, None]
                        tile_in = ((d + 31) // 32) * 32
                        if tile_in != d:
                            noise = torch.cat([noise, torch.zeros(B, N, tile_in - d)], dim=-1)
                        x0_dev[dm] = to_tt(noise)
                    t0 = time.perf_counter()
                    p_x = sampler(x0_dev, mask_tt, pair_mask_tt, pmb_tt,
                                  nsteps=nsteps, n=N, self_cond=True)
                    ttnn.synchronize_device(dev)
                    t1 = time.perf_counter()
                    _ = ttnn.to_torch(p_x["bb_ca"]).float()[..., :1]
                finally:
                    torch.randn = real_randn
                return t1 - t0
            run(max(NSTEPS))  # warmup (capture + prime)
            ttnn.synchronize_device(dev)
            print("")
            for nsteps in NSTEPS:
                ts = sorted([run(nsteps) for _ in range(REPEATS)])
                med = ts[len(ts) // 2]
                print(f"nsteps={nsteps}  total_ms={med*1e3:.2f}  per_step={med*1e3/nsteps:.2f}")
            sampler._release()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
