#!/usr/bin/env python3
# La-Proteina FeatureFactory / PairReprBuilder -- isolation parity harness (pass 5).
#
# Parity-checks the ported feature pipeline (TTFeatureFactory seq init_repr +
# cond, TTPairReprBuilder) against the vendored FeatureFactory /
# PairReprBuilder IN ISOLATION (a single forward, no loop, no trunk) -- so this
# separates feature-pipeline error from sampler-loop compounding. Builds
# seqs / c_pre / pair_rep from a fixed (x_t, t, mask) and compares PCC.
#
# Run on qb2 card 0:
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_feature_factory_parity.py
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import importlib.util
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
# reuse the loop harness module (stubs + loads + CFG + build_port_sd)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "loop_harness",
    WORKTREE / "scripts" / "la_proteina_sampler_loop_parity.py",
)
h = importlib.util.module_from_spec(spec)
h.__name__ = "loop_harness"
spec.loader.exec_module(h)

import torch

B, N = h.B, h.N
BAR = 0.999
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")
SEED = 1234


def main():
    print(f"[setup] dtype={DTYPE}")
    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False
    mask_f = mask.float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    torch.manual_seed(SEED)
    g_nn = h.LocalLatentsTransformer(**h.CFG).eval()
    port_sd = h.build_port_sd(g_nn)

    # fixed x_t / t (random)
    torch.manual_seed(SEED + 7)
    x_t = {
        "bb_ca": torch.randn(B, N, h.LATENT_DIMS["bb_ca"]),
        "local_latents": torch.randn(B, N, h.LATENT_DIMS["local_latents"]),
    }
    t = {"bb_ca": 0.5 * torch.ones(B), "local_latents": 0.5 * torch.ones(B)}
    # x_sc present (step>=1 case): use a random x_sc
    x_sc = {
        "bb_ca": torch.randn(B, N, h.LATENT_DIMS["bb_ca"]),
        "local_latents": torch.randn(B, N, h.LATENT_DIMS["local_latents"]),
    }
    batch = {"x_t": x_t, "t": t, "mask": mask, "x_sc": x_sc}

    # golden factory outputs (vendored FeatureFactory / PairReprBuilder)
    with torch.no_grad():
        g_c = g_nn.cond_factory(batch)               # [B, N, dim_cond]
        g_seqs = g_nn.init_repr_factory(batch)      # [B, N, token_dim]
        g_pair = g_nn.pair_repr_builder(batch)     # [B, N, N, pair_dim]

    results = []
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

        # port factories
        from tt_bio.la_proteina.feature_factory import TTFeatureFactory, TTPairReprBuilder
        cond_fac = TTFeatureFactory(
            dev, ck, port_sd["cond_factory"], mode="seq",
            feats=h.CFG["feats_cond_seq"], dim_out=h.CFG["dim_cond"],
            feat_dims=h.FEAT_DIMS, use_ln_out=False, dtype=dtype,
        )
        init_fac = TTFeatureFactory(
            dev, ck, port_sd["init_repr_factory"], mode="seq",
            feats=h.CFG["feats_seq"], dim_out=h.CFG["token_dim"],
            feat_dims=h.FEAT_DIMS, use_ln_out=False, dtype=dtype,
        )
        pair_bld = TTPairReprBuilder(
            dev, ck, port_sd["pair_repr_builder"],
            feats_repr=h.CFG["feats_pair_repr"], feats_cond=h.CFG["feats_pair_cond"],
            dim_feats_out=h.CFG["pair_repr_dim"], dim_cond_pair=h.CFG["dim_cond"],
            feat_dims=h.FEAT_DIMS, dtype=dtype,
        )
        # device batch: x_t/x_sc as device tensors (tile-padded to 32); t scalars
        def to_tt_pad(t):
            d = t.shape[-1]
            tile_in = ((d + 31) // 32) * 32
            if tile_in != d:
                t = torch.cat([t, torch.zeros(B, N, tile_in - d)], dim=-1)
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
        dev_batch = {
            "x_t": {dm: to_tt_pad(x_t[dm]) for dm in h.DATA_MODES},
            "t": {"bb_ca": 0.5, "local_latents": 0.5},
            "mask": mask_tt,
            "x_sc": {dm: to_tt_pad(x_sc[dm]) for dm in h.DATA_MODES},
        }
        p_c_tt = cond_fac(dev_batch, mask_tt, pair_mask_tt, B, N)
        p_seqs_tt = init_fac(dev_batch, mask_tt, pair_mask_tt, B, N)
        p_pair_tt = pair_bld(dev_batch, mask_tt, pair_mask_tt, B, N)
        p_c = ttnn.to_torch(p_c_tt).float()[..., : h.CFG["dim_cond"]]
        p_seqs = ttnn.to_torch(p_seqs_tt).float()[..., : h.CFG["token_dim"]]
        p_pair = ttnn.to_torch(p_pair_tt).float()[..., : h.CFG["pair_repr_dim"]]
        results.append(("cond_factory (c_pre)", h._pcc(p_c, g_c)))
        results.append(("init_repr_factory (seqs)", h._pcc(p_seqs, g_seqs)))
        results.append(("pair_repr_builder (pair_rep)", h._pcc(p_pair, g_pair)))
    finally:
        ttnn.close_device(dev)

    print("")
    print(f"{'component':<40} {'PCC':<12} {'bar':<6} result")
    print("-" * 70)
    all_ok = True
    for name, pcc in results:
        ok = pcc >= BAR
        all_ok = all_ok and ok
        print(f"{name:<40} {pcc:<12.6f} {BAR:<6} {'PASS' if ok else 'FAIL'}")
    print("-" * 70)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
