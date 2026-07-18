#!/usr/bin/env python3
# La-Proteina denoiser — core attention block parity harness (pass 2).
#
# Compares the ttnn port (tt_bio.la_proteina.denoiser.TTPairBiasAttentionAdaLN)
# against the vendored PyTorch reference (MultiHeadBiasedAttentionADALN_MM) on
# identical random weights + identical fixed inputs, component-level PCC.
#
# The golden IS the unmodified vendored reference code (loaded straight from
# _vendor/la-proteina-ref via importlib, bypassing the heavy proteinfoundation
# package __init__ which pulls graphein/torch_geometric/openfold). So parity is
# purely a function of the ttnn op math, not a re-implementation.
#
# Run on qb2 card 1:
#   TT_VISIBLE_DEVICES=1 python3 scripts/la_proteina_attn_parity.py
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
import importlib.util
from pathlib import Path

import torch
import ttnn

# --- locate the vendored reference ------------------------------------------
HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
VENDOR = WORKTREE / "tt_bio" / "la_proteina" / "_vendor" / "la-proteina-ref"
NN_MOD = VENDOR / "proteinfoundation" / "nn" / "modules"
assert NN_MOD.is_dir(), f"vendor missing: {NN_MOD}"

# Stub the package chain so the vendored modules' absolute imports resolve
# without running proteinfoundation/__init__.py (which imports datasets/utils
# and pulls graphein/torch_geometric/openfold).
for name, path in [
    ("proteinfoundation", VENDOR / "proteinfoundation"),
    ("proteinfoundation.nn", VENDOR / "proteinfoundation" / "nn"),
    ("proteinfoundation.nn.modules", NN_MOD),
]:
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m


def _load(mod_name: str, file: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


adaptive_ln = _load("proteinfoundation.nn.modules.adaptive_ln_scale",
                   NN_MOD / "adaptive_ln_scale.py")
pair_bias = _load("proteinfoundation.nn.modules.pair_bias_attn",
                  NN_MOD / "pair_bias_attn.py")

MultiHeadBiasedAttentionADALN_MM = pair_bias.MultiHeadBiasedAttentionADALN_MM

# --- ttnn port --------------------------------------------------------------
sys.path.insert(0, str(WORKTREE))
from tt_bio.la_proteina.denoiser import TTPairBiasAttentionAdaLN, _pcc  # noqa: E402

# --- config (160M denoiser, configs/nn/local_latents_score_nn_160M.yaml) -----
TOKEN_DIM = 768
PAIR_DIM = 256
NHEADS = 12
DIM_COND = 256
USE_QKLN = True
B, N = 1, 64  # tile-aligned seq len
SEED = 1234
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")


def build_golden():
    torch.manual_seed(SEED)
    m = MultiHeadBiasedAttentionADALN_MM(
        dim_token=TOKEN_DIM, dim_pair=PAIR_DIM, nheads=NHEADS,
        dim_cond=DIM_COND, use_qkln=USE_QKLN,
    ).eval()
    return m


def main():
    print(f"[setup] torch={torch.__version__} ttnn={getattr(ttnn,'__version__','?')} dtype={DTYPE}")
    # ---- golden (PyTorch reference) ----
    golden = build_golden()
    sd = {k: v.detach().clone() for k, v in golden.state_dict().items()}

    g = torch.Generator().manual_seed(SEED)
    def rt(*shape):
        return torch.randn(*shape, generator=g)
    x = rt(B, N, TOKEN_DIM)
    pair_rep = rt(B, N, N, PAIR_DIM)
    cond = rt(B, N, DIM_COND)
    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False  # mask the last half of the sequence
    pair_mask = mask[:, :, None] * mask[:, None, :]  # [B, N, N]

    with torch.no_grad():
        golden_out = golden(x, pair_rep, cond, mask)
    print(f"[golden] out {tuple(golden_out.shape)} mean={golden_out.mean():.6f} std={golden_out.std():.6f}")

    # ---- device ----
    dev = ttnn.open_device(device_id=0)
    try:
        arch = dev.arch()
        ck = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        dev.enable_program_cache()
        dtype = ttnn.bfloat16 if DTYPE == "bf16" else ttnn.float32

        port = TTPairBiasAttentionAdaLN(
            dev, ck, sd, token_dim=TOKEN_DIM, pair_dim=PAIR_DIM,
            nheads=NHEADS, dim_cond=DIM_COND, use_qkln=USE_QKLN, dtype=dtype,
        )

        def to_tt(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

        x_tt = to_tt(x)
        pair_tt = to_tt(pair_rep)
        cond_tt = to_tt(cond)
        mask_f = mask.float()
        mask_tt = to_tt(mask_f.unsqueeze(-1))  # [B, N, 1]
        # pair mask as additive bias: 0 where True, -1e4 where False.
        mask_tt = to_tt(mask.float().unsqueeze(-1))
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        out_tt = port(x_tt, pair_tt, cond_tt, mask_tt, pmb_tt)
        out_th = ttnn.to_torch(out_tt).float()
    finally:
        ttnn.close_device(dev)

    pcc = _pcc(out_th, golden_out)
    print(f"[port]   out {tuple(out_th.shape)} mean={out_th.mean():.6f} std={out_th.std():.6f}")
    print(f"[parity] PCC = {pcc:.6f}  (target >= 0.999)")
    # per-channel diagnostic
    if pcc < 0.999:
        import math
        max_abs = (out_th - golden_out).abs().max().item()
        print(f"[diag] max|abs diff| = {max_abs:.6e}")
    ok = pcc >= 0.999
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
