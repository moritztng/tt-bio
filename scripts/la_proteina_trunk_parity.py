#!/usr/bin/env python3
# La-Proteina denoiser — trunk parity harness (pass 3).
#
# Extends pass-2's component-by-component discipline to the rest of the
# denoiser trunk. The golden IS the unmodified vendored reference code, loaded
# straight from _vendor/la-proteina-ref via importlib (stubbing the heavy
# proteinfoundation and openfold package __init__ chains). Same random-weight
# PCC bar (>= 0.999) as pass 2.
#
# Components parity-checked here (160M denoiser config, B=1 N=64, bf16,
# HiFi4 + fp32_dest_acc, random weights seeded identically on both sides):
#   1. TransitionADALN                       (seq_transition_af3.TransitionADALN)
#   2. conditioning path (transition_c_1/2) (seq_transition_af3.Transition, exp=2)
#   3. output head: local_latents_linear    (LN + Linear 768 -> 8)
#   4. output head: ca_linear                (LN + Linear 768 -> 3)
#   5. MultiheadAttnAndTransition           (full trunk layer stitch)
#   6. PairReprUpdate (use_tri_mult=False)   (pair_update.PairReprUpdate)
#
# Run on qb2 card 1:
#   TT_VISIBLE_DEVICES=1 python3 scripts/la_proteina_trunk_parity.py
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
import importlib.util
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
VENDOR = WORKTREE / "tt_bio" / "la_proteina" / "_vendor" / "la-proteina-ref"
NN_MOD = VENDOR / "proteinfoundation" / "nn" / "modules"
OF_MOD = VENDOR / "openfold" / "model"
OF_UTILS = VENDOR / "openfold" / "utils"
assert NN_MOD.is_dir(), f"vendor missing: {NN_MOD}"


def _stub_pkgs():
    for name, path in [
        ("proteinfoundation", VENDOR / "proteinfoundation"),
        ("proteinfoundation.nn", VENDOR / "proteinfoundation" / "nn"),
        ("proteinfoundation.nn.modules", NN_MOD),
        ("openfold", VENDOR / "openfold"),
        ("openfold.model", OF_MOD),
        ("openfold.utils", OF_UTILS),
    ]:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m


_stub_pkgs()


def _load(mod_name: str, file: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# openfold light deps first (primitives imports checkpointing + tensor_utils)
_load("openfold.utils.checkpointing", OF_UTILS / "checkpointing.py")
_load("openfold.utils.tensor_utils", OF_UTILS / "tensor_utils.py")
_load("openfold.model.primitives", OF_MOD / "primitives.py")
of_pair_transition = _load("openfold.model.pair_transition", OF_MOD / "pair_transition.py")
of_trimul = _load(
    "openfold.model.triangular_multiplicative_update",
    OF_MOD / "triangular_multiplicative_update.py",
)

# proteinfoundation modules
adaptive_ln = _load("proteinfoundation.nn.modules.adaptive_ln_scale",
                    NN_MOD / "adaptive_ln_scale.py")
_load("proteinfoundation.nn.modules.swiglu", NN_MOD / "swiglu.py")
seq_trans = _load("proteinfoundation.nn.modules.seq_transition_af3",
                  NN_MOD / "seq_transition_af3.py")
pair_bias = _load("proteinfoundation.nn.modules.pair_bias_attn",
                  NN_MOD / "pair_bias_attn.py")
attn_n_trans = _load("proteinfoundation.nn.modules.attn_n_transition",
                      NN_MOD / "attn_n_transition.py")
pair_update = _load("proteinfoundation.nn.modules.pair_update",
                    NN_MOD / "pair_update.py")

# checkpoint is a memory opt only; make it a passthrough so the golden runs
# cleanly under no_grad (mathematically identity).
pair_update.checkpoint = lambda fn, *a, **k: fn(*a, **k)

TransitionADALN = seq_trans.TransitionADALN
Transition = seq_trans.Transition
MultiheadAttnAndTransition = attn_n_trans.MultiheadAttnAndTransition
PairReprUpdate = pair_update.PairReprUpdate

sys.path.insert(0, str(WORKTREE))
from tt_bio.la_proteina.denoiser import (  # noqa: E402
    TTTransitionADALN, TTTransition, TTMultiheadAttnAndTransition,
    TTPairReprUpdate, TTLocalLatentsHead, TTCaHead, _pcc,
)

TOKEN_DIM = 768
PAIR_DIM = 256
NHEADS = 12
DIM_COND = 256
LATENT_DIM = 8
USE_QKLN = True
B, N = 1, 64
SEED = 1234
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")
BAR = 0.999


def _scope(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main():
    print(f"[setup] torch={torch.__version__} ttnn={getattr(ttnn,'__version__','?')} dtype={DTYPE}")
    g = torch.Generator().manual_seed(SEED)
    def rt(*shape):
        return torch.randn(*shape, generator=g)

    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False
    pair_mask = mask[:, :, None] * mask[:, None, :]  # [B,N,N]

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

        mask_tt = to_tt(mask.float().unsqueeze(-1))  # [B,N,1]
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        # 1. TransitionADALN
        torch.manual_seed(SEED)
        g_tr = TransitionADALN(dim=TOKEN_DIM, dim_cond=DIM_COND, expansion_factor=4).eval()
        sd = {k: v.detach().clone() for k, v in g_tr.state_dict().items()}
        x = rt(B, N, TOKEN_DIM); cond = rt(B, N, DIM_COND)
        with torch.no_grad():
            go = g_tr(x, cond, mask)
        port_sd = {
            "adaln": _scope(sd, "adaln."),
            "transition": _scope(sd, "transition."),
            "scale_output": _scope(sd, "scale_output."),
        }
        port = TTTransitionADALN(dev, ck, port_sd, dim=TOKEN_DIM, dim_cond=DIM_COND,
                                  expansion_factor=4, dtype=dtype)
        po = ttnn.to_torch(port(to_tt(x), to_tt(cond), mask_tt)).float()
        results.append(("TransitionADALN", _pcc(po, go)))

        # 2. conditioning path (transition_c_1 -> transition_c_2)
        torch.manual_seed(SEED)
        g_tc1 = Transition(DIM_COND, expansion_factor=2).eval()
        g_tc2 = Transition(DIM_COND, expansion_factor=2).eval()
        sd1 = {k: v.detach().clone() for k, v in g_tc1.state_dict().items()}
        sd2 = {k: v.detach().clone() for k, v in g_tc2.state_dict().items()}
        c = rt(B, N, DIM_COND)
        with torch.no_grad():
            gc = g_tc2(g_tc1(c, mask), mask)
        tc1 = TTTransition(dev, ck, sd1, dim=DIM_COND, expansion_factor=2,
                            layer_norm=False, dtype=dtype)
        tc2 = TTTransition(dev, ck, sd2, dim=DIM_COND, expansion_factor=2,
                            layer_norm=False, dtype=dtype)
        c_tt = tc2(tc1(to_tt(c), mask_tt), mask_tt)
        pc = ttnn.to_torch(c_tt).float()
        results.append(("conditioning (transition_c_1/2)", _pcc(pc, gc)))

        # 3. output head: local_latents_linear (LN + Linear 768 -> 8)
        torch.manual_seed(SEED)
        g_llh = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_DIM),
            torch.nn.Linear(TOKEN_DIM, LATENT_DIM, bias=False),
        ).eval()
        sd = {k: v.detach().clone() for k, v in g_llh.state_dict().items()}
        xs = rt(B, N, TOKEN_DIM)
        with torch.no_grad():
            go = g_llh(xs) * mask[..., None]
        port = TTLocalLatentsHead(dev, ck, sd, dim=TOKEN_DIM,
                                   latent_dim=LATENT_DIM, dtype=dtype)
        po = ttnn.to_torch(port(to_tt(xs), mask_tt)).float()
        results.append(("head: local_latents_linear", _pcc(po, go)))

        # 4. output head: ca_linear (LN + Linear 768 -> 3)
        torch.manual_seed(SEED)
        g_cah = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_DIM),
            torch.nn.Linear(TOKEN_DIM, 3, bias=False),
        ).eval()
        sd = {k: v.detach().clone() for k, v in g_cah.state_dict().items()}
        with torch.no_grad():
            go = g_cah(xs) * mask[..., None]
        port = TTCaHead(dev, ck, sd, dim=TOKEN_DIM, dtype=dtype)
        po = ttnn.to_torch(port(to_tt(xs), mask_tt)).float()
        results.append(("head: ca_linear", _pcc(po, go)))

        # 5. MultiheadAttnAndTransition (full trunk layer stitch)
        torch.manual_seed(SEED)
        g_layer = MultiheadAttnAndTransition(
            dim_token=TOKEN_DIM, dim_pair=PAIR_DIM, nheads=NHEADS, dim_cond=DIM_COND,
            residual_mha=True, residual_transition=True, parallel_mha_transition=False,
            use_attn_pair_bias=True, use_qkln=USE_QKLN,
        ).eval()
        sd = {k: v.detach().clone() for k, v in g_layer.state_dict().items()}
        x = rt(B, N, TOKEN_DIM); pair_rep = rt(B, N, N, PAIR_DIM); cond = rt(B, N, DIM_COND)
        with torch.no_grad():
            go = g_layer(x, pair_rep, cond, mask)
        port_sd = {
            "mhba": {
                "adaln": _scope(sd, "mhba.adaln."),
                "mha": _scope(sd, "mhba.mha."),
                "scale_output": _scope(sd, "mhba.scale_output."),
            },
            "transition": {
                "adaln": _scope(sd, "transition.adaln."),
                "transition": _scope(sd, "transition.transition."),
                "scale_output": _scope(sd, "transition.scale_output."),
            },
        }
        port = TTMultiheadAttnAndTransition(
            dev, ck, port_sd, token_dim=TOKEN_DIM, pair_dim=PAIR_DIM, nheads=NHEADS,
            dim_cond=DIM_COND, use_qkln=USE_QKLN, expansion_factor=4,
            residual_mha=True, residual_transition=True, parallel=False, dtype=dtype,
        )
        po = ttnn.to_torch(port(to_tt(x), to_tt(pair_rep), to_tt(cond), mask_tt, pmb_tt)).float()
        results.append(("MultiheadAttnAndTransition (layer)", _pcc(po, go)))

        # 6. PairReprUpdate (use_tri_mult=False)
        torch.manual_seed(SEED)
        g_pru = PairReprUpdate(token_dim=TOKEN_DIM, pair_dim=PAIR_DIM,
                                expansion_factor_transition=2, use_tri_mult=False).eval()
        sd = {k: v.detach().clone() for k, v in g_pru.state_dict().items()}
        x = rt(B, N, TOKEN_DIM); pair_rep = rt(B, N, N, PAIR_DIM)
        with torch.no_grad():
            go = g_pru(x, pair_rep, mask)
        pru_sd = {
            "layer_norm_in.weight": sd["layer_norm_in.weight"],
            "layer_norm_in.bias": sd["layer_norm_in.bias"],
            "linear_x.weight": sd["linear_x.weight"],
            "transition_out": _scope(sd, "transition_out."),
        }
        port = TTPairReprUpdate(dev, ck, pru_sd, token_dim=TOKEN_DIM,
                                 pair_dim=PAIR_DIM, expansion_factor_transition=2,
                                 use_tri_mult=False, dtype=dtype)
        po = ttnn.to_torch(port(to_tt(x), to_tt(pair_rep), mask_tt)).float()
        results.append(("PairReprUpdate (no tri-mult)", _pcc(po, go)))
    finally:
        ttnn.close_device(dev)

    _print(results)


def _print(results):
    print("")
    print(f"{'component':<42} {'PCC':<12} {'bar':<6} result")
    print("-" * 70)
    all_ok = True
    for name, pcc in results:
        ok = pcc >= BAR
        all_ok = all_ok and ok
        print(f"{name:<42} {pcc:<12.6f} {BAR:<6} {'PASS' if ok else 'FAIL'}")
    print("-" * 70)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
