#!/usr/bin/env python3
# La-Proteina denoiser -- full multi-layer trunk parity harness (pass 4).
#
# Extends pass 3's component discipline to the FULL trunk forward:
#   - 14x MultiheadAttnAndTransition (the 160M trunk stack)
#   - the conditioning stack (transition_c_1 -> transition_c_2)
#   - both output heads (local_latents_linear -> 8, ca_linear -> 3)
#   - AND the tri-mult pair-representation update path (PairReprUpdate with
#     use_tri_mult=True), exercised both as a component and as part of the
#     full `_tri` trunk (update_pair_repr=True, every_n=2, use_tri_mult=True).
#
# Two configs are parity-checked (random weights, B=1 N=64, bf16, HiFi4 +
# fp32_dest_acc, both all-True and partial masks):
#   A. 160M        : update_pair_repr=False, nlayers=14 (the shipped 160M trunk)
#   B. 160M_tri    : update_pair_repr=True, every_n=2, use_tri_mult=True, nlayers=14
#
# The golden is the unmodified vendored reference code (Transition,
# MultiheadAttnAndTransition, PairReprUpdate, openfold TriangleMultiplication*,
# PairTransition, nn.Sequential heads), orchestrated identically to
# `local_latents_transformer.LocalLatentsTransformer.forward`. Inputs are
# injected at the post-builder interface (seqs, pair_rep, c_pre, mask) so the
# full trunk + cond + heads run without porting the 1990-line feature_factory.
#
# Run on qb2 card 0:
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_full_trunk_parity.py
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


_load("openfold.utils.checkpointing", OF_UTILS / "checkpointing.py")
_load("openfold.utils.tensor_utils", OF_UTILS / "tensor_utils.py")
_load("openfold.model.primitives", OF_MOD / "primitives.py")
_load("openfold.model.pair_transition", OF_MOD / "pair_transition.py")
of_trimul = _load(
    "openfold.model.triangular_multiplicative_update",
    OF_MOD / "triangular_multiplicative_update.py",
)
_load("proteinfoundation.nn.modules.swiglu", NN_MOD / "swiglu.py")
seq_trans = _load("proteinfoundation.nn.modules.seq_transition_af3",
                  NN_MOD / "seq_transition_af3.py")
_load("proteinfoundation.nn.modules.adaptive_ln_scale",
      NN_MOD / "adaptive_ln_scale.py")
pair_bias = _load("proteinfoundation.nn.modules.pair_bias_attn",
                  NN_MOD / "pair_bias_attn.py")
attn_n_trans = _load("proteinfoundation.nn.modules.attn_n_transition",
                      NN_MOD / "attn_n_transition.py")
pair_update = _load("proteinfoundation.nn.modules.pair_update",
                    NN_MOD / "pair_update.py")
# checkpoint is a memory opt only; passthrough so the golden runs under no_grad.
pair_update.checkpoint = lambda fn, *a, **k: fn(*a, **k)

Transition = seq_trans.Transition
MultiheadAttnAndTransition = attn_n_trans.MultiheadAttnAndTransition
PairReprUpdate = pair_update.PairReprUpdate
TriOut = of_trimul.TriangleMultiplicationOutgoing
TriIn = of_trimul.TriangleMultiplicationIncoming

sys.path.insert(0, str(WORKTREE))
from tt_bio.la_proteina.denoiser import (  # noqa: E402
    TTTransition, TTMultiheadAttnAndTransition, TTPairReprUpdate,
    TTLocalLatentsHead, TTCaHead, TTLocalLatentsTransformer, _pcc,
)

TOKEN_DIM = 768
PAIR_DIM = 256
NHEADS = 12
DIM_COND = 256
LATENT_DIM = 8
NLAYERS = 14
USE_QKLN = True
B, N = 1, 64
SEED = 1234
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")
BAR = 0.999


def _scope(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _layer_port_sd(sd):
    return {
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


def _build_golden_layers(nlayers, use_tri_mult):
    layers = [
        MultiheadAttnAndTransition(
            dim_token=TOKEN_DIM, dim_pair=PAIR_DIM, nheads=NHEADS, dim_cond=DIM_COND,
            residual_mha=True, residual_transition=True, parallel_mha_transition=False,
            use_attn_pair_bias=True, use_qkln=USE_QKLN,
        ).eval()
        for _ in range(nlayers)
    ]
    return layers


def _golden_trunk(g_tc1, g_tc2, g_layers, g_pair_updates, g_llh, g_cah,
                  seqs, pair_rep, c_pre, mask, update_pair_repr):
    with torch.no_grad():
        c = g_tc1(c_pre, mask)
        c = g_tc2(c, mask)
        s = seqs * mask[..., None]
        pr = pair_rep
        for i in range(len(g_layers)):
            s = g_layers[i](s, pr, c, mask)
            if update_pair_repr and i < len(g_layers) - 1:
                upd = g_pair_updates[i]
                if upd is not None:
                    pr = upd(s, pr, mask)
        ll = g_llh(s) * mask[..., None]
        ca = g_cah(s) * mask[..., None]
    return ll, ca


def _build_port_sd(g_tc1, g_tc2, g_layers, g_pair_updates, g_llh, g_cah,
                   update_pair_repr, use_tri_mult):
    sd = {
        "transition_c_1": {k: v.clone() for k, v in g_tc1.state_dict().items()},
        "transition_c_2": {k: v.clone() for k, v in g_tc2.state_dict().items()},
        "transformer_layers": [_layer_port_sd(l.state_dict()) for l in g_layers],
        "local_latents_linear": {k: v.clone() for k, v in g_llh.state_dict().items()},
        "ca_linear": {k: v.clone() for k, v in g_cah.state_dict().items()},
    }
    if update_pair_repr:
        sd["pair_update_layers"] = []
        for upd in g_pair_updates:
            if upd is None:
                sd["pair_update_layers"].append(None)
            else:
                flat = {k: v.clone() for k, v in upd.state_dict().items()}
                scoped = {
                    "layer_norm_in.weight": flat["layer_norm_in.weight"],
                    "layer_norm_in.bias": flat["layer_norm_in.bias"],
                    "linear_x.weight": flat["linear_x.weight"],
                    "transition_out": _scope(flat, "transition_out."),
                }
                if "tri_mult_out.layer_norm_in.weight" in flat:
                    scoped["tri_mult_out"] = _scope(flat, "tri_mult_out.")
                    scoped["tri_mult_in"] = _scope(flat, "tri_mult_in.")
                sd["pair_update_layers"].append(scoped)
    return sd


def main():
    print(f"[setup] torch={torch.__version__} ttnn={getattr(ttnn,'__version__','?')} dtype={DTYPE}")
    g = torch.Generator().manual_seed(SEED)

    def rt(*shape):
        return torch.randn(*shape, generator=g)

    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False
    pair_mask = mask[:, :, None] * mask[:, None, :]

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

        mask_tt = to_tt(mask.float().unsqueeze(-1))      # [B,N,1]
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        def run_config(name, update_pair_repr, every_n, use_tri_mult):
            torch.manual_seed(SEED)
            g_tc1 = Transition(DIM_COND, expansion_factor=2).eval()
            g_tc2 = Transition(DIM_COND, expansion_factor=2).eval()
            g_layers = _build_golden_layers(NLAYERS, use_tri_mult)
            g_llh = torch.nn.Sequential(
                torch.nn.LayerNorm(TOKEN_DIM),
                torch.nn.Linear(TOKEN_DIM, LATENT_DIM, bias=False),
            ).eval()
            g_cah = torch.nn.Sequential(
                torch.nn.LayerNorm(TOKEN_DIM),
                torch.nn.Linear(TOKEN_DIM, 3, bias=False),
            ).eval()
            g_pair_updates = []
            if update_pair_repr:
                for i in range(NLAYERS - 1):
                    if i % every_n == 0:
                        g_pair_updates.append(
                            PairReprUpdate(token_dim=TOKEN_DIM, pair_dim=PAIR_DIM,
                                           expansion_factor_transition=2,
                                           use_tri_mult=use_tri_mult).eval()
                        )
                    else:
                        g_pair_updates.append(None)

            # fresh random inputs (deterministic via the seeded generator)
            torch.manual_seed(SEED + 1)
            seqs_in = torch.randn(B, N, TOKEN_DIM)
            pair_rep_in = torch.randn(B, N, N, PAIR_DIM)
            c_pre = torch.randn(B, N, DIM_COND)

            g_ll, g_ca = _golden_trunk(
                g_tc1, g_tc2, g_layers, g_pair_updates, g_llh, g_cah,
                seqs_in, pair_rep_in, c_pre, mask, update_pair_repr,
            )

            port_sd = _build_port_sd(
                g_tc1, g_tc2, g_layers, g_pair_updates, g_llh, g_cah,
                update_pair_repr, use_tri_mult,
            )
            port = TTLocalLatentsTransformer(
                dev, ck, port_sd, token_dim=TOKEN_DIM, pair_dim=PAIR_DIM,
                nheads=NHEADS, dim_cond=DIM_COND, latent_dim=LATENT_DIM,
                nlayers=NLAYERS, use_qkln=USE_QKLN,
                update_pair_repr=update_pair_repr,
                update_pair_repr_every_n=every_n,
                use_tri_mult=use_tri_mult, dtype=dtype,
            )
            p_ll_tt, p_ca_tt = port(
                to_tt(seqs_in), to_tt(pair_rep_in), to_tt(c_pre), mask_tt, pmb_tt,
            )
            p_ll = ttnn.to_torch(p_ll_tt).float()
            p_ca = ttnn.to_torch(p_ca_tt).float()
            results.append((f"{name}: local_latents", _pcc(p_ll, g_ll)))
            results.append((f"{name}: ca", _pcc(p_ca, g_ca)))

        # also a standalone tri-mult PairReprUpdate component check
        def run_trimul_component():
            torch.manual_seed(SEED)
            g_pru = PairReprUpdate(token_dim=TOKEN_DIM, pair_dim=PAIR_DIM,
                                   expansion_factor_transition=2,
                                   use_tri_mult=True).eval()
            x = torch.randn(B, N, TOKEN_DIM)
            pr = torch.randn(B, N, N, PAIR_DIM)
            with torch.no_grad():
                go = g_pru(x, pr, mask)
            pru_flat = {k: v.clone() for k, v in g_pru.state_dict().items()}
            pru_sd = {
                "layer_norm_in.weight": pru_flat["layer_norm_in.weight"],
                "layer_norm_in.bias": pru_flat["layer_norm_in.bias"],
                "linear_x.weight": pru_flat["linear_x.weight"],
                "transition_out": _scope(pru_flat, "transition_out."),
                "tri_mult_out": _scope(pru_flat, "tri_mult_out."),
                "tri_mult_in": _scope(pru_flat, "tri_mult_in."),
            }
            port = TTPairReprUpdate(
                dev, ck, pru_sd, token_dim=TOKEN_DIM, pair_dim=PAIR_DIM,
                expansion_factor_transition=2, use_tri_mult=True, dtype=dtype,
            )
            po = ttnn.to_torch(port(to_tt(x), to_tt(pr), mask_tt)).float()
            results.append(("PairReprUpdate (tri-mult, component)", _pcc(po, go)))

        run_trimul_component()
        run_config("160M (no pair update)", False, 3, False)
        run_config("160M_tri (tri-mult pair update)", True, 2, True)
    finally:
        ttnn.close_device(dev)

    _print(results)


def _print(results):
    print("")
    print(f"{'component':<46} {'PCC':<12} {'bar':<6} result")
    print("-" * 74)
    all_ok = True
    for name, pcc in results:
        ok = pcc >= BAR
        all_ok = all_ok and ok
        print(f"{name:<46} {pcc:<12.6f} {BAR:<6} {'PASS' if ok else 'FAIL'}")
    print("-" * 74)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
