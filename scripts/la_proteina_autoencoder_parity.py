#!/usr/bin/env python3
# La-Proteina autoencoder -- encoder + decoder parity harness (pass 4).
#
# Parity-checks the AE encoder and decoder trunks + heads (ported in
# `tt_bio/la_proteina/autoencoder.py`) against the unmodified vendored
# reference submodules (Transition, MultiheadAttnAndTransition, nn.Sequential
# heads), orchestrated identically to `EncoderTransformer.forward` /
# `DecoderTransformer.forward`. Inputs injected at the post-builder interface
# (seqs, pair_rep, c_pre, mask) so the full 12-layer trunk + cond + heads run
# without porting the FeatureFactory / PairRepBuilder dataset feature pipeline.
#
# Encoder: latent head (LN+Linear 768->16, chunk -> mean/log_scale, z = mean +
# eps*exp(log_scale), ln_z=Identity) -- stochastic, so `eps` is a SHARED draw
# (per memory `diffusion-port-parity-shared-draws`): identical eps fed to
# device port and CPU reference (torch.randn_like patched to return it).
#
# Decoder: logit head (LN+Linear 768->20) and struct head (LN+Linear 768->111,
# reshape [B,N,37,3], abs_coors post-process). Config abs_coors=False: CA slot
# zeroed, then all atoms += ca_coors_nm (parameter-free host math, applied
# identically on both sides).
#
# Config (configs/nn_ae/nn_130m.yaml): nlayers=12, token_dim=768, nheads=12,
# dim_cond=128, pair_repr_dim=256, update_pair_repr=False, use_qkln=True,
# latent_z_dim=8. Random weights, B=1 N=64, bf16, HiFi4 + fp32_dest_acc, both
# all-True and partial masks.
#
# Run on qb2 card 0:
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_autoencoder_parity.py
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
_load("openfold.model.triangular_multiplicative_update",
      OF_MOD / "triangular_multiplicative_update.py")
_load("proteinfoundation.nn.modules.swiglu", NN_MOD / "swiglu.py")
seq_trans = _load("proteinfoundation.nn.modules.seq_transition_af3",
                  NN_MOD / "seq_transition_af3.py")
_load("proteinfoundation.nn.modules.adaptive_ln_scale",
      NN_MOD / "adaptive_ln_scale.py")
_load("proteinfoundation.nn.modules.pair_bias_attn",
      NN_MOD / "pair_bias_attn.py")
attn_n_trans = _load("proteinfoundation.nn.modules.attn_n_transition",
                      NN_MOD / "attn_n_transition.py")
pair_update = _load("proteinfoundation.nn.modules.pair_update",
                    NN_MOD / "pair_update.py")
pair_update.checkpoint = lambda fn, *a, **k: fn(*a, **k)

Transition = seq_trans.Transition
MultiheadAttnAndTransition = attn_n_trans.MultiheadAttnAndTransition

sys.path.insert(0, str(WORKTREE))
from tt_bio.la_proteina.autoencoder import (  # noqa: E402
    TTEncoderTransformer, TTDecoderTransformer,
)
from tt_bio.la_proteina.denoiser import _pcc  # noqa: E402

TOKEN_DIM = 768
PAIR_DIM = 256
NHEADS = 12
DIM_COND = 128
LATENT_DIM = 8
NLAYERS = 12
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


def _build_golden_trunk(dim_cond, nlayers):
    g_tc1 = Transition(dim_cond, expansion_factor=2).eval()
    g_tc2 = Transition(dim_cond, expansion_factor=2).eval()
    g_layers = [
        MultiheadAttnAndTransition(
            dim_token=TOKEN_DIM, dim_pair=PAIR_DIM, nheads=NHEADS, dim_cond=dim_cond,
            residual_mha=True, residual_transition=True, parallel_mha_transition=False,
            use_attn_pair_bias=True, use_qkln=USE_QKLN,
        ).eval()
        for _ in range(nlayers)
    ]
    return g_tc1, g_tc2, g_layers


def _golden_trunk_run(g_tc1, g_tc2, g_layers, seqs, pair_rep, c_pre, mask):
    with torch.no_grad():
        c = g_tc1(c_pre, mask)
        c = g_tc2(c, mask)
        s = seqs * mask[..., None]
        for lay in g_layers:
            s = lay(s, pair_rep, c, mask)
    return s


def _trunk_port_sd(g_tc1, g_tc2, g_layers):
    return {
        "transition_c_1": {k: v.clone() for k, v in g_tc1.state_dict().items()},
        "transition_c_2": {k: v.clone() for k, v in g_tc2.state_dict().items()},
        "transformer_layers": [_layer_port_sd(l.state_dict()) for l in g_layers],
    }


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

        mask_tt = to_tt(mask.float().unsqueeze(-1))
        pmb = torch.where(pair_mask.unsqueeze(1), torch.zeros(B, 1, N, N),
                          torch.full((B, 1, N, N), -1e4))
        pmb_tt = to_tt(pmb)

        # ---------------- Encoder ----------------
        torch.manual_seed(SEED)
        g_tc1, g_tc2, g_layers = _build_golden_trunk(DIM_COND, NLAYERS)
        g_latent = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_DIM),
            torch.nn.Linear(TOKEN_DIM, 2 * LATENT_DIM, bias=False),
        ).eval()
        torch.manual_seed(SEED + 1)
        seqs_in = torch.randn(B, N, TOKEN_DIM)
        pair_rep_in = torch.randn(B, N, N, PAIR_DIM)
        c_pre = torch.randn(B, N, DIM_COND)
        eps = torch.randn(B, N, LATENT_DIM)            # shared stochastic draw

        s_gold = _golden_trunk_run(g_tc1, g_tc2, g_layers, seqs_in, pair_rep_in, c_pre, mask)
        with torch.no_grad():
            flat = g_latent(s_gold) * mask[..., None]
            g_mean, g_log_scale = torch.chunk(flat, 2, dim=-1)
            real_randn_like = torch.randn_like
            torch.randn_like = lambda x: eps
            try:
                g_z = g_mean + torch.randn_like(g_log_scale) * torch.exp(g_log_scale)
            finally:
                torch.randn_like = real_randn_like
            g_z = g_z * mask[..., None]

        enc_sd = _trunk_port_sd(g_tc1, g_tc2, g_layers)
        enc_sd["latent_decoder_mean_n_log_scale"] = {
            k: v.clone() for k, v in g_latent.state_dict().items()
        }
        enc = TTEncoderTransformer(
            dev, ck, enc_sd, token_dim=TOKEN_DIM, pair_dim=PAIR_DIM, nheads=NHEADS,
            dim_cond=DIM_COND, nlayers=NLAYERS, latent_dim=LATENT_DIM,
            use_qkln=USE_QKLN, normalize_latent=False, dtype=dtype,
        )
        p_mean_tt, p_log_tt, p_z_tt = enc(
            to_tt(seqs_in), to_tt(pair_rep_in), to_tt(c_pre), mask_tt, pmb_tt, to_tt(eps),
        )
        p_mean = ttnn.to_torch(p_mean_tt).float()[..., :LATENT_DIM]
        p_log = ttnn.to_torch(p_log_tt).float()[..., :LATENT_DIM]
        p_z = ttnn.to_torch(p_z_tt).float()[..., :LATENT_DIM]
        results.append(("encoder: mean", _pcc(p_mean, g_mean)))
        results.append(("encoder: log_scale", _pcc(p_log, g_log_scale)))
        results.append(("encoder: z_latent (shared eps)", _pcc(p_z, g_z)))

        # ---------------- Decoder ----------------
        torch.manual_seed(SEED)
        g_tc1, g_tc2, g_layers = _build_golden_trunk(DIM_COND, NLAYERS)
        g_logit = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_DIM),
            torch.nn.Linear(TOKEN_DIM, 20, bias=False),
        ).eval()
        g_struct = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_DIM),
            torch.nn.Linear(TOKEN_DIM, 111, bias=False),
        ).eval()
        torch.manual_seed(SEED + 1)
        seqs_in = torch.randn(B, N, TOKEN_DIM)
        pair_rep_in = torch.randn(B, N, N, PAIR_DIM)
        c_pre = torch.randn(B, N, DIM_COND)
        ca_coors_nm = torch.randn(B, N, 3)

        s_gold = _golden_trunk_run(g_tc1, g_tc2, g_layers, seqs_in, pair_rep_in, c_pre, mask)
        with torch.no_grad():
            g_logits = g_logit(s_gold) * mask[..., None]
            coors_flat = g_struct(s_gold) * mask[..., None]
            g_coors = coors_flat.reshape(B, N, 37, 3)
            # abs_coors=False: CA slot zeroed, then all atoms += ca_coors_nm
            g_coors = g_coors.clone()
            g_coors[..., 1, :] = 0.0
            g_coors = g_coors + ca_coors_nm[:, :, None, :]

        dec_sd = _trunk_port_sd(g_tc1, g_tc2, g_layers)
        dec_sd["logit_linear"] = {k: v.clone() for k, v in g_logit.state_dict().items()}
        dec_sd["struct_linear"] = {k: v.clone() for k, v in g_struct.state_dict().items()}
        dec = TTDecoderTransformer(
            dev, ck, dec_sd, token_dim=TOKEN_DIM, pair_dim=PAIR_DIM, nheads=NHEADS,
            dim_cond=DIM_COND, nlayers=NLAYERS, use_qkln=USE_QKLN, dtype=dtype,
        )
        p_logits_tt, p_coors_flat_tt = dec(
            to_tt(seqs_in), to_tt(pair_rep_in), to_tt(c_pre), mask_tt, pmb_tt,
        )
        p_logits = ttnn.to_torch(p_logits_tt).float()[..., :20]
        p_coors_flat = ttnn.to_torch(p_coors_flat_tt).float()[..., :111]
        p_coors = p_coors_flat.reshape(B, N, 37, 3)
        p_coors = p_coors.clone()
        p_coors[..., 1, :] = 0.0
        p_coors = p_coors + ca_coors_nm[:, :, None, :]
        results.append(("decoder: seq_logits", _pcc(p_logits, g_logits)))
        results.append(("decoder: coors_nm (abs_coors=False)", _pcc(p_coors, g_coors)))
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
