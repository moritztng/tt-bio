#!/usr/bin/env python3
# La-Proteina full nsteps flow-matching sampler loop -- parity harness (pass 5).
#
# Parity-checks the full sampler loop (denoiser trunk + Euler integrator +
# FeatureFactory/PairReprBuilder feature pipeline, wired end-to-end) against
# the unmodified vendored reference loop. The golden IS the vendored
# LocalLatentsTransformer (with the vendored FeatureFactory/PairReprBuilder)
# driven by the vendored RDNFlowMatcher per data mode, orchestrated identically
# to product_space_flow_matcher.full_simulation (guidance_w=1.0, no CFG/AG,
# n_recycle=0 -- the uncond 160M config). The device port is TTLaProteinaSampler
# (TTLaProteinaDenoiser + TTEulerStep).
#
# Stochastic draws (initial noise + per-step SDE eps) are SHARED: torch.randn is
# patched to draw from a seeded generator, and the device loop draws eps via
# the same patched torch.randn at the same conditional points + per-data-mode
# order as the reference simulation_step; the generator is reset between the
# golden and device runs so the draws are identical (per memory
# diffusion-port-parity-shared-draws).
#
# Random weights, B=1 N=64, bf16, HiFi4 + fp32_dest_acc, both all-True and partial
# masks, across a few (seed, nsteps) pairs. PCC bar 0.999 on final coordinates
# per data mode.
#
# Run on qb2 card 0:
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_sampler_loop_parity.py
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
NN_DIR = VENDOR / "proteinfoundation" / "nn"
UTILS = VENDOR / "proteinfoundation" / "utils"
FM = VENDOR / "proteinfoundation" / "flow_matching"
OF_MOD = VENDOR / "openfold" / "model"
OF_UTILS = VENDOR / "openfold" / "utils"
assert NN_MOD.is_dir(), f"vendor missing: {NN_MOD}"


def _stub_pkgs():
    # jaxtyping is not installed; used only for type annotations.
    jx = types.ModuleType("jaxtyping")
    class _Any:
        def __getitem__(self, *a, **k):
            return object
    jx.Float = _Any()
    jx.Bool = _Any()
    sys.modules["jaxtyping"] = jx
    # torch_scatter is not installed; only used by features we do not wire.
    sc = types.ModuleType("torch_scatter")
    def _scatter_mean(*a, **k):
        raise RuntimeError("scatter_mean stub hit -- feature not in scope")
    sc.scatter_mean = _scatter_mean
    sys.modules["torch_scatter"] = sc
    for name, path in [
        ("proteinfoundation", VENDOR / "proteinfoundation"),
        ("proteinfoundation.nn", NN_DIR),
        ("proteinfoundation.nn.modules", NN_MOD),
        ("proteinfoundation.utils", UTILS),
        ("proteinfoundation.flow_matching", FM),
        ("openfold", VENDOR / "openfold"),
        ("openfold.model", OF_MOD),
        ("openfold.utils", OF_UTILS),
        ("openfold.np", VENDOR / "openfold" / "np"),
        ("openfold.data", VENDOR / "openfold" / "data"),
    ]:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
    # Stub openfold.np.residue_constants (real one needs `tree`, not installed;
    # only RESTYPE_ATOM37_MASK + atom_types are referenced, by features / helpers
    # we do not exercise).
    rc = types.ModuleType("openfold.np.residue_constants")
    rc.atom_types = []
    rc.RESTYPE_ATOM37_MASK = None
    sys.modules["openfold.np.residue_constants"] = rc
    # Stub openfold.data.data_transforms (heavy; only atom37_to_torsion_angles
    # referenced, by the sidechain feature we do not wire).
    dt = types.ModuleType("openfold.data.data_transforms")
    class _Dummy:
        def atom37_to_torsion_angles(self, *a, **k):
            raise RuntimeError("data_transforms stub hit")
    dt.atom37_to_torsion_angles = lambda *a, **k: _Dummy()
    sys.modules["openfold.data.data_transforms"] = dt


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
of_trimul = _load("openfold.model.triangular_multiplicative_update",
                  OF_MOD / "triangular_multiplicative_update.py")
_load("proteinfoundation.utils.angle_utils", UTILS / "angle_utils.py")
_load("proteinfoundation.utils.align_utils", UTILS / "align_utils.py")
_load("proteinfoundation.utils.fold_utils", UTILS / "fold_utils.py")
_load("proteinfoundation.nn.modules.swiglu", NN_MOD / "swiglu.py")
seq_trans = _load("proteinfoundation.nn.modules.seq_transition_af3",
                  NN_MOD / "seq_transition_af3.py")
_load("proteinfoundation.nn.modules.adaptive_ln_scale", NN_MOD / "adaptive_ln_scale.py")
pair_bias = _load("proteinfoundation.nn.modules.pair_bias_attn", NN_MOD / "pair_bias_attn.py")
attn_n_trans = _load("proteinfoundation.nn.modules.attn_n_transition",
                     NN_MOD / "attn_n_transition.py")
pair_update = _load("proteinfoundation.nn.modules.pair_update", NN_MOD / "pair_update.py")
pair_update.checkpoint = lambda fn, *a, **k: fn(*a, **k)  # passthrough under no_grad
ff = _load("proteinfoundation.nn.feature_factory", NN_DIR / "feature_factory.py")
pri = _load("proteinfoundation.nn.modules.pair_rep_initial", NN_MOD / "pair_rep_initial.py")
llt = _load("proteinfoundation.nn.local_latents_transformer", NN_DIR / "local_latents_transformer.py")
_load("proteinfoundation.flow_matching.base_flow_matcher", FM / "base_flow_matcher.py")
rdn = _load("proteinfoundation.flow_matching.rdn_flow_matcher", FM / "rdn_flow_matcher.py")

FeatureFactory = ff.FeatureFactory
PairReprBuilder = pri.PairReprBuilder
LocalLatentsTransformer = llt.LocalLatentsTransformer
RDNFlowMatcher = rdn.RDNFlowMatcher

sys.path.insert(0, str(WORKTREE))
from tt_bio.la_proteina.feature_factory import (  # noqa: E402
    TTLaProteinaDenoiser, TTFeatureFactory, TTPairReprBuilder,
)
from tt_bio.la_proteina.sampler import TTLaProteinaSampler  # noqa: E402


def _pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


# ---------------------------------------------------------------------------
# 160M uncond denoiser config (configs/nn/local_latents_score_nn_160M.yaml)
# ---------------------------------------------------------------------------
CFG = {
    "name": "local_latents_transformer",
    "output_parameterization": {"bb_ca": "v", "local_latents": "v"},
    "nlayers": 14, "token_dim": 768, "nheads": 12,
    "parallel_mha_transition": False,
    "strict_feats": False,
    "feats_seq": ["xt_bb_ca", "xt_local_latents", "x_sc_bb_ca",
                  "x_sc_local_latents", "optional_ca_coors_nm_seq_feat",
                  "optional_res_type_seq_feat"],
    "feats_cond_seq": ["time_emb_bb_ca", "time_emb_local_latents"],
    "dim_cond": 256, "idx_emb_dim": 256, "t_emb_dim": 256,
    "feats_pair_repr": ["rel_seq_sep", "xt_bb_ca_pair_dists",
                        "x_sc_bb_ca_pair_dists", "optional_ca_pair_dist"],
    "feats_pair_cond": ["time_emb_bb_ca", "time_emb_local_latents"],
    "xt_pair_dist_dim": 30, "xt_pair_dist_min": 0.1, "xt_pair_dist_max": 3,
    "x_sc_pair_dist_dim": 30, "x_sc_pair_dist_min": 0.1, "x_sc_pair_dist_max": 3,
    "seq_sep_dim": 127, "pair_repr_dim": 256,
    "update_pair_repr": False, "update_pair_repr_every_n": 3, "use_tri_mult": False,
    "use_qkln": True, "latent_dim": 8,
}
FEAT_DIMS = {
    "latent_dim": 8, "t_emb_dim": 256, "idx_emb_dim": 256, "seq_sep_dim": 127,
    "xt_pair_dist_dim": 30, "xt_pair_dist_min": 0.1, "xt_pair_dist_max": 3,
    "x_sc_pair_dist_dim": 30, "x_sc_pair_dist_min": 0.1, "x_sc_pair_dist_max": 3,
}
DATA_MODES = ["bb_ca", "local_latents"]
LATENT_DIMS = {"bb_ca": 3, "local_latents": 8}

# inference_base.yaml `model` section (uncond 160M sampling args)
SAMPLING_MODEL_ARGS = {
    "bb_ca": {
        "schedule": {"mode": "log", "p": 2.0},
        "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "sc", "sc_scale_noise": 0.1, "sc_scale_score": 1.0,
            "t_lim_ode": 0.98, "t_lim_ode_below": 0.02, "center_every_step": True,
        },
    },
    "local_latents": {
        "schedule": {"mode": "power", "p": 2.0},
        "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "sc", "sc_scale_noise": 0.1, "sc_scale_score": 1.0,
            "t_lim_ode": 0.98, "t_lim_ode_below": 0.02, "center_every_step": False,
        },
    },
}

B, N = 1, 64
BAR = 0.999
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")


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


def build_port_sd(g_nn):
    gsd = g_nn.state_dict()

    def lin(key):
        # PyTorch Linear weight is [out, in]; the port factory expects [in, out].
        return gsd[key].t().contiguous().clone()
    sd = {
        "cond_factory": {"linear_out.weight": lin("cond_factory.linear_out.weight")},
        "init_repr_factory": {"linear_out.weight": lin("init_repr_factory.linear_out.weight")},
        "pair_repr_builder": {
            "init_repr_factory": {
                "linear_out.weight": lin("pair_repr_builder.init_repr_factory.linear_out.weight"),
                "ln_out.weight": gsd["pair_repr_builder.init_repr_factory.ln_out.weight"].clone(),
                "ln_out.bias": gsd["pair_repr_builder.init_repr_factory.ln_out.bias"].clone(),
            },
            "cond_factory": {
                "linear_out.weight": lin("pair_repr_builder.cond_factory.linear_out.weight"),
                "ln_out.weight": gsd["pair_repr_builder.cond_factory.ln_out.weight"].clone(),
                "ln_out.bias": gsd["pair_repr_builder.cond_factory.ln_out.bias"].clone(),
            },
            "adaln": {
                "norm_cond.weight": gsd["pair_repr_builder.adaln.norm_cond.weight"].clone(),
                "norm_cond.bias": gsd["pair_repr_builder.adaln.norm_cond.bias"].clone(),
                "to_gamma.0.weight": gsd["pair_repr_builder.adaln.to_gamma.0.weight"].clone(),
                "to_gamma.0.bias": gsd["pair_repr_builder.adaln.to_gamma.0.bias"].clone(),
                "to_beta.weight": gsd["pair_repr_builder.adaln.to_beta.weight"].clone(),
            },
        },
        "transition_c_1": _scope(gsd, "transition_c_1."),
        "transition_c_2": _scope(gsd, "transition_c_2."),
        "transformer_layers": [
            _layer_port_sd(_scope(gsd, f"transformer_layers.{i}."))
            for i in range(CFG["nlayers"])
        ],
        "local_latents_linear": _scope(gsd, "local_latents_linear."),
        "ca_linear": _scope(gsd, "ca_linear."),
    }
    # Pad the head Linear out-dim to tile-aligned (32) so the head output v is
    # [B, N, 32] (3 / 8 real lanes + zero pad), matching the tile-aligned x_t
    # the loop feeds back. Real lanes are unchanged (zero rows contribute 0).
    for hk in ("local_latents_linear", "ca_linear"):
        w = sd[hk]["1.weight"]                      # [out, 768]
        out_dim = w.shape[0]
        tile_out = ((out_dim + 31) // 32) * 32
        if tile_out != out_dim:
            pad = torch.zeros(tile_out - out_dim, w.shape[1], dtype=w.dtype)
            sd[hk]["1.weight"] = torch.cat([w, pad], dim=0)
    return sd


def _patched_randn(gen_holder):
    real = torch.randn

    def patched(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            shape = tuple(args[0])
        else:
            shape = tuple(args)
        return real(*shape, generator=gen_holder[0], dtype=torch.float32)
    return real, patched


def golden_loop(g_nn, nsteps, mask, self_cond=True):
    fms = {dm: RDNFlowMatcher(dim=LATENT_DIMS[dm]) for dm in DATA_MODES}
    from tt_bio.la_proteina.sampler import _get_schedule, _get_gt
    ts = {
        dm: _get_schedule(SAMPLING_MODEL_ARGS[dm]["schedule"]["mode"], nsteps,
                          SAMPLING_MODEL_ARGS[dm]["schedule"]["p"]) for dm in DATA_MODES
    }
    gt = {
        dm: _get_gt(ts[dm][:-1], SAMPLING_MODEL_ARGS[dm]["gt"]["mode"],
                     SAMPLING_MODEL_ARGS[dm]["gt"]["p"],
                     SAMPLING_MODEL_ARGS[dm]["gt"]["clamp_val"]) for dm in DATA_MODES
    }
    cpu = torch.device("cpu")
    x = {dm: fms[dm].sample_noise(N, cpu, shape=(B,), mask=mask) for dm in DATA_MODES}
    x_1_pred = None
    with torch.no_grad():
        for step in range(nsteps):
            t = {dm: ts[dm][step] * torch.ones(B) for dm in DATA_MODES}
            dt = {dm: ts[dm][step + 1] - ts[dm][step] for dm in DATA_MODES}
            gt_step = {dm: gt[dm][step] for dm in DATA_MODES}
            batch = {"x_t": x, "t": t, "mask": mask}
            if self_cond and step > 0:
                batch["x_sc"] = x_1_pred
            nn_out = g_nn(batch)
            for dm in DATA_MODES:
                nn_out[dm] = fms[dm].nn_out_add_clean_sample_prediction(
                    x_t=x[dm], t=t[dm], mask=mask, nn_out=nn_out[dm])
            x_1_pred = {dm: nn_out[dm]["x_1"] for dm in DATA_MODES}
            sim_params = {
                dm: SAMPLING_MODEL_ARGS[dm]["simulation_step_params"] for dm in DATA_MODES
            }
            x = {
                dm: fms[dm].simulation_step(
                    x_t=x[dm], nn_out=nn_out[dm], t=t[dm], dt=dt[dm],
                    gt=gt_step[dm], mask=mask, simulation_step_params=sim_params[dm])
                for dm in DATA_MODES
            }
    return x


def main():
    print(f"[setup] torch={torch.__version__} ttnn={getattr(ttnn,'__version__','?')} dtype={DTYPE}")
    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False
    mask_f = mask.float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    # (seed, nsteps) pairs -- a few seeds / timestep counts.
    CASES = [(1234, 3), (1234, 5), (7777, 4), (2026, 6)]

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

        mask_tt = to_tt(mask_f.unsqueeze(-1))           # [B,N,1]
        pair_mask_tt = to_tt(pair_mask.float().unsqueeze(-1))  # [B,N,N,1]
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        gen_holder = [None]
        real_randn, patched = _patched_randn(gen_holder)

        for seed, nsteps in CASES:
            torch.manual_seed(seed)
            g_nn = LocalLatentsTransformer(**CFG).eval()
            port_sd = build_port_sd(g_nn)
            fdt = os.environ.get("LA_PROTEINA_FACTORY_DTYPE", DTYPE)
            factory_dtype = ttnn.float32 if fdt == "fp32" else ttnn.bfloat16
            edt = os.environ.get("LA_PROTEINA_EULER_DTYPE", "fp32")
            euler_dtype = ttnn.float32 if edt == "fp32" else ttnn.bfloat16
            denoiser = TTLaProteinaDenoiser(
                dev, ck, port_sd, CFG, FEAT_DIMS, dtype=dtype,
                factory_dtype=factory_dtype,
            )
            sampler = TTLaProteinaSampler(
                dev, ck, denoiser, DATA_MODES, SAMPLING_MODEL_ARGS,
                LATENT_DIMS, dtype=dtype, math_dtype=euler_dtype)

            # ---- golden run ----
            gen_holder[0] = torch.Generator().manual_seed(seed + 1)
            torch.randn = patched
            try:
                g_x = golden_loop(g_nn, nsteps, mask, self_cond=True)
            finally:
                torch.randn = real_randn

            # ---- device run (same draws: reset generator to same seed) ----
            gen_holder[0] = torch.Generator().manual_seed(seed + 1)
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
                p_x = sampler(x0_dev, mask_tt, pair_mask_tt, pmb_tt,
                               nsteps=nsteps, n=N, self_cond=True)
            finally:
                torch.randn = real_randn

            for dm in DATA_MODES:
                d = LATENT_DIMS[dm]
                p = ttnn.to_torch(p_x[dm]).float()[..., :d]
                g = g_x[dm].float()
                tag = f"seed={seed} nsteps={nsteps} {dm}(d={d})"
                results.append((tag, _pcc(p, g)))
    finally:
        ttnn.close_device(dev)

    _print(results)


def _print(results):
    print("")
    print(f"{'case':<46} {'PCC':<12} {'bar':<6} result")
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
