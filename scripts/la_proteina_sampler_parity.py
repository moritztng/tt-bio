#!/usr/bin/env python3
# La-Proteina Euler sampler -- integrator parity harness (pass 4).
#
# Parity-checks the per-step Euler integrator (`RDNFlowMatcher.simulation_step`)
# ported in `tt_bio/la_proteina/sampler.py`, against the unmodified vendored
# reference. The golden IS the vendored `simulation_step` (with `torch.randn`
# patched to return a pre-drawn `eps`, so the stochastic draw is IDENTICAL on
# both sides -- per memory `diffusion-port-parity-shared-draws`, a stochastic
# sampler must be compared device-vs-reference sharing the same noise draw,
# never device-vs-an-independently-sampled-golden).
#
# Covers both data modalities (d=3 bb_ca, d=8 local_latents) and all four
# sampling modes (vf, vf_ss, sc, vf_ss_sc_sn), at t values chosen to exercise
# each branch (t < t_lim_ode_below, middle, t > t_lim_ode), with and without
# center_every_step. Random weights, B=1 N=64, bf16, HiFi4 + fp32_dest_acc.
#
# NOTE: this is the integrator slice of the sampler, NOT the full nsteps loop
# around the denoiser. The full loop is gated on the FeatureFactory /
# PairReprBuilder dataset feature-pipeline port (the denoiser NN itself is
# already parity-verified in denoiser.py).
#
# Run on qb2 card 0:
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_sampler_parity.py
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
FM = VENDOR / "proteinfoundation" / "flow_matching"
UTILS = VENDOR / "proteinfoundation" / "utils"


def _stub_pkgs():
    # jaxtyping is not installed in the venv; it is used only for type
    # annotations in the vendored flow matcher, so stub it.
    jx = types.ModuleType("jaxtyping")
    class _Any:
        def __getitem__(self, *a, **k):
            return object
    jx.Float = _Any()
    jx.Bool = _Any()
    sys.modules["jaxtyping"] = jx
    for name, path in [
        ("proteinfoundation", VENDOR / "proteinfoundation"),
        ("proteinfoundation.nn", VENDOR / "proteinfoundation" / "nn"),
        ("proteinfoundation.nn.modules", VENDOR / "proteinfoundation" / "nn" / "modules"),
        ("proteinfoundation.utils", UTILS),
        ("proteinfoundation.flow_matching", FM),
        ("openfold", VENDOR / "openfold"),
        ("openfold.model", VENDOR / "openfold" / "model"),
        ("openfold.utils", VENDOR / "openfold" / "utils"),
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


_load("proteinfoundation.utils.align_utils", UTILS / "align_utils.py")
_load("proteinfoundation.flow_matching.base_flow_matcher", FM / "base_flow_matcher.py")
rdn = _load("proteinfoundation.flow_matching.rdn_flow_matcher", FM / "rdn_flow_matcher.py")

RDNFlowMatcher = rdn.RDNFlowMatcher

sys.path.insert(0, str(WORKTREE))
from tt_bio.la_proteina.sampler import TTEulerStep, _pcc  # noqa: E402

B, N = 1, 64
SEED = 1234
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")
BAR = 0.999

# (data_mode, d)
MODES = [("bb_ca", 3), ("local_latents", 8)]
# (sampling_mode, t, center_every_step) -- t chosen to hit each branch
CASES = [
    ("vf", 0.5, False),
    ("vf", 0.5, True),
    ("vf_ss", 0.03, False),   # t < t_lim_ode_below -> SDE noise-scaling branch
    ("vf_ss", 0.5, False),    # middle -> ODE score-scaling branch
    ("sc", 0.97, False),      # t > t_lim_ode -> low-temp ODE branch
    ("sc", 0.5, True),        # middle -> SDE noise-scaling branch + center
    ("vf_ss_sc_sn", 0.97, False),   # t > t_lim_ode
    ("vf_ss_sc_sn", 0.03, True),    # t < t_lim_ode_below + center
    ("vf_ss_sc_sn", 0.5, False),    # middle -> ODE scaled + SDE scaled
]
DT = 0.05
GT = 0.5
SC_NOISE = 0.6
SC_SCORE = 1.2
T_LIM_ODE = 0.93
T_LIM_ODE_BELOW = 0.07


def _golden_step(rdn_fm, x_t, v, eps, mask, t, sampling_mode, center):
    params = {
        "sampling_mode": sampling_mode,
        "sc_scale_noise": SC_NOISE,
        "sc_scale_score": SC_SCORE,
        "t_lim_ode": T_LIM_ODE,
        "t_lim_ode_below": T_LIM_ODE_BELOW,
        "center_every_step": center,
    }
    real_randn = torch.randn
    torch.randn = lambda *a, **k: eps
    try:
        out = rdn_fm.simulation_step(
            x_t=x_t, nn_out={"v": v}, t=torch.tensor([t]), dt=DT,
            gt=torch.tensor(GT), mask=mask, simulation_step_params=params,
        )
    finally:
        torch.randn = real_randn
    return out


def main():
    print(f"[setup] torch={torch.__version__} ttnn={getattr(ttnn,'__version__','?')} dtype={DTYPE}")
    g = torch.Generator().manual_seed(SEED)

    def rt(*shape):
        return torch.randn(*shape, generator=g)

    mask = torch.ones(B, N, dtype=torch.bool)
    if os.environ.get("LA_PROTEINA_MASK") == "1":
        mask[:, N // 2:] = False

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

        mask_tt = to_tt(mask.float().unsqueeze(-1))   # [B,N,1]
        port = TTEulerStep(dev, ck, dtype=dtype)

        for mode_name, d in MODES:
            rdn_fm = RDNFlowMatcher(dim=d)
            for sampling_mode, t, center in CASES:
                torch.manual_seed(SEED)
                x_t = torch.randn(B, N, d)
                v = torch.randn(B, N, d)
                eps = torch.randn(B, N, d)
                g_out = _golden_step(rdn_fm, x_t, v, eps, mask, t, sampling_mode, center)
                p_out_tt = port(
                    to_tt(x_t), to_tt(v), to_tt(eps), mask_tt,
                    t=t, dt=DT, gt=GT, sampling_mode=sampling_mode,
                    sc_scale_noise=SC_NOISE, sc_scale_score=SC_SCORE,
                    t_lim_ode=T_LIM_ODE, t_lim_ode_below=T_LIM_ODE_BELOW,
                    center_every_step=center,
                )
                p_out = ttnn.to_torch(p_out_tt).float()[..., :d]
                tag = f"{mode_name}(d={d}) {sampling_mode} t={t} center={center}"
                results.append((tag, _pcc(p_out, g_out)))
    finally:
        ttnn.close_device(dev)

    _print(results)


def _print(results):
    print("")
    print(f"{'case':<52} {'PCC':<12} {'bar':<6} result")
    print("-" * 80)
    all_ok = True
    for name, pcc in results:
        ok = pcc >= BAR
        all_ok = all_ok and ok
        print(f"{name:<52} {pcc:<12.6f} {BAR:<6} {'PASS' if ok else 'FAIL'}")
    print("-" * 80)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
