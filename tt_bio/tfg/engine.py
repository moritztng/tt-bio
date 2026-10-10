"""TFG config parsing and the denoiser-free guidance update (host torch).

Port of OpenDDE v1.2.0 `opendde/tfg/config.py` and `opendde/tfg/engine.py` (Apache-2.0,
Copyright (c) 2026 Aureka AI Research, Copyright 2024 ByteDance and/or its affiliates), modified:
upstream's `TFGEngine.step` calls the denoiser itself; tt-bio's samplers own the denoiser call, so
`TFGEngine.update` takes the denoised `x0` and does the rest of upstream's step (projection,
x0 hook, mu refinement, predictor-corrector). Only rho == 0 and tfg_outer == 1 are supported.

The potentials define E(x); guidance treats p(x0) ~ exp(-E(x0)) and does gradient ascent on log p.
"""

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from tt_bio.tfg import potentials
from tt_bio.tfg.potentials import Potential

logger = logging.getLogger(__name__)

_DEFAULT_GUIDANCE: dict[str, Any] = {
    "enable": False,
    "log_last_step_energy": True,
    "rho": 0.0,
    "mu": 0.1,
    "mc": {"std": 0.0, "batch": 1},
    "steps": {"tfg_outer": 1, "tfg_inner": 20, "projection_outer": 2, "projection_inner": 10},
    "terms": {
        "VinaStericPotential": {"interval": 1, "weight": 0.1, "buffer": 0.225},
        "ExperimentalTorsionPotential": {"interval": 1, "weight": 0.0015},
        "InterchainBondPotential": {"interval": 1, "weight": 0.15, "buffer": 2.0},
        "UserDistanceRestraintPotential": {"interval": 1, "weight": 0.5, "enable_projection": True},
        "PairwiseDistancePotential": {
            "interval": 1,
            "weight": 0.5,
            "enable_projection": True,
            "bond_buffer": 0.00,
            "angle_buffer": 0.00,
            "clash_buffer": 0.00,
        },
        "ChiralAtomPotential": {"interval": 1, "weight": 0.0, "enable_projection": True, "buffer": 0.6155},
        "StereoBondPotential": {"interval": 1, "weight": 0.25, "buffer": 0.52360},
        "PlanarImproperPotential": {"interval": 1, "weight": 0.12},
        "LinearBondPotential": {"interval": 1, "weight": 0.25, "buffer": 0.08726646259},
    },
}


def default_guidance_config() -> dict[str, Any]:
    """OpenDDE v1.2.0 `sample_diffusion.guidance` (config/model_base.py), as a fresh deep copy.

    `enable` is False there; callers that want guidance set it True.
    """
    return copy.deepcopy(_DEFAULT_GUIDANCE)


class Schedule:
    """Maps normalized time t in [0, 1] to a float."""

    def __call__(self, t: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Schedule):
    value: float

    def __call__(self, t: float) -> float:
        return float(self.value)


@dataclass(frozen=True)
class ExponentialInterpolation(Schedule):
    """start + (end - start) * w(t); w(t) = t if alpha == 0 else (e^(alpha t) - 1) / (e^alpha - 1)."""

    start: float
    end: float
    alpha: float = 0.0

    def __call__(self, t: float) -> float:
        if self.alpha == 0.0:
            return float(self.start + (self.end - self.start) * t)
        num = math.exp(self.alpha * t) - 1.0
        den = math.exp(self.alpha) - 1.0
        return float(self.start + (self.end - self.start) * (num / den))


def schedule_from_cfg(obj: Any) -> Schedule:
    """Schedule from a Schedule, a number (constant), or `{"type": "const"|"exp_interpolation", ...}`."""
    if isinstance(obj, Schedule):
        return obj
    if isinstance(obj, (int, float)):
        return Constant(float(obj))
    if isinstance(obj, Mapping):
        cfg = dict(obj)
        if "type" not in cfg:
            raise KeyError("Schedule config must contain key 'type'. Examples: {'type': 'const', 'value': 1.0}")
        t = str(cfg["type"]).lower()
        if t == "const":
            return Constant(float(cfg["value"]))
        if t == "exp_interpolation":
            return ExponentialInterpolation(
                start=float(cfg["start"]), end=float(cfg["end"]), alpha=float(cfg.get("alpha", 0.0))
            )
        raise ValueError(f"Unknown schedule type: {t}")
    raise TypeError(f"Unsupported schedule config type: {type(obj)}")


# Feature keys each potential reads; checked once at step 0.
_REQUIRED_FEATURES: dict[str, set[str]] = {
    "PairwiseDistancePotential": {
        "pairwise_distance_index",
        "pairwise_distance_is_bond",
        "pairwise_distance_is_angle",
        "pairwise_distance_upper_bound",
        "pairwise_distance_lower_bound",
        "ref_element",
    },
    "InterchainBondPotential": {"interchain_bond_index"},
    "UserDistanceRestraintPotential": {
        "user_distance_restraint_index",
        "user_distance_restraint_lower_bound",
        "user_distance_restraint_upper_bound",
    },
    "VinaStericPotential": {"asym_id", "atom_to_token_idx", "ref_element", "interchain_bond_index"},
    "StereoBondPotential": {"stereo_bond_index", "stereo_bond_orientation"},
    "ChiralAtomPotential": {"chiral_index", "chiral_orientation", "asym_id", "atom_to_token_idx"},
    "PlanarImproperPotential": {"planar_improper_index", "planar_improper_is_carbonyl"},
    "LinearBondPotential": {"linear_triple_bond_index"},
    "ExperimentalTorsionPotential": {
        "experimental_torsion_index",
        "experimental_torsion_force_constant",
        "experimental_torsion_sign",
    },
}


@dataclass
class Term:
    """One potential with an interval (active when step_i % interval == 0), a weight schedule
    (scales energy and gradient) and parameters that may be Schedules."""

    name: str
    interval: int
    weight: Schedule
    param_templates: dict[str, Any]
    _potential: Any
    enable_projection: bool = True

    def required_features(self) -> set[str]:
        return set(_REQUIRED_FEATURES.get(self.name, set()))

    def active(self, step_i: int) -> bool:
        return self.interval > 0 and (step_i % self.interval == 0)

    def inert(self, feats) -> bool:
        """True when the potential has nothing to act on in `feats` (exact zeros everywhere); cached per feats."""
        if getattr(self, "_inert_for", None) is not feats:
            self._inert_for, self._inert = feats, self._potential.inert(feats)
        return self._inert

    def _params_at(self, t: float) -> dict[str, Any]:
        return {k: v(t) if isinstance(v, Schedule) else v for k, v in self.param_templates.items()}

    def energy(self, coords, feats, t: float) -> torch.Tensor:
        w = float(self.weight(t))
        if w == 0.0:
            return torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)
        return self._potential.energy(coords, feats, self._params_at(t)) * w

    def energy_and_grad(self, coords, feats, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        w = float(self.weight(t))
        if w == 0.0:
            return (
                torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype),
                torch.zeros_like(coords),
            )
        e, g = self._potential.energy_and_grad(coords, feats, self._params_at(t))
        return e * w, g * w

    def project(self, coords, feats, t: float) -> torch.Tensor:
        """Projection delta; not weighted (weight 0 still projects)."""
        if not hasattr(self._potential, "project"):
            return torch.zeros_like(coords)
        return self._potential.project(coords, feats, self._params_at(t))


def build_terms(term_cfg: Mapping[str, Any] | None) -> list[Term]:
    """Terms from `{PotentialName: {interval=1, weight=0.0, enable_projection=True, **params}}`."""
    terms: list[Term] = []
    if term_cfg is None:
        return terms
    if not isinstance(term_cfg, Mapping):
        raise TypeError(f"terms must be a mapping of term_name -> term_config, got {type(term_cfg)}")
    for name, raw in term_cfg.items():
        raw = dict(raw or {})
        interval = int(raw.pop("interval", 1))
        weight = schedule_from_cfg(raw.pop("weight", 0.0))
        enable_projection = bool(raw.pop("enable_projection", True))
        param_templates = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "type" in v:
                try:
                    param_templates[k] = schedule_from_cfg(v)
                    continue
                except Exception:
                    pass  # not a schedule spec: kept as a raw dict, as upstream does
            param_templates[k] = v
        if name not in potentials.CLASS_REGISTRY:
            raise KeyError(f"Unknown potential '{name}'. Available: {sorted(potentials.CLASS_REGISTRY.keys())}")
        terms.append(
            Term(
                name=name,
                interval=interval,
                weight=weight,
                param_templates=param_templates,
                _potential=potentials.CLASS_REGISTRY[name](),
                enable_projection=enable_projection,
            )
        )
    return terms


def validate_features(feats: Mapping[str, Any], terms: Iterable[Term]) -> None:
    """Raise KeyError listing, per term, the required feature keys missing from `feats`."""
    missing: dict[str, list[str]] = {}
    for term in terms:
        miss = [k for k in term.required_features() if k not in feats]
        if miss:
            missing[term.name] = miss
    if missing:
        lines = ["TFG is missing required input features:"]
        lines += [f"- {name}: {miss}" for name, miss in missing.items()]
        raise KeyError("\n".join(lines))


@dataclass(frozen=True)
class TFGConfig:
    enable: bool
    rho: float
    mu: float
    eps_std: float
    eps_batch: int
    outer_steps: int
    inner_steps: int
    projection_outer_steps: int
    projection_inner_steps: int
    terms: tuple[Term, ...]
    log_last_step_energy: bool = False


def parse_tfg_config(guidance_cfg: Mapping[str, Any] | None) -> TFGConfig:
    """TFGConfig from a guidance mapping (keys: enable, rho, mu, mc, steps, terms, log_last_step_energy).

    None means disabled. Unknown top-level keys raise. Missing step counts default to tfg_outer 1,
    tfg_inner 10, projection_outer 2, projection_inner 10 (upstream's parser defaults).
    """
    if guidance_cfg is None:
        return TFGConfig(
            enable=False,
            rho=0.0,
            mu=0.0,
            eps_std=0.0,
            eps_batch=1,
            outer_steps=1,
            inner_steps=0,
            projection_outer_steps=0,
            projection_inner_steps=0,
            terms=(),
        )
    cfg = dict(guidance_cfg)
    extra = set(cfg) - {"enable", "rho", "mu", "mc", "steps", "terms", "log_last_step_energy"}
    if extra:
        raise KeyError(f"Unsupported keys in TFG config: {sorted(extra)}")
    mc = dict(cfg.get("mc", {}))
    steps = dict(cfg.get("steps", {}))
    enable = bool(cfg.get("enable", False))
    terms = tuple(build_terms(cfg.get("terms", {})))
    if enable and len(terms) == 0:
        raise ValueError("TFG is enabled but no terms are configured")
    return TFGConfig(
        enable=enable,
        rho=float(cfg.get("rho", 0.0)),
        mu=float(cfg.get("mu", 0.0)),
        eps_std=float(mc.get("std", 0.0)),
        eps_batch=max(1, int(mc.get("batch", 1))),
        outer_steps=max(1, int(steps.get("tfg_outer", 1))),
        inner_steps=max(0, int(steps.get("tfg_inner", 10))),
        projection_outer_steps=max(0, int(steps.get("projection_outer", 2))),
        projection_inner_steps=max(0, int(steps.get("projection_inner", 10))),
        terms=terms,
        log_last_step_energy=bool(cfg.get("log_last_step_energy", False)),
    )


def _sample_eps(std, shape, *, k, device, dtype, generator=None) -> torch.Tensor:
    """`[k, *shape]` Gaussian perturbations of std `std`; std 0 returns `[1, *shape]` zeros, no draw."""
    if std == 0.0:
        return torch.zeros((1, *shape), device=device, dtype=dtype)
    return std * torch.randn((k, *shape), device=device, dtype=dtype, generator=generator)


def _logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def _normalized_time(step_i: int, num_diffusion_steps: int) -> float:
    """1 at the first (noisiest) step, -> 0 at the last; potentials' schedules read this t."""
    return 1.0 - float(step_i) / float(max(1, num_diffusion_steps))


class TFGEngine:
    """Evaluates the configured terms and applies them to one diffusion step."""

    def __init__(self, cfg: TFGConfig, *, device=None, dtype=torch.float32):
        if cfg.rho != 0.0:
            raise ValueError(
                f"TFG rho={cfg.rho}: denoiser-path guidance needs autograd through the denoiser, "
                "which the device denoiser does not provide; only rho == 0 is supported"
            )
        if cfg.outer_steps != 1:
            raise ValueError(f"TFG tfg_outer={cfg.outer_steps}: only tfg_outer == 1 is supported")
        self.cfg = cfg
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        # Fixed projection order: chirality, then pairwise distances, then the rest (stable sort).
        order = {"ChiralAtomPotential": 0, "PairwiseDistancePotential": 1}
        self._projection_terms_sorted = sorted(cfg.terms, key=lambda x: order.get(x.name, 2))

    def _energy_and_grad(self, coords, feats, *, t: float, step_i: int):
        """Sum of active weighted terms: `(E [*batch], dE/dx [*batch, N, 3])`."""
        energy = torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)
        grad = torch.zeros_like(coords)
        for term in self.cfg.terms:
            if not term.active(step_i) or term.inert(feats):      # an inert term adds exact zeros
                continue
            e, g = term.energy_and_grad(coords, feats, t)
            energy = energy + e
            grad = grad + g
        return energy, grad

    def _logp_and_grad_x0(self, x0, eps, feats, *, t: float, step_i: int, log_components: bool = False):
        """Monte-Carlo `log p(x0)` and `d log p / d x0` over perturbations `eps [K, *x0.shape]`."""
        k = eps.shape[0]
        x0_eps = (x0.unsqueeze(0) + eps).reshape(-1, *x0.shape[-2:])
        e, g = self._energy_and_grad(x0_eps, feats, t=t, step_i=step_i)
        e = e.reshape(k, *x0.shape[:-2])
        g = g.reshape(k, *x0.shape)
        logp = -e
        if k == 1:
            avg_logp = logp.squeeze(0)
            grad = (-g).squeeze(0)
        else:
            avg_logp = _logmeanexp(logp, dim=0)
            w = torch.softmax(logp, dim=0)
            grad = (w.unsqueeze(-1).unsqueeze(-1) * -g).sum(dim=0)
        if log_components:
            with torch.no_grad():
                for term in self.cfg.terms:
                    if term.active(step_i):
                        e_t = term.energy(x0, feats, t)
                        logger.info(f"TFG last-step energy {term.name}: {e_t.detach().cpu().tolist()}")
        return avg_logp, grad

    def _project(self, coords, feats, *, t: float, step_i: int) -> torch.Tensor:
        """Accumulated projection delta: projection_outer x (each projecting term x projection_inner)."""
        cfg = self.cfg
        if (not cfg.enable) or cfg.projection_outer_steps <= 0 or cfg.projection_inner_steps <= 0:
            return torch.zeros_like(coords)
        delta = torch.zeros_like(coords)
        sorted_terms = [
            term
            for term in self._projection_terms_sorted
            if type(term._potential)._project is not Potential._project
        ]
        for _ in range(cfg.projection_outer_steps):
            for term in sorted_terms:
                if (not term.active(step_i)) or (not term.enable_projection) or term.inert(feats):
                    continue
                for _ in range(cfg.projection_inner_steps):
                    d = term.project(coords + delta, feats, t)
                    if d is not None:
                        delta = delta + d
        return delta

    def project(self, coords, feats, *, step_i: int, num_diffusion_steps: int) -> torch.Tensor:
        """Projection delta for `coords` at step `step_i` (no refinement, no update)."""
        return self._project(coords, feats, t=_normalized_time(step_i, num_diffusion_steps), step_i=step_i)

    def update(
        self,
        x_noisy: torch.Tensor,
        x0: torch.Tensor,
        *,
        t_hat,
        c_tau,
        step_scale_eta: float,
        step_i: int,
        num_diffusion_steps: int,
        feats: Mapping[str, Any],
        x0_hook: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Guided `x_noisy -> x_next` given the denoiser output `x0` for `x_noisy` at `t_hat`.

        Equals upstream `TFGEngine.step`'s returned x_next with rho 0 and tfg_outer 1:
        1. x0 += projection delta;
        2. x0 = x0_hook(x0, step_i) if given (upstream calls epitope_guidance.guide_x0 here);
        3. tfg_inner steps of x0 += mu * d log p / d x0 (skipped when mu == 0);
        4. x_next = x_noisy + eta * (c_tau - t_hat) * (x_noisy - x0) / t_hat.
        t_hat and c_tau are tensors shaped `[*batch]` (or scalars); x_noisy, x0 are `[*batch, N, 3]`.
        Upstream also draws `sigma * randn` for the next outer iteration's x_work; with tfg_outer 1
        it is discarded, so it is not drawn here (the generator is not advanced by it). `generator`
        is used only when mc.std > 0.
        """
        cfg = self.cfg
        if step_i == 0:
            validate_features(feats, cfg.terms)
        t_hat = torch.as_tensor(t_hat, device=x_noisy.device, dtype=x_noisy.dtype)
        c_tau = torch.as_tensor(c_tau, device=x_noisy.device, dtype=x_noisy.dtype)
        t = _normalized_time(step_i, num_diffusion_steps)
        eps = _sample_eps(
            cfg.eps_std, x_noisy.shape, k=cfg.eps_batch, device=self.device, dtype=self.dtype, generator=generator
        )
        x0_ref = x0.detach()
        x0_ref = x0_ref + self._project(x0_ref, feats, t=t, step_i=step_i)
        if x0_hook is not None:
            x0_ref = x0_hook(x0_ref, step_i)
        last_log = cfg.log_last_step_energy and step_i == num_diffusion_steps - 1
        if cfg.mu != 0.0:
            for inner in range(cfg.inner_steps):
                log_components = last_log and inner == cfg.inner_steps - 1
                _, grad_x0 = self._logp_and_grad_x0(
                    x0_ref, eps, feats, t=t, step_i=step_i, log_components=log_components
                )
                # Without perturbations the terms are a function of x0_ref alone, so a zero gradient is a fixed
                # point: every remaining inner step would add exact zeros (the last step still logs its energies).
                if cfg.eps_std == 0.0 and not last_log and not grad_x0.any():
                    break
                x0_ref = x0_ref + grad_x0 * float(cfg.mu)
        # Upstream adds a zero x_t shift (rho == 0) to x_noisy in both places; x + 0 == x.
        direction = (x_noisy - x0_ref) / t_hat[..., None, None]
        dt = c_tau - t_hat
        return x_noisy + float(step_scale_eta) * dt[..., None, None] * direction
