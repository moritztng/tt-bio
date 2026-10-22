"""RFD4-Proteina's flow-matching sampler: schedules, the update rule, the RNG and the batching.

This is everything between the features and the network. The network (the flow trunk and the
autoencoder) plugs in behind one call:

    predict(x_t, t, recycle_depth) -> {modality: {"v" | "x_1": array}, "sequence": {...}, ...}

and the loop never looks inside it. Two modalities move together: ``bb_ca`` (the backbone CA
coordinates, 3 channels) and ``local_latents`` (the autoencoder's per-residue latent, 32), each
with its own time schedule, its own noise-injection schedule and its own TSR constants, stepped
with one shared step index. The released recipe is 60 steps and 180 trunk passes.

Three things are one mechanism under ``vf_tsr`` and are ported together, because porting the step
rule without them matches every shape and still samples the wrong distribution:

* the time schedule starts at ``eps = 1e-3`` rather than 0 (``get_schedule_tsr_safe``),
* the noise-injection schedule is divided by ``tsr_ratio ** 2``,
* the score is rescaled by ``tsr_ratio`` inside the step.

**Arithmetic order is upstream's, deliberately.** The step rule has a shorter closed form (under
``vf_tsr`` the rescaled field is ``((1 - r) / t) * x_t + r * v``), but the loop here evaluates the
same expressions in the same order as ``rdn_flow_matcher.simulation_step`` so a seeded run matches
upstream exactly. The work is elementwise on ``[designs, tokens, 3]`` and ``[designs, tokens, 32]``:
nothing is bought by folding it.

**The device seam.** The step rule only uses ``+ - * /`` against scalars and a mask multiply, so the
state never has to come back to the host between steps: give it arrays whose operators run on
device and the whole trajectory stays there. The one place that needs more than operators is the
Gaussian draw, which goes through :class:`DesignRng` and is the only part a device backend has to
supply. ARCH section 3 prices 60 host round trips as a real cost; this is how we avoid them.

**One generator per design.** Upstream draws every design's noise from one global stream, so a
design's output depends on how many designs were in the batch with it. We give each design its own
stream keyed by ``(seed, design index)``: a design is then reproducible on its own, in any batch, at
any chunk width, which is what makes the quality grade and the seed floor mean anything.
``rng="upstream"`` reproduces upstream's draw order instead and is what the parity test runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence

import torch

BB_CA = "bb_ca"
LOCAL_LATENTS = "local_latents"
DATA_MODES = (BB_CA, LOCAL_LATENTS)


# --------------------------------------------------------------------------------------
# The sampling spec (upstream's configs/generation/inference_base.yaml + the release flow config)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalitySpec:
    """One modality's schedule, noise injection, step rule and noise-draw behaviour.

    The field values on :data:`RELEASED` are the released recipe's, read from upstream's
    ``configs/generation/inference_base.yaml`` (schedule / gt / simulation_step_params),
    ``configs/training_proteina.yaml`` (the flow-matcher constructor args) and
    ``configs/releases/flow/2026_10_19_hero_pretrain.yaml`` (the per-group centering override).
    """

    dim: int
    schedule_mode: str  # "uniform" | "power" | "log"
    schedule_p: float
    tail_steps: int = 0
    tail_step_multiplier: int = 1
    gt_mode: str = "1/t"  # "1/t" | "1-t/t" | "tan" | "ode"
    gt_p: float = 1.0
    gt_clamp: float | None = None
    sampling_mode: str = "vf_tsr"  # "vf" | "sc" | "vf_tsr"
    sc_scale_noise: float = 1.0
    t_lim_ode: float = 0.95
    t_lim_ode_below: float = 0.02
    center_every_step: bool = False
    tsr_k: float = 1.0
    tsr_sigma: float = 1.0
    zero_com_noise: bool = False
    stochastic_centering_mode: str = "none"  # "none" | "x_0" | "x_1" | "x_t"
    stochastic_centering_scale: float | None = None
    stochastic_centering_scope: str = "without_absolute_coords"
    stochastic_centering_group_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SamplingSpec:
    """The two modalities plus the trajectory-level settings the loop reads."""

    modalities: dict[str, ModalitySpec]
    nsteps: int = 60
    n_recycle_schedule: tuple[tuple[int, int], ...] | None = ((5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1))
    n_recycle: int = 0
    #: Classifier-free guidance weight. 1.0 is off and is the only value the release supports;
    #: see :func:`check_guidance_w`.
    guidance_w: float = 1.0

    @property
    def data_modes(self) -> tuple[str, ...]:
        return tuple(self.modalities)


#: The released ``thinking_medium`` recipe: 60 steps, 180 trunk passes.
RELEASED = SamplingSpec(
    modalities={
        BB_CA: ModalitySpec(
            dim=3,
            schedule_mode="log",
            schedule_p=2.0,
            tail_steps=10,
            tail_step_multiplier=2,
            gt_mode="1/t",
            gt_p=1.0,
            gt_clamp=None,
            sampling_mode="vf_tsr",
            tsr_k=16.0,
            tsr_sigma=3.0,
            zero_com_noise=True,
            stochastic_centering_mode="x_0",
            stochastic_centering_scale=0.2,
            stochastic_centering_scope="per_group",
            stochastic_centering_group_keys=("molecule_iid",),
        ),
        LOCAL_LATENTS: ModalitySpec(
            dim=32,
            schedule_mode="power",
            schedule_p=2.0,
            tail_steps=10,
            tail_step_multiplier=2,
            gt_mode="tan",
            gt_p=1.0,
            gt_clamp=None,
            sampling_mode="vf_tsr",
            tsr_k=4.0,
            tsr_sigma=12.0,
            zero_com_noise=False,
            stochastic_centering_mode="none",
        ),
    }
)

#: Every shipped recycle schedule, by its config path. ``thinking_{low,medium,high}`` are aliases
#: for the matching ``front_loaded`` file; ``flat`` and ``midpeak`` are reachable by name only.
RECYCLE_SCHEDULES: dict[str, tuple[tuple[int, int], ...]] = {
    "low/flat": ((60, 2),),
    "low/front_loaded": ((10, 4), (10, 3), (10, 2), (30, 1)),
    "low/midpeak": ((10, 1), (5, 2), (5, 3), (10, 4), (5, 3), (5, 2), (20, 1)),
    "medium/flat": ((60, 3),),
    "medium/front_loaded": ((5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1)),
    "medium/midpeak": ((5, 1), (5, 2), (5, 2), (5, 4), (10, 5), (5, 4), (5, 2), (5, 2), (15, 1)),
    "high/flat": ((60, 5),),
    "high/front_loaded": ((10, 9), (10, 8), (10, 5), (20, 3), (10, 2)),
    "high/midpeak": ((5, 1), (5, 2), (5, 3), (5, 5), (10, 6), (5, 5), (5, 3), (5, 2), (15, 1)),
    "thinking_low": ((10, 4), (10, 3), (10, 2), (30, 1)),
    "thinking_medium": ((5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1)),
    "thinking_high": ((10, 9), (10, 8), (10, 5), (20, 3), (10, 2)),
}


def recycle_strategy(name: str, spec: SamplingSpec = RELEASED) -> SamplingSpec:
    """``spec`` with a named recycle strategy. ``trunk_passes`` prices it."""
    if name not in RECYCLE_SCHEDULES:
        raise ValueError(f"unknown recycle strategy {name!r}; have {sorted(RECYCLE_SCHEDULES)}")
    return replace(spec, n_recycle_schedule=RECYCLE_SCHEDULES[name])


def recycle_depths(spec: SamplingSpec, nsteps: int, full_nsteps: int | None = None) -> tuple[int, ...]:
    """The recycle depth of each of the ``nsteps`` steps actually run.

    A schedule is written over the full trajectory. Partial diffusion runs only its tail, so with
    ``nsteps < full_nsteps`` the schedule is expanded over ``full_nsteps`` and its last ``nsteps``
    depths are returned. Upstream ``proteina.Proteina.recycle_depths``.
    """
    full_nsteps = nsteps if full_nsteps is None else full_nsteps
    if spec.n_recycle_schedule is None:
        depths = (spec.n_recycle,) * nsteps
    else:
        depths = tuple(depth for width, depth in spec.n_recycle_schedule for _ in range(width))
        if len(depths) != full_nsteps:
            raise ValueError(f"recycle schedule expands to {len(depths)} steps; expected {full_nsteps}")
        depths = depths[len(depths) - nsteps:]
    if any(not isinstance(d, int) or d < 0 for d in depths):
        raise ValueError("recycle depths must be non-negative integers")
    return depths


def trunk_passes(spec: SamplingSpec = RELEASED, nsteps: int | None = None) -> int:
    """Trunk passes a trajectory costs: the recycle depths summed. 180 for the released recipe.

    The network runs ``depth`` passes at a step (``for cycle in range(n_recycle)``), so this is the
    model's own work and the one number a speed lever must never reduce.
    """
    nsteps = spec.nsteps if nsteps is None else nsteps
    return sum(recycle_depths(spec, nsteps))


# --------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------


def _schedule_with_tail_multiplier(schedule_fn: Callable[[int], torch.Tensor], nsteps: int,
                                   tail_steps: int, multiplier: int) -> torch.Tensor:
    """Sample the last ``tail_steps`` intervals at ``multiplier`` times the density.

    The whole trajectory is laid out on a ``base_steps * multiplier`` grid and then subsampled:
    every ``multiplier``-th point up to the tail, every point inside it. The step count stays
    ``nsteps``, so the tail buys its density from the body.
    """
    if tail_steps < 1:
        raise ValueError("tail_steps must be positive")
    try:
        integer_multiplier = int(multiplier)
    except (TypeError, ValueError, OverflowError):
        integer_multiplier = 0
    if integer_multiplier != multiplier or integer_multiplier < 2:
        raise ValueError("tail_step_multiplier must be an integer >= 2")
    multiplier = integer_multiplier

    base_steps = nsteps - (multiplier - 1) * tail_steps
    if base_steps < tail_steps:
        raise ValueError("nsteps must be at least tail_steps * tail_step_multiplier")

    fine_steps = base_steps * multiplier
    tail_start = (base_steps - tail_steps) * multiplier
    indices = [*range(0, tail_start, multiplier), *range(tail_start, fine_steps + 1)]
    return schedule_fn(fine_steps)[torch.tensor(indices)]


def schedule_times(mode: str, nsteps: int, *, p1: float | None = None, tsr_safe: bool = False,
                   eps: float | None = None, tail_steps: int = 0,
                   tail_step_multiplier: int = 1) -> torch.Tensor:
    """The ``nsteps + 1`` times the field is evaluated at, from ~0 to 1.

    ``tsr_safe`` starts the schedule at ``eps`` (1e-3) instead of 0, which ``vf_tsr`` needs: the
    score conversion divides by t. Upstream ``get_schedule`` / ``get_schedule_tsr_safe``.
    """
    if eps is None:
        eps = 1e-3 if tsr_safe else 1e-5
    if tail_steps:
        return _schedule_with_tail_multiplier(
            lambda steps: schedule_times(mode, steps, p1=p1, tsr_safe=tsr_safe, eps=eps),
            nsteps, tail_steps, tail_step_multiplier,
        )
    if mode == "uniform":
        return torch.linspace(eps if tsr_safe else 0.0, 1, nsteps + 1)
    if mode == "power":
        if p1 is None:
            raise ValueError("p1 cannot be None for the power schedule")
        t = torch.linspace(eps ** (1 / p1), 1, nsteps + 1) if tsr_safe else torch.linspace(0, 1, nsteps + 1)
        return t ** p1
    if mode == "log":
        if p1 is None or p1 <= 0:
            raise ValueError(f"p1 must be > 0 for the log schedule, got {p1}")
        t = 1.0 - torch.logspace(-p1, eps if tsr_safe else 0, nsteps + 1).flip(0)
        t = t - (torch.min(t) - eps) if tsr_safe else t - torch.min(t)
        return t / torch.max(t)
    raise ValueError(f"schedule mode not recognized {mode}")


def noise_injection(t: torch.Tensor, mode: str, param: float, clamp_val: float | None = None,
                    eps: float = 1e-2) -> torch.Tensor:
    """``g(t)``, the Langevin drift / noise-injection schedule. Upstream ``get_gt``.

    ``"ode"`` is identically zero, which zeroes both the drift and the noise and recovers plain
    ODE integration whatever the sampling mode.
    """
    if mode == "ode":
        return torch.zeros_like(t)

    def transform(gt: torch.Tensor, f_pow: float) -> torch.Tensor:
        if f_pow == 1.0:
            return gt
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        normalized = torch.sigmoid(log_gt - mean_log_gt) ** f_pow
        return torch.exp(torch.logit(normalized, eps=1e-6) + mean_log_gt)

    t = torch.clamp(t, 0, 1 - 1e-5)
    if mode == "1-t/t":
        gt = (1.0 - t) / (t + eps)
    elif mode == "tan":
        half_pi = torch.pi / 2.0
        gt = half_pi * torch.sin((1.0 - t) * half_pi) / (torch.cos((1.0 - t) * half_pi) + eps)
    elif mode == "1/t":
        gt = 1.0 / (t + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    return torch.clamp(transform(gt, param), 0, clamp_val)


def tsr_ratio(t: torch.Tensor | float, tsr_k: float, tsr_sigma: float) -> torch.Tensor | float:
    """The temporal-score-rescaling factor. Upstream ``compute_tsr_ratio``.

    ``tsr_k`` sets how low the temperature goes, ``tsr_sigma`` how fast it gets there.
    """
    snr = t ** 2 / (1 - t) ** 2
    return (snr * tsr_sigma ** 2 + 1) / (snr * tsr_sigma ** 2 / tsr_k + 1)


def sample_schedule(spec: SamplingSpec, nsteps: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(ts, gt)`` per modality: times ``[nsteps + 1]`` and noise injection ``[nsteps]``.

    Under ``vf_tsr`` the schedule is the TSR-safe one and ``gt`` is divided by ``tsr_ratio ** 2``,
    so the Langevin drift and the injected noise stay matched to the rescaled score.
    """
    ts: dict[str, torch.Tensor] = {}
    gt: dict[str, torch.Tensor] = {}
    for mode, m in spec.modalities.items():
        ts[mode] = schedule_times(
            m.schedule_mode, int(nsteps), p1=m.schedule_p, tsr_safe=m.sampling_mode == "vf_tsr",
            tail_steps=m.tail_steps, tail_step_multiplier=m.tail_step_multiplier,
        )
        g = noise_injection(ts[mode][:-1], m.gt_mode, m.gt_p, m.gt_clamp)
        if m.sampling_mode == "vf_tsr":
            g = g / tsr_ratio(ts[mode][:-1], m.tsr_k, m.tsr_sigma) ** 2
        gt[mode] = g
    return ts, gt


# --------------------------------------------------------------------------------------
# RNG: one stream per design
# --------------------------------------------------------------------------------------


def design_seed(seed: int, design_index: int) -> int:
    """A design's own seed, stable across runs, machines and batch sizes.

    RFD3 derives its per-design seed as ``seed + design_idx`` (``rfd3/design.py:401``) and the
    generator-per-design plumbing here is the same shape as its. The derivation is not: under
    ``seed + i``, design *i* of seed 7 is design *i - 1* of seed 8, so two arms seeded a few apart
    share most of their designs. The quality grade compares arms over >= 30 designs and reads a
    seed floor off them, and that aliasing would quietly correlate the arms it compares, so the
    index is hashed in instead.
    """
    digest = hashlib.sha256(f"rfd4:{int(seed)}:{int(design_index)}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class DesignRng:
    """Gaussian draws for a batch of designs.

    ``per_design`` (the default) gives each design its own generator, so design *i* sees the same
    noise whatever else is in the batch and whatever width the batch was chunked at. ``upstream``
    draws the whole batch from the ambient global stream in upstream's order, which is what a
    seeded comparison against upstream's loop needs.

    A device backend overrides :meth:`randn` alone: everything else in the sampler is operators.
    """

    def __init__(self, seed: int, n_designs: int, *, mode: str = "per_design",
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32, first: int = 0):
        """``first`` is the design index of row 0, so designs ``first .. first + n_designs - 1`` of
        a larger run draw exactly what they draw in it."""
        if mode not in ("per_design", "upstream"):
            raise ValueError(f"unknown rng mode {mode!r}")
        self.mode = mode
        self.seed = int(seed)
        self.n_designs = int(n_designs)
        self.device = torch.device(device)
        self.dtype = dtype
        self._generators = (
            [torch.Generator(device=self.device).manual_seed(design_seed(seed, first + i))
             for i in range(n_designs)]
            if mode == "per_design" else None
        )

    def randn(self, shape: Sequence[int]) -> torch.Tensor:
        """Standard normal of shape ``[n_designs, *shape]``."""
        shape = tuple(shape)
        if self.mode == "upstream":
            return torch.randn((self.n_designs, *shape), device=self.device, dtype=self.dtype)
        rows = [torch.randn(shape, generator=g, device=self.device, dtype=self.dtype) for g in self._generators]
        return torch.stack(rows, dim=0)

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.n_designs:
            raise ValueError(f"leading dim {x.shape[0]} is not the design count {self.n_designs}")
        return self.randn(x.shape[1:]).to(dtype=x.dtype)


# --------------------------------------------------------------------------------------
# Noise, centering and interpolation
# --------------------------------------------------------------------------------------


def _masked(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero a modality outside the tokens that carry it."""
    return x if mask is None else x * mask[..., None].to(x.dtype)


def _force_zero_com(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Centre over the token axis, counting only masked tokens."""
    if mask is None:
        return x - torch.mean(x, dim=-2, keepdim=True)
    m = mask[..., None].to(x.dtype)
    mean = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(min=1)
    return (x - mean) * m


def compose_group_ids(groups: dict[str, torch.Tensor | None], keys: Sequence[str]) -> torch.Tensor:
    """One integer per token for each distinct combination of the requested group tensors."""
    tensors = []
    for k in keys:
        t = groups.get(k)
        if t is None:
            raise ValueError(f"stochastic centering asked for group key {k!r}, which the batch does not carry")
        tensors.append(t.long())
    stacked = torch.stack(tensors, dim=-1)
    _, inverse = torch.unique(stacked.reshape(-1, stacked.shape[-1]), dim=0, return_inverse=True)
    return inverse.reshape(stacked.shape[:-1])


def add_stochastic_offsets(x: torch.Tensor, m: ModalitySpec, rng: DesignRng, *,
                           emit_mask: torch.Tensor | None,
                           coord_cond_mask: torch.Tensor | None = None,
                           groups: dict[str, torch.Tensor | None] | None = None) -> torch.Tensor:
    """Translate each group of generated tokens by its own random offset.

    The release conditions the backbone per ``molecule_iid``, so each molecule gets an independent
    translation while absolute-coordinate-conditioned tokens stay where they are. One draw per
    group, groups taken in ascending id order: that order is part of the seeded result.
    """
    offset_mask = emit_mask
    if m.stochastic_centering_scope != "all" and coord_cond_mask is not None and emit_mask is not None:
        offset_mask = emit_mask & ~coord_cond_mask
    if offset_mask is None:
        return x

    if m.stochastic_centering_scope == "per_group":
        if not m.stochastic_centering_group_keys:
            raise ValueError("stochastic_centering_scope='per_group' needs stochastic_centering_group_keys")
        group_ids = compose_group_ids(groups or {}, m.stochastic_centering_group_keys)
        for g in group_ids[offset_mask].unique():
            group_mask = (offset_mask & (group_ids == g))[..., None].to(x.dtype)
            noise = rng.randn((x.shape[-1],)).to(dtype=x.dtype)
            x = x + m.stochastic_centering_scale * noise.unsqueeze(-2) * group_mask
    elif m.stochastic_centering_scope in ("all", "without_absolute_coords"):
        noise = rng.randn((x.shape[-1],)).to(dtype=x.dtype)
        x = x + m.stochastic_centering_scale * noise.unsqueeze(-2) * offset_mask[..., None].to(x.dtype)
    else:
        raise ValueError(f"unknown stochastic_centering_scope {m.stochastic_centering_scope!r}")
    return _masked(x, emit_mask)


def sample_noise(spec: SamplingSpec, n_tokens: int, rng: DesignRng, *,
                 emit_masks: dict[str, torch.Tensor | None],
                 coord_cond_mask: torch.Tensor | None = None,
                 groups: dict[str, torch.Tensor | None] | None = None,
                 training: bool = False) -> dict[str, torch.Tensor]:
    """``x_0`` for every modality: masked standard normal, centred and offset as the spec says.

    Modality order is the spec's, and so is the order the draws are taken in.
    """
    x: dict[str, torch.Tensor] = {}
    for mode, m in spec.modalities.items():
        noise = _masked(rng.randn((n_tokens, m.dim)), emit_masks.get(mode))
        if m.zero_com_noise:
            noise = _force_zero_com(noise, emit_masks.get(mode))
        if m.stochastic_centering_mode == "x_0" and not training:
            noise = add_stochastic_offsets(noise, m, rng, emit_mask=emit_masks.get(mode),
                                           coord_cond_mask=coord_cond_mask, groups=groups)
        x[mode] = noise
    return x


def interpolate(spec: SamplingSpec, x_0: dict[str, torch.Tensor], x_1: dict[str, torch.Tensor],
                t: dict[str, torch.Tensor], *, emit_masks: dict[str, torch.Tensor | None],
                rng: DesignRng | None = None, coord_cond_mask: torch.Tensor | None = None,
                groups: dict[str, torch.Tensor | None] | None = None) -> dict[str, torch.Tensor]:
    """``x_t = (1 - t) x_0 + t x_1``, with the spec's stochastic centering applied to whichever
    endpoint it names. This is the forward marginal partial diffusion starts from."""
    out: dict[str, torch.Tensor] = {}
    for mode, m in spec.modalities.items():
        a = _masked(x_0[mode], emit_masks.get(mode))
        b = _masked(x_1[mode], emit_masks.get(mode))
        if m.stochastic_centering_mode in ("x_0", "x_1"):
            if rng is None:
                raise ValueError("stochastic centering needs an rng")
            target = a if m.stochastic_centering_mode == "x_0" else b
            moved = add_stochastic_offsets(target, m, rng, emit_mask=emit_masks.get(mode),
                                           coord_cond_mask=coord_cond_mask, groups=groups)
            a, b = (moved, b) if m.stochastic_centering_mode == "x_0" else (a, moved)
        tt = t[mode][..., None, None]
        x_t = (1.0 - tt) * a + tt * b
        if m.stochastic_centering_mode == "x_t":
            if rng is None:
                raise ValueError("stochastic centering needs an rng")
            x_t = add_stochastic_offsets(x_t, m, rng, emit_mask=emit_masks.get(mode),
                                         coord_cond_mask=coord_cond_mask, groups=groups)
        out[mode] = x_t
    return out


# --------------------------------------------------------------------------------------
# The update rule
# --------------------------------------------------------------------------------------


def vf_to_score(x_t, v, t):
    """``s = (t v - x_t) / (1 - t)``. Needs ``t < 1``."""
    return (t * v - x_t) / (1.0 - t)


def score_to_vf(x_t, score, t):
    """``v = (x_t + (1 - t) s) / t``. Needs ``t > 0``."""
    return (x_t + (1.0 - t) * score) / t


def add_clean_prediction(m_out: dict, x_t, t: float):
    """``x_1`` from ``v`` when the network parameterises the field, and vice versa.

    The released heads emit ``x_1``; the loop integrates ``v``. Whichever is missing is derived
    here so the step rule only ever sees both.
    """
    if "x_1" not in m_out:
        if "v" not in m_out:
            raise KeyError(f"cannot derive the clean prediction from {sorted(m_out)}")
        m_out["x_1"] = x_t + (1.0 - t) * m_out["v"]
    if "v" not in m_out:
        m_out["v"] = (m_out["x_1"] - x_t) / ((1.0 - t) + 1e-5)
    return m_out


def _f32(x: float) -> torch.Tensor:
    """A scalar as float32.

    Every coefficient the step rule multiplies the state by is computed in float32, the dtype the
    schedules are in, rather than in Python doubles: a chain like ``sqrt(2 g s dt)`` rounded once
    at the end is up to an ulp away from the same chain in float32, which is enough to make a
    seeded trajectory diverge from upstream's. Coefficients are host scalars either way, so this
    costs nothing and keeps bit-exactness as a free regression signal.
    """
    return torch.tensor(x, dtype=torch.float32)


def _brownian_scale(gt: float, sc_scale_noise: float, dt: float) -> float:
    """``sqrt(2 g(t) * sc_scale_noise * dt)``, the injected-noise amplitude of one Euler step."""
    return float(torch.sqrt(2 * _f32(gt) * sc_scale_noise * _f32(dt)))


def simulation_step(m: ModalitySpec, x_t, v, *, t: float, dt: float, gt: float,
                    rng: DesignRng, emit_mask=None):
    """One Euler step of this modality, in upstream's arithmetic order.

    * ``vf``  — plain ODE, ``x += v dt``.
    * ``sc``  — SDE with noise scaling, and a 1.5x-score low-temperature ODE above ``t_lim_ode``.
    * ``vf_tsr`` — the score is rescaled by ``tsr_ratio(t)``; deterministic outside
      ``[t_lim_ode_below, t_lim_ode]``, Langevin drift plus matched noise inside, and a zero step
      at ``t == 0`` where the score conversion is undefined.
    """
    if m.sampling_mode not in ("vf", "sc", "vf_tsr"):
        raise ValueError(f"invalid sampling mode {m.sampling_mode!r}")
    if m.sc_scale_noise < 0:
        raise ValueError(f"sc_scale_noise must be >= 0, got {m.sc_scale_noise}")
    if gt < 0:
        raise ValueError(f"gt must be >= 0, got {gt}")

    endpoint_score_scale = 1.5  # keeps the regular SDE's low-temperature ODE endpoint
    noise_scale = _brownian_scale(gt, m.sc_scale_noise, dt)

    if m.sampling_mode == "vf":
        delta_x = v * dt
    elif m.sampling_mode == "sc":
        if t > m.t_lim_ode:
            score = vf_to_score(x_t, v, t)
            delta_x = score_to_vf(x_t, score * endpoint_score_scale, t) * dt
        else:
            score = vf_to_score(x_t, v, t)
            eps = rng.randn_like(x_t)
            delta_x = (v + gt * score) * dt + noise_scale * eps
    else:
        ratio = float(tsr_ratio(_f32(t), m.tsr_k, m.tsr_sigma))
        if t == 0.0:
            delta_x = x_t * 0.0
        else:
            score = vf_to_score(x_t, v, t)
            scaled_score = score * ratio
            v_scaled = score_to_vf(x_t, scaled_score, t)
            if t > m.t_lim_ode or t < m.t_lim_ode_below:
                delta_x = v_scaled * dt
            else:
                eps = rng.randn_like(x_t)
                delta_x = (v_scaled + gt * scaled_score) * dt + noise_scale * eps

    x_next = _masked(x_t + delta_x, emit_mask)
    if m.center_every_step:
        x_next = _force_zero_com(x_next, emit_mask)
    return x_next


# --------------------------------------------------------------------------------------
# Partial diffusion and inverse folding
# --------------------------------------------------------------------------------------

RESAMPLED_FORWARD_MARGINAL = "resampled_forward_marginal"
INVERSE_FOLDING_METHODS = frozenset({RESAMPLED_FORWARD_MARGINAL})


@dataclass(frozen=True)
class PartialStart:
    """Where a trajectory starts: per-modality ``t``, and how many of ``nsteps`` are left to run.

    ``t == {}`` is a full generation from pure noise.
    """

    t: dict[str, float] = field(default_factory=dict)
    nsteps: int | None = None
    inverse_folding_method: str | None = None
    #: True only for ``partial_trajectory_fraction_steps``, which is a real tail of the configured
    #: schedule. Explicit times are not a tail of anything and get a re-expanded schedule instead.
    from_fraction: bool = False

    @property
    def is_partial(self) -> bool:
        return bool(self.t)


def resolve_partial_start(spec: SamplingSpec, *, partial_t_backbone: float | None = None,
                          partial_t_latent: float | None = None, couple_partial_t: bool = False,
                          partial_trajectory_fraction_steps: float | None = None,
                          inverse_folding: bool = False,
                          inverse_folding_method: str = RESAMPLED_FORWARD_MARGINAL,
                          ref_nsteps: int = 10000) -> PartialStart:
    """Read the partial-diffusion / inverse-folding arguments the way upstream reads them.

    Inverse folding starts both modalities at ``t = 0`` and excludes every partial argument. A
    fraction maps to a step index ``k = nsteps - round(frac * nsteps)`` on the real schedule.
    Explicit times come in pairs unless ``couple_partial_t`` derives the other from a dense
    ``ref_nsteps`` schedule. Any ``t >= 1`` is rejected: the score conversion divides by ``1 - t``.
    """
    if inverse_folding:
        if inverse_folding_method not in INVERSE_FOLDING_METHODS:
            raise ValueError(f"inverse_folding_method must be one of {sorted(INVERSE_FOLDING_METHODS)}, "
                             f"got {inverse_folding_method!r}")
        if (partial_t_backbone is not None or partial_t_latent is not None or couple_partial_t
                or partial_trajectory_fraction_steps is not None):
            raise ValueError("inverse_folding is mutually exclusive with partial_t_backbone, "
                             "partial_t_latent, couple_partial_t and partial_trajectory_fraction_steps")
        return PartialStart(t={mode: 0.0 for mode in spec.data_modes}, nsteps=spec.nsteps,
                            inverse_folding_method=inverse_folding_method)

    if partial_trajectory_fraction_steps is not None:
        frac = partial_trajectory_fraction_steps
        if partial_t_backbone is not None or partial_t_latent is not None or couple_partial_t:
            raise ValueError("partial_trajectory_fraction_steps is mutually exclusive with "
                             "partial_t_backbone, partial_t_latent and couple_partial_t")
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"partial_trajectory_fraction_steps must be in (0, 1], got {frac}")
        remaining = round(frac * spec.nsteps)
        if remaining < 1:
            raise ValueError(f"partial_trajectory_fraction_steps={frac} with nsteps={spec.nsteps} "
                             "rounds to zero denoising steps")
        ref_ts, _ = sample_schedule(spec, spec.nsteps)
        k = spec.nsteps - remaining
        return _validated(PartialStart(t={dm: ref_ts[dm][k].item() for dm in spec.data_modes},
                                       nsteps=remaining, from_fraction=True))

    n_set = (partial_t_backbone is not None) + (partial_t_latent is not None)
    if n_set == 0:
        return PartialStart()
    if n_set == 2:
        if couple_partial_t:
            raise ValueError("couple_partial_t=true takes exactly one of partial_t_backbone or partial_t_latent")
        return _validated(PartialStart(t={BB_CA: partial_t_backbone, LOCAL_LATENTS: partial_t_latent},
                                       nsteps=spec.nsteps))
    if not couple_partial_t:
        which = "partial_t_backbone" if partial_t_backbone is not None else "partial_t_latent"
        raise ValueError(f"only {which} is set: give both, or set couple_partial_t=true to derive the other")

    ref_ts, _ = sample_schedule(spec, ref_nsteps)
    ref_sched, ref_val = ((ref_ts[BB_CA], partial_t_backbone) if partial_t_backbone is not None
                          else (ref_ts[LOCAL_LATENTS], partial_t_latent))
    above = ref_sched >= ref_val - 1e-6
    k = above.nonzero(as_tuple=True)[0][0].item() if above.any() else ref_nsteps
    return _validated(PartialStart(t={dm: ref_ts[dm][k].item() for dm in spec.data_modes}, nsteps=spec.nsteps))


def _validated(start: PartialStart) -> PartialStart:
    at_or_above_one = {dm: t for dm, t in start.t.items() if t >= 1.0}
    if at_or_above_one:
        raise ValueError("partial-diffusion start time must be < 1 (t=1 is the clean state, and the "
                         f"score conversion divides by 1 - t); resolved {at_or_above_one}")
    return start


def partial_schedule(spec: SamplingSpec, start: PartialStart, nsteps: int,
                     *, full_nsteps: int | None = None) -> tuple[dict, dict]:
    """Per-modality schedules covering ``[partial_t, 1]`` in exactly ``nsteps`` steps.

    A fractional restart slices the configured full schedule, so it is the tail of the real
    trajectory. Explicit times cannot be a tail of anything, so the schedule is re-expanded until
    its last ``nsteps`` intervals cover the requested range, estimated on a dense reference.
    """
    if full_nsteps is not None:
        if not 1 <= nsteps <= full_nsteps:
            raise ValueError(f"partial step count must be in [1, {full_nsteps}], got {nsteps}")
        full_ts, full_gt = sample_schedule(spec, full_nsteps)
        ts = {dm: full_ts[dm][-(nsteps + 1):] for dm in spec.data_modes}
        gt = {dm: full_gt[dm][-nsteps:] for dm in spec.data_modes}
        for dm in spec.data_modes:
            if ts[dm][0] != ts[dm].new_tensor(start.t[dm]):
                raise ValueError(f"{dm}: partial corruption time must match the first time of the schedule tail")
        return ts, gt

    nsteps_dense = max(nsteps * 100, 1000)
    dense_ts, _ = sample_schedule(spec, nsteps_dense)
    ts: dict[str, torch.Tensor] = {}
    gt: dict[str, torch.Tensor] = {}
    for dm in spec.data_modes:
        pt = float(start.t.get(dm, 0.0))
        above = dense_ts[dm] >= pt - 1e-6
        i_start = above.nonzero(as_tuple=True)[0][0].item() if above.any() else nsteps_dense
        n_above = max(nsteps_dense - i_start, 1)
        expanded = round(nsteps * nsteps_dense / n_above)
        full_ts, full_gt = sample_schedule(spec, expanded)
        ts[dm] = full_ts[dm][-(nsteps + 1):]
        gt[dm] = full_gt[dm][-nsteps:]
    return ts, gt


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


class Network(Protocol):
    """What the loop needs from the model, and nothing else.

    One call per step. ``recycle_depth`` is how many trunk passes this step runs, and the network
    owns the recycling: the loop only decides the schedule and counts the passes. ``x_t`` and the
    returned fields stay wherever the network keeps them, so a device implementation never has to
    hand the state back.

    Returns ``{modality: {"v" | "x_1": array}}``, optionally with ``"sequence": {"logits": ...}``
    and ``"insertion": {"logits": ...}``.
    """

    def __call__(self, x_t: dict[str, Any], t: dict[str, float], *, recycle_depth: int,
                 step: int) -> dict[str, dict[str, Any]]:
        ...


class ChunkedNetwork:
    """Run a step's network call a few designs at a time, narrowing when DRAM refuses.

    The whole sample axis goes through ``sample_chunks.denoise_in_chunks``, the same chunker
    Boltz-2, Protenix, OpenDDE and PXDesign denoise through: a refused chunk halves the width,
    drops whatever device state the width keyed, and reruns from its saved input.

    Chunking cannot change a design. Every draw a design sees comes from its own generator, taken
    on the host before the step runs, so the width decides how many designs are resident and
    nothing else. ``tests/test_rfd4_sampler.py`` holds that to a bit-exact comparison across
    widths; it is the property that lets the engine pick a width from free DRAM.

    The modalities are concatenated along the channel axis for the chunker's single-tensor
    interface and split again inside, which also means one device call per chunk rather than two.
    """

    def __init__(self, network: Network, spec: SamplingSpec = RELEASED, *,
                 max_parallel_samples: int | None = 5, reset: Callable | None = None):
        self.network = network
        self.spec = spec
        self.max_parallel_samples = max_parallel_samples
        self.reset = reset
        self.width: int | None = None

    def __call__(self, x_t: dict[str, Any], t: dict[str, float], *, recycle_depth: int, step: int):
        from tt_bio.sample_chunks import denoise_in_chunks, resolve_sample_chunk_width

        modes = tuple(self.spec.data_modes)
        dims = [x_t[mode].shape[-1] for mode in modes]
        packed = torch.cat([x_t[mode] for mode in modes], dim=-1)
        if self.width is None:
            self.width = resolve_sample_chunk_width(packed.shape[0], self.max_parallel_samples)
        extras: list[dict] = []

        def run(chunk, width):
            parts = torch.split(chunk, dims, dim=-1)
            out = self.network({mode: part for mode, part in zip(modes, parts)}, t,
                               recycle_depth=recycle_depth, step=step)
            extras.append({k: v for k, v in out.items() if k not in modes})
            fields = [out[mode] for mode in modes]
            key = "v" if all("v" in f for f in fields) else "x_1"
            return torch.cat([f[key] for f in fields], dim=-1), key

        keys: list[str] = []

        def run_tensor(chunk, width):
            tensor, key = run(chunk, width)
            keys.append(key)
            return tensor

        packed_out, self.width = denoise_in_chunks(packed, self.width, run_tensor,
                                                   reset=self.reset, tag="rfd4")
        parts = torch.split(packed_out, dims, dim=-1)
        out = {mode: {keys[0]: part} for mode, part in zip(modes, parts)}
        for head in {k for e in extras for k in e}:
            pieces = [e[head] for e in extras if head in e]
            out[head] = {k: torch.cat([p[k] for p in pieces], dim=0) for k in pieces[0]}
        return out


@dataclass
class SamplerOutput:
    """What a trajectory produces."""

    x: dict[str, Any]                      # the final state, per modality
    x_1_pred: dict[str, Any]               # the last clean prediction
    sequence_logits: Any = None            # [designs, tokens, 147] from the final step
    insertion_logits: Any = None           # [designs, tokens, tokens] from the final step
    trajectory: list[dict] = field(default_factory=list)
    trunk_passes: int = 0


def run_trajectory(spec: SamplingSpec, network: Network, **kwargs) -> SamplerOutput:
    """Integrate from ``ts[*][0]`` to 1 in ``nsteps`` steps, one network call each.
    The arguments are :func:`trajectory`'s."""
    return _drive(trajectory(spec, **kwargs), network)


def trajectory(spec: SamplingSpec, *, x: dict[str, Any],
               ts: dict[str, torch.Tensor], gt: dict[str, torch.Tensor], nsteps: int,
               depths: Sequence[int], rng: DesignRng,
               emit_masks: dict[str, torch.Tensor | None],
               save_trajectory_every: int = 0, dump_noisy_trajectories: bool = False,
               resample_target_backbone: bool = False, target_backbone=None,
               coord_cond_mask: torch.Tensor | None = None,
               groups: dict[str, torch.Tensor | None] | None = None,
               guidance: Callable | None = None):
    """The loop, as a generator: it yields each step's network request ``(x_t, t, recycle_depth,
    step)``, is sent the network's output, and returns the :class:`SamplerOutput`. Who answers the
    requests, and when, is the driver's business (:func:`run_trajectory`, :func:`run_lanes`).

    This is both of upstream's loops (``full_simulation`` and ``partial_simulation``): they differ
    only in where ``x`` starts and in the inverse-folding resampling, so they are one function
    here. ``depths`` must be as long as ``nsteps`` — the recycle schedule is not negotiable, it is
    the model's own work.

    With ``resample_target_backbone``, the backbone is redrawn from its forward marginal at every
    step and set to the clean target on the last one, while the model moves only the sequence
    latents. That is inverse folding, and upstream's only method for it.
    """
    if len(depths) != nsteps:
        raise ValueError(f"expected {nsteps} recycle depths, got {len(depths)}")

    x = {mode: _masked(x[mode], emit_masks.get(mode)) for mode in spec.data_modes}
    if resample_target_backbone:
        if target_backbone is None:
            raise ValueError("inverse folding needs the clean target backbone")
        target_backbone = target_backbone.clone()
        x[BB_CA] = _noised_target_backbone(spec, target_backbone, float(ts[BB_CA][0]), rng,
                                           emit_masks=emit_masks, coord_cond_mask=coord_cond_mask,
                                           groups=groups)

    trajectory: list[dict] = []
    x_1_pred: dict[str, Any] = {}
    out: dict[str, dict[str, Any]] = {}
    for step in range(nsteps):
        t = {mode: float(ts[mode][step]) for mode in spec.data_modes}
        dt = {mode: float(ts[mode][step + 1] - ts[mode][step]) for mode in spec.data_modes}

        out = yield x, t, depths[step], step
        for mode in spec.data_modes:
            out[mode] = add_clean_prediction(out[mode], x[mode], t[mode])
            out[mode]["x_1"] = _masked(out[mode]["x_1"], emit_masks.get(mode))
            out[mode]["v"] = _masked(out[mode]["v"], emit_masks.get(mode))
        x_1_pred = {mode: out[mode]["x_1"] for mode in spec.data_modes}

        x = {
            mode: simulation_step(spec.modalities[mode], x[mode], out[mode]["v"], t=t[mode], dt=dt[mode],
                                  gt=float(gt[mode][step]), rng=rng, emit_mask=emit_masks.get(mode))
            for mode in spec.data_modes
        }

        if guidance is not None:
            for mode, delta in guidance(x_1_pred=x_1_pred, t=t, step=step).items():
                x[mode] = x[mode] + delta

        if resample_target_backbone:
            x[BB_CA] = (target_backbone.clone() if step == nsteps - 1 else
                        _noised_target_backbone(spec, target_backbone, float(ts[BB_CA][step + 1]), rng,
                                                emit_masks=emit_masks, coord_cond_mask=coord_cond_mask,
                                                groups=groups))

        x = {mode: _masked(x[mode], emit_masks.get(mode)) for mode in spec.data_modes}

        if save_trajectory_every > 0 and (step % save_trajectory_every == 0 or step == nsteps - 1):
            snapshot = {"step": step, "x_1": {m: x_1_pred[m] for m in x_1_pred},
                        "sequence_logits": out.get("sequence", {}).get("logits")}
            if dump_noisy_trajectories:
                snapshot["x_t"] = dict(x)
            trajectory.append(snapshot)

    return SamplerOutput(
        x=x, x_1_pred=x_1_pred,
        sequence_logits=out.get("sequence", {}).get("logits"),
        insertion_logits=out.get("insertion", {}).get("logits"),
        trajectory=trajectory, trunk_passes=int(sum(depths)),
    )


def _noised_target_backbone(spec: SamplingSpec, target_backbone, t: float, rng: DesignRng, *,
                            emit_masks, coord_cond_mask, groups):
    """Draw the fixed backbone from its forward marginal at ``t``.

    Only the backbone modality is touched, and ``training=True`` on the draw so the x_0 offset is
    applied once, inside ``interpolate``, exactly as corruption does it.
    """
    bb = {BB_CA: spec.modalities[BB_CA]}
    sub = replace(spec, modalities=bb)
    n_tokens = target_backbone.shape[-2]
    x_0 = sample_noise(sub, n_tokens, rng, emit_masks=emit_masks, coord_cond_mask=coord_cond_mask,
                       groups=groups, training=True)
    t_vec = torch.full((target_backbone.shape[0],), t, dtype=target_backbone.dtype,
                       device=target_backbone.device)
    return interpolate(sub, x_0, {BB_CA: target_backbone}, {BB_CA: t_vec}, emit_masks=emit_masks,
                       rng=rng, coord_cond_mask=coord_cond_mask, groups=groups)[BB_CA]


def check_guidance_w(spec: SamplingSpec) -> None:
    """Classifier-free guidance is off in the release, and cannot be turned on from it.

    `v_guided = w * v + (1 - w) * v_ucond` needs a second, unconditional network call, and
    upstream builds that call by dropping `cath_code` / `x_motif` from the batch
    (`rfproteina/proteina.py::predict_for_sampling`, which asserts one of them is present).
    The released RFD4 pipeline carries neither: CATH arrives as the `condition_cath-label`
    palette entry and the motif as coordinate conditioning, so upstream's own unconditional
    branch cannot run on a release batch, and its BioIR backend rejects `guidance_w` outright.
    A weight other than 1.0 is therefore refused here rather than silently ignored.
    """
    if spec.guidance_w != 1.0:
        raise ValueError(
            f"guidance_w={spec.guidance_w}: classifier-free guidance is not available on the "
            "released RFD4 checkpoint. Upstream's unconditional pass drops cath_code/x_motif, "
            "and the release conditions through the condition palette instead, so there is no "
            "unconditional model to blend with. Use metric gradient guidance "
            "(tt_bio.rfd4.guidance) for a steered run.")


def generate(spec: SamplingSpec, network: Network, *, n_designs: int, n_tokens: int, seed: int,
             emit_masks: dict[str, torch.Tensor | None], rng_mode: str = "per_design",
             start: PartialStart | None = None, x_1: dict[str, Any] | None = None,
             coord_cond_mask: torch.Tensor | None = None,
             groups: dict[str, torch.Tensor | None] | None = None,
             device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
             lane_width: int | None = None, **kwargs) -> SamplerOutput:
    """One batch of designs, from pure noise or from a partly noised structure.

    ``start`` comes from :func:`resolve_partial_start`. A partial start needs ``x_1``: the clean
    structure to corrupt, per modality. Everything else is :func:`trajectory`.

    ``lane_width`` splits the designs into lanes of that many, each its own trajectory, and runs
    the lanes through :func:`run_lanes`, so one lane's host work overlaps another's network call.
    A lane is exactly the chunk ``ChunkedNetwork`` would run at that width, with the same draws, so
    the designs are the same; it needs ``rng_mode="per_design"``, because under ``"upstream"`` a
    design's noise depends on the order the whole batch is drawn in.
    """
    check_guidance_w(spec)
    common = dict(n_tokens=n_tokens, seed=seed, rng_mode=rng_mode, start=start or PartialStart(),
                  device=device, dtype=dtype, **kwargs)
    if lane_width is None or lane_width >= n_designs:
        return _drive(_designs(spec, n_designs=n_designs, emit_masks=emit_masks, x_1=x_1,
                               coord_cond_mask=coord_cond_mask, groups=groups, **common), network)
    if rng_mode != "per_design":
        raise ValueError("lanes need rng_mode='per_design'")
    if kwargs.get("guidance") is not None:
        raise ValueError("guidance sees the whole batch, so it cannot run in lanes")
    lanes = []
    for first in range(0, n_designs, lane_width):
        k = min(lane_width, n_designs - first)
        lanes.append(_designs(spec, n_designs=k, first=first,
                              emit_masks=_rows(emit_masks, first, k, n_designs),
                              x_1=_rows(x_1, first, k, n_designs),
                              coord_cond_mask=_rows(coord_cond_mask, first, k, n_designs),
                              groups=_rows(groups, first, k, n_designs), **common))
    return _cat(run_lanes(lanes, network))


def _designs(spec: SamplingSpec, *, n_designs: int, n_tokens: int, seed: int,
             emit_masks: dict[str, torch.Tensor | None], rng_mode: str, start: PartialStart,
             x_1: dict[str, Any] | None, coord_cond_mask: torch.Tensor | None,
             groups: dict[str, torch.Tensor | None] | None, device, dtype, first: int = 0,
             **kwargs):
    """The :func:`trajectory` of designs ``first .. first + n_designs - 1``."""
    rng = DesignRng(seed, n_designs, mode=rng_mode, device=device, dtype=dtype, first=first)

    if not start.is_partial:
        depths = recycle_depths(spec, spec.nsteps)
        ts, gt = sample_schedule(spec, spec.nsteps)
        x = sample_noise(spec, n_tokens, rng, emit_masks=emit_masks, coord_cond_mask=coord_cond_mask,
                         groups=groups, training=False)
        return trajectory(spec, x=x, ts=ts, gt=gt, nsteps=spec.nsteps, depths=depths,
                          rng=rng, emit_masks=emit_masks, coord_cond_mask=coord_cond_mask,
                          groups=groups, **kwargs)

    if x_1 is None:
        raise ValueError("a partial start needs x_1, the clean structure to corrupt")
    nsteps = start.nsteps or spec.nsteps
    depths = recycle_depths(spec, nsteps, full_nsteps=spec.nsteps)
    ts, gt = partial_schedule(spec, start, nsteps,
                              full_nsteps=spec.nsteps if start.from_fraction else None)

    t_vec = {mode: torch.full((n_designs,), start.t[mode], dtype=dtype, device=device)
             for mode in spec.data_modes}
    x_0 = sample_noise(spec, n_tokens, rng, emit_masks=emit_masks, coord_cond_mask=coord_cond_mask,
                       groups=groups, training=True)
    x = interpolate(spec, x_0, x_1, t_vec, emit_masks=emit_masks, rng=rng,
                    coord_cond_mask=coord_cond_mask, groups=groups)

    inverse = start.inverse_folding_method == RESAMPLED_FORWARD_MARGINAL
    return trajectory(spec, x=x, ts=ts, gt=gt, nsteps=nsteps, depths=depths, rng=rng,
                      emit_masks=emit_masks, resample_target_backbone=inverse,
                      target_backbone=x_1[BB_CA] if inverse else None,
                      coord_cond_mask=coord_cond_mask, groups=groups, **kwargs)


def _drive(steps, network: Network) -> SamplerOutput:
    """Answer one trajectory's requests in order with ``network``."""
    try:
        request = next(steps)
        while True:
            x, t, depth, step = request
            request = steps.send(network(x, t, recycle_depth=depth, step=step))
    except StopIteration as done:
        return done.value


def run_lanes(lanes: Sequence, network: Network) -> list[SamplerOutput]:
    """Drive independent trajectories (:func:`trajectory` generators) so the host never waits idle.

    A network call is a host half (features, conditioning), a device half (the trunk) and a host
    tail (heads, then the sampler's update). One design's next call needs its last one's output,
    so within a lane the halves are serial; across lanes they are not. While one lane's trunk runs,
    the host finishes the previous lane's step and prepares its next call, and the next trunk is
    launched the moment the current one returns, so the card goes from one lane to the next
    without waiting on the host.

    The network may split its call into ``prepare(x_t, t, *, recycle_depth, step)``,
    ``launch(prepared)`` and ``finish(launched)``, where ``launch`` only enqueues device work and
    ``finish`` waits for it. One launched call is in flight at a time, so the card holds one
    lane's trunk, as it would running them one after another. A network with no stages is called
    as it is and the lanes simply run in turn.

    Each lane computes exactly what it computes alone: the order lanes run in is not an input to
    any of them.
    """
    from collections import deque

    prepare = getattr(network, "prepare", None)
    if prepare is None:
        def prepare(x, t, *, recycle_depth, step):
            return network(x, t, recycle_depth=recycle_depth, step=step)
        launch = finish = lambda value: value
    else:
        launch, finish = network.launch, network.finish

    results: list[SamplerOutput | None] = [None] * len(lanes)
    ready: deque = deque()

    def advance(i, out=None):
        try:
            x, t, depth, step = next(lanes[i]) if out is None else lanes[i].send(out)
        except StopIteration as done:
            results[i] = done.value
            return
        ready.append((i, prepare(x, t, recycle_depth=depth, step=step)))

    for i in range(len(lanes)):
        advance(i)
    flight = None
    while ready or flight:
        if flight is None:
            i, prepared = ready.popleft()
            flight = i, launch(prepared)
        i, launched = flight
        out = finish(launched)
        flight = None
        if ready:
            j, prepared = ready.popleft()
            flight = j, launch(prepared)
        advance(i, out)
    return results


def _rows(value, first: int, k: int, n: int):
    """Rows ``first .. first + k - 1`` of whatever carries the design axis ``n``; a leading 1 is a
    broadcast and is kept."""
    if isinstance(value, dict):
        return {key: _rows(v, first, k, n) for key, v in value.items()}
    if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == n:
        return value[first:first + k]
    return value


def _cat(parts: Sequence):
    """Join lanes back into one batch, in lane order."""
    head = parts[0]
    if isinstance(head, SamplerOutput):
        return SamplerOutput(x=_cat([p.x for p in parts]), x_1_pred=_cat([p.x_1_pred for p in parts]),
                             sequence_logits=_cat([p.sequence_logits for p in parts]),
                             insertion_logits=_cat([p.insertion_logits for p in parts]),
                             trajectory=[_cat(list(snaps)) for snaps in zip(*(p.trajectory for p in parts))],
                             trunk_passes=head.trunk_passes)
    if isinstance(head, dict):
        return {key: _cat([p[key] for p in parts]) for key in head}
    if torch.is_tensor(head):
        return torch.cat(parts, dim=0)
    return head
