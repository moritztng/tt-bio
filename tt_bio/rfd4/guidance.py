"""Metric gradient guidance: a differentiable metric steers the sampler.

Upstream's `GuidanceModel` runs differentiable metrics on the clean prediction each step, builds
a weighted scalar objective and backpropagates it to the per-modality state, adding the gradient
to the integrated step. Guidance always *maximizes* the objective, so a negative weight minimizes
the metric. Only one metric upstream ships is differentiable, `ContractCAMetric`, and it ships as
the mechanics smoke test (`configs/guidance/contract_ca.yaml`).

Everything here is host arithmetic on the clean prediction, so it runs with or without a card.
`run_trajectory(guidance=...)` takes the callable this module builds.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from tt_bio.rfd4.sampler import BB_CA

#: Upstream's differentiable metrics, keyed `"<metric>.<output>"` the way a term names one.
METRICS: dict[str, Callable[[dict[str, torch.Tensor]], torch.Tensor]] = {
    # `rfproteina/metrics/contract_ca_metric.py`: the summed squared CA norm. A negative weight
    # pulls the chain towards the origin.
    "contract_ca.sq_norm": lambda state: (state[BB_CA] ** 2).sum(),
}


def constant(t: float, *, w: float) -> float:
    """`rfproteina.guidance.schedules.constant`: the same weight at every step."""
    return w


def window(t: float, *, w: float, t_start: float, t_end: float) -> float:
    """`rfproteina.guidance.schedules.window`: `w` inside `[t_start, t_end]`, 0 outside."""
    return w if t_start <= t <= t_end else 0.0


def gradient_guidance(terms: dict[str, dict[str, Callable[[float], float]]],
                      emit_masks: dict[str, torch.Tensor | None] | None = None) -> Callable:
    """A `run_trajectory(guidance=...)` callable from `{term: {modality: schedule}}`.

    A term names a differentiable metric output; its schedule gives the signed weight at this
    step's time for one modality, and a modality the term does not name contributes nothing. The
    gradient is taken against detached copies of the clean prediction, so guidance cannot leak
    into the sampler's own graph, and a non-finite gradient contributes zero rather than
    poisoning the state.
    """
    for key in terms:
        if key not in METRICS:
            raise ValueError(f"no differentiable metric named {key}; have {sorted(METRICS)}")

    def guidance(*, x_1_pred: dict[str, torch.Tensor], t: dict[str, float], step: int) -> dict:
        weights = {key: {mode: float(sched(t[mode])) for mode, sched in modes.items() if mode in t}
                   for key, modes in terms.items()}
        active = [key for key, w in weights.items() if any(v != 0.0 for v in w.values())]
        if not active:
            return {}

        with torch.enable_grad():
            state = {mode: value.detach().clone().requires_grad_(True)
                     for mode, value in x_1_pred.items()}
            delta: dict[str, torch.Tensor] = {}
            for key in active:
                modes = [mode for mode, w in weights[key].items() if w != 0.0]
                grads = torch.autograd.grad(METRICS[key](state), [state[m] for m in modes],
                                            retain_graph=True, allow_unused=True)
                for mode, grad in zip(modes, grads):
                    if grad is None:
                        continue
                    contrib = weights[key][mode] * torch.nan_to_num(grad, nan=0.0, posinf=0.0,
                                                                    neginf=0.0)
                    delta[mode] = contrib if mode not in delta else delta[mode] + contrib

        if emit_masks:
            for mode, value in delta.items():
                mask = emit_masks.get(mode)
                if mask is not None:
                    delta[mode] = value * mask[..., None].to(value.dtype)
        return delta

    return guidance
