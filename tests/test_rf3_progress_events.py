"""RF3 reports its trunk recycles and diffusion steps, host-only.

The live view reads `stage` events: a trunk tick with `total=0`, or no diffusion tick at
all, renders as a bar that never fills and a phase that never appears. That is what
`--model rf3` shipped with until the v0.6.6 UX gate scored it: the worker emitted one
bare `report_progress("trunk")` and handed no callback to `predict`, so no loop ticked.

The trunk test drives the real `RF3.trunk` loop with the device calls stubbed, rather
than a mirror of it, so a change to the loop is a change to what is tested.
"""
from __future__ import annotations

import inspect
import types

import pytest
import torch

from tt_bio.rf3 import model as rf3_model
from tt_bio.rf3.model import RF3
from tt_bio.rf3.sampler import DiffusionSampler


def _recorder():
    ticks = []

    def progress_fn(stage, step=0, total=0):
        ticks.append((stage, step, total))
    return progress_fn, ticks


# ── the diffusion rollout ─────────────────────────────────────────────────


def _stub_denoise(x_noisy, t):
    return x_noisy


@pytest.mark.parametrize("partial_t", [0, 190])
def test_sampler_ticks_once_per_step_with_a_real_total(partial_t):
    s = DiffusionSampler()
    progress_fn, ticks = _recorder()
    s.sample(_stub_denoise, torch.randn(1, 16, 3), 1,
             partial_t=partial_t, progress_fn=progress_fn)

    steps = s.num_timesteps - partial_t - 1
    assert [t[0] for t in ticks] == ["diffusion"] * steps, \
        "one diffusion tick per rollout step"
    assert [t[1] for t in ticks] == list(range(steps)), "step index must advance"
    assert {t[2] for t in ticks} == {steps}, \
        "total must be the real step count, not 0 (the empty-bar bug)"


def test_sampler_without_a_callback_is_the_same_structure():
    """The callback is reporting, not arithmetic: passing one must not move a number."""
    s = DiffusionSampler()
    coord = torch.randn(2, 24, 3)
    torch.manual_seed(0)
    a, _ = s.sample(_stub_denoise, coord, 2)
    progress_fn, ticks = _recorder()
    torch.manual_seed(0)
    b, _ = s.sample(_stub_denoise, coord, 2, progress_fn=progress_fn)
    assert torch.equal(a, b)
    assert ticks, "the callback did run"


# ── the trunk recycling loop ──────────────────────────────────────────────


def _trunk_host(n_msa=1):
    """The attributes `RF3.trunk` reads off HostInputs, as sentinels."""
    return types.SimpleNamespace(
        single_in=None, pair_in=None, pair_v=None, keys_indexing=None,
        atom_to_token_mean=None, window_mask=None, n_atom_padded=0,
        token_feats=None, relpos_feat=None, bond_feat=None,
        template_feats=None, msa_stack=[object() for _ in range(n_msa)])


def _fake_rf3(recycles_seen):
    """A bare RF3 whose device collaborators are counted instead of run."""
    m = object.__new__(RF3)
    m.feature_initializer = lambda *a: ("s_inputs", "s_init", "z_init")
    m.recycler = types.SimpleNamespace(
        template_embedder=types.SimpleNamespace(
            embed_template_feats=lambda _f: "template_channels"))

    def recycler(*a):
        recycles_seen.append(1)
        return "s", "z"
    recycler.template_embedder = m.recycler.template_embedder
    m.recycler = recycler
    return m


@pytest.fixture
def no_device_ttnn(monkeypatch):
    """`trunk` zeroes its carry with ttnn.mul; nothing else there touches a device."""
    monkeypatch.setattr(rf3_model, "ttnn",
                        types.SimpleNamespace(mul=lambda t, _s: t))


@pytest.mark.parametrize("n_recycles", [1, 4, 10])
def test_trunk_ticks_once_per_recycle_with_a_real_total(no_device_ttnn, n_recycles):
    seen = []
    progress_fn, ticks = _recorder()
    _fake_rf3(seen).trunk(_trunk_host(), n_recycles, progress_fn=progress_fn)

    assert len(seen) == n_recycles, "the real loop ran every recycle"
    assert [t[0] for t in ticks] == ["trunk"] * n_recycles
    assert [t[1] for t in ticks] == list(range(n_recycles)), "step index must advance"
    assert {t[2] for t in ticks} == {n_recycles}, \
        "total must be n_recycles, not 0 — a zero total is the empty trunk bar"


def test_trunk_without_a_callback_still_runs(no_device_ttnn):
    seen = []
    _fake_rf3(seen).trunk(_trunk_host(), 3)
    assert len(seen) == 3


# ── the wiring that carries the callback from the worker to both loops ────


def test_predict_forwards_the_callback_to_both_loops():
    src = inspect.getsource(RF3.predict)
    assert "progress_fn" in inspect.signature(RF3.predict).parameters
    assert src.count("progress_fn=progress_fn") == 2, \
        "predict must forward the callback to the trunk AND the sampler"


def test_the_worker_hands_the_callback_to_predict():
    """The defect the UX gate caught was here, not in the model: a one-shot
    report_progress("trunk") with no callback threaded into predict()."""
    from tt_bio.worker import _WorkerState
    src = inspect.getsource(_WorkerState._predict_rf3_one)
    assert "progress_fn=report_progress" in src, \
        "the rf3 fold path must hand report_progress to predict, not tick once by hand"
