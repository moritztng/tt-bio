"""The RFD4 curve leg takes upstream's optimizer step, not just upstream's data.

The fixture is upstream's own `build_optimizers` split over the released model (282 Muon, 276
Adam) beside the 558 names the TT trunk trains, read from a TT curve-leg checkpoint. TT names its
host-side heads ``glue.X`` where upstream says ``nn.X``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tt_bio.train.optim import Muon
from tt_bio.train.rfd4_objective import MUON_EXCLUDE, UPDATE_RULE, update_rule

FX = json.loads((Path(__file__).parent / "fixtures/rfd4_upstream_optimizer_split.json").read_text())


def upstream_name(n):
    return "nn." + n[len("glue."):] if n.startswith("glue.") else n


def split(exclude):
    m = Muon(exclude=exclude)
    muon = sorted(upstream_name(n) for n, s in FX["tt_names"].items() if m.takes(n, s))
    adam = sorted(upstream_name(n) for n, s in FX["tt_names"].items() if not m.takes(n, s))
    return muon, adam


def test_the_tt_leg_trains_upstreams_parameters():
    assert sorted(map(upstream_name, FX["tt_names"])) == sorted(FX["muon"] + FX["adam"])


def test_routing_matches_upstreams_split_name_for_name():
    muon, adam = split(MUON_EXCLUDE)
    assert (len(muon), len(adam)) == (282, 276)
    assert muon == sorted(FX["muon"]) and adam == sorted(FX["adam"])


def test_the_routing_check_fails_on_upstreams_patterns_read_literally():
    """Upstream's head pattern anchors on ``nn.``; applied verbatim to TT names it misses the
    glue heads and routes them to Muon. The check above has to catch that."""
    muon, _ = split(tuple(FX["patterns"]))
    assert muon != sorted(FX["muon"])
    assert "nn.insertion_q_proj.weight" in muon


def test_hyperparameters_are_upstreams():
    a, m = FX["args"]["Adam"], FX["args"]["GroupedMuon"]
    mu = Muon()
    assert (mu.momentum, mu.nesterov, mu.eps, mu.ns_steps, mu.weight_decay, mu.adjust_lr_fn) == (
        m["momentum"], m["nesterov"], m["eps"], m["ns_steps"], m["weight_decay"], m["adjust_lr_fn"])
    assert tuple(mu.ns_coefficients) == tuple(m["ns_coefficients"])
    assert a["weight_decay"] == 0 and tuple(a["betas"]) == (0.9, 0.999) and a["eps"] == 1e-8
    assert a["lr"] == m["lr"] == 1e-4 and UPDATE_RULE["clip"] == 1.0


class _Val(np.ndarray):
    def device(self):
        return None


class _P:
    def __init__(self, shape):
        self.value = np.ones(shape, np.float32).view(_Val)
        self.grad = None


def test_update_rule_builds_upstreams_optimizer():
    p = {"nn.transformer_layers.0.mhba.mha.to_g.weight": _P((8, 8)),
         "glue.insertion_q_proj.weight": _P((8, 8)),
         "glue.init_repr_factory.linear_out.weight": _P((8, 4)),
         "nn.s_cycle_norm.weight": _P((8,))}
    opt = update_rule(p)
    assert opt.muon_names == {"nn.transformer_layers.0.mhba.mha.to_g.weight"}
    assert (opt.lr, opt.weight_decay, opt.clip_norm, opt.schedule) == (1e-4, 0.0, 1.0, None)


def _steps(opt, ps, grads):
    from tt_bio.train import optim
    keep = optim.to_device
    optim.to_device = lambda arr, dev, dtype=None: np.array(arr, np.float32).view(_Val)
    try:
        for g in grads:
            for n, p in ps.items():
                p.grad = g[n]
            opt.step()
    finally:
        optim.to_device = keep


def test_a_resumed_muon_run_takes_the_unbroken_runs_steps(tmp_path):
    """The curve leg checkpoints every 100 steps. Muon's state is its momentum buffer; an adapter
    that drops it resumes at zero momentum and the resumed curve is a different run."""
    pytest.importorskip("safetensors")
    from tt_bio.train import checkpoint
    shapes = {"nn.a.weight": (8, 8), "nn.s_cycle_norm.weight": (8,)}
    rng = np.random.default_rng(0)
    grads = [{n: rng.normal(size=s).astype(np.float32) for n, s in shapes.items()} for _ in range(6)]

    ps = {n: _P(s) for n, s in shapes.items()}
    whole = update_rule(ps)
    _steps(whole, ps, grads)

    ps = {n: _P(s) for n, s in shapes.items()}
    first = update_rule(ps)
    _steps(first, ps, grads[:3])
    checkpoint.save_adapter(tmp_path / "a.safetensors", first)
    ps = {n: _P(s) for n, s in shapes.items()}
    resumed = update_rule(ps)
    keep = checkpoint.to_device
    checkpoint.to_device = lambda arr, dev, dtype=None: np.array(arr, np.float32).view(_Val)
    try:
        checkpoint.load_adapter(tmp_path / "a.safetensors", ps, None, opt=resumed)
        _steps(resumed, ps, grads[3:])
        for n in shapes:
            np.testing.assert_array_equal(resumed.master[n], whole.master[n])

        # an adapter written without Muon (the AdamW leg's) cannot seed a Muon run
        from tt_bio.train.optim import AdamW
        plain = AdamW({n: _P(s) for n, s in shapes.items()}, lr=1e-4, weight_decay=0.0)
        checkpoint.save_adapter(tmp_path / "b.safetensors", plain)
        with pytest.raises(ValueError, match="no Muon momentum"):
            checkpoint.load_adapter(tmp_path / "b.safetensors", {n: _P(s) for n, s in shapes.items()},
                                    None, opt=update_rule({n: _P(s) for n, s in shapes.items()}))
    finally:
        checkpoint.to_device = keep
