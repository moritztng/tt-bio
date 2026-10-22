"""The next step's conditioning, prepared while the card runs this one (`FlowNetwork(plan=)`).

The conditioning and the pair AdaLN's modulation read only `t`, the CATH code and the remaining-pass
count, so `FlowNetwork.launch` computes step k+1's after enqueueing step k's trunk, and lanes share
one. These tests hold that to the same tensors as computing them in `prepare`, on the released
weights with a stub trunk that reads every input, and hold `sampler.step_plan` to what a trajectory
actually asks for.

Run: python3 -m pytest tests/test_rfd4_hostlane.py (no card; RFD4_WEIGHTS / RFD4_SPEC as in
test_rfd4_pair_device.py for the bundle-backed cases).
"""
import os
from pathlib import Path

import pytest
import torch

from tt_bio.rfd4 import sampler as S

WEIGHTS = Path(os.environ.get("RFD4_WEIGHTS", Path.home() / "rfd4-bundle"))
SPEC = Path(os.environ.get("RFD4_SPEC", Path.home() / "rfd4-inputs" / "pdl1-128.yaml"))
needs_bundle = pytest.mark.skipif(not (WEIGHTS.is_dir() and SPEC.is_file()),
                                  reason=f"released bundle {WEIGHTS} or spec {SPEC} missing")


class _Recorder:
    def __init__(self):
        self.asked = []

    def __call__(self, x_t, t, *, recycle_depth, step):
        self.asked.append((dict(t), recycle_depth))
        return {mode: {"v": torch.zeros_like(x)} for mode, x in x_t.items()}


@pytest.mark.parametrize("start", [S.PartialStart(),
                                   S.PartialStart(t={"bb_ca": 0.4, "local_latents": 0.4}, nsteps=20)],
                         ids=["full", "partial"])
def test_step_plan_is_what_the_trajectory_asks_for(start):
    spec = S.recycle_strategy("thinking_medium")
    rec = _Recorder()
    n = 16
    x_1 = {S.BB_CA: torch.randn(1, n, 3), S.LOCAL_LATENTS: torch.randn(1, n, 32)}
    S.generate(spec, rec, n_designs=1, n_tokens=n, seed=0, emit_masks={}, start=start,
               x_1=x_1 if start.is_partial else None)
    assert rec.asked == S.step_plan(spec, start)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N

    designs = rfd4.prepare_examples(str(SPEC), str(tmp_path_factory.mktemp("inputs")))
    feats, _ = rfd4.featurize(designs[0].cif, seed=0)
    sd, cfg = N.load_checkpoint(WEIGHTS)
    return N.FlowGlue(sd, cfg), E.preprocess(feats)


class _StubTrunk:
    """Reads everything the card would: s_init, every recycle's conditioning, the pair inputs."""
    builds_pair = True

    def launch(self, s_init, z_init, conds, mask):
        mix = sum(c.mean(-1, keepdim=True) for c in conds) / len(conds)
        pair = (z_init.gamma.mean(-1) + z_init.beta.mean(-1))[:, None, None] + z_init.scale.mean(-1, keepdim=True)
        return torch.tanh(s_init + mix + pair) * mask[..., None]

    def collect(self, pending):
        return pending, None


class _NoMemo(dict):
    """The reference: every prepare computes its own conditioning, as before the memo."""

    def __setitem__(self, key, value):
        pass


def _run(case, plan, n_designs, calls):
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N

    glue, batch = case
    spec = S.recycle_strategy("thinking_medium")
    net = N.FlowNetwork(glue, _StubTrunk(), batch, plan=S.step_plan(spec) if plan else None)
    if not plan:
        net._conditioned = _NoMemo()
    inner = glue.conditioning

    def counted(*a, **k):
        calls.append(1)
        return inner(*a, **k)

    glue.conditioning = counted
    try:
        return S.generate(spec, net, n_designs=n_designs, n_tokens=batch["padding_mask"].shape[1],
                          seed=3, rng_mode="per_design", emit_masks=compute_flow_emit_masks(batch),
                          groups=E._groups(batch), coord_cond_mask=batch.get("coord_cond_mask"),
                          lane_width=1 if n_designs > 1 else None)
    finally:
        glue.conditioning = inner


@needs_bundle
@pytest.mark.parametrize("n_designs", [1, 3], ids=["one", "three_lanes"])
def test_prefetched_conditioning_gives_the_same_designs(case, n_designs):
    base_calls, plan_calls = [], []
    base = _run(case, False, n_designs, base_calls)
    ahead = _run(case, True, n_designs, plan_calls)
    for mode in base.x:
        assert torch.equal(base.x[mode], ahead.x[mode])
    assert torch.equal(base.sequence_logits, ahead.sequence_logits)
    assert torch.equal(base.insertion_logits, ahead.insertion_logits)
    # One conditioning per recycle of the plan, whatever the lane count: 180 for thinking_medium.
    assert len(base_calls) == 180 * n_designs
    assert len(plan_calls) == 180
