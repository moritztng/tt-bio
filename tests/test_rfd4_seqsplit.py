"""The fixed half of `init_repr_factory`, computed once a design (`tt_bio.rfd4.seq`).

19 of the released net's 21 sequence features read the featurized example and not the noisy state.
These tests hold the split projection to upstream's whole one on the released weights, and hold the
partition itself: every creator the split calls constant must ignore `x_t`.

Run: python3 -m pytest tests/test_rfd4_seqsplit.py (no card; RFD4_WEIGHTS / RFD4_SPEC as in
test_rfd4_hostlane.py for the bundle-backed cases).
"""
import os
from pathlib import Path

import pytest
import torch

WEIGHTS = Path(os.environ.get("RFD4_WEIGHTS", Path.home() / "rfd4-bundle"))
SPEC = Path(os.environ.get("RFD4_SPEC", Path.home() / "rfd4-inputs" / "pdl1-128.yaml"))
needs_bundle = pytest.mark.skipif(not (WEIGHTS.is_dir() and SPEC.is_file()),
                                  reason=f"released bundle {WEIGHTS} or spec {SPEC} missing")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N

    designs = rfd4.prepare_examples(str(SPEC), str(tmp_path_factory.mktemp("inputs")))
    feats, _ = rfd4.featurize(designs[0].cif, seed=0)
    sd, cfg = N.load_checkpoint(WEIGHTS)
    return N.FlowGlue(sd, cfg), E.preprocess(feats)


def _with_state(batch, seed):
    """`batch` with a noisy state written in, as the sampler writes it."""
    g = torch.Generator().manual_seed(seed)
    n = batch["padding_mask"].shape[1]
    latent = batch["x_t"]["local_latents"].shape[-1] if "x_t" in batch else 32
    out = dict(batch)
    out["x_t"] = {"bb_ca": torch.randn(1, n, 3, generator=g),
                  "local_latents": torch.randn(1, n, latent, generator=g)}
    # `t` is what the sampler writes beside the state; the pair builder and the conditioning read it.
    out["t"] = {mode: torch.full((1,), 0.1 * seed) for mode in out["x_t"]}
    return out


@needs_bundle
def test_the_split_projection_is_upstreams(case):
    """`W_const f_const + W_step f_step` against `W cat(f_const, f_step)`, two different states."""
    glue, batch = case
    assert glue.seq_split is not None
    first = _with_state(batch, 11)
    constant = glue.seq_split.constant(first)
    for seed in (11, 12):
        b = _with_state(batch, seed)
        with torch.no_grad():
            want = glue.init_repr_factory(b)
        got = glue.seq_split.s_init(b, constant)
        assert got.shape == want.shape
        # One fp32 sum split in two, as `PairSplit` does; the tokens are O(1) and 768 wide.
        assert torch.allclose(got, want, atol=2e-5, rtol=2e-5), (got - want).abs().max()


@needs_bundle
def test_the_constant_half_really_ignores_the_state(case):
    """Every creator the split calls constant must give the same tensor at two different states."""
    glue, batch = case
    a, b = _with_state(batch, 1), _with_state(batch, 2)
    assert not torch.equal(a["x_t"]["bb_ca"], b["x_t"]["bb_ca"])
    for creator in glue.seq_split.const_creators:
        with torch.no_grad():
            assert torch.equal(creator(a), creator(b)), type(creator).__name__


@needs_bundle
def test_a_design_sees_the_same_initial_states(case):
    """`initial_states` with the design's constant equals `initial_states` without it."""
    from tt_bio.rfd4 import network as N

    glue, batch = case
    net = N.FlowNetwork(glue, lambda *a: None, batch)
    b = _with_state(batch, 7)
    constant = glue.seq_split.constant(b)
    want, _ = glue.initial_states(b, None, None, None)
    got, _ = glue.initial_states(b, None, None, constant)
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5), (got - want).abs().max()
    assert net._seq_constant == {}
