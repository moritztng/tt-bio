"""The initial pair built on the card (`tt_bio.rfd4.pair`) against upstream's builder in float64.

The split itself is graded on the host: the constant projection plus the gathered step rows,
normalised and modulated, against `PairReprBuilder` run in float64 on the same released weights and
the same featurized example. The card is then graded against the same float64 pair, beside the
error of upstream's own fp32 path, which is what a design runs today.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_rfd4_pair_device.py
(RFD4_WEIGHTS: the released bundle, default ~/rfd4-bundle; RFD4_SPEC: an input spec, default
~/rfd4-inputs/pdl1-128.yaml.)
"""
import itertools
import os
from pathlib import Path

import pytest
import torch

from tt_bio.rfd4.pair import PairSplit, split_bf16

WEIGHTS = Path(os.environ.get("RFD4_WEIGHTS", Path.home() / "rfd4-bundle"))
SPEC = Path(os.environ.get("RFD4_SPEC", Path.home() / "rfd4-inputs" / "pdl1-128.yaml"))
needs_bundle = pytest.mark.skipif(not (WEIGHTS.is_dir() and SPEC.is_file()),
                                  reason=f"released bundle {WEIGHTS} or spec {SPEC} missing")


def test_bf16_parts_sum_back_exactly_in_any_order():
    w = torch.randn(30, 384) * torch.logspace(-6, 2, 384)
    parts = split_bf16(w).float()
    for i, j, k in itertools.permutations(range(3)):
        assert torch.equal((parts[i] + parts[j]) + parts[k], w)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N

    designs = rfd4.prepare_examples(str(SPEC), str(tmp_path_factory.mktemp("inputs")))
    feats, _ = rfd4.featurize(designs[0].cif, seed=0)
    sd, cfg = N.load_checkpoint(WEIGHTS)
    glue = N.FlowGlue(sd, cfg)
    batch = E.preprocess(feats)
    return glue, batch


def _step(batch, seed, t):
    n = batch["padding_mask"].shape[1]
    g = torch.Generator().manual_seed(seed)
    b = dict(batch)
    b["x_t"] = {"bb_ca": torch.randn(1, n, 3, generator=g) * (2.0 - t),
                "local_latents": torch.randn(1, n, 8, generator=g)}
    b["t"] = {"bb_ca": torch.full((1,), t), "local_latents": torch.full((1,), t)}
    return b


def _ref64(builder, b):
    """`PairReprBuilder` in float64. Its features are one-hots and small integers, exact in fp32, so
    they are taken from upstream's creators and every op after them runs in double."""
    n = b["padding_mask"].shape[1]

    def feats(factory):
        out = []
        for c in factory.feat_creators:
            x = c(b).double()
            if x.ndim == 2 and factory.mode == "pair" and not factory.compact:
                x = x.reshape(x.shape[0], 1, 1, -1).expand(-1, n, n, -1)
            out.append(x)
        x = torch.cat(out, -1) @ factory.linear_out.weight.double().t()
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), factory.ln_out.weight.double(),
                                            factory.ln_out.eps)

    ada = builder.adaln
    nc = torch.nn.functional.rms_norm(feats(builder.cond_factory), (ada.norm_cond.weight.shape[0],),
                                      ada.norm_cond.weight.double(), ada.norm_cond.eps)
    gamma = torch.sigmoid(nc @ ada.to_gamma[0].weight.double().t() + ada.to_gamma[0].bias.double())
    beta = nc @ ada.to_beta.weight.double().t()
    x = feats(builder.init_repr_factory)
    x = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, ada.norm.eps)
    return x * gamma[:, None, None] + beta[:, None, None]


@needs_bundle
def test_the_split_is_upstreams_builder(case):
    glue, batch = case
    split = glue.pair_split
    constant = split.constant(batch)
    for seed, t in ((0, 0.05), (1, 0.6), (2, 0.97)):
        b = _step(batch, seed, t)
        with torch.no_grad():
            ref = _ref64(glue.pair_repr_builder, b)
            host = glue.pair_repr_builder(b).double()
            got = split.assemble(split.inputs(b, constant)).double()
        # Upstream's own fp32 path sits ~3e-7 from float64; the split is one sum reordered.
        assert (host - ref).abs().max() < 2e-6
        assert (got - ref).abs().max() < 2e-6
        assert (got - host).abs().max() < 2e-6


@needs_bundle
def test_the_constant_does_not_move_with_the_step(case):
    glue, batch = case
    split = glue.pair_split
    for a, b in ((_step(batch, 0, 0.1), _step(batch, 1, 0.9)), (batch, _step(batch, 0, 0.1))):
        ca, cb = split.constant(a), split.constant(b)
        assert torch.equal(ca.proj, cb.proj) and torch.equal(ca.scales, cb.scales)


@needs_bundle
def test_the_bins_are_upstreams(case):
    from tt_bio._vendor.rfd4_prep._cif.nn_feature_factory import bin_pairwise_distances

    glue, batch = case
    split = glue.pair_split
    c = split.step_creator
    b = _step(batch, 3, 0.5)
    onehot = bin_pairwise_distances(b["x_t"]["bb_ca"], c.min_dist, c.max_dist, c.dim)
    assert torch.equal(onehot.argmax(-1), split.bins(b))


@pytest.mark.device
@needs_bundle
@pytest.mark.parametrize("npad", [0, 32])
def test_the_card_builds_the_pair(case, npad):
    import ttnn

    from tt_bio import tenstorrent as T
    from tt_bio.rfd4.pair import PairOnCard

    glue, batch = case
    split = glue.pair_split
    dev = T.get_device()
    card = PairOnCard(split, dev)
    constant = split.constant(batch)
    n = batch["padding_mask"].shape[1]
    b = _step(batch, 0, 0.6)
    with torch.no_grad():
        p = split.inputs(b, constant)
        ref = _ref64(glue.pair_repr_builder, b)
        host = glue.pair_repr_builder(b).double()
    bins = torch.nn.functional.pad(p.bins, [0, npad, 0, npad]) if npad else p.bins
    rows = ttnn.to_torch(card.rows(bins))[:, :n, :n]
    # a one-hot matmul over the three bf16 parts: ~1e-4 at |w| ~ 0.1, not bit for bit
    assert (rows - split.w_step[p.bins]).abs().max() < 1e-3, "the gather is off"
    got = ttnn.to_torch(card(p, npad)).double()
    assert got.shape[1] == n + npad and torch.isfinite(got).all()
    got = got[:, :n, :n]
    err, host_err = (got - ref).abs(), (host - ref).abs()
    rel_rms = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
    print(f"npad {npad}: card max {err.max():.3e} rel_rms {rel_rms:.3e}; host fp32 max {host_err.max():.3e}")
    # The card adds and multiplies in fp32 and reduces nothing, so it should sit near fp32's own
    # error, like the host. A reduction on the card (ttnn.sum unpacks to tf32) read 2.1e-4 here.
    assert rel_rms < 1e-4
    card.release()
