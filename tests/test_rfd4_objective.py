"""RFD4-Proteina objective: every analytic seed against float64 central differences, card-free.

The parity against upstream's own torch code lives in `perf/r4t/objective_parity.py` (it needs
upstream's environment); this file needs numpy only and proves the reference's gradient, plus
that the check is able to fail.
"""

import numpy as np
import pytest

from tt_bio.train import rfd4_objective as R
from tt_bio.train.checks import FD_BAR
from tt_bio.train.objectives import objective


def synthetic_batch(seed=0, n_ex=2, k=4, n=14, d_lat=8, n_cls=11):
    """A small batch that exercises every mask and every term, with k candidates per example."""
    rng = np.random.default_rng(seed)
    b = n_ex * k
    pad = np.ones((b, n), bool)
    pad[:, -2:] = False                                   # padding tail
    ctx = np.zeros((b, n), bool)
    ctx[:, 0] = True                                      # one in-context conditioning token
    sll = np.ones((b, n), bool)
    sll[:, 3] = False                                     # a token without local latents
    res = np.ones((b, n), bool)
    res[:, 5] = False                                     # an unresolved token
    mol = np.zeros((b, n), np.int64)
    mol[:, n // 2:] = 1                                   # two molecules
    ins = -np.ones((b, n), np.int64)
    ins[:, 9] = 2
    ins[:, 10] = 4                                        # two unindexed tokens -> indexed sources
    tet = -np.ones((b, n, 4), np.int64)
    tet[:, 6] = [1, 2, 4, 6]
    tet[:, 7] = [6, 8, 9, 7]
    seq = np.zeros((b, n), np.int64)
    seq[:, 2] = 5                                         # a sequence-conditioned token
    lw = np.ones((b, n))
    lw[:, 8] = 2.5
    rep = lambda a: np.repeat(a[::k], k, 0)               # candidates share the example
    x1 = {"bb_ca": rep(rng.normal(size=(b, n, 3))), "local_latents": rep(rng.normal(size=(b, n, d_lat)))}
    t = {m: np.repeat(rng.uniform(0.05, 0.95, n_ex), k) for m in R.MODES}
    batch = {"padding_mask": pad, "is_in_context_conditioning": ctx, "supports_local_latent": sll,
             "is_resolved_mask": res, "molecule_iid": mol, "insertion_source_idx": ins,
             "tetrahedral_vertices": tet, "condition_sequence_1_residue": seq, "loss_weight": lw,
             "residue_type": rep(rng.integers(0, n_cls, (b, n))), "exploration_k": k}
    eps = {m: rng.normal(size=x1[m].shape) for m in R.MODES}
    offsets = {g: rng.normal(size=(b, 3)) for g in (0, 1)}
    batch.update(R.corrupt(batch, x1, t, eps, offsets))
    batch["t"] = t
    outputs = {"v_bb_ca": rng.normal(size=(b, n, 3)), "v_local_latents": rng.normal(size=(b, n, d_lat)),
               "sequence_logits": rng.normal(size=(b, n, n_cls)),
               "insertion_logits": rng.normal(size=(b, n, n))}
    return batch, outputs


def fd_worst(fn, outputs, seeds, h=1e-4, n_probe=60, seed=0, mag_floor=1e-6):
    """Worst relative disagreement of the analytic seeds with central differences.

    h = 1e-4 because the loss is ~40: at 1e-6 the quotient measures float64 roundoff
    (eps * L / h ~ 1e-8 absolute, 1e-5 relative on small seeds), while the O(h^2) truncation
    at 1e-4 measured below 1e-7 on every seed.
    """
    rng = np.random.default_rng(seed)
    worst, probed = 0.0, 0
    for key, g in seeds.items():
        flat = np.abs(g).ravel()
        elig = np.flatnonzero(flat > mag_floor * flat.max())
        for i in rng.choice(elig, min(n_probe, elig.size), replace=False):
            hi = {**outputs, key: outputs[key].copy()}
            lo = {**outputs, key: outputs[key].copy()}
            hi[key].ravel()[i] += h
            lo[key].ravel()[i] -= h
            num = (fn(hi) - fn(lo)) / (2 * h)
            worst = max(worst, abs(num - g.ravel()[i]) / max(abs(num), abs(g.ravel()[i]), 1e-12))
            probed += 1
    return worst, probed


def test_registered_and_every_term_fires():
    batch, outputs = synthetic_batch(1)          # seed 1 violates a handedness margin
    obj = objective("rfd4")
    total, br, seeds = obj(batch, outputs)
    for name in R.TERMS:
        assert br[name]["value"] is not None and br[name]["value"] > 0, (name, br[name])
    assert set(seeds) == {"v_bb_ca", "v_local_latents", "sequence_logits", "insertion_logits"}
    assert np.isclose(total, sum(br[n]["contribution"] for n in R.TERMS))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seeds_match_central_differences(seed):
    batch, outputs = synthetic_batch(seed)
    total_fn = lambda o: R.rfd4_loss(batch, o, R.WEIGHTS)[0]
    _, _, seeds = R.rfd4_loss(batch, outputs, R.WEIGHTS)
    worst, probed = fd_worst(total_fn, outputs, seeds, seed=seed)
    assert probed >= 150
    assert worst < FD_BAR, f"reference vs central differences {worst:.2e} >= {FD_BAR:.0e}"


def test_losers_get_exactly_zero_seed():
    batch, outputs = synthetic_batch()
    _, br, seeds = R.rfd4_loss(batch, outputs, R.WEIGHTS)
    losers = np.setdiff1d(np.arange(outputs["v_bb_ca"].shape[0]), br["winners"])
    for g in seeds.values():
        assert not g[losers].any()


def test_check_can_fail():
    """A seed missing the (1 - t) chain factor must fail the same check (the broken arm)."""
    batch, outputs = synthetic_batch()
    total_fn = lambda o: R.rfd4_loss(batch, o, R.WEIGHTS)[0]
    _, _, seeds = R.rfd4_loss(batch, outputs, R.WEIGHTS)
    t = batch["t"]["bb_ca"][:, None, None]
    broken = {**seeds, "v_bb_ca": seeds["v_bb_ca"] / (1.0 - t)}
    worst, _ = fd_worst(total_fn, outputs, broken)
    assert worst > 100 * FD_BAR


def test_recycle_depth_is_upstreams_draw():
    # Poisson(3) clamped to [1, 9] from default_rng(0x5EED0000 + step): pinned values, so a
    # change of the draw (or of numpy's Poisson) fails here instead of in a curve.
    got = [R.recycle_depth(s) for s in range(8)]
    ref = [int(np.clip(np.random.default_rng(0x5EED0000 + s).poisson(3.0), 1, 9)) for s in range(8)]
    assert got == ref and all(1 <= d <= 9 for d in got)


def test_t_sampler_moments():
    rng = np.random.default_rng(0)
    for m, (p1, p2, p3) in R.T_DISTRIBUTION.items():
        t = R.sample_t(rng, (200_000,), m)
        mean = p3 * 0.5 + (1 - p3) * p1 / (p1 + p2)
        assert abs(t.mean() - mean) < 3e-3 and t.min() >= 0 and t.max() <= 1


def test_fixture_is_upstreams_crops():
    """The committed fixture: upstream's own pipeline output, 16 examples x k=4 at crop 256."""
    from tt_bio.train import rfd4_data

    b = rfd4_data.load()
    assert b["exploration_k"] == 4 and int(b["crop"]) == 256
    assert b["padding_mask"].shape == (64, 256) and len(set(b["example_ids"])) == 16
    # k adjacent rows are the SAME example: upstream's collate replicates, it does not re-crop
    for key in ("padding_mask", "coords", "residue_type", "loss_weight"):
        v = b[key].reshape(16, 4, *b[key].shape[1:])
        assert (v == v[:, :1]).all(), key
    m = R.masks(b)
    assert (m["sup"]["bb_ca"] <= m["emit"]["bb_ca"]).all()
    assert (m["emit"]["local_latents"] <= m["emit"]["bb_ca"]).all()
    assert m["sup"]["bb_ca"].sum() > 0 and (b["loss_weight"] > 1).any()


def test_objective_runs_on_the_fixture():
    from tt_bio.train import rfd4_data

    b = rfd4_data.load()
    with pytest.raises(KeyError, match="autoencoder"):      # never a zero latent
        rfd4_data.clean_samples(b)
    rng = np.random.default_rng(0)
    # A stand-in for the frozen AE: this exercises the plumbing, NOT upstream's target.
    enc = lambda batch: rng.normal(size=batch["coords"].shape[:2] + (rfd4_data.LATENT_PLACEHOLDER,))
    x1 = rfd4_data.clean_samples(b, encode=enc)
    nb = b["padding_mask"].shape[0]
    t = {m: np.repeat(R.sample_t(rng, (nb // 4,), m), 4) for m in R.MODES}
    eps = {m: rng.normal(size=x1[m].shape) for m in R.MODES}
    mol = b["molecule_iid"].astype(np.int64)
    offsets = {int(g): rng.normal(size=(nb, 3)) for g in np.unique(mol)}
    batch = {**b, "molecule_iid": mol}
    batch.update(R.corrupt(batch, x1, t, eps, offsets))
    batch["t"] = t
    outputs = {"v_bb_ca": rng.normal(size=x1["bb_ca"].shape),
               "v_local_latents": rng.normal(size=x1["local_latents"].shape),
               "sequence_logits": rng.normal(size=(nb, 256, 147)),
               "insertion_logits": rng.normal(size=(nb, 256, 256))}
    total, br, seeds = R.rfd4_loss(batch, outputs, R.WEIGHTS)
    assert np.isfinite(total) and total > 0
    assert len(br["winners"]) == 16
    for name in ("bb_ca", "local_latents", "sequence_prediction"):
        assert br[name]["value"] > 0, (name, br[name])
    for g in seeds.values():
        assert np.isfinite(g).all()
