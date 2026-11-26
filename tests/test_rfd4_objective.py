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
