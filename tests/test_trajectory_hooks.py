"""The trajectory hooks the SC26 demo reads are off by default and only read.

ESMFold2's sampler and Boltz-2's AtomDiffusion.sample hand every sampler state to an optional
callback (step -1 is the initial noise). A fold with the hook on must be bit-identical to one
without it, and with no hook installed nothing is called at all.
"""

import torch

from tt_bio import boltz2, esmfold2


def _fold():
    # a denoiser that pulls every atom halfway to the origin: cheap, deterministic, nontrivial
    return esmfold2.sample_structure(lambda x, t: 0.5 * x, 6, torch.ones(1, 6), steps=5, seed=3)


def test_esmfold2_hook_is_off_by_default():
    assert esmfold2._DUMP is None


def test_esmfold2_hook_sees_every_step_and_changes_nothing():
    plain = _fold()
    seen = []
    esmfold2.set_trajectory_dump(lambda step, x, x_den: seen.append((step, x.clone(), x_den)))
    try:
        hooked = _fold()
    finally:
        esmfold2.set_trajectory_dump(None)
    assert torch.equal(plain, hooked)
    assert [s for s, _, _ in seen] == [-1, 0, 1, 2, 3]
    assert seen[0][2] is None and all(d is not None for _, _, d in seen[1:])
    assert torch.equal(seen[-1][1], hooked)


def test_boltz2_hook_needs_the_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_BIO_TRAJECTORY_DIR", raising=False)
    assert boltz2._trajectory_dump_from_env() is None
    monkeypatch.setenv("TT_BIO_TRAJECTORY_DIR", str(tmp_path))
    dump = boltz2._trajectory_dump_from_env()
    dump(-1, torch.zeros(1, 2, 3), None)
    dump(0, torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_-01.npy", "step_000.npy", "x0_000.npy"]
