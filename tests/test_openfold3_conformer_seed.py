"""The reference-conformer ETKDG seed: a retry stays inside its own residue.

`_pooled_reference_molecules` hands residue i the seed the sequential loop would have drawn
for it. A retry used to draw its second seed from python's global `random`, which stole the
next residue's seed and made the caller recompute all 108 conformers off a stream position
that depended on how many residues had retried -- a different `ref_pos`, which is a model
input, from one transient 30 s ETKDG timeout. Measured on `examples/fkg_ligand.yaml`: every
one of 864 atoms moved, RMSD 0.14-0.28 A, one structure per retry count.
"""
import random

import pytest

from tt_bio._vendor.openfold3.core.data.primitives.structure import conformer as C


@pytest.fixture(autouse=True)
def _clean_pool():
    C.pool_reset()
    yield
    C.pool_reset()


def test_first_draw_is_the_pooled_seed():
    C.pool_seed(1234)
    assert C._etkdg_seed() == 1234


def test_retry_comes_from_the_residue_s_own_stream():
    C.pool_seed(1234)
    C._etkdg_seed()
    expected = random.Random(1234).randint(0, 10**9)
    assert C._etkdg_seed() == expected
    # And again, from the same stream rather than a fresh one.
    assert C._etkdg_seed() != expected


def test_retry_does_not_move_the_global_stream():
    """The next residue's seed is unaffected by this residue retrying."""
    random.seed(0)
    before = [random.randint(0, 10**9) for _ in range(3)]

    random.seed(0)
    C.pool_seed(77)
    C._etkdg_seed()
    for _ in range(5):
        C._etkdg_seed()
    C.pool_reset()
    after = [random.randint(0, 10**9) for _ in range(3)]

    assert before == after


def test_unpooled_caller_still_draws_from_the_global_stream():
    random.seed(0)
    expected = random.randint(0, 10**9)
    random.seed(0)
    assert C._etkdg_seed() == expected


def test_timed_out_embedding_retries_the_same_strategy_and_seed():
    """A 30 s budget expires only on a starved host, so the retry must be the same
    embedding, not a different strategy: otherwise the geometry is a function of the load.
    """
    from func_timeout import FunctionTimedOut
    from rdkit import Chem

    seeds = []
    calls = []

    def fake_func_timeout(timeout, func, args=(), kwargs=None):
        seeds.append(args[1].randomSeed)
        calls.append(timeout)
        if len(calls) == 1:
            raise FunctionTimedOut("test: forced ETKDG timeout")
        return func(*args, **(kwargs or {}))

    C.pool_seed(4242)
    old = C.func_timeout
    C.func_timeout = fake_func_timeout
    try:
        mol, conf_id = C._compute_conformer(Chem.MolFromSmiles("CCO"), timeout=30.0)
    finally:
        C.func_timeout = old

    assert len(seeds) == 2, seeds
    assert seeds[0] == seeds[1] == 4242
    assert calls == [30.0, 30.0]
    assert conf_id == 0
    assert mol.GetNumConformers() == 1
