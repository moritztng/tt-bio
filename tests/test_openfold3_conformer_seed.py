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


def test_retry_after_a_timeout_that_still_embedded_returns_conformer_zero():
    """The timed-out embedding keeps running, and the retry must not inherit its conformer.

    `func_timeout` cannot interrupt `AllChem.EmbedMolecule`: it is a C++ call that never
    returns to the interpreter to receive the async exception, so it finishes and deposits
    its conformer on the molecule anyway. `strategy.clearConfs` is False, so a retry on
    that same molecule appends a SECOND conformer and returns conf_id 1 --
    `assert conf_id == 0` in `processed_reference_molecule_from_mol` then kills the fold
    with a bare AssertionError. Measured over fkg_ligand's 107-residue protein: 1 to 4
    residues per fold returned conf_id 1 at every budget from 1 ms to 8 ms.

    The sibling test above raises the timeout INSTEAD of embedding, which is why it could
    not see this: a timeout double that skips the wrapped call cannot model the side effect
    a real timeout leaves behind.
    """
    from func_timeout import FunctionTimedOut
    from rdkit import Chem

    calls = []

    def timed_out_but_still_embedded(timeout, func, args=(), kwargs=None):
        calls.append(args[1].randomSeed)
        if len(calls) == 1:
            func(*args, **(kwargs or {}))  # what the uninterruptible C++ call does anyway
            raise FunctionTimedOut("test: budget expired while the embedding finished")
        return func(*args, **(kwargs or {}))

    C.pool_seed(99)
    old = C.func_timeout
    C.func_timeout = timed_out_but_still_embedded
    try:
        mol, conf_id = C._compute_conformer(Chem.MolFromSmiles("CCO"), timeout=120.0)
    finally:
        C.func_timeout = old

    assert calls == [99, 99], calls          # same strategy, same seed, as before
    assert conf_id == 0, f"retry inherited the timed-out attempt's conformer: {conf_id}"
    assert mol.GetNumConformers() == 1
