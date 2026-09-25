#!/usr/bin/env python3
"""The card's trunk and the JAX side around it must be the same checkpoint.

No card and no weights: the invariant is about WHICH NAME each side gets, and that is
decided before any arithmetic happens. `predict` and `sequence_gradients` resolve
`model` themselves (`bindcraft/af2.py:287,381`), and resolving `None` samples a name and
splits `self.key`. Resolving it a second time to choose the card's trunk therefore drew a
SECOND name, and the two sides disagreed four times in five.

Run: python3 perf/bcx_predictor/test_pool_checkpoint_match.py
"""
import pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import bc2_state  # noqa: F401  -- puts BindCraft 2 on the path
from bindcraft.af2 import AlphaFoldDesignModel
from multimer_pool import POOL
from ttbio_predictor import TTBioAlphaFoldDesignModel

DRAWS = 20


class FakePool:
    """Records what the splice would have been pointed at. No device, no weights."""
    def __init__(self):
        self.used = []

    def use(self, name):
        self.used.append(name)


def instance(pool):
    """A predictor with only the fields resolution reads, so nothing has to be loaded."""
    m = object.__new__(TTBioAlphaFoldDesignModel)
    m.key = jax.random.PRNGKey(0)
    m.models = POOL
    m.model_families = {name: ('multimer', 'v3') for name in POOL}
    m.pool = pool
    m.trunk = 'device'
    m.card = None
    m._device = None
    return m


def with_stubbed_base(fn):
    """Stand in for BindCraft 2's own predict/sequence_gradients, which resolve `model`
    themselves. What they resolve is what the heads, templates and structure module use."""
    seen = []

    def predict(self, protein_states, model=None, *a, **k):
        seen.append(self._resolve_model_name(model))
        return {}

    def sequence_gradients(self, protein_states, losses, model=None, *a, **k):
        seen.append(self._resolve_model_name(model))
        return {}, {}

    real = AlphaFoldDesignModel.predict, AlphaFoldDesignModel.sequence_gradients
    AlphaFoldDesignModel.predict = predict
    AlphaFoldDesignModel.sequence_gradients = sequence_gradients
    try:
        fn(seen)
    finally:
        AlphaFoldDesignModel.predict, AlphaFoldDesignModel.sequence_gradients = real
    return seen


def check(seen_holder):
    pool = FakePool()
    m = instance(pool)
    for i in range(DRAWS):
        if i % 2:
            m.predict({}, None)
        else:
            m.sequence_gradients({}, {}, None)
    seen_holder.append(('card', pool.used))


def witness(seen_holder):
    """The defect this test is here for, reproduced on purpose: resolve twice and the two
    sides disagree. Without it a green test could just mean the pool is never consulted."""
    pool = FakePool()
    m = instance(pool)
    for _ in range(DRAWS):
        pool.use(m._resolve_model_name(None))   # what _select used to do
        m.predict({}, None)                     # and BindCraft 2 drew again
    seen_holder.append(('card', pool.used))


def no_pool_draws_nothing_extra(seen_holder):
    m = instance(None)
    for _ in range(DRAWS):
        m.predict({}, None)
    seen_holder.append(('key', m.key))


failures = []

seen = []
jax_side = with_stubbed_base(lambda s: (check(seen), s))
card_side = dict(seen)['card']
if list(card_side) != list(jax_side):
    failures.append(f'fixed arm disagrees: card {card_side} vs jax {jax_side}')
if len(set(jax_side)) < 2:
    failures.append(f'the pool never varied over {DRAWS} draws: {jax_side}')

seen = []
w_jax = with_stubbed_base(lambda s: (witness(seen), s))
w_card = dict(seen)['card']
mismatches = sum(1 for a, b in zip(w_card, w_jax) if a != b)
if mismatches == 0:
    failures.append('the witness did not reproduce the defect, so the check above proves nothing')

seen = []
with_stubbed_base(lambda s: (no_pool_draws_nothing_extra(seen), s))
key_after = dict(seen)['key']
baseline = instance(None)
for _ in range(DRAWS):
    baseline._resolve_model_name(None)
if not (key_after == baseline.key).all():
    failures.append('the no-pool arm consumed a draw it did not consume before the fix')

print(f'fixed arm      : card and JAX agree on all {DRAWS} draws, '
      f'{len(set(jax_side))} distinct checkpoints used')
print(f'witness (bug)  : {mismatches}/{DRAWS} draws had the card on a different checkpoint')
print(f'no-pool arm    : key unmoved, {DRAWS} draws')
if failures:
    print('FAIL: ' + '; '.join(failures))
    sys.exit(1)
print('PASS')
