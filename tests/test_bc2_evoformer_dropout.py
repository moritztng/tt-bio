"""The on-card Evoformer draws AlphaFold 2's dropout, and draws exactly what the host would.

Issue #17. BindCraft 2 designs with `design_dropout` true by default and turns it off only for
`harden` (`bindcraft/trajectory.py:301`), so every default campaign regularises its design loop
with dropout. tt-bio's device stack replaces the whole `layer_stack` body and applied none of it,
in any stage, and the card did not say so: host and card ran different regularisers.

What is pinned here is the draw, against the reference's own `modules.dropout_wrapper` rather than
against a re-derivation of it. The mask a block applies is `dropout_wrapper`'s mask for the same
sub-key, element for element, which is stronger than "dropout of the right rate": a trajectory on
a card takes the same path as that trajectory on host JAX, so the two arms stay comparable.

No card and no weights: the draw is JAX, and the device multiply it feeds is the card leg.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("bindcraft")

from bindcraft.af.alphafold.model import modules, prng  # noqa: E402

from tt_bio import bindcraft2  # noqa: E402

BLOCKS, N, C_PAIR, C_MSA, ROWS = 3, 7, 5, 4, 2


class _Op:
    """A module whose update is all ones, so `dropout_wrapper`'s output IS the mask."""

    def __init__(self, rate, orientation):
        self.config = type("c", (), {"dropout_rate": rate, "orientation": orientation,
                                     "shared_dropout": True})()

    def __call__(self, act, mask, **kw):
        return jax.numpy.ones_like(act)


def _reference(key, use_dropout, opm_first):
    """The masks the host stack draws, from the host's own dropout code."""
    import jax.numpy as jnp

    msa = jnp.zeros((ROWS, N, C_MSA))
    pair = jnp.zeros((N, N, C_PAIR))
    safe_key = prng.SafeKey(key)
    out = []
    for _ in range(BLOCKS):
        safe_key, sub = safe_key.split()
        _unused, *sub_keys = sub.split(10)
        block = {}
        for op, idx, rate, axis in bindcraft2.DROPOUT_DRAWS:
            shift = 1 if (opm_first and op == "msa_row_attn") else 0
            act = msa if op == "msa_row_attn" else pair
            block[op] = modules.dropout_wrapper(
                _Op(rate, "per_row" if axis == 0 else "per_column"),
                act, None, safe_key=sub_keys[idx + shift], global_config=None,
                use_dropout=jnp.asarray(use_dropout))
        out.append(block)
    return out, safe_key._key


@pytest.mark.parametrize("use_dropout", [True, False])
@pytest.mark.parametrize("opm_first", [False, True])
def test_the_masks_are_the_hosts_own(use_dropout, opm_first):
    key = jax.random.PRNGKey(7)
    msa_keep, pair_keep, scales, key_out = bindcraft2.dropout_masks(
        key, BLOCKS, (ROWS, N, C_MSA), (N, N, C_PAIR), use_dropout, opm_first)
    want, want_key = _reference(key, use_dropout, opm_first)

    for b, block in enumerate(want):
        got = [np.asarray(msa_keep[b, 0]) * float(scales[0])]
        got += [np.asarray(pair_keep[b, i]) * float(scales[i + 1])
                for i in range(len(bindcraft2.DROPOUT_PAIR_OPS))]
        for (op, _, _, axis), mine in zip(bindcraft2.DROPOUT_DRAWS, got):
            # The reference's mask is the update's shape with one axis shared; tt-bio ships the
            # shared row or column alone and broadcasts it on card.
            theirs = np.asarray(block[op])
            expect = theirs[0] if axis == 0 else theirs[:, 0]
            assert mine.shape == expect.shape
            np.testing.assert_allclose(mine, expect, rtol=0, atol=0,
                                       err_msg=f"block {b}, {op}")
    np.testing.assert_array_equal(np.asarray(key_out), np.asarray(want_key))


def test_dropout_off_keeps_everything():
    """`design_dropout: false` must leave the fold exactly as it was before this landed."""
    msa_keep, pair_keep, scales, _ = bindcraft2.dropout_masks(
        jax.random.PRNGKey(3), BLOCKS, (ROWS, N, C_MSA), (N, N, C_PAIR), False, False)
    assert np.asarray(msa_keep).all() and np.asarray(pair_keep).all()
    np.testing.assert_allclose(np.asarray(scales), np.ones(len(bindcraft2.DROPOUT_DRAWS)))


class _StubTrunk:
    """`_Trunk.up` without a card: the same leading axis, nothing else."""

    @staticmethod
    def up(t):
        return t.unsqueeze(0)


def test_the_uploaded_mask_carries_the_rescale_and_pads_with_keeps():
    """What goes to the card is one shared row or column per draw, padded to the token axis."""
    import torch

    msa_keep, pair_keep, scales, _ = bindcraft2.dropout_masks(
        jax.random.PRNGKey(5), BLOCKS, (ROWS, N, C_MSA), (N, N, C_PAIR), True, False)
    n32 = 32
    blocks = bindcraft2.upload_dropout(_StubTrunk, bindcraft2.DROPOUT_DRAWS[1:], pair_keep,
                                      scales, n32, msa_keep)

    assert len(blocks) == BLOCKS
    for b, block in enumerate(blocks):
        assert set(block) == {op for op, _, _, _ in bindcraft2.DROPOUT_DRAWS}
        for op, _, _, axis in bindcraft2.DROPOUT_DRAWS:
            t = block[op]
            c = C_MSA if op == "msa_row_attn" else C_PAIR
            assert tuple(t.shape) == ((1, n32, 1, c) if axis else (1, 1, n32, c))
            flat = t.reshape(n32, c)
            drawn = (np.asarray(msa_keep[b, 0]) if op == "msa_row_attn"
                     else np.asarray(pair_keep[b, bindcraft2.DROPOUT_PAIR_OPS.index(op)]))
            scale = float(scales[0] if op == "msa_row_attn"
                          else scales[bindcraft2.DROPOUT_PAIR_OPS.index(op) + 1])
            np.testing.assert_allclose(flat[:N].numpy(), drawn * scale)
            # The pad is masked out of every update anyway, so it keeps rather than drops.
            assert torch.equal(flat[N:], torch.full((n32 - N, c), scale))


def test_a_drawn_mask_drops_at_the_configured_rate():
    """The rates are AF2's: 0.15 on the row attention, 0.25 on the four pair ops."""
    n = 256
    msa_keep, pair_keep, _, _ = bindcraft2.dropout_masks(
        jax.random.PRNGKey(11), 8, (ROWS, n, n), (n, n, n), True, False)
    assert abs(float(np.asarray(msa_keep).mean()) - 0.85) < 0.01
    for i in range(len(bindcraft2.DROPOUT_PAIR_OPS)):
        assert abs(float(np.asarray(pair_keep[:, i]).mean()) - 0.75) < 0.01


@pytest.mark.parametrize("use_dropout", [True, False])
def test_the_template_stacks_masks_are_the_hosts_own(use_dropout):
    """multimer_v3's template pair stack: per-block split, then twenty ways, multiplications first.

    That stack is on card too, so it owed the same draws. Graded the same way, against
    `dropout_wrapper` fed the keys `template_iteration_fn` and `TemplateEmbeddingIteration` hand out
    (`modules_multimer.py:650` and `:706`).
    """
    import jax.numpy as jnp

    key, c_t = jax.random.PRNGKey(13), 6
    _, pair_keep, scales, key_out = bindcraft2.stack_dropout_masks(
        key, BLOCKS, bindcraft2.TEMPLATE_KEY_FAN, bindcraft2.TEMPLATE_DROPOUT_DRAWS,
        (N, N, c_t), use_dropout)

    act = jnp.zeros((N, N, c_t))
    safe_key = prng.SafeKey(key)
    for b in range(BLOCKS):
        safe_key, sub = safe_key.split()
        _unused, *sub_keys = sub.split(20)
        for i, (op, idx, rate, axis) in enumerate(bindcraft2.TEMPLATE_DROPOUT_DRAWS):
            theirs = np.asarray(modules.dropout_wrapper(
                _Op(rate, "per_row" if axis == 0 else "per_column"), act, None,
                safe_key=sub_keys[idx], global_config=None, use_dropout=jnp.asarray(use_dropout)))
            expect = theirs[0] if axis == 0 else theirs[:, 0]
            mine = np.asarray(pair_keep[b, i]) * float(scales[i])
            np.testing.assert_allclose(mine, expect, rtol=0, atol=0, err_msg=f"block {b}, {op}")
    np.testing.assert_array_equal(np.asarray(key_out), np.asarray(safe_key._key))
