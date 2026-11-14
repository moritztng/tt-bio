"""`bindcraft2.ipa_point_einsum`: same parameters, same values to float32 rounding, restored on exit.

Card-free: runs BindCraft 2's own `InvariantPointAttention` on the host CPU.
"""

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
hk = pytest.importorskip("haiku")
fm = pytest.importorskip("bindcraft.af.alphafold.model.folding_multimer")

import jax.numpy as jnp  # noqa: E402
import ml_collections  # noqa: E402
from bindcraft.af.alphafold.model import geometry  # noqa: E402

from tt_bio import bindcraft2  # noqa: E402

N = 37


def _model():
    cfg = ml_collections.ConfigDict(dict(num_head=12, num_scalar_qk=16, num_point_qk=4,
                                         num_scalar_v=16, num_point_v=8, num_channel=384))
    gc = ml_collections.ConfigDict(dict(zero_init=False, bfloat16=False))

    def f(s, z, mask, q, t):
        rot = geometry.Rot3Array.from_quaternion(q[:, 0], q[:, 1], q[:, 2], q[:, 3],
                                                 normalize=True)
        rigid = geometry.Rigid3Array(rot, geometry.Vec3Array(t[:, 0], t[:, 1], t[:, 2]))
        return fm.InvariantPointAttention(cfg, gc)(s, z, mask, rigid)

    return hk.without_apply_rng(hk.transform(f))


def _inputs(dtype):
    rng = np.random.default_rng(0)
    mask = np.ones((N, 1))
    mask[-5:] = 0
    return [jnp.asarray(x, dtype) for x in (
        rng.standard_normal((N, 384)), rng.standard_normal((N, N, 128)), mask,
        rng.standard_normal((N, 4)), 10 * rng.standard_normal((N, 3)))]


def _value_and_grad(params, args):
    tf = _model()
    w = jnp.asarray(np.random.default_rng(1).standard_normal((N, 384)), args[0].dtype)

    def loss(p, s, z):
        return jnp.sum(tf.apply(p, s, z, *args[2:]) * w)

    out = jax.value_and_grad(loss, argnums=(0, 1, 2))(params, args[0], args[1])
    return [np.asarray(x, np.float64) for x in jax.tree_util.tree_leaves(out)]


def test_einsum_is_as_close_to_float64_as_the_point_sum(monkeypatch):
    """Judged against a float64 run of the original, leaf by leaf: the einsum may round
    differently, but it must not land further from the exact value than the f32 sum does
    (allowing for rounding noise on leaves whose gradient is itself near zero)."""
    args = _inputs(jnp.float32)
    params = _model().init(jax.random.PRNGKey(0), *args)
    ref32 = _value_and_grad(params, args)
    # BindCraft 2's geometry dataclasses assert float32 on construction; the reference run
    # needs float64 through them.
    for cls in (geometry.Vec3Array, geometry.Rot3Array, geometry.Rigid3Array):
        monkeypatch.setattr(cls, "__post_init__", lambda self: None)
    with jax.enable_x64(True):
        to64 = functools.partial(jax.tree_util.tree_map, lambda x: jnp.asarray(x, jnp.float64))
        exact = _value_and_grad(to64(params), _inputs(jnp.float64))
    monkeypatch.undo()
    original = fm.InvariantPointAttention
    with bindcraft2.ipa_point_einsum(True):
        assert fm.InvariantPointAttention is not original
        assert jax.tree_util.tree_structure(_model().init(jax.random.PRNGKey(0), *args)) == \
            jax.tree_util.tree_structure(params)
        got = _value_and_grad(params, args)
    assert fm.InvariantPointAttention is original
    for e, a, b in zip(exact, ref32, got):
        assert np.isfinite(b).all()
        scale = max(np.linalg.norm(e), 1e-30)
        err_sum, err_ein = np.linalg.norm(a - e) / scale, np.linalg.norm(b - e) / scale
        assert err_ein <= 2 * err_sum + 1e-5, (err_sum, err_ein, e.shape)


def test_disabled_is_a_no_op():
    original = fm.InvariantPointAttention
    with bindcraft2.ipa_point_einsum(False) as on:
        assert on is False and fm.InvariantPointAttention is original
