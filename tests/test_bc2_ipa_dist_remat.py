"""`bindcraft2.ipa_dist_remat`: as close to float64 as the original, fewer saved residuals.

Not bit-exact: with the differences no longer materialised XLA fuses the subtraction into the
reduction and rounds the logits differently (loss rel 8e-7 at 37 tokens), so each leaf is
judged against a float64 run of the original, as `test_bc2_ipa_einsum.py` does.

Card-free: runs BindCraft 2's own `InvariantPointAttention` on the host CPU, inside an
`hk.scan` over layers as the structure module runs it, because the scan's stacked residuals
are what the rewrite removes.
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

N, LAYERS = 37, 3


def _model():
    cfg = ml_collections.ConfigDict(dict(num_head=12, num_scalar_qk=16, num_point_qk=4,
                                         num_scalar_v=16, num_point_v=8, num_channel=384))
    gc = ml_collections.ConfigDict(dict(zero_init=False, bfloat16=False))

    def f(s, z, mask, q, t):
        rot = geometry.Rot3Array.from_quaternion(q[:, 0], q[:, 1], q[:, 2], q[:, 3],
                                                 normalize=True)
        rigid = geometry.Rigid3Array(rot, geometry.Vec3Array(t[:, 0], t[:, 1], t[:, 2]))
        ipa = fm.InvariantPointAttention(cfg, gc)

        def layer(s, _):
            s = s + ipa(s, z, mask, rigid)
            return s / (1. + jnp.sqrt(jnp.mean(s * s))), None

        return hk.scan(layer, s, None, length=LAYERS)[0]

    return hk.without_apply_rng(hk.transform(f))


def _inputs(dtype=jnp.float32):
    rng = np.random.default_rng(0)
    mask = np.ones((N, 1))
    mask[-5:] = 0
    return [jnp.asarray(x, dtype) for x in (
        rng.standard_normal((N, 384)), rng.standard_normal((N, N, 128)), mask,
        rng.standard_normal((N, 4)), 10 * rng.standard_normal((N, 3)))]


def _loss(params, args):
    tf = _model()
    w = jnp.asarray(np.random.default_rng(1).standard_normal((N, 384)), args[0].dtype)
    return lambda p, s, z: jnp.sum(tf.apply(p, s, z, *args[2:]) * w)


def _value_and_grad(params, args):
    out = jax.jit(jax.value_and_grad(_loss(params, args), argnums=(0, 1, 2)))(
        params, args[0], args[1])
    return [np.asarray(x, np.float64) for x in jax.tree_util.tree_leaves(out)]


def _residual_bytes(params, args):
    """Bytes the linearised forward keeps for the backward."""
    _, vjp = jax.vjp(_loss(params, args), params, args[0], args[1])
    return sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(vjp)
               if hasattr(x, "dtype"))


@pytest.mark.parametrize("einsum", [False, True])
def test_remat_is_as_close_to_float64_and_keeps_less(einsum, monkeypatch):
    args = _inputs()
    params = _model().init(jax.random.PRNGKey(0), *args)
    for cls in (geometry.Vec3Array, geometry.Rot3Array, geometry.Rigid3Array):
        monkeypatch.setattr(cls, "__post_init__", lambda self: None)
    with jax.enable_x64(True):
        to64 = functools.partial(jax.tree_util.tree_map, lambda x: jnp.asarray(x, jnp.float64))
        exact = _value_and_grad(to64(params), _inputs(jnp.float64))
    monkeypatch.undo()
    with bindcraft2.ipa_rewrites(einsum=einsum):
        ref = _value_and_grad(params, args)
        ref_bytes = _residual_bytes(params, args)
    original = fm.InvariantPointAttention
    with bindcraft2.ipa_rewrites(einsum=einsum, dist_remat=True) as on:
        assert on is True and fm.InvariantPointAttention is not original
        assert jax.tree_util.tree_structure(_model().init(jax.random.PRNGKey(0), *args)) == \
            jax.tree_util.tree_structure(params)
        got = _value_and_grad(params, args)
        got_bytes = _residual_bytes(params, args)
    assert fm.InvariantPointAttention is original
    assert len(exact) == len(ref) == len(got)
    worst_ref = worst_got = 0.0
    floor = 1e-6 * max(np.linalg.norm(e) for e in exact)
    for e, a, b in zip(exact, ref, got):
        assert np.isfinite(b).all()
        scale = max(np.linalg.norm(e), 1e-30)
        err_ref, err_got = np.linalg.norm(a - e) / scale, np.linalg.norm(b - e) / scale
        if np.linalg.norm(e) > floor:   # leaves whose exact gradient is ~0 are rounding noise
            worst_ref, worst_got = max(worst_ref, err_ref), max(worst_got, err_got)
        assert err_got <= 2 * err_ref + 1e-5, (err_ref, err_got, e.shape)
    print(f"einsum={einsum} worst leaf rel L2 vs float64: original {worst_ref:.2e}, "
          f"remat {worst_got:.2e}; residuals "
          f"{ref_bytes} -> {got_bytes} bytes")
    # Three [N, N, heads, points] f32 differences a layer are no longer kept.
    dropped = LAYERS * 3 * N * N * 12 * 4 * 4
    assert ref_bytes - got_bytes >= dropped, (ref_bytes, got_bytes, dropped)


def test_disabled_is_a_no_op():
    original = fm.InvariantPointAttention
    with bindcraft2.ipa_dist_remat(False) as on:
        assert on is False and fm.InvariantPointAttention is original
