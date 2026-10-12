"""`_untaped_recycles`: the discarded recycles take the plain forward, the gradient is unchanged."""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from tt_bio.bindcraft2 import _untaped_recycles


def _bc2_loop(runner, params, key, model_inputs, num_recycle):
    # `bindcraft/af2.py:136`, the loop the wrapper relies on.
    prev = model_inputs["prev"]
    for _ in range(num_recycle):
        prev = jax.lax.stop_gradient(runner.apply(params, key, {**model_inputs, "prev": prev})["prev"])
    return runner.apply(params, key, {**model_inputs, "prev": prev})


def _runner(calls):
    @jax.custom_vjp
    def stack(x):
        jax.debug.callback(lambda: calls.append("primal"))
        return jnp.sin(x)

    def fwd(x):
        jax.debug.callback(lambda: calls.append("taped"))
        return jnp.sin(x), x

    stack.defvjp(fwd, lambda x, g: (g * jnp.cos(x),))

    class Runner:
        def apply(self, params, key, x):
            out = stack(x["seq"] * params + x["prev"])
            return {"prev": out, "loss": (out ** 2).sum()}
    return Runner()


@pytest.mark.parametrize("num_recycle", [1, 3])
def test_only_the_last_recycle_is_taped_and_the_gradient_is_unchanged(num_recycle):
    seq = jnp.array([0.3, -0.7, 1.1])
    grads, taped = [], []
    for loop in (_bc2_loop, _untaped_recycles(_bc2_loop)):
        calls = []
        runner = _runner(calls)
        g = jax.grad(lambda s: loop(runner, 2.0, None, {"seq": s, "prev": jnp.zeros(3)},
                                    num_recycle)["loss"])(seq)
        jax.effects_barrier()
        grads.append(g)
        taped.append(calls.count("taped"))
    assert taped == [num_recycle + 1, 1]
    assert jnp.array_equal(grads[0], grads[1])
