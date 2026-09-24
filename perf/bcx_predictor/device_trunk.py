#!/usr/bin/env python3
"""tt-bio's AF2 trunk as a JAX value with a gradient: the primitive the predictor needs.

BindCraft 2 differentiates one jitted program w.r.t. the sequences (`af2.py:328-378`), so a
non-JAX trunk has to appear to JAX as a differentiable function. This is that function: a
`jax.custom_vjp` whose forward runs `(msa, pair)` through tt-bio's extra-MSA and Evoformer
blocks under `taped_ttnn.tape()` and returns `(single, pair)`, and whose backward seeds that
tape with the cotangents and reads `(d_msa, d_pair)` back off the leaves.

`bcx-e2e` proved that this is a genuine graph cut: `single = Linear(msa_final[0])` and
`pair_final` never reads `single`, so neither root descends from the other and the one-sided
arms sum back to the full gradient to 0.25%. The `single_activations` projection sits on the
DEVICE side, which is what makes the cut `(single, pair)` rather than `(msa, pair)` -- seeding
at `(msa, pair)` hands the MSA track exactly zero and that bug arrives disguised as a speedup.

THE TAPE IS THE RESIDUAL, and JAX residuals must be JAX types, so the tape is held here in
`_LIVE` and the residual is an int32 token into it. `custom_vjp` guarantees one backward per
forward; `release()` drops the entry either way so a refused step cannot leak a tape.
"""
import numpy as np
import jax
import jax.numpy as jnp
import torch

_LIVE: dict[int, dict] = {}
_NEXT = [0]


class DeviceTrunk:
    """One device context, reused across steps. tt-bio refuses an unpinned open."""

    def __init__(self, dev, k_extra: int = 4, k_evo: int = 48, checkpoint: bool = True):
        self.dev, self.k_extra, self.k_evo, self.checkpoint = dev, k_extra, k_evo, checkpoint
        self.c_single = 384

    # ------------------------------------------------------------------ the two halves

    def _forward_notape(self, msa_np, pair_np):
        """The primal with no tape at all.

        `predict` is forward-only and `MPNN_stage.py:125` calls it once per validation
        model, so a primal that banks a tape is an out-of-memory bug, not a leak to tidy
        later: `bcx-ckpt` measured 5.33 GB of tape per Evoformer block. Caught by
        `live_tapes_after` reading 1 in test_device_trunk.py, which is why that counter is
        asserted rather than printed.
        """
        dev = self.dev
        m0 = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z0 = torch.from_numpy(np.asarray(pair_np).copy()).float()
        n = z0.shape[0]
        # Raw ttnn tensors, not `autograd.Tensor`: the blocks only accept the wrapper
        # inside `taped_ttnn.tape()`, which patches the ops to understand it. Outside a
        # tape this is tt-bio's ordinary inference path.
        mo, zo = dev.stack(dev.up(m0), dev.up(z0), self.k_extra, self.k_evo, ckpt=False)
        so = dev.dm.device_single(mo)
        dev.sync()
        return (dev.down(so, (n, self.c_single)).numpy(),
                dev.down(zo, (n, n, z0.shape[-1])).numpy())

    def _forward(self, msa_np, pair_np):
        dev = self.dev
        m0 = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z0 = torch.from_numpy(np.asarray(pair_np).copy()).float()
        n = z0.shape[0]
        ml, zl = dev.leaf(m0), dev.leaf(z0)
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, self.k_extra, self.k_evo, ckpt=self.checkpoint)
            so = dev.dm.device_single(mo)
        dev.sync()
        single = dev.down(so.value, (n, self.c_single))
        pair = dev.down(zo.value, (n, n, z0.shape[-1]))
        token = _NEXT[0]
        _NEXT[0] += 1
        _LIVE[token] = {"roots": [so, zo], "leaves": [ml, zl],
                        "shapes": [tuple(m0.shape), tuple(z0.shape)]}
        return single.numpy(), pair.numpy(), np.int32(token)

    def _backward(self, token, g_single_np, g_pair_np):
        entry = _LIVE.pop(int(token), None)
        if entry is None:
            raise RuntimeError(f"no live tape for token {int(token)}; a backward ran twice or "
                               "the forward was released before its gradient was taken")
        dev = self.dev
        so, zo = entry["roots"]
        ml, zl = entry["leaves"]
        gs = torch.from_numpy(np.asarray(g_single_np).copy()).float()
        gp = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        dev.ag.backward([so, zo], [dev.seed(gs, so), dev.seed(gp, zo)])
        dev.sync()
        g_msa = dev.grad(ml, entry["shapes"][0])
        g_pair = dev.grad(zl, entry["shapes"][1])
        dev.ag.release_pins()
        return g_msa.numpy(), g_pair.numpy()

    def release(self, token) -> None:
        _LIVE.pop(int(token), None)

    @staticmethod
    def live_tapes() -> int:
        return len(_LIVE)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self):
        """`(msa, pair) -> (single, pair)`, differentiable, callable under `jax.jit`."""

        @jax.custom_vjp
        def trunk(msa, pair):
            # The primal banks no tape -- see `_forward_notape`.
            return _primal_callback(self, msa, pair)

        def trunk_fwd(msa, pair):
            single, out_pair, token = _fwd_callback(self, msa, pair)
            return (single, out_pair), token

        def trunk_bwd(token, cotangents):
            g_single, g_pair = cotangents
            return _bwd_callback(self, token, g_single, g_pair)

        trunk.defvjp(trunk_fwd, trunk_bwd)
        return trunk


def _primal_callback(trunk: DeviceTrunk, msa, pair):
    n = pair.shape[0]
    shapes = (jax.ShapeDtypeStruct((n, trunk.c_single), jnp.float32),
              jax.ShapeDtypeStruct(pair.shape, jnp.float32))
    return jax.pure_callback(trunk._forward_notape, shapes,
                             msa.astype(jnp.float32), pair.astype(jnp.float32))


def _fwd_callback(trunk: DeviceTrunk, msa, pair):
    n = pair.shape[0]
    shapes = (jax.ShapeDtypeStruct((n, trunk.c_single), jnp.float32),
              jax.ShapeDtypeStruct(pair.shape, jnp.float32),
              jax.ShapeDtypeStruct((), jnp.int32))
    return jax.pure_callback(trunk._forward, shapes,
                             msa.astype(jnp.float32), pair.astype(jnp.float32))


def _bwd_callback(trunk: DeviceTrunk, token, g_single, g_pair):
    # The cotangent shapes are the leaf shapes, which the tape remembers; JAX needs them
    # declared, so they are rebuilt from the cotangents it hands back.
    n = g_pair.shape[0]
    shapes = (jax.ShapeDtypeStruct((1, n, 256), jnp.float32),
              jax.ShapeDtypeStruct(g_pair.shape, jnp.float32))
    return jax.pure_callback(trunk._backward, shapes, token,
                             g_single.astype(jnp.float32), g_pair.astype(jnp.float32))
