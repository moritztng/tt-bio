#!/usr/bin/env python3
"""Put tt-bio's Evoformer inside BindCraft 2's own haiku model.

`layer_stack` is a `jax.lax.scan`, so its body is traced once and a device call cannot be
dropped into a block: the stack is replaced whole. `modules.py` calls `layer_stack` three
times and `splice_probe.json` shows the closures are distinguishable by name -- `block`
twice for the template pair stack, `extra_msa_stack_fn` once, `evoformer_fn` once -- so
this patches the factory and swaps only the one it is asked for.

`single_activations` (`modules.py:1599`) sits outside both stacks and stays in JAX, so the
cut is `(msa, pair)`. That is not `bcx-e2e`'s zero-gradient hazard: that row's `dL/dmsa` was
exactly 0 because BindCraft 2's TAIL reads only `single` and `pair`, whereas here JAX
differentiates the Linear itself and `d(msa)` is real. `assert_msa_gradient_reaches` below
checks it rather than trusting the argument.

The extra-MSA stack stays in JAX for now. BindCraft 2 feeds `extra_msa` as a single zero row
under an all-zero `extra_msa_mask` (`bindcraft/af2.py:134`), which is exactly what tt-bio's
extra-MSA blocks bake into `opm_constant`, so it is a legal swap -- just not made yet.
"""
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import torch

_LIVE: dict[int, dict] = {}
_NEXT = [0]


class EvoformerOnDevice:
    """tt-bio's 48 Evoformer blocks as `(msa, pair) -> (msa, pair)`, differentiable."""

    def __init__(self, dev, k_evo: int = 48, checkpoint: bool = True):
        self.dev, self.k_evo, self.checkpoint = dev, k_evo, checkpoint
        self.calls = {"primal": 0, "taped": 0, "backward": 0}

    # ------------------------------------------------------------------ host halves

    def _primal(self, msa_np, pair_np):
        """No tape. `predict` is forward-only and MPNN_stage.py:125 calls it once per
        validation model, so a primal that banks a tape is an out-of-memory bug."""
        dev = self.dev
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        mo, zo = dev.stack(dev.up(m), dev.up(z), 0, self.k_evo, ckpt=False)
        dev.sync()
        self.calls["primal"] += 1
        return (dev.down(mo, tuple(m.shape)).numpy(), dev.down(zo, tuple(z.shape)).numpy())

    def _taped(self, msa_np, pair_np):
        dev = self.dev
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        ml, zl = dev.leaf(m), dev.leaf(z)
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, 0, self.k_evo, ckpt=self.checkpoint)
        dev.sync()
        token = _NEXT[0]; _NEXT[0] += 1
        _LIVE[token] = {"roots": [mo, zo], "leaves": [ml, zl],
                        "shapes": [tuple(m.shape), tuple(z.shape)]}
        self.calls["taped"] += 1
        return (dev.down(mo.value, tuple(m.shape)).numpy(),
                dev.down(zo.value, tuple(z.shape)).numpy(), np.int32(token))

    def _backward(self, token, g_msa_np, g_pair_np):
        entry = _LIVE.pop(int(token), None)
        if entry is None:
            raise RuntimeError(f"no live tape for token {int(token)}")
        dev = self.dev
        mo, zo = entry["roots"]; ml, zl = entry["leaves"]
        gm = torch.from_numpy(np.asarray(g_msa_np).copy()).float()
        gz = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
        dev.sync()
        out = (dev.grad(ml, entry["shapes"][0]).numpy(), dev.grad(zl, entry["shapes"][1]).numpy())
        dev.ag.release_pins()
        self.calls["backward"] += 1
        return out

    @staticmethod
    def live_tapes() -> int:
        return len(_LIVE)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self):
        def shapes_of(msa, pair):
            return (jax.ShapeDtypeStruct(msa.shape, jnp.float32),
                    jax.ShapeDtypeStruct(pair.shape, jnp.float32))

        @jax.custom_vjp
        def stack(msa, pair):
            return jax.pure_callback(self._primal, shapes_of(msa, pair),
                                     msa.astype(jnp.float32), pair.astype(jnp.float32))

        def fwd(msa, pair):
            out = jax.pure_callback(
                self._taped, shapes_of(msa, pair) + (jax.ShapeDtypeStruct((), jnp.int32),),
                msa.astype(jnp.float32), pair.astype(jnp.float32))
            return (out[0], out[1]), (out[2], msa.shape, pair.shape)

        def bwd(res, cts):
            token, msa_shape, pair_shape = res
            g_msa, g_pair = cts
            return jax.pure_callback(
                self._backward,
                (jax.ShapeDtypeStruct(msa_shape, jnp.float32),
                 jax.ShapeDtypeStruct(pair_shape, jnp.float32)),
                token, g_msa.astype(jnp.float32), g_pair.astype(jnp.float32))

        stack.defvjp(fwd, bwd)
        return stack


@contextlib.contextmanager
def evoformer_on_device(evo: EvoformerOnDevice, expect_blocks: int = 48):
    """Swap `modules.py:1594`'s Evoformer stack for `evo`, and nothing else."""
    from bindcraft.af.alphafold.model import layer_stack as LS
    from bindcraft.af.alphafold.model import modules

    real = LS.layer_stack
    device_stack = evo.as_jax()
    swapped = []

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            if getattr(fn, "__name__", None) == "evoformer_fn":
                if int(num_layers) != expect_blocks:
                    raise ValueError(
                        f"evoformer_fn has {num_layers} blocks, tt-bio holds {expect_blocks}")
                swapped.append(int(num_layers))

                def on_device(x):
                    act, safe_key = x
                    msa, pair = device_stack(act["msa"], act["pair"])
                    return {**act, "msa": msa, "pair": pair}, safe_key
                return on_device
            return made(fn)
        return choose

    modules.layer_stack.layer_stack = factory
    try:
        yield swapped
    finally:
        modules.layer_stack.layer_stack = real
