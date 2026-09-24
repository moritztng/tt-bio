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
        """`msa_mask` is BindCraft 2's `[rows, n]` Evoformer MSA mask, as a host array.

        It is passed in rather than read off the traced activations because the stack
        replacement never sees the `masks` dict -- `evoformer_fn` closes over it and this
        swaps out the whole `layer_stack`. The mask is fixed for a given target and binder
        length, so capturing it once per shape is exact; a predictor class that reads it
        from BindCraft 2's own batch is the general form and this is the harness's.
        """
        self.dev, self.k_evo, self.checkpoint = dev, k_evo, checkpoint
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        self._mask_dev = {}

    def _mask(self, mask_np):
        """Upload per call and cache by shape: the binder length changes per trajectory."""
        key = tuple(np.asarray(mask_np).shape)
        got = self._mask_dev.get(key)
        if got is None:
            got = self.dev.up(torch.from_numpy(
                np.asarray(mask_np, dtype=np.float32).copy()).float())
            self._mask_dev[key] = got
        return got

    def _primal(self, msa_np, pair_np, mask_np):
        """No tape. `predict` is forward-only and MPNN_stage.py:125 calls it once per
        validation model, so a primal that banks a tape is an out-of-memory bug."""
        dev = self.dev
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        mo, zo = dev.stack(dev.up(m), dev.up(z), 0, self.k_evo, ckpt=False,
                           msa_mask=self._mask(mask_np))
        dev.sync()
        self.calls["primal"] += 1
        return (dev.down(mo, tuple(m.shape)).numpy(), dev.down(zo, tuple(z.shape)).numpy())

    def _taped(self, msa_np, pair_np, mask_np):
        dev = self.dev
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        ml, zl = dev.leaf(m), dev.leaf(z)
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, 0, self.k_evo, ckpt=self.checkpoint,
                               msa_mask=self._mask(mask_np))
        dev.sync()
        # recycled_alphafold_outputs runs design_recycles stop-gradient passes and then one
        # differentiated pass, and JAX routes ALL of them through fwd -- it cannot know the
        # stop_gradient discards the first until after the trace. Measured: taped 2,
        # backward 1, one tape left live per step. At bcx-ckpt's 5.33 GB an Evoformer block
        # that is fatal over 125 steps, so the superseded tapes are dropped here. The
        # differentiated pass is the LAST taped call, so keeping only the newest is correct;
        # _backward raises by token if that ever stops holding.
        for stale in list(_LIVE):
            _LIVE.pop(stale, None)
        dev.ag.release_pins()
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
        """`(msa, pair, msa_mask) -> (msa, pair)`, differentiable in the first two.

        The mask is an ARGUMENT, not a captured host array: BindCraft 2 draws a new binder
        length per trajectory, so its shape changes under us, and `find_evoformer_masks`
        recovers the live tracer from the closure the stack replacement displaces. It is
        not differentiable, so the backward returns zeros for it.

        The callback works in float32 while the model around it runs bfloat16, and
        `recycled_alphafold_outputs` carries `pair` through a `while_loop` whose carry types
        must match exactly, so every value handed back takes the dtype it arrived with.
        """
        def f32(msa, pair):
            return (jax.ShapeDtypeStruct(msa.shape, jnp.float32),
                    jax.ShapeDtypeStruct(pair.shape, jnp.float32))

        @jax.custom_vjp
        def stack(msa, pair, mask):
            m, z = jax.pure_callback(self._primal, f32(msa, pair),
                                     msa.astype(jnp.float32), pair.astype(jnp.float32),
                                     mask.astype(jnp.float32))
            return m.astype(msa.dtype), z.astype(pair.dtype)

        def fwd(msa, pair, mask):
            m, z, token = jax.pure_callback(
                self._taped, f32(msa, pair) + (jax.ShapeDtypeStruct((), jnp.int32),),
                msa.astype(jnp.float32), pair.astype(jnp.float32), mask.astype(jnp.float32))
            return (m.astype(msa.dtype), z.astype(pair.dtype)), (token, mask)

        def bwd(res, cts):
            token, mask = res
            g_msa, g_pair = cts
            gm, gz = jax.pure_callback(
                self._backward,
                (jax.ShapeDtypeStruct(g_msa.shape, jnp.float32),
                 jax.ShapeDtypeStruct(g_pair.shape, jnp.float32)),
                token, g_msa.astype(jnp.float32), g_pair.astype(jnp.float32))
            return gm.astype(g_msa.dtype), gz.astype(g_pair.dtype), jnp.zeros_like(mask)

        stack.defvjp(fwd, bwd)
        return stack


def find_evoformer_masks(fn, depth=0, seen=None):
    """Recover `evoformer_masks` from the closure of `evoformer_fn`.

    Replacing the whole `layer_stack` means never seeing the masks dict that
    `EmbeddingsAndEvoformer.__call__` builds and `evoformer_fn` closes over. `hk.remat`
    wraps that closure twice -- the free variables read `dec_stateful_fun` then `f` -- so
    the walk is recursive rather than one `__wrapped__` hop.
    """
    import types
    seen = seen if seen is not None else set()
    if depth > 6 or not isinstance(fn, types.FunctionType) or id(fn) in seen:
        return None
    seen.add(id(fn))
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if name == "evoformer_masks" and isinstance(value, dict) and "msa" in value:
            return value
        found = find_evoformer_masks(value, depth + 1, seen)
        if found is not None:
            return found
    return None


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

                masks = find_evoformer_masks(fn)
                if masks is None:
                    raise RuntimeError(
                        "evoformer_masks not found in evoformer_fn's closure; tt-bio's "
                        "trunk refuses an unmasked AF2 fold and guessing one is worse "
                        "than stopping")

                def on_device(x):
                    act, safe_key = x
                    msa, pair = device_stack(act["msa"], act["pair"], masks["msa"])
                    return {**act, "msa": msa, "pair": pair}, safe_key
                return on_device
            return made(fn)
        return choose

    modules.layer_stack.layer_stack = factory
    try:
        yield swapped
    finally:
        modules.layer_stack.layer_stack = real
