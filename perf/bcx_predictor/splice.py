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
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import torch

_NANLOG_ON = bool(os.environ.get("BCX_NANLOG"))
#: When set, the FIRST backward that returns a non-finite gradient writes its exact inputs
#: here. The NaN appears after 3 to 5 optimisation rounds and the round varies run to run,
#: so a trajectory is a 6-minute stochastic reproducer; this makes it a deterministic
#: seconds-long one that replays the single failing call.
_CAPTURE_DIR = os.environ.get("BCX_NANCAP")


BUCKET = 32


def _pad32(n: int) -> int:
    return -(-n // BUCKET) * BUCKET


def _pad_inputs(m, z, mask, pair_mask):
    """Pad the token axis up to a multiple of 32 and mask what was added.

    BindCraft 2 samples a binder length per trajectory and at `length_bucket_size` 1 the
    complex is whatever it drew: n = 186, 253, 261, 291, 295 on the five live reference
    trajectories, not one of them a multiple of 32. tt-bio's token axis wants 32, and this
    row measured 211 padded up to 224 running 3.29x FASTER on the trunk forward while doing
    strictly more arithmetic. So pad, mask the padding, and slice back -- more work, not
    less, and the masked ops that make it exact are now all in `tt_bio/af2.py`.
    """
    n = z.shape[0]
    n32 = _pad32(n)
    if n32 == n:
        return m, z, mask, pair_mask, n, n32
    pad = n32 - n
    m = torch.nn.functional.pad(m, (0, 0, 0, pad))
    z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
    mask = torch.nn.functional.pad(mask, (0, pad))
    pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
    return m, z, mask, pair_mask, n, n32


#: Per-call finiteness at the seam, filled only when BCX_NANLOG is set. The gradient goes
#: non-finite mid-trajectory (trace_exit.json) and this says which SIDE it is born on: a
#: cotangent that arrives NaN is BindCraft 2's tail, one that leaves NaN is our backward.
NANLOG: list = []


def _nan(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0 or np.isfinite(a).all():
        return None
    return {"nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
            "size": int(a.size), "absmax_finite": float(
                np.abs(a[np.isfinite(a)]).max()) if np.isfinite(a).any() else None}


_CAPTURED: list = []
_LIVE: dict[int, dict] = {}
_NEXT = [0]


class EvoformerOnDevice:
    """tt-bio's 48 Evoformer blocks as `(msa, pair) -> (msa, pair)`, differentiable."""

    def __init__(self, dev, k_evo: int = 48, checkpoint: bool = True, trace: bool = False):
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
        self._pair_mask_dev = {}
        # `trace=True` captures the taped forward and the backward once per shape and replays
        # them (`trace_wire.py`). Default OFF: it is a program change, and a program change
        # between two seeds of a matched pair makes the pair meaningless. An argument, not an
        # environment variable, so a run records it in its own stamp.
        self.wire = None
        #: Flipped per round by `perf/bcx_tracewire/round_ab.py` so both arms run in one
        #: process on the same trajectory. A backward follows the arm its own forward took,
        #: read off the token, never off this flag.
        self.trace_on = bool(trace)
        if trace:
            from trace_wire import TraceWire
            self.wire = TraceWire(dev)

    @staticmethod
    def _mask_key(mask_np):
        return tuple(np.asarray(mask_np).shape)

    @staticmethod
    def _pair_mask_key(pm):
        return (tuple(pm.shape), float(pm.sum()))

    def _mask(self, mask_np):
        """Upload per call and cache by shape: the binder length changes per trajectory."""
        key = self._mask_key(mask_np)
        got = self._mask_dev.get(key)
        if got is None:
            got = self.dev.up(torch.from_numpy(
                np.asarray(mask_np, dtype=np.float32).copy()).float())
            self._mask_dev[key] = got
        return got

    def _pair_mask(self, pm):
        """`af2_pair_masks` for this shape, cached. `(None, None)` on an all-ones mask.

        BindCraft 2 buckets the token axis to 32 and masks the pad out of `seq_mask`, so
        `masks["pair"]` is all ones only when the complex happens to land on a multiple of
        32 -- 77 + 115 = 192 does, and PD-L1's two single-chain folds (115 and 77) do not.
        Dropping it read pLDDT 0.534 against BindCraft 2's own 0.950 on the 115-residue
        target alone, a natural protein with a known fold, and 0.9541 with it
        (`perf/bcx_mono/masked_fold.json`).
        """
        from tt_bio.af2 import af2_pair_masks
        key = self._pair_mask_key(pm)
        got = self._pair_mask_dev.get(key)
        if got is None:
            got = af2_pair_masks(pm, self.dev.device)
            self._pair_mask_dev[key] = got
        return got

    def _primal(self, msa_np, pair_np, mask_np, pair_mask_np):
        """No tape. `predict` is forward-only and MPNN_stage.py:125 calls it once per
        validation model, so a primal that banks a tape is an out-of-memory bug."""
        dev = self.dev
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        mk = torch.from_numpy(np.asarray(mask_np).copy()).float()
        pmk = torch.from_numpy(np.asarray(pair_mask_np).copy()).float()
        m, z, mk, pmk, n, n32 = _pad_inputs(m, z, mk, pmk)
        mo, zo = dev.stack(dev.up(m), dev.up(z), 0, self.k_evo, ckpt=False,
                           msa_mask=self._mask(mk.numpy()),
                           pair_masks=self._pair_mask(pmk))
        dev.sync()
        out_m = dev.down(mo, tuple(m.shape))[:, :n]
        out_z = dev.down(zo, tuple(z.shape))[:n, :n]
        self.calls["primal"] += 1
        return out_m.numpy(), out_z.numpy()

    def _taped(self, msa_np, pair_np, mask_np, pair_mask_np):
        dev = self.dev
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        mk = torch.from_numpy(np.asarray(mask_np).copy()).float()
        pmk = torch.from_numpy(np.asarray(pair_mask_np).copy()).float()
        m, z, mk, pmk, n, n32 = _pad_inputs(m, z, mk, pmk)
        if self.wire is not None and self.trace_on:
            return self._taped_traced(m, z, mk, pmk, n)
        ml, zl = dev.leaf(m), dev.leaf(z)
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, 0, self.k_evo, ckpt=self.checkpoint,
                               msa_mask=self._mask(mk.numpy()),
                               pair_masks=self._pair_mask(pmk))
        dev.sync()
        # recycled_alphafold_outputs stop_gradients every recycle but the last and JAX
        # routes all of them through fwd, so superseded tapes are dropped here: at
        # bcx-ckpt's 5.33 GB an Evoformer block, keeping them is fatal over 125 steps.
        for stale in list(_LIVE):
            _LIVE.pop(stale, None)
        dev.ag.release_pins()
        token = _NEXT[0]; _NEXT[0] += 1
        _LIVE[token] = {"roots": [mo, zo], "leaves": [ml, zl],
                        "shapes": [tuple(m.shape), tuple(z.shape)], "n": n,
                        "msa_in": m.numpy() if _CAPTURE_DIR else None,
                        "pair_in": z.numpy() if _CAPTURE_DIR else None,
                        "mask_in": mk.numpy() if _CAPTURE_DIR else None}
        self.calls["taped"] += 1
        if _NANLOG_ON:
            NANLOG.append({"op": "taped", "call": self.calls["taped"],
                           "in_msa": _nan(msa_np), "in_pair": _nan(pair_np),
                           "out_msa": _nan(dev.down(mo.value, tuple(m.shape)).numpy()),
                           "out_pair": _nan(dev.down(zo.value, tuple(z.shape)).numpy())})
        return (dev.down(mo.value, tuple(m.shape))[:, :n].numpy(),
                dev.down(zo.value, tuple(z.shape))[:n, :n].numpy(), np.int32(token))

    def _taped_traced(self, m, z, mk, pmk, n):
        """The same taped forward, replayed from a capture (`trace_wire.py`).

        The tape is the capture's, not this call's, so nothing is dropped or released here:
        the backward trace reads the addresses the forward trace writes, and both live until
        the shape changes. BindCraft 2 runs two forwards per round -- the stop-gradient recycle
        and the differentiated one -- and both replay the same trace into the same buffers, so
        the backward sees the second one's residuals, which is the pair eager produces too.
        """
        dev = self.dev
        msa_mask, pair_masks = self._mask(mk.numpy()), self._pair_mask(pmk)
        key = (tuple(m.shape), tuple(z.shape), self._mask_key(mk.numpy()),
               self._pair_mask_key(pmk), self.k_evo, self.checkpoint)

        def body(ml, zl):
            return dev.stack(ml, zl, 0, self.k_evo, ckpt=self.checkpoint,
                             msa_mask=msa_mask, pair_masks=pair_masks)

        out_m, out_z = self.wire.forward(key, [m, z], body, [tuple(m.shape), tuple(z.shape)])
        for stale in list(_LIVE):
            _LIVE.pop(stale, None)
        token = _NEXT[0]; _NEXT[0] += 1
        _LIVE[token] = {"key": key, "shapes": [tuple(m.shape), tuple(z.shape)], "n": n,
                        "msa_in": None, "pair_in": None, "mask_in": None}
        self.calls["taped"] += 1
        return (out_m[:, :n].numpy(), out_z[:n, :n].numpy(), np.int32(token))

    def _backward(self, token, g_msa_np, g_pair_np):
        entry = _LIVE.pop(int(token), None)
        if entry is None:
            raise RuntimeError(f"no live tape for token {int(token)}")
        dev = self.dev
        m_shape, z_shape = entry["shapes"]
        n = entry["n"]
        gm = torch.zeros(m_shape)
        gz = torch.zeros(z_shape)
        gm[:, :n] = torch.from_numpy(np.asarray(g_msa_np).copy()).float()
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        if entry.get("key") is not None:
            g_m, g_z = self.wire.backward(entry["key"], [gm, gz], [m_shape, z_shape])
            out = (g_m[:, :n].numpy(), g_z[:n, :n].numpy())
        else:
            mo, zo = entry["roots"]; ml, zl = entry["leaves"]
            dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
            dev.sync()
            out = (dev.grad(ml, m_shape)[:, :n].numpy(), dev.grad(zl, z_shape)[:n, :n].numpy())
            dev.ag.release_pins()
        self.calls["backward"] += 1
        if _CAPTURE_DIR and not _CAPTURED and not (
                np.isfinite(out[0]).all() and np.isfinite(out[1]).all()):
            _CAPTURED.append(1)
            d = pathlib.Path(_CAPTURE_DIR); d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                d / "nan_backward_inputs.npz",
                msa_leaf=entry["msa_in"], pair_leaf=entry["pair_in"],
                mask=entry["mask_in"], cot_msa=np.asarray(g_msa_np),
                cot_pair=np.asarray(g_pair_np),
                grad_msa_out=out[0], grad_pair_out=out[1],
                backward_call=np.int64(self.calls["backward"]), n=np.int64(n))
            print(f"captured the failing backward (call {self.calls['backward']}) to "
                  f"{d / 'nan_backward_inputs.npz'}", flush=True)
        if _NANLOG_ON:
            NANLOG.append({"op": "backward", "call": self.calls["backward"],
                           "cotangent_msa_in": _nan(g_msa_np),
                           "cotangent_pair_in": _nan(g_pair_np),
                           "grad_msa_out": _nan(out[0]), "grad_pair_out": _nan(out[1])})
        return out

    @staticmethod
    def live_tapes() -> int:
        return len(_LIVE)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self):
        """`(msa, pair, msa_mask, pair_mask) -> (msa, pair)`, differentiable in the first two.

        Both masks are ARGUMENTS, not captured host arrays: BindCraft 2 draws a new binder
        length per trajectory, so their shapes change under us, and `find_evoformer_masks`
        recovers the live tracers from the closure the stack replacement displaces. Neither
        is differentiable, so the backward returns zeros for both.

        The callback works in float32 while the model around it runs bfloat16, and
        `recycled_alphafold_outputs` carries `pair` through a `while_loop` whose carry types
        must match exactly, so every value handed back takes the dtype it arrived with.
        """
        def f32(msa, pair):
            return (jax.ShapeDtypeStruct(msa.shape, jnp.float32),
                    jax.ShapeDtypeStruct(pair.shape, jnp.float32))

        @jax.custom_vjp
        def stack(msa, pair, mask, pair_mask):
            m, z = jax.pure_callback(self._primal, f32(msa, pair),
                                     msa.astype(jnp.float32), pair.astype(jnp.float32),
                                     mask.astype(jnp.float32),
                                     pair_mask.astype(jnp.float32))
            return m.astype(msa.dtype), z.astype(pair.dtype)

        def fwd(msa, pair, mask, pair_mask):
            m, z, token = jax.pure_callback(
                self._taped, f32(msa, pair) + (jax.ShapeDtypeStruct((), jnp.int32),),
                msa.astype(jnp.float32), pair.astype(jnp.float32), mask.astype(jnp.float32),
                pair_mask.astype(jnp.float32))
            return (m.astype(msa.dtype), z.astype(pair.dtype)), (token, mask, pair_mask)

        def bwd(res, cts):
            token, mask, pair_mask = res
            g_msa, g_pair = cts
            gm, gz = jax.pure_callback(
                self._backward,
                (jax.ShapeDtypeStruct(g_msa.shape, jnp.float32),
                 jax.ShapeDtypeStruct(g_pair.shape, jnp.float32)),
                token, g_msa.astype(jnp.float32), g_pair.astype(jnp.float32))
            return (gm.astype(g_msa.dtype), gz.astype(g_pair.dtype),
                    jnp.zeros_like(mask), jnp.zeros_like(pair_mask))

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
                    msa, pair = device_stack(act["msa"], act["pair"],
                                             masks["msa"], masks["pair"])
                    return {**act, "msa": msa, "pair": pair}, safe_key
                return on_device
            return made(fn)
        return choose

    modules.layer_stack.layer_stack = factory
    try:
        yield swapped
    finally:
        modules.layer_stack.layer_stack = real
