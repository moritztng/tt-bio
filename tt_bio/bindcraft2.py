"""Run BindCraft 2's binder design loop with tt-bio's AlphaFold 2 Evoformer on a Tenstorrent card.

BindCraft 2 (https://github.com/PacesaLab/BindCraft2) designs binders by differentiating
AlphaFold 2 through a sequence. It takes its predictor as a parameter everywhere and constructs
one in a single place, so a backend is a class rather than a fork of the design loop. This module
is that class and the device trunk behind it::

    import bindcraft.campaign as campaign
    from tt_bio import bindcraft2

    with bindcraft2.campaign_predictor(card=0):
        campaign.run_campaign(settings, project, af2_weights=params, mpnn_weights=mpnn)

Everything between BindCraft 2's protein states and the trunk stays BindCraft 2's own code:
padding, templates, the input features, the structure module, the Kabsch alignment, the
confidence heads, the filters, the ranking and the step budget. Only AlphaFold 2's 48 Evoformer
blocks move to the card, which is where every O(L^3) op in the trunk lives.

``trunk="jax"`` is the control arm. Same class, same call path, same bucket rounding, BindCraft 2's
own trunk, so a device result is read against that rather than against a differently shaped
program.

A checkpoint the card does not hold folds on that same host trunk instead of stopping the
campaign, and the validation ensemble folds there by default: it is the instrument that decides
whether a design is accepted, and running it on card would put device numerics inside the
measurement that grades the device.

tt-bio neither ships BindCraft 2 nor serves it. Install it yourself; this module only binds to it
if it is importable.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import threading
from collections.abc import Mapping
from typing import Callable, Iterator

import numpy as np
import torch

#: tt-bio's token axis buckets to 32, and rounding a design UP is faster than running it ragged:
#: the PD-L1 complex at 211 tokens costs 4.504 s on the trunk forward and the same design padded
#: to 224 costs 1.369 s. The splice pads, masks what it added and slices the result back.
TOKEN_BUCKET = 32

#: AlphaFold 2's Evoformer depth. The splice refuses a stack of any other length rather than
#: running a partial trunk.
EVOFORMER_BLOCKS = 48


def _pad32(n: int) -> int:
    return -(-n // TOKEN_BUCKET) * TOKEN_BUCKET


def pin_card(card: int | str) -> None:
    """Pin this process to one physical card.

    ttnn reads ``TT_VISIBLE_DEVICES`` when it is imported and brings up every card it names, so
    the choice has to be made before that import and cannot be changed afterwards. Calling this
    once ttnn is loaded raises rather than pretending the argument was honoured.
    """
    want = str(card)
    have = os.environ.get("TT_VISIBLE_DEVICES")
    if have == want:
        return
    if "ttnn" in sys.modules:
        raise RuntimeError(
            f"ttnn is already imported with TT_VISIBLE_DEVICES={have!r}, so card={card} cannot "
            f"be honoured. Pin the card before importing ttnn, or start the process with "
            f"TT_VISIBLE_DEVICES={card} in its environment.")
    os.environ["TT_VISIBLE_DEVICES"] = want


# ----------------------------------------------------------------- the checkpoints, on card


def _is_multimer(path: pathlib.Path) -> bool:
    """Whether an AlphaFold 2 npz is a `multimer_v3` checkpoint, read off the file itself.

    The two families need a different weight remap and a different `AF2Model`, and loading one
    as the other raises a `KeyError` from inside the remap rather than returning a wrong model.
    The tell is the relative encoding: multimer_v3 embeds residue offsets through
    `~_relative_encoding/position_activations`, the monomer through `pair_activiations`
    (AlphaFold's own spelling). Reading the array names rather than the file name keeps a
    renamed or re-exported checkpoint from being loaded as the wrong family.

    The shipped `examples/pdl1.json` designs on all five `multimer_v3` checkpoints, so for
    BindCraft 2 this is the common case rather than the exotic one.
    """
    with np.load(path, allow_pickle=False) as npz:
        return any("~_relative_encoding" in name for name in npz.files)


class _Trunk:
    """One AlphaFold 2 checkpoint's Evoformer blocks on the card, under tt-bio's tape."""

    def __init__(self, path: pathlib.Path):
        import ttnn

        from tt_bio import autograd, taped_ttnn
        from tt_bio.af2 import load_af2_device_model
        from tt_bio.af2_weights import load_af2_state_dict

        self.ttnn, self.ag, self.taped = ttnn, autograd, taped_ttnn
        self.multimer = _is_multimer(path)
        self.model = load_af2_device_model(load_af2_state_dict(str(path),
                                                               multimer=self.multimer),
                                           template=False, multimer=self.multimer,
                                           trunk_dtype=torch.bfloat16)
        self.device = self.model._device
        self.blocks = len(self.model.device_evoformer)
        self.extra_blocks = len(self.model.device_extra_msa)

    def up(self, t: torch.Tensor):
        return self.ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),
                                    layout=self.ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=self.ttnn.bfloat16)

    def down(self, t, shape) -> torch.Tensor:
        x = torch.Tensor(self.ttnn.to_torch(t)).float()
        while x.dim() > len(shape) and x.shape[0] == 1:
            x = x.squeeze(0)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"the trunk returned {tuple(x.shape)}, expected {tuple(shape)}")
        return x

    def leaf(self, t: torch.Tensor):
        return self.ag.Tensor(self.up(t), requires_grad=True)

    def sync(self) -> None:
        self.ttnn.synchronize_device(self.device)

    def evoformer(self, m, z, msa_mask, pair_masks, recompute: bool):
        for block in self.model.device_evoformer:
            if recompute:
                m, z = self.ag.checkpoint(
                    lambda a, b, blk=block: blk(a, b, msa_mask, *pair_masks), m, z)
            else:
                m, z = block(m, z, msa_mask, *pair_masks)
        return m, z

    def extra_msa(self, z, pair_masks, recompute: bool):
        """The extra-MSA stack's four pair blocks, `pair -> pair`, left on card throughout.

        `AF2DeviceModel.extra_msa_stack` is this same loop with the dead MSA track written for
        its taps and the pair brought back to host at the end. Here the pair stays on card and
        under the tape, which is what makes it differentiable.

        The MSA track is not computed and none is owed. BindCraft 2 feeds `extra_msa` as a
        single zero row under an all-zero `extra_msa_mask` (`bindcraft/af2.py:134`), so the MSA
        track reaches `pair` only through an outer product mean that collapses to
        `proj_o.bias / eps` -- which is exactly the `opm_constant` injected here.
        """
        model = self.model

        # Built per execution, never captured. `_residual` DEALLOCATES its `update` operand
        # (`tt_bio/af2.py::_residual`), and under `recompute` the checkpoint runs this same
        # closure a second time in the backward, so a constant hoisted out of the loop body is
        # a freed buffer by then and `ttnn.typecast` raises "Buffer is not allocated". The
        # Evoformer loop above has no such operand, which is why only this stack broke.
        def const(index):
            return model._up(model.opm_constant[index].reshape(1, 1, -1))

        for index, block in enumerate(model.device_extra_msa):
            if recompute:
                z = self.ag.checkpoint(
                    lambda t, blk=block, i=index: blk(blk._residual(t, const(i)), *pair_masks), z)
            else:
                z = block(block._residual(z, const(index)), *pair_masks)
        return z

    def seed(self, t: torch.Tensor, like):
        """A cotangent in the root's own device shape."""
        shape = [int(d) for d in like.value.shape]
        return self.ttnn.from_torch(t.detach().reshape(shape).to(torch.bfloat16),
                                    layout=self.ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=self.ttnn.bfloat16)

    def grad(self, leaf, shape) -> torch.Tensor:
        return self.down(leaf.grad, shape) if leaf.grad is not None else torch.zeros(shape)


class TrunkPool:
    """The AlphaFold 2 checkpoints a campaign may draw from, on card, selected by name.

    BindCraft 2 samples one design model per gradient step, so a device trunk pinned to one
    checkpoint cannot run a multi-model campaign. Checkpoints load lazily, on the first step that
    reaches one, and the pool keeps what it loads.

    ``resident`` caps how many stay on card and evicts least-recently-used. Five AF2 trunks is
    about 910 MB of weights, and holding all five brought a backward-pass allocator refusal
    forward at n=288 that ``resident=1`` ran past, so cap it if a long run dies in the allocator.
    Eviction drops the only Python reference and relies on ttnn freeing the weight buffers when
    it is collected; tt-bio has no model-level deallocate, and that release has not been read off
    the allocator directly.

    ``source`` is a directory of ``params_<name>.npz`` (the layout ``tt-bio weights --download
    af2ig`` writes, and the one BindCraft 2's own ``data_dir`` uses), a mapping of model name to
    parameters file, a single parameters file to use for every name, or None for tt-bio's own
    weights cache. A checkpoint the source cannot supply is recorded in ``absent`` rather than
    refused, and the predictor folds it on BindCraft 2's own JAX trunk.
    """

    def __init__(self, source=None, *, resident: int | None = None):
        self._source = source
        self.resident = int(resident) if resident else None
        self.paths: dict[str, pathlib.Path] = {}
        self.absent: dict[str, str] = {}
        self.selections: dict[str, int] = {}
        self._trunks: dict[str, _Trunk] = {}
        self._order: list[str] = []
        # BindCraft 2 drives this pool from TWO threads: `campaign.py:182` starts
        # `compile_next_length_bucket`, a daemon thread that calls `sequence_gradients(...,
        # compile_only=True)` on the same model while `run_trajectory` folds on the main thread.
        # So the trunk cache and the selection are shared state and move under a lock, and the
        # load is deferred to the thread that folds.
        self._lock = threading.RLock()
        self._current: str | None = None
        if isinstance(source, Mapping):
            self.require(source)

    # ------------------------------------------------------------------ the checkpoint set

    def require(self, names) -> None:
        """Resolve every name in `names` against the source; record what does not resolve.

        Called with the pool BindCraft 2 actually asks for. A campaign builds a design model and
        a validation model separately and they draw from different pools, so the set grows.

        A name the source cannot supply lands in `absent` rather than raising. BindCraft 2 holds
        validation out of design on purpose, so the shipped `examples/pdl1.json` designs on all
        five multimer checkpoints and validates on the monomer pool, and refusing there is fatal
        to a campaign that has already done all of its work. What the card does not hold folds on
        BindCraft 2's own JAX trunk instead; the predictor decides that and says so.
        """
        pairs = names.items() if isinstance(names, Mapping) else ((n, None) for n in names)
        for name, given in pairs:
            if name in self.paths or name in self.absent:
                continue
            try:
                path = pathlib.Path(os.path.expanduser(str(
                    given if given is not None else self._path_for(name))))
            except (KeyError, FileNotFoundError) as why:
                self.absent[name] = str(why)
                continue
            if path.exists():
                self.paths[name] = path
            else:
                self.absent[name] = str(path)

    def _path_for(self, name: str) -> pathlib.Path:
        source = self._source
        if isinstance(source, Mapping):
            raise KeyError(f"{name!r} is not in the checkpoint mapping {sorted(source)}; the "
                           f"campaign draws from a model this pool was never given")
        if source is None:
            from tt_bio import weights
            source = weights.resolve("af2-params")
            if source is None:
                raise FileNotFoundError(
                    "no AlphaFold 2 parameters in tt-bio's weights cache; run "
                    "`tt-bio weights --download af2ig`, or pass checkpoints=<dir>")
        path = pathlib.Path(os.path.expanduser(str(source)))
        return path / f"params_{name}.npz" if path.is_dir() else path

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, name) -> bool:
        return name in self.paths

    def holds(self, name: str) -> bool:
        """Whether the card can fold `name`, i.e. whether the weights resolved."""
        return name in self.paths

    # ------------------------------------------------------------------ selection

    def use(self, name: str) -> None:
        """Make `name` the checkpoint the trunk runs.

        An unknown name is an error, not a default. Folding a design on a checkpoint BindCraft 2
        did not pick is invisible downstream, because the shapes agree and the loss still falls.

        Selecting does NOT bring the trunk on card. The load is deferred to `current`, i.e. to the
        thread that actually folds, because BindCraft 2 selects from its compile thread too and
        that call never executes: `bindcraft/af2.py:404` returns after `lower().compile()`, before
        the `pure_callback` runs. Loading here put `_Trunk` construction, and with it a device
        bring-up and the first `tt_bio` import, on a thread racing the main fold -- which aborted
        the process on `context_id ... is out of range` from
        `Device::init_command_queue_device_with_topology`.
        """
        if name not in self.paths:
            raise KeyError(f"{name!r} is not in the trunk pool {self.names}")
        with self._lock:
            self._current = name
            self.selections[name] = self.selections.get(name, 0) + 1

    def _load(self, name: str) -> _Trunk:
        with self._lock:
            trunk = self._trunks.get(name)
            if trunk is None:
                trunk = self._trunks[name] = _Trunk(self.paths[name])
            if name in self._order:
                self._order.remove(name)
            self._order.append(name)
            while len(self._order) > (self.resident or len(self.paths)):
                self._trunks.pop(self._order.pop(0), None)
            return trunk

    @property
    def current(self) -> _Trunk:
        if self._current is None:
            if not self.paths:
                raise RuntimeError("the trunk pool is empty; nothing has asked it for a model")
            self.use(self.names[0])
        return self._load(self._current)


# ----------------------------------------------------------------- the Evoformer on card


class EvoformerOnDevice:
    """BindCraft 2's Evoformer stack, run on card, differentiable.

    ``layer_stack`` is a ``jax.lax.scan``, so its body is traced once and a device call cannot be
    dropped into one block: the stack is replaced whole. The cut is ``(msa, pair)``.
    ``single_activations`` sits outside the stack and stays in JAX, so JAX differentiates that
    projection itself and the MSA cotangent arriving here is a real one.

    ``recompute`` is gradient checkpointing on the device blocks. It trades speed for memory and
    is on by default: an uncheckpointed 48-block tape does not fit at the sizes BindCraft 2 draws.
    """

    def __init__(self, pool: TrunkPool, *, blocks: int = EVOFORMER_BLOCKS,
                 recompute: bool = True):
        self.pool = pool
        self.blocks = blocks
        self.recompute = recompute
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        #: True while `on_host` is open. Read where haiku TRACES, by the stack replacement.
        self.host_only = False
        #: Folds handed back to BindCraft 2's own trunk, per checkpoint name.
        self.host_folds: dict[str, int] = {}
        self._mask_dev: dict = {}
        self._pair_mask_dev: dict = {}
        self._live: dict[int, dict] = {}
        self._next = 0

    # ------------------------------------------------------------------ which side folds

    @contextlib.contextmanager
    def on_host(self, name: str = ""):
        """Fold inside this block on BindCraft 2's own JAX trunk, leaving the card idle.

        A checkpoint the card does not hold has to fold somewhere, and refusing is fatal rather
        than slow: on the shipped `examples/pdl1.json` all five multimer checkpoints are design
        models, so BindCraft 2 moves validation to the monomer pool (`campaign.py:77-83`) and a
        refusal reaches every acceptance after all five design stages have already passed.
        Standing the card down for those folds also makes the validation stage numerically
        BindCraft 2's own rather than an approximation of it, which is what an acceptance
        measurement wants.

        The switch is read where haiku traces, and the route is then baked into a compiled
        program that BindCraft 2 caches by model FAMILY, so it is safe only at family
        granularity. `TenstorrentAlphaFoldDesignModel._route_by_family` is what enforces that.
        """
        was = self.host_only
        self.host_only = True
        if name and name not in self.host_folds:
            print(f"[tt_bio.bindcraft2] {name} is not on card; folding it on BindCraft 2's own "
                  f"JAX trunk", flush=True)
        if name:
            self.host_folds[name] = self.host_folds.get(name, 0) + 1
        try:
            yield
        finally:
            self.host_only = was

    # ------------------------------------------------------------------ host <-> card

    @staticmethod
    def _pad(m, z, mask, pair_mask):
        """Pad the token axis to a multiple of 32 and mask what was added.

        BindCraft 2 samples a binder length per trajectory, so the complex is whatever it drew and
        is rarely a multiple of 32. Padding does strictly more arithmetic and is faster; the mask
        keeps it exact and the caller slices the result back.
        """
        n = z.shape[0]
        n32 = _pad32(n)
        if n32 == n:
            return m, z, mask, pair_mask, n
        pad = n32 - n
        m = torch.nn.functional.pad(m, (0, 0, 0, pad))
        z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
        return m, z, mask, pair_mask, n

    def _inputs(self, msa_np, pair_np, mask_np, pair_mask_np):
        as_t = lambda a: torch.from_numpy(np.asarray(a).copy()).float()  # noqa: E731
        return self._pad(as_t(msa_np), as_t(pair_np), as_t(mask_np), as_t(pair_mask_np))

    @staticmethod
    def _key(mask: torch.Tensor):
        """A cache key that is the mask itself, not its shape.

        Shape is not enough and the difference is not exotic. BindCraft 2 draws binder lengths in
        a range and the token axis rounds up, so two trajectories of different length routinely
        land on the same padded size: 200 residues and 211 both become 224, with 24 and 13 masked
        tokens respectively and the same `(1, 224)` mask shape. A shape-keyed cache hands the
        second trajectory the first one's mask and folds its padding as real residues.
        """
        return tuple(mask.shape), mask.numpy().tobytes()

    def _msa_mask(self, trunk, mask):
        """Upload per mask and cache: the binder length changes from trajectory to trajectory."""
        key = self._key(mask)
        got = self._mask_dev.get(key)
        if got is None:
            got = self._mask_dev[key] = trunk.up(mask)
        return got

    def _pair_masks(self, trunk, pair_mask):
        """`af2_pair_masks` for this mask: the pair track's multiply and its key bias.

        `(None, None)` on an all-ones mask. BindCraft 2 buckets the token axis and zeroes the pad
        out of `seq_mask`, so the pair mask is all ones only when the complex happens to land on
        a multiple of the bucket. Dropping it read 0.534 pLDDT against BindCraft 2's own 0.950 on
        a 115-residue natural target, and 0.9541 with it.
        """
        from tt_bio.af2 import af2_pair_masks
        key = self._key(pair_mask)
        got = self._pair_mask_dev.get(key)
        if got is None:
            got = self._pair_mask_dev[key] = af2_pair_masks(pair_mask, trunk.device)
        return got

    # ------------------------------------------------------------------ forward and backward

    def _trunk(self) -> _Trunk:
        trunk = self.pool.current
        if trunk.blocks != self.blocks:
            raise ValueError(f"{self.pool._current!r} holds {trunk.blocks} Evoformer blocks and "
                             f"this splice was built for {self.blocks}")
        return trunk

    def _primal(self, msa_np, pair_np, mask_np, pair_mask_np):
        """No tape. `predict` is forward-only and a validation refold calls it once per model, so
        a primal that banks a tape is an out-of-memory bug."""
        trunk = self._trunk()
        m, z, mask, pair_mask, n = self._inputs(msa_np, pair_np, mask_np, pair_mask_np)
        mo, zo = trunk.evoformer(trunk.up(m), trunk.up(z), self._msa_mask(trunk, mask),
                                 self._pair_masks(trunk, pair_mask), recompute=False)
        trunk.sync()
        self.calls["primal"] += 1
        return (trunk.down(mo, tuple(m.shape))[:, :n].numpy(),
                trunk.down(zo, tuple(z.shape))[:n, :n].numpy())

    def _taped(self, msa_np, pair_np, mask_np, pair_mask_np):
        trunk = self._trunk()
        m, z, mask, pair_mask, n = self._inputs(msa_np, pair_np, mask_np, pair_mask_np)
        ml, zl = trunk.leaf(m), trunk.leaf(z)
        with trunk.taped.tape():
            mo, zo = trunk.evoformer(ml, zl, self._msa_mask(trunk, mask),
                                     self._pair_masks(trunk, pair_mask),
                                     recompute=self.recompute)
        trunk.sync()
        # AlphaFold 2 stops the gradient on every recycle but the last and JAX still routes all
        # of them through the forward rule, so the superseded tapes are dropped here. At several
        # GB an Evoformer block, keeping them is fatal within one trajectory.
        self._live.clear()
        trunk.ag.release_pins()
        token, self._next = self._next, self._next + 1
        self._live[token] = {"roots": (mo, zo), "leaves": (ml, zl),
                             "shapes": (tuple(m.shape), tuple(z.shape)), "n": n}
        self.calls["taped"] += 1
        return (trunk.down(mo.value, tuple(m.shape))[:, :n].numpy(),
                trunk.down(zo.value, tuple(z.shape))[:n, :n].numpy(), np.int32(token))

    def _backward(self, token, g_msa_np, g_pair_np):
        entry = self._live.pop(int(token), None)
        if entry is None:
            raise RuntimeError(f"no live tape for token {int(token)}")
        trunk = self.pool.current
        mo, zo = entry["roots"]
        ml, zl = entry["leaves"]
        m_shape, z_shape = entry["shapes"]
        n = entry["n"]
        gm, gz = torch.zeros(m_shape), torch.zeros(z_shape)
        gm[:, :n] = torch.from_numpy(np.asarray(g_msa_np).copy()).float()
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        trunk.ag.backward([mo, zo], [trunk.seed(gm, mo), trunk.seed(gz, zo)])
        trunk.sync()
        out = (trunk.grad(ml, m_shape)[:, :n].numpy(), trunk.grad(zl, z_shape)[:n, :n].numpy())
        trunk.ag.release_pins()
        self.calls["backward"] += 1
        return out

    def live_tapes(self) -> int:
        return len(self._live)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self):
        """`(msa, pair, msa_mask, pair_mask) -> (msa, pair)`, differentiable in the first two.

        Both masks are arguments rather than captured host arrays: BindCraft 2 draws a new binder
        length per trajectory, so their shapes change under us. Neither is differentiable, so the
        backward returns zeros for both.

        The callback works in float32 while the model around it runs bfloat16, and AlphaFold 2
        carries the pair representation through a `while_loop` whose carry types must match
        exactly, so every value handed back takes the dtype it arrived with.
        """
        import jax
        import jax.numpy as jnp

        def shapes(msa, pair):
            return (jax.ShapeDtypeStruct(msa.shape, jnp.float32),
                    jax.ShapeDtypeStruct(pair.shape, jnp.float32))

        @jax.custom_vjp
        def stack(msa, pair, mask, pair_mask):
            m, z = jax.pure_callback(self._primal, shapes(msa, pair),
                                     msa.astype(jnp.float32), pair.astype(jnp.float32),
                                     mask.astype(jnp.float32), pair_mask.astype(jnp.float32))
            return m.astype(msa.dtype), z.astype(pair.dtype)

        def fwd(msa, pair, mask, pair_mask):
            m, z, token = jax.pure_callback(
                self._taped, shapes(msa, pair) + (jax.ShapeDtypeStruct((), jnp.int32),),
                msa.astype(jnp.float32), pair.astype(jnp.float32), mask.astype(jnp.float32),
                pair_mask.astype(jnp.float32))
            return (m.astype(msa.dtype), z.astype(pair.dtype)), (token, mask, pair_mask)

        def bwd(res, cotangents):
            token, mask, pair_mask = res
            g_msa, g_pair = cotangents
            gm, gz = jax.pure_callback(
                self._backward,
                (jax.ShapeDtypeStruct(g_msa.shape, jnp.float32),
                 jax.ShapeDtypeStruct(g_pair.shape, jnp.float32)),
                token, g_msa.astype(jnp.float32), g_pair.astype(jnp.float32))
            return (gm.astype(g_msa.dtype), gz.astype(g_pair.dtype),
                    jnp.zeros_like(mask), jnp.zeros_like(pair_mask))

        stack.defvjp(fwd, bwd)
        return stack


class ExtraMsaOnDevice:
    """BindCraft 2's extra-MSA stack, run on card, differentiable in `pair` alone.

    `EvoformerOnDevice` replaces the 48-block trunk; this replaces the 4-block extra-MSA stack
    that runs before it. The cut is `pair` alone: `modules.py:1530` reads only `pair` out of the
    stack, and the MSA track collapses to a constant under BindCraft 2's all-zero
    `extra_msa_mask`, so no gradient into `extra_msa` comes back and none is owed.

    `_check_mask` refuses any nonzero mask rather than folding it against the wrong constant, so
    a featurisation that ever carries a real extra MSA stops here instead of agreeing quietly.

    Its tapes live in their own registry. JAX runs the extra-MSA forward, then the Evoformer
    forward, then the two backwards in reverse, so a registry shared with `EvoformerOnDevice`
    would have the Evoformer's stale-tape sweep drop this stack's live tape before its backward.
    """

    def __init__(self, pool: TrunkPool, *, recompute: bool = True):
        self.pool = pool
        self.recompute = recompute
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        #: What the mask actually carried, so a refusal can be explained after the fact.
        self.mask_seen = {"calls": 0, "abs_max": 0.0}
        self.swapped: list[int] = []
        self._pair_mask_dev: dict = {}
        self._live: dict[int, dict] = {}
        self._next = 0

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _pad(z, pair_mask):
        """The pair half of `EvoformerOnDevice._pad`, which is all this swap hands the card."""
        n = z.shape[0]
        n32 = _pad32(n)
        if n32 == n:
            return z, pair_mask, n
        pad = n32 - n
        z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
        return z, pair_mask, n

    def _check_mask(self, extra_mask_np):
        a = np.asarray(extra_mask_np)
        self.mask_seen["calls"] += 1
        self.mask_seen["abs_max"] = max(self.mask_seen["abs_max"], float(np.abs(a).max()))
        if a.any():
            raise ValueError(
                f"extra_msa_mask carries {int((a != 0).sum())} nonzero entries; this swap "
                f"injects the outer product mean an all-zero mask collapses to and is wrong "
                f"for anything else")

    def _inputs(self, pair_np, extra_mask_np, pair_mask_np):
        self._check_mask(extra_mask_np)
        as_t = lambda a: torch.from_numpy(np.asarray(a).copy()).float()  # noqa: E731
        return self._pad(as_t(pair_np), as_t(pair_mask_np))

    def _trunk(self) -> _Trunk:
        trunk = self.pool.current
        if trunk.extra_blocks == 0:
            raise ValueError(f"{self.pool._current!r} holds no extra-MSA blocks on card")
        return trunk

    def _pair_masks(self, trunk, pair_mask):
        """`af2_pair_masks` for this mask, keyed on the mask itself for the reason
        `EvoformerOnDevice._key` gives: two binder lengths routinely pad to one shape."""
        from tt_bio.af2 import af2_pair_masks
        key = EvoformerOnDevice._key(pair_mask)
        got = self._pair_mask_dev.get(key)
        if got is None:
            got = self._pair_mask_dev[key] = af2_pair_masks(pair_mask, trunk.device)
        return got

    # ------------------------------------------------------------------ forward and backward

    def _primal(self, pair_np, extra_mask_np, pair_mask_np):
        trunk = self._trunk()
        z, pair_mask, n = self._inputs(pair_np, extra_mask_np, pair_mask_np)
        zo = trunk.extra_msa(trunk.up(z), self._pair_masks(trunk, pair_mask), recompute=False)
        trunk.sync()
        self.calls["primal"] += 1
        return trunk.down(zo, tuple(z.shape))[:n, :n].numpy()

    def _taped(self, pair_np, extra_mask_np, pair_mask_np):
        trunk = self._trunk()
        z, pair_mask, n = self._inputs(pair_np, extra_mask_np, pair_mask_np)
        zl = trunk.leaf(z)
        with trunk.taped.tape():
            zo = trunk.extra_msa(zl, self._pair_masks(trunk, pair_mask),
                                 recompute=self.recompute)
        trunk.sync()
        # Every recycle but the last is stop_gradient'ed and still goes through the forward
        # rule, so the superseded tapes are dropped here -- `EvoformerOnDevice._taped`'s reason.
        self._live.clear()
        trunk.ag.release_pins()
        token, self._next = self._next, self._next + 1
        self._live[token] = {"root": zo, "leaf": zl, "shape": tuple(z.shape), "n": n}
        self.calls["taped"] += 1
        return trunk.down(zo.value, tuple(z.shape))[:n, :n].numpy(), np.int32(token)

    def _backward(self, token, g_pair_np):
        entry = self._live.pop(int(token), None)
        if entry is None:
            raise RuntimeError(f"no live extra-MSA tape for token {int(token)}")
        trunk = self.pool.current
        shape, n = entry["shape"], entry["n"]
        gz = torch.zeros(shape)
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        trunk.ag.backward([entry["root"]], [trunk.seed(gz, entry["root"])])
        trunk.sync()
        out = trunk.grad(entry["leaf"], shape)[:n, :n].numpy()
        trunk.ag.release_pins()
        self.calls["backward"] += 1
        return out

    def live_tapes(self) -> int:
        return len(self._live)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self):
        """`(pair, extra_msa_mask, pair_mask) -> pair`, differentiable in `pair` alone."""
        import jax
        import jax.numpy as jnp

        def f32(pair):
            return jax.ShapeDtypeStruct(pair.shape, jnp.float32)

        def args(pair, extra_mask, pair_mask):
            return (pair.astype(jnp.float32), extra_mask.astype(jnp.float32),
                    pair_mask.astype(jnp.float32))

        @jax.custom_vjp
        def stack(pair, extra_mask, pair_mask):
            z = jax.pure_callback(self._primal, f32(pair), *args(pair, extra_mask, pair_mask))
            return z.astype(pair.dtype)

        def fwd(pair, extra_mask, pair_mask):
            z, token = jax.pure_callback(
                self._taped, (f32(pair), jax.ShapeDtypeStruct((), jnp.int32)),
                *args(pair, extra_mask, pair_mask))
            return z.astype(pair.dtype), (token, extra_mask, pair_mask)

        def bwd(res, g_pair):
            token, extra_mask, pair_mask = res
            gz = jax.pure_callback(self._backward, f32(g_pair), token,
                                   g_pair.astype(jnp.float32))
            return (gz.astype(g_pair.dtype), jnp.zeros_like(extra_mask),
                    jnp.zeros_like(pair_mask))

        stack.defvjp(fwd, bwd)
        return stack


def find_evoformer_masks(fn):
    """Recover the Evoformer masks from the closure of AlphaFold 2's `evoformer_fn`.

    Replacing the whole `layer_stack` means never seeing the masks dict that
    `EmbeddingsAndEvoformer` builds and `evoformer_fn` closes over. `hk.remat` wraps that closure
    twice, so the walk is recursive rather than one `__wrapped__` hop.
    """
    return _free_variable(fn, "evoformer_masks", lambda v: isinstance(v, dict) and "msa" in v)


def find_extra_msa_masks(fn):
    """The two masks `extra_msa_stack_fn` uses, read off its closure.

    Unlike `evoformer_fn` there is no masks dict to recover: `modules.py:1522` builds the stack's
    masks inline from `batch` and `mask_2d`, so those are what the walk looks for.
    """
    batch = _free_variable(fn, "batch", lambda v: isinstance(v, dict) and "extra_msa_mask" in v)
    mask_2d = _free_variable(fn, "mask_2d", lambda v: hasattr(v, "shape"))
    if batch is None or mask_2d is None:
        return None
    return {"msa": batch["extra_msa_mask"], "pair": mask_2d}


def _free_variable(fn, want, accept, depth: int = 0, seen=None):
    """The value `want` names in `fn`'s closure, followed through `hk.remat`'s wrappers."""
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
        if name == want and accept(value):
            return value
        found = _free_variable(value, want, accept, depth + 1, seen)
        if found is not None:
            return found
    return None


#: The splice `evoformer_on_device` currently has installed, or None. The patch is on
#: `modules.layer_stack`, which is process-global, so a predictor has to be able to find it:
#: a `trunk="jax"` model built inside a live campaign would otherwise be spliced too.
_INSTALLED: EvoformerOnDevice | None = None


def installed() -> EvoformerOnDevice | None:
    """The splice currently patched into AlphaFold 2, or None if the trunk is BindCraft 2's."""
    return _INSTALLED


@contextlib.contextmanager
def evoformer_on_device(evo: EvoformerOnDevice,
                        extra_msa: "ExtraMsaOnDevice | None" = None):
    """Swap AlphaFold 2's Evoformer stack for `evo` for the duration, and nothing else.

    `modules.py` calls `layer_stack` three times, twice for the template pair stack and once
    each for the extra-MSA and Evoformer stacks. The closures are distinguishable by name, so
    this patches the factory and swaps only the ones it is asked for.

    `extra_msa` is the second swap and is independent of the first: the default None leaves the
    extra-MSA stack in JAX, which is the program every Evoformer-only comparison was graded on.
    """
    from bindcraft.af.alphafold.model import modules

    real = modules.layer_stack.layer_stack
    device_stack = evo.as_jax()
    extra_stack = extra_msa.as_jax() if extra_msa is not None else None
    swapped: list[int] = []

    def choose_extra(fn, made, num_layers):
        blocks = extra_msa.pool.current.extra_blocks
        if num_layers != blocks:
            raise ValueError(f"extra_msa_stack_fn has {num_layers} blocks, tt-bio holds "
                             f"{blocks}")
        masks = find_extra_msa_masks(fn)
        if masks is None:
            raise RuntimeError(
                "batch/mask_2d not found in extra_msa_stack_fn's closure. The stack needs the "
                "extra-MSA and pair masks to fold a padded complex and guessing one is worse "
                "than stopping.")
        extra_msa.swapped.append(num_layers)

        def extra_on_device(x):
            activations, safe_key = x
            pair = extra_stack(activations["pair"], masks["msa"], masks["pair"])
            # The scan splits the key once per block and carries the first half on, so what
            # follows the stack sees the key it would have seen.
            for _ in range(num_layers):
                safe_key, _unused = safe_key.split()
            return {**activations, "pair": pair}, safe_key
        return extra_on_device

    def factory(num_layers, *args, **kwargs):
        made = real(num_layers, *args, **kwargs)

        def choose(fn):
            name = getattr(fn, "__name__", None)
            if extra_stack is not None and name == "extra_msa_stack_fn":
                return choose_extra(fn, made, int(num_layers))
            if name != "evoformer_fn":
                return made(fn)
            if evo.host_only:
                # This fold runs on a checkpoint the card does not hold, or is the control arm
                # standing down. Hand back BindCraft 2's own stack; see `EvoformerOnDevice.on_host`.
                return made(fn)
            if int(num_layers) != evo.blocks:
                raise ValueError(f"evoformer_fn has {num_layers} blocks, tt-bio holds "
                                 f"{evo.blocks}")
            swapped.append(int(num_layers))
            masks = find_evoformer_masks(fn)
            if masks is None:
                raise RuntimeError(
                    "evoformer_masks not found in evoformer_fn's closure. The trunk needs the "
                    "MSA and pair masks to fold a padded complex and guessing one is worse than "
                    "stopping.")

            def on_device(x):
                activations, safe_key = x
                msa, pair = device_stack(activations["msa"], activations["pair"],
                                         masks["msa"], masks["pair"])
                return {**activations, "msa": msa, "pair": pair}, safe_key
            return on_device
        return choose

    global _INSTALLED
    was, _INSTALLED = _INSTALLED, evo
    modules.layer_stack.layer_stack = factory
    try:
        yield swapped
    finally:
        modules.layer_stack.layer_stack = real
        _INSTALLED = was


# ----------------------------------------------------------------- the predictor


_MODEL_CLASS = None


def design_model_class():
    """`AlphaFoldDesignModel` with its Evoformer trunk selectable, built on first use.

    The class is built lazily because it subclasses BindCraft 2's, and tt-bio does not depend on
    BindCraft 2.
    """
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    try:
        from bindcraft.af2 import AlphaFoldDesignModel
    except ImportError as exc:
        raise ImportError(
            "BindCraft 2 is not importable. tt-bio does not ship it: install it from "
            "https://github.com/PacesaLab/BindCraft2 and put it on sys.path.") from exc

    class TenstorrentAlphaFoldDesignModel(AlphaFoldDesignModel):
        """BindCraft 2's design model with tt-bio's Evoformer trunk on a Tenstorrent card.

        `trunk="jax"` keeps BindCraft 2's own trunk and is the control arm: same class, same call
        path, same bucket rounding, so a device result is compared against this rather than
        against a differently shaped program.
        """

        def __init__(self, *args, trunk: str = "device", pool: TrunkPool | None = None,
                     **kwargs):
            if trunk not in ("jax", "device"):
                raise ValueError(f"trunk must be 'jax' or 'device', not {trunk!r}")
            if trunk == "device" and pool is None:
                raise ValueError("trunk='device' needs a TrunkPool to fold on")
            super().__init__(*args, **kwargs)
            self.trunk = trunk
            self.pool = pool
            #: Checkpoint name -> "device" or "jax", decided once, per family.
            self.routes: dict[str, str] = {}
            if pool is not None:
                pool.require(sorted(set(self.presets) | set(self.models)))
                if not pool.paths:
                    # Not a route: an empty pool means the source is wrong, and a campaign that
                    # runs entirely on the host while the caller believes it is on a card
                    # misattributes every number it produces. A pool that holds SOMETHING and
                    # is missing this model's checkpoints is the ordinary case, and falls back.
                    raise FileNotFoundError(
                        "trunk='device' but no AlphaFold 2 parameters resolved: " +
                        ", ".join(f"{n} at {w}" for n, w in sorted(pool.absent.items())) +
                        ". Pass checkpoints=<directory or {name: file}> to name them, or run "
                        "`tt-bio weights --download af2ig` for the monomer checkpoint.")
                self.routes = self._route_by_family()

        def _route_by_family(self) -> dict[str, str]:
            """Decide which folds go on card per model FAMILY, not per checkpoint.

            The route is read where haiku traces and is then baked into a compiled program that
            BindCraft 2 caches keyed on the family (`af2.py:271` for `predict`, `:331` for
            `sequence_gradients`). `alphafold_model_family` collapses all five multimer
            checkpoints to `('multimer',)`, and `model_1_ptm` and `model_2_ptm` to one monomer
            family because their `CONFIG_DIFFS` entries are identical. So two checkpoints of one
            family cannot take different routes: the second would silently reuse the first one's
            compiled program and fold on the wrong trunk, which nothing downstream can see
            because the shapes agree and the loss still falls.

            A family therefore goes on card only if the card holds EVERY checkpoint this model
            can draw from it. Splitting the shipped campaign the way it actually splits, five
            multimer design models on card and the monomer validation pool on host, is a split
            along family lines and is safe; three of five multimer checkpoints is not, and this
            sends all five to the host rather than fold two of them wrong.
            """
            members: dict[tuple, list[str]] = {}
            for name in sorted(set(self.presets) | set(self.models)):
                # An unknown name has no family and cannot collide, so give it its own.
                members.setdefault(self.model_families.get(name, ("?", name)), []).append(name)
            return {name: ("device" if all(self.pool.holds(n) for n in names) else "jax")
                    for names in members.values() for name in names}

        def _pick(self, model):
            """The checkpoint this call folds on, and the `model` argument to hand `super()`.

            `predict` and `sequence_gradients` resolve `model` themselves, and for `model=None`
            that does not look a name up, it SAMPLES one and splits `self.key` doing it. So
            resolving a second time here draws a second, independent name: the card runs one
            checkpoint's 48 Evoformer blocks while the embedder, the template stack, the
            structure module and the heads around them come from another. Nothing downstream can
            see it, because the shapes agree and the loss still falls.

            A multi-model pool therefore resolves once here and passes the name on. A pool of one
            reads its single name without resolving and passes `None` straight through, because
            sampling from a pool of one still splits `self.key` and the control arm's stream has
            to stay identical for a device result to be read against it.
            """
            if model is not None:
                return self._resolve_model_name(model), model
            if len(self.models) == 1:
                return self.models[0], None
            picked = self._resolve_model_name(None)
            return picked, picked

        @contextlib.contextmanager
        def _route(self, name):
            """Fold this call on the card or on BindCraft 2's own trunk, and say which.

            `name=None` is the control arm standing down. `evoformer_on_device` patches
            `modules.layer_stack` for the whole process, so a `trunk="jax"` model built inside a
            live campaign is spliced too unless it says otherwise, and a validation ensemble is
            exactly that model.
            """
            evo = installed()
            if evo is None:
                yield  # nothing is spliced, every fold is already BindCraft 2's own
                return
            if name is not None and self.routes.get(name, "device") == "device":
                self.pool.use(name)
                yield
                return
            with evo.on_host(name or ""):
                yield

        def predict(self, protein_states, model=None, *args, **kwargs):
            if self.trunk == "jax":
                with self._route(None):
                    return super().predict(protein_states, model, *args, **kwargs)
            name, passed = self._pick(model)
            with self._route(name):
                return super().predict(protein_states, passed, *args, **kwargs)

        def sequence_gradients(self, protein_states, losses, model=None, *args, **kwargs):
            if self.trunk == "jax":
                with self._route(None):
                    return super().sequence_gradients(protein_states, losses, model,
                                                      *args, **kwargs)
            name, passed = self._pick(model)
            with self._route(name):
                return super().sequence_gradients(protein_states, losses, passed,
                                                  *args, **kwargs)

    _MODEL_CLASS = TenstorrentAlphaFoldDesignModel
    return _MODEL_CLASS


def _factory(*, trunk: str, pool: TrunkPool | None, evoformer: EvoformerOnDevice | None = None,
             extra_msa: "ExtraMsaOnDevice | None" = None, exact: bool = False):
    cls = design_model_class()

    def build(*args, **kwargs):
        return cls(*args, trunk=trunk, pool=pool, **kwargs)

    build.trunk = trunk
    build.pool = pool
    build.evoformer = evoformer
    #: The extra-MSA swap, or None when the stack stayed in BindCraft 2's JAX. Its `calls`,
    #: `swapped` and `mask_seen` counters are how a caller checks the on-card path ran.
    build.extra_msa = extra_msa
    #: Whether the tape this factory's models open runs softmax and layer norm exact. Inert on
    #: `trunk="jax"`, which opens no tt-bio tape at all.
    build.exact = exact
    return build


@contextlib.contextmanager
def predictor(*, trunk: str = "device", card: int | str | None = None, checkpoints=None,
              resident: int | None = None, blocks: int = EVOFORMER_BLOCKS,
              recompute: bool = True,
              extra_msa: bool = False,
              exact: bool = False) -> Iterator[Callable[..., object]]:
    """Put tt-bio's Evoformer on card for the duration and yield a predictor factory.

    The factory takes BindCraft 2's own `AlphaFoldDesignModel` arguments (`presets`, `data_dir`,
    `num_recycle`, `length_bucket_size` and the rest) and returns a
    `DifferentiableProteinPredictor`::

        with bindcraft2.predictor(card=0) as build:
            model = build(presets="model_1_ptm", data_dir=params, length_bucket_size=32)
            gradients, prediction = model.sequence_gradients(protein_states, losses)

    `card` pins the chip and must be set before ttnn is imported; leave it None to accept
    whatever `TT_VISIBLE_DEVICES` already says. `checkpoints` is a `TrunkPool`, a directory of
    `params_<name>.npz`, a mapping of model name to file, a single file, or None for tt-bio's own
    weights cache. `resident` caps how many trunks stay on card at once.

    `extra_msa` additionally runs the 4-block extra-MSA stack on card. It is off by default and
    independent of the Evoformer swap, so a comparison graded on the Evoformer alone keeps the
    program it was graded on. Read `build.extra_msa.calls` to check the on-card path ran.

    `exact` runs softmax and layer norm on the host in float64 inside the tape, which
    reproduces AlphaFold 2's own gradient most closely. It is off by default because it is a
    diagnostic and it is expensive: one `sequence_gradients` call on a PD-L1 draw at n=192 takes
    479.59 s with it on against 19.285 s with it off, 24.87x (`perf/bcx_exact/ROUND_AB.json`).
    That is the gradient call, not the whole design round, which also carries BindCraft 2's own
    JAX work. What it buys is 1.1 % on the worst gradient tensor: at n=192 the MSA cotangent out
    of Evoformer block 1 sits 0.087998 from a float64 reference with it on and 0.088985 with it
    off, where bfloat16 alone already carries 0.075483 of that
    (`perf/bcx_exact/grade/VJP_on_n192.json` and `VJP_off_n192.json`). A PD-L1 campaign on the
    off setting accepts binders (`perf/bcx_exact/ACCEPT_GRADE.json`), which is the bar a design
    loop is graded on. Set `exact=True` to reproduce a training-style gradient bar, which
    BindCraft 2 does not have. Read `tt_bio.autograd.EXACT_SOFTMAX_STATS` to confirm which one
    ran.

    `trunk="jax"` opens no device and touches no card. It runs BindCraft 2's own trunk through
    this same class, which is the control arm every device result should be read against.

    With `trunk="device"`, a checkpoint whose weights the source cannot supply folds on that host
    trunk rather than raising, decided per model family and announced on the first such fold. If
    the source supplies nothing at all, building the model raises instead: an empty pool is a
    misconfiguration, not a route.
    """
    if trunk == "jax":
        yield _factory(trunk="jax", pool=None, exact=exact)
        return
    if card is not None:
        pin_card(card)
    # After `pin_card`: importing autograd imports ttnn, and a pin after that raises.
    from tt_bio import autograd

    pool = checkpoints if isinstance(checkpoints, TrunkPool) else TrunkPool(
        checkpoints, resident=resident)
    evo = EvoformerOnDevice(pool, blocks=blocks, recompute=recompute)
    extra = ExtraMsaOnDevice(pool, recompute=recompute) if extra_msa else None
    # `_EXACT_TRAINING` is a process-wide stack, not thread-local, so this covers every tape
    # opened for the duration -- both `_taped` calls and the backward's recompute -- without
    # either swap having to know about it.
    with autograd.exact_training(exact), evoformer_on_device(evo, extra):
        yield _factory(trunk="device", pool=pool, evoformer=evo, extra_msa=extra, exact=exact)


@contextlib.contextmanager
def campaign_predictor(*, validation: str = "jax",
                       **kwargs) -> Iterator[Callable[..., object]]:
    """`predictor`, with BindCraft 2's own predictor construction rebound to it.

    `campaign.py` builds `AlphaFoldDesignModel` in one place for the design model and one for the
    validation model, so rebinding that name is the whole integration::

        with bindcraft2.campaign_predictor(card=0):
            campaign.run_campaign(settings, project, af2_weights=params, mpnn_weights=mpnn)

    `validation` is the trunk the validation ensemble folds on and it defaults to `"jax"`,
    BindCraft 2's own. The validation ensemble is the instrument that decides whether a design is
    accepted, so folding it on card would put device numerics inside the measurement that grades
    the device; on host JAX that stage is bit-for-bit the reference's own and the thing under
    test stays the gradient loop. Any accepted count quoted as a result should come from this
    default, and should say so. Pass `validation="device"` to fold it on card as well, and
    re-measure the accepted count on that path before quoting it.

    The design model is the first predictor `run_campaign` builds and every later one is a
    validation ensemble, including the one `desperate_prediction_pools` rebuilds mid-campaign
    when validation has to move to another pool, so "every build after the first" is the rule.

    The design model's checkpoints come from one pool, so `resident` caps the card across it.

    Everything `predictor` takes passes through, `extra_msa` and `exact` included, and the swap
    it builds is re-exposed as `build.extra_msa` so a campaign can read its counters.
    """
    if validation not in ("jax", "device"):
        raise ValueError(f"validation must be 'jax' or 'device', not {validation!r}")
    with predictor(**kwargs) as build:
        from bindcraft import campaign
        control = None
        built = []

        def build_for_campaign(*args, **kw):
            nonlocal control
            if built and validation == "jax" and build.trunk == "device":
                if control is None:
                    control = _factory(trunk="jax", pool=None)
                made = control(*args, **kw)
            else:
                made = build(*args, **kw)
            built.append(made)
            return made

        build_for_campaign.trunk = build.trunk
        build_for_campaign.pool = build.pool
        build_for_campaign.evoformer = build.evoformer
        build_for_campaign.extra_msa = build.extra_msa
        build_for_campaign.exact = build.exact
        build_for_campaign.validation = validation
        build_for_campaign.built = built

        real = campaign.AlphaFoldDesignModel
        campaign.AlphaFoldDesignModel = build_for_campaign
        try:
            yield build_for_campaign
        finally:
            campaign.AlphaFoldDesignModel = real
