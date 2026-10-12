"""Run the full ESMFold2 model on Tenstorrent hardware end-to-end.

The vendored ESMFold2 reference (`tt_bio._vendor`, see its NOTICE) does the
host-side work that is *not* the neural network — sequence tokenisation, CCD
reference-conformer lookup, the
atom/token featurisation, the parcae state-space recurrence, the LM BOS/EOS
chain wrapping, and assembling the predicted `mmCIF` structure. This module
keeps all of that and swaps **every learnable submodule** for its ttnn
implementation (`tt_bio.esmc` / `tt_bio.esmfold2`), so the entire neural
forward — ESMC-6B language model, folding trunk, encoders, diffusion structure
head and confidence head — executes on the TT device.

Usage:
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio.esmfold2_runtime import patch_esmfold2
    from tt_bio.weights import hf_revision

    repo = "biohub/ESMFold2"
    model = ESMFold2Model.from_pretrained(
        repo, load_esmc=False, revision=hf_revision(repo))
    patch_esmfold2(model)                      # replace nn modules with ttnn
    # then drive it through the normal input builder / .forward

Always pass the pinned revision. On 2026-09-14 upstream re-published this repo's
`main` with a config schema the vendored loader cannot parse, and every fold on
the public service failed for three hours. `load_ttnn_esmfold2` below does it for
you; see `weights.HF_REVISIONS`.

`load_esmc=False` skips the 24 GB CPU ESMC checkpoint — the ttnn ESMC-6B is
loaded from its own sharded safetensors instead.
"""

from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F

from tt_bio import esmfold2 as E
from tt_bio.token_axis import pad_amount
from tt_bio.esmc import ESMCLanguageModel
from tt_bio.tenstorrent import dram_peak, is_wormhole


# The ESMFold2 host-side reference (featurization + mmCIF assembly) and the
# ESMFold2 `transformers` model are vendored under `tt_bio._vendor` (see that
# package's NOTICE), so there are no external clones / sys.path shims: they
# import like any other dependency, on top of the stock `transformers` wheel.


class _ESMCAdapter:
    """Wrap the ttnn ESMC-6B to the `transformers` ESMC call contract.

    `forward(input_ids, sequence_id, output_hidden_states=True)` returns an
    object exposing `.hidden_states` of shape `[n_layers+1, B, L, d_model]`.
    `sequence_id` (chain ids, -1 for PAD) becomes an additive attention mask so
    tokens only attend within their own chain — matching ESMC's chain-aware
    attention for multi-chain / padded batches.
    """

    def __init__(self, repo: str, persistent: bool = True, fast: bool = False):
        self._repo = repo
        self._persistent = persistent
        self._fast = fast
        self.lm = None  # loaded lazily on first fold

    def preload(self):
        """Load the ESMC-6B weights now (≈60 s) instead of lazily on the first
        fold — lets the CLI attribute that time to the 'loading' stage."""
        if self.lm is None:
            # --fast keeps the 6B's projection/FFN weights in block-fp8 (6.4 GB instead of
            # 12.8 GB resident, which is what fits a 12 GiB Wormhole chip). The fold itself runs
            # under lever_scope with the older bfp8 mode off, so the mode is set here, at build.
            from tt_bio import tenstorrent
            prev = tenstorrent._FAST_MODE
            tenstorrent.set_fast_mode(self._fast)
            try:
                self.lm = ESMCLanguageModel.from_pretrained(self._repo)
            finally:
                tenstorrent.set_fast_mode(prev)
            dram_peak("esmfold2/lm-loaded")

    # No progress stage of its own: this single forward is a small slice of the band the
    # CLI shows as "Featurize". Measured on ESMFold2-Fast at 20 aa on a p150a: 0.60 s of a
    # 7.94 s first-fold prep band (8%) and 0.10 s of a 0.17 s warm one, 3-5% of the whole
    # fold either way. A label the user sees for a tenth of a second is not worth a stage
    # every other model then has to skip over in the bar.
    def __call__(self, input_ids, sequence_id=None, output_hidden_states=True, **_):
        attn_mask = None
        if sequence_id is not None:
            sid = sequence_id
            same = sid[:, :, None] == sid[:, None, :]          # [B,L,L]
            valid = sid[:, :, None] >= 0
            allow = same & valid & (sid[:, None, :] >= 0)
            attn_mask = torch.where(allow, 0.0, float("-inf"))  # [B,L,L]
        self.preload()
        hs = self.lm(input_ids, attn_mask=attn_mask)            # [n_layers+1,B,L,D]
        dram_peak("esmfold2/lm-forward")
        if not self._persistent:
            # Memory-conservative mode: free the ~12.8 GB of 6B device weights
            # after the single LM forward (reloaded lazily next fold). fold_complex
            # turns this on for an MSA fold on Wormhole, where the resident LM and
            # the MSA encoder's activation do not both fit a 12 GiB chip.
            self.lm.release()
            self.lm = None
            dram_peak("esmfold2/lm-released")
        return types.SimpleNamespace(hidden_states=hs)


def _to_t(x):
    return x.float() if torch.is_tensor(x) and x.is_floating_point() else x


class _Adapter(torch.nn.Module):
    """Bridge a reference submodule call to a ttnn TorchWrapper's positional forward.

    `argnames` lists the reference kwarg names in the order the wrapper expects;
    floating tensors are cast to fp32. With no `argnames`, the reference's first
    positional arg is forwarded (single-input modules like the trunk / shim /
    distogram head, which ignore any extra kwargs such as `pair_attention_mask`).
    """

    def __init__(self, mod, *argnames):
        super().__init__()
        self.m = mod
        self.argnames = argnames

    def forward(self, *args, **kw):
        if self.argnames:
            return self.m(*[_to_t(kw[n]) for n in self.argnames])
        return self.m(_to_t(args[0]))

    # No-ops so a reference module that owns this wrapper (e.g. the confidence
    # head owning a folding trunk) can still call these on it.
    def set_kernel_backend(self, backend):
        pass

    def set_chunk_size(self, chunk_size):
        pass


# Largest diffusion batch B·L² we attempt in one pass. Best-of-N replicates the
# per-sample conditioning, so cost grows ~B·L² and a single B=N pass exceeds the
# device (static circular-buffer clash / L1 OOM, and a hard process abort at very
# large N·L) well before the recommended N=32 fits past short lengths. Calibrated
# on a Blackhole p150a: max safe single-pass batch is K=24@L128, 4@L256, 1@L512;
# 300000 yields caps {32,32,18,4,1,1} with no first-try failure at any length.
# The clash is shape-specific (not a clean budget), so shrink-on-OOM below is the
# real safety net; this just sets a good starting chunk, halved on small grids
# (e.g. Wormhole 8x8, ~55% of Blackhole's aggregate L1). Override per card with
# TT_ESMFOLD2_DIFFUSION_BUDGET.
_DIFFUSION_BUDGET = 300000


def _diffusion_budget() -> int:
    """Largest B·L² to attempt per best-of-N pass: env override wins, else the
    Blackhole-calibrated default, halved on small grids (Wormhole). Shrink-on-OOM
    is the real safety net, so this only needs to be a sensible starting point."""
    env = os.environ.get("TT_ESMFOLD2_DIFFUSION_BUDGET")
    if env:
        return int(env)
    from tt_bio import tenstorrent
    return _DIFFUSION_BUDGET // 2 if getattr(tenstorrent, "_IS_SMALL_GRID", False) else _DIFFUSION_BUDGET


def _is_oom(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in ("out of memory", "circular buffer", "clash", "not enough space", "allocate"))


class _StructureHeadAdapter(_Adapter):
    """Diffusion sampler adapter — runs best-of-N in memory-safe batched chunks.

    A single B=1 sample underutilizes the device, so batching the N samples is a
    big speedup — but the conditioning is replicated per sample, so memory grows
    ~B·L² and a full B=N pass OOMs / hard-crashes for large N·L (e.g. the
    recommended N=32 past short lengths). We therefore run the largest sub-batch
    that fits a B·L² budget, looping until all N are drawn, and shrink-on-OOM as
    a safety net. Distinct per-chunk seeds keep samples independent. Small N·L
    still runs in one pass. Output is [N, n_atoms, 3] — the contract the
    confidence head and best-of-N selection expect."""

    def sample(self, *, z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge,
               ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx,
               num_diffusion_samples: int = 1, num_sampling_steps=None, seed: int | None = None,
               **_ignored):
        steps = 20 if num_sampling_steps is None else int(num_sampling_steps)
        if seed is None:
            # The vendored forward never threads a fold seed into this call. Take the chunk
            # base from the global generator's seed instead: builder.fold(seed=...) sets it
            # via _seed_context. (Previously the default 0 made --seed a no-op for esmfold2:
            # every fold drew from a private generator seeded 0+done regardless of the seed
            # the caller asked for.)
            seed = torch.initial_seed()
        n = max(1, num_diffusion_samples)
        L = int(s_inputs.shape[1])  # residue (token) count
        args = (z_trunk.float(), s_inputs.float(), relative_position_encoding.float(),
                ref_pos.float(), ref_charge.float(), ref_mask.float(), ref_element.float(),
                ref_atom_name_chars.float(), ref_space_uid, tok_idx)
        budget = _diffusion_budget()
        chunk = max(1, min(n, budget // (L * L)))
        out, done = [], 0
        self.m.prepare(*args)  # the conditioning is the molecule's, shared by every chunk
        try:
            while done < n:
                k = min(chunk, n - done)
                try:
                    out.append(self.m.draw(steps=steps, seed=seed + done, multiplicity=k))
                    done += k
                    dram_peak(f"esmfold2/diffusion chunk [k={k}]")
                except RuntimeError as exc:
                    if k == 1 or not _is_oom(exc):
                        raise
                    chunk = max(1, k // 2)  # too big for this length/card — halve and retry
        finally:
            self.m.release()
        dram_peak("esmfold2/diffusion-done")
        return {"sample_atom_coords": torch.cat(out, dim=0)}  # [N, n_atoms, 3]


# ---------------------------------------------------------------------------
# Device-resident pair handoffs
# ---------------------------------------------------------------------------
# Several places hand a [1,L,L,256] pair tensor from one ttnn module to the next
# through a host round trip. The device side is bf16 and the host side is fp32, so
# bf16 -> fp32 -> bf16 is lossless and dropping the round trip is bit-exact by
# construction. One of them is gone: the LM shim -> LM encoder handoff inside the
# resident trunk loop.
#
# What cannot go: `parcae_input_norm` and `confidence_head.z_norm` are host fp32
# LayerNorms, and moving a reduction onto the device changes the arithmetic, so the
# pair still comes back to the host once.
#
# The parcae_coda -> distogram handoff was built, measured at -0.242 s at 512 aa, and
# then deleted: keeping `z` on the device means `z + z.transpose(-2, -3)` runs as
# `ttnn.add` on bf16 instead of a host fp32 add, and that moves 11.1 % of
# `distogram_logits` by up to one bf16 ULP at 298, 512 and 640 aa
# (`perf/esmbeat/a_distogram_parity.json`). The whole win IS that add, so there is no
# reduced version to keep. Note what did not catch it: `distogram_logits` reaches
# neither the CIF nor plDDT, so 24 folds of fold-level A/B read the arm as clean.

DEVICE_LM_HANDOFF = True
_DEVICE_LM_HANDOFF = bool(int(os.environ.get(
    "TT_ESMFOLD2_DEVICE_LM_HANDOFF", "1" if DEVICE_LM_HANDOFF else "0")))

# [served, declined]. A zero in `served` means the lever never fired and any A/B
# against it is vacuous.
LM_HANDOFF_STATS = [0, 0]


def set_device_lm_handoff(on: bool) -> None:
    global _DEVICE_LM_HANDOFF
    _DEVICE_LM_HANDOFF = bool(on)


class _DevPair:
    """A pair tensor left on the device, carried through code that only stores it."""

    __slots__ = ("t", "seq_len")

    def __init__(self, t, seq_len):
        self.t, self.seq_len = t, int(seq_len)

    @property
    def shape(self):
        return (1, self.seq_len, self.seq_len, self.t.shape[-1])


def _trunk_on_device(ftw, dev, to_host=True):
    """`E.FoldingTrunk.forward` on an already-resident input, minus the upload.

    The incoming bf16 device tensor is exactly what `_from_torch` would have
    produced from the host fp32 copy, so this is the same computation. With
    `to_host=False` the output stays on the device, still padded to the trunk's
    tile bucket.
    """
    import ttnn

    seq_len = dev.seq_len
    pad = pad_amount(seq_len, E.PAD_MULTIPLE)
    z, mask = dev.t, None
    if pad:
        n = seq_len + pad
        z = ttnn.pad(dev.t, [(0, 0), (0, pad), (0, pad), (0, 0)], value=0.0)
        real = torch.zeros(1, n, n)
        real[:, :seq_len, :seq_len] = 1.0
        mask = ftw._from_torch(real)
    out = ftw.module(z, mask)
    for t in (z if pad else None, mask):
        if t is not None:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
    if not to_host:
        return out
    return ftw._to_torch(out)[:, :seq_len, :seq_len, :]


class _ShimAdapter(_Adapter):
    """LM shim that hands its pair tensor straight to the LM encoder (lever B)."""

    def forward(self, *args, **kw):
        if not _DEVICE_LM_HANDOFF:
            LM_HANDOFF_STATS[1] += 1
            return super().forward(*args, **kw)
        m = self.m
        hs = _to_t(args[0])
        LM_HANDOFF_STATS[0] += 1
        return _DevPair(m.module(m._from_torch(hs), torch.softmax(m._combine, 0)),
                        int(hs.shape[1]))


# reference attribute -> (ttnn wrapper, state-dict prefix, reference kwarg order).
# An empty kwarg tuple means "forward the first positional arg" (trunk/shim/
# distogram). Block counts and MSA head width follow the checkpoint config, so
# the same code loads both ESMFold2 (48 trunk blocks, msa_head_width 16) and
# ESMFold2-Fast (24 trunk blocks, msa_head_width 32) — see _spec().
def _spec(config):
    """Build the component spec for this checkpoint's config (variant-aware)."""
    ft, lm, pc, msa = (config.folding_trunk, config.lm_encoder,
                       config.parcae, config.msa_encoder)
    spec = {
        "inputs_embedder": (lambda: E.InputsEmbedder(), "inputs_embedder.",
            ("aatype", "profile", "deletion_mean", "ref_pos", "atom_attention_mask",
             "ref_space_uid", "ref_charge", "ref_element", "ref_atom_name_chars", "atom_to_token")),
        "rel_pos": (lambda: E.RelPosEncoding(), "rel_pos.",
            ("residue_index", "asym_id", "sym_id", "entity_id", "token_index")),
        "language_model": (lambda: E.LanguageModelShim(), "language_model.", ()),
        "lm_encoder": (lambda: E.FoldingTrunk(lm.n_layers), "lm_encoder.", ()),
        "folding_trunk": (lambda: E.FoldingTrunk(ft.n_layers), "folding_trunk.", ()),
        "parcae_coda": (lambda: E.FoldingTrunk(pc.coda_n_layers), "parcae_coda.", ()),
        "distogram_head": (lambda: E.DistogramHeadModel(), "distogram_head.", ()),
        "structure_head": (lambda: E.StructureHead(sigma_data=16.0), "structure_head.", None),
    }
    # The MSA encoder exists only when enabled (ESMFold2-Fast ships without it),
    # mirroring the reference (model.msa_encoder is None when disabled). Build it
    # only then — patch_esmfold2 likewise uses it only if model.msa_encoder is set.
    if getattr(msa, "enabled", True):
        spec["msa_encoder"] = (
            lambda: E.MSAEncoder(msa.n_layers, msa.n_heads_msa, msa.msa_head_width),
            "msa_encoder.",
            ("x_pair", "x_inputs", "msa_oh", "has_deletion", "deletion_value", "msa_attention_mask"))
    return spec


def _components(sd, config):
    sub = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    built = {}
    for name, (factory, prefix, argnames) in _spec(config).items():
        mod = factory()
        mod.load_state_dict(sub(prefix), strict=False)
        cls = _StructureHeadAdapter if argnames is None else _Adapter
        cls = _ShimAdapter if name == "language_model" else cls
        built[name] = cls(mod, *(argnames or ()))
    return built


def _msa_per_loop(kw, total_steps, max_depth, col_rate):
    """The MSA encoder's inputs for each trunk loop, the way esm 3.4.1 draws them
    (`model.py` forward + `_run_one_loop`, `layers.py::maybe_apply_msa_column_masking` /
    `maybe_subsample_msa`): `col_rate` of the columns masked once in every row but the query,
    then, when the MSA is deeper than `max_depth`, a fresh `max_depth`-row subsample per loop
    that always keeps the query. Draws come from the global torch RNG, which `fold()` seeds.

    `kw` holds the vendored forward's tensors, rows on axis 2: `msa_oh` [B,L,M,C] and
    `has_deletion` / `deletion_value` / `msa_attention_mask` [B,L,M]. Returns one kwargs dict
    per loop, or a single one when every loop would see the same MSA.
    """
    mask, oh = kw["msa_attention_mask"], kw["msa_oh"]
    B, L, M = mask.shape
    if col_rate > 0 and M > 1:
        keep = (torch.rand(B, L) >= col_rate)[:, :, None].expand(B, L, M).clone()
        keep[:, :, 0] = True
        mask = (mask.bool() & keep.to(mask.device)).to(mask.dtype)
        # the bias-free embed needs masked entries zeroed, as upstream does before the encoder
        oh = oh * mask.unsqueeze(-1).to(oh.dtype)
    kw = dict(kw, msa_attention_mask=mask, msa_oh=oh)
    if not max_depth or M <= max_depth:
        return [kw]
    loops = []
    for _ in range(total_steps):
        idx = torch.zeros(max_depth, dtype=torch.long)
        idx[1:] = torch.randperm(M - 1)[: max_depth - 1] + 1
        idx = idx.sort().values
        loops.append(dict(kw, **{k: kw[k].index_select(2, idx.to(kw[k].device))
                                 for k in ("msa_oh", "has_deletion", "deletion_value",
                                           "msa_attention_mask")}))
    return loops


def _lm_dropout(lm_z, p, keep=None):
    """esm 3.4.1's per-loop `F.dropout(lm_z, p, training=True)` (`model.py::_run_one_loop`).

    The keep mask comes from the global torch RNG, which `fold()` seeds, so the draw follows the
    seed and not the card. A device-resident pair is masked on the device: the 0/1 mask is exact
    in bf16, the 1/(1-p) scale is applied after it rather than baked into the mask. `keep` is an
    already drawn bf16 host mask (`_KeepDraws`).
    """
    if not isinstance(lm_z, _DevPair):
        return lm_z * torch.empty(lm_z.shape).bernoulli_(1.0 - p).to(lm_z.dtype) / (1.0 - p)
    import ttnn

    t = lm_z.t
    if keep is None:
        keep = torch.empty(tuple(t.shape)).bernoulli_(1.0 - p).to(torch.bfloat16)
    # Uploaded row-major and tiled on the device: a tile-layout from_torch tilizes the pair-sized
    # mask on the host, 0.24 s a loop at 512 aa (cProfile, p300c), with the device waiting.
    rm = ttnn.from_torch(keep, device=t.device(), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    m = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm)
    kept = ttnn.multiply(t, m)
    ttnn.deallocate(m)
    out = ttnn.multiply(kept, 1.0 / (1.0 - p))
    ttnn.deallocate(kept)
    return _DevPair(out, lm_z.seq_len)


class _KeepDraws:
    """The fold's per-loop LM dropout masks, each drawn on a host thread while the device runs
    the previous loop. Only torch runs on the thread; the upload stays on the caller's.

    The generator is seeded with one draw from the global torch RNG, so the masks follow the
    fold's seed and not the card. Drawing them inline cost ~0.4 s of host time per loop at 512 aa.
    """

    def __init__(self, shape, p, n):
        from concurrent.futures import ThreadPoolExecutor

        self.gen = torch.Generator().manual_seed(int(torch.randint(2 ** 62, ())))
        self.shape, self.p, self.left = shape, p, n
        self.pool = ThreadPoolExecutor(1)
        self.nxt = self.pool.submit(self._draw)

    def _draw(self):
        return (torch.rand(self.shape, generator=self.gen) >= self.p).to(torch.bfloat16)

    def take(self):
        keep = self.nxt.result()
        self.left -= 1
        self.nxt = self.pool.submit(self._draw) if self.left > 0 else None
        if self.nxt is None:
            self.pool.shutdown(wait=False)
        return keep


# The per-loop injection on the device. Since the LM dropout is redrawn every trunk loop
# (43f72be60) the injection `b_mat @ LN(msa_pair + lm_encoder(dropout(lm_z)))` is per-loop,
# and on the host each loop paid the LM encoder's download, a host LayerNorm and linear over
# L^2 x 256, and the upload. On the device the LM encoder's output never leaves it. Declined
# (host path) when the MSA draw itself changes per loop.
_DEVICE_INJECTION = True
INJECTION_STATS = [0, 0]  # [loops served on the device, loops on the host]


def set_device_injection(on: bool) -> None:
    global _DEVICE_INJECTION
    _DEVICE_INJECTION = bool(on)


# The resident trunk's pair-sized transfers (z, the injection, its base, the trunk output) tilize on the chip
# (tenstorrent.upload / download, bit for bit the host permutation). Off = host tilize in ttnn.from_torch.
DEVICE_UPLOAD = True

# Off = esm 3.4.1's inline draw from the global RNG (perf/spd_gap_esmf/ab.py flips it between folds).
PAIR_STATE_THREAD = True


class _PairStateDraw:
    """The trunk's initial pair state, esm 3.4.1's `_init_pair_state`: a truncated normal over the
    [B, L, L, c_z] pair in float32, ~1 s of host at 512 aa that the device used to wait for. It is
    drawn on a host thread from the fold's first line, while the device runs the input embedding,
    ESMC-6B and the LM projection, from a copy of the global RNG taken there. `take` then moves the
    global RNG to where the inline draw would have left it, so the state and every later draw are
    bit for bit the inline ones. If anything drew from the global RNG in between, it draws inline.
    """

    def __init__(self):
        self.pool = self.key = self.nxt = self.rng = None

    @staticmethod
    def _draw(shape, rng):
        std = (2.0 / (5.0 * shape[-1])) ** 0.5
        g = torch.Generator()
        g.set_state(rng)
        state = torch.nn.init.trunc_normal_(torch.empty(shape, dtype=torch.float32), mean=0.0, std=std,
                                            a=-3 * std, b=3 * std, generator=g)
        return state, g.get_state()

    def start(self, shape):
        from concurrent.futures import ThreadPoolExecutor

        if self.pool is None:
            self.pool = ThreadPoolExecutor(1)
        self.key, self.rng = tuple(shape), torch.get_rng_state()
        self.nxt = self.pool.submit(self._draw, self.key, self.rng)

    def take(self, ref, inline):
        nxt, key, rng, self.nxt = self.nxt, self.key, self.rng, None
        if nxt is None or key != tuple(ref.shape) or not torch.equal(torch.get_rng_state(), rng):
            return inline(ref)
        state, after = nxt.result()
        torch.set_rng_state(after)
        return state.to(ref.dtype)


def _install_pair_state_draw(model):
    """Start `_PairStateDraw` at forward entry; `_init_pair_state` takes the result."""
    import functools
    import inspect
    import types

    draw = _PairStateDraw()
    fwd, inline = model.forward, model._init_pair_state
    sig = inspect.signature(fwd)
    c_z = model.config.d_pair

    @functools.wraps(fwd)
    def forward(*args, **kwargs):
        mask = sig.bind(*args, **kwargs).arguments["token_attention_mask"]
        if PAIR_STATE_THREAD:
            draw.start((mask.shape[0], mask.shape[1], mask.shape[1], c_z))
        return fwd(*args, **kwargs)

    model.forward = forward
    model._init_pair_state = types.MethodType(
        lambda self, ref: draw.take(ref, inline) if PAIR_STATE_THREAD else inline(ref), model)


def _install_resident_trunk_loop(model):
    """Replace the reference `_run_one_loop` with an on-device, resident-z version.

    Two wins over the per-module reference loop:
      * Each loop draws what esm 3.4.1 draws: the LM pair input gets a fresh
        `lm_encoder.lm_dropout` mask (upstream keeps it on in eval, as in training;
        `_lm_dropout`) and the MSA encoder a fresh row subsample when the MSA is
        deeper than `msa_encoder.max_depth` (`_msa_per_loop`). Only what no loop
        redraws is computed once.
      * The pair state z stays resident on the TT device across all trunk
        iterations: the parcae recurrence (a*z + inject) and the folding trunk
        both run on-device, so the ~L²·256 pair tensor is never round-tripped
        (host<->device, with tile-layout conversion) per loop.
    """
    import types

    import ttnn

    from tt_bio import esmfold2 as E

    ftw = model.folding_trunk.m  # _Adapter.m -> E.FoldingTrunk TorchWrapper
    overwrite = bool(getattr(model.config, "msa_encoder_overwrite", True))

    msa_cfg = getattr(model.config, "msa_encoder", None)
    max_depth = getattr(msa_cfg, "max_depth", None)
    col_rate = float(getattr(msa_cfg, "column_mask_rate", 0.0) or 0.0)
    lm_cfg = getattr(model.config, "lm_encoder", None)
    lm_p = float(getattr(lm_cfg, "lm_dropout", 0.0) or 0.0) \
        if getattr(lm_cfg, "per_loop_lm_dropout", False) else 0.0

    def _run_one_loop(self, z, z_init, lm_z, _msa_kwargs, pair_mask, a, b_mat, total_steps):
        use_lm = lm_z is not None and self.lm_encoder is not None

        def lm_refined():
            x = _lm_dropout(lm_z, lm_p) if lm_p else lm_z
            if isinstance(x, _DevPair):
                out = _trunk_on_device(self.lm_encoder.m, x)
                if x is not lm_z:
                    ttnn.deallocate(x.t)
            else:
                out = self.lm_encoder(x.to(z_init.dtype), pair_attention_mask=pair_mask)
            return out.to(z_init.dtype)

        msa_loops = None
        if self.msa_encoder is not None and _msa_kwargs is not None:
            msa_loops = _msa_per_loop(_msa_kwargs, total_steps, max_depth, col_rate)

        def msa_inject(msa_kw):
            # reference passes x_pair (the current pair state) separately from _msa_kwargs
            msa_pair = self.msa_encoder(x_pair=z_init, **msa_kw).to(z_init.dtype)
            dram_peak("esmfold2/msa-encoder")
            return msa_pair if overwrite else (z_init + msa_pair)

        # The MSA encoder reads z_init and the loop's MSA draw. With one draw for every loop
        # (an MSA no deeper than max_depth) its output is the same in every loop, so it is
        # encoded once even when the LM dropout makes the rest of the injection per-loop.
        msa_fixed = msa_inject(msa_loops[0]) if msa_loops is not None and len(msa_loops) == 1 \
            else None

        def inject_proj(msa_kw):
            refined = lm_refined() if use_lm else None
            z_inject = z_init
            if msa_fixed is not None:
                z_inject = msa_fixed
            elif msa_kw is not None:
                z_inject = msa_inject(msa_kw)
            if refined is not None:
                z_inject = z_inject + refined
            injected = self.parcae_input_norm(z_inject)
            return F.linear(injected.to(z.dtype), b_mat)  # [1,L,L,256] (host)

        from tt_bio import tenstorrent as T

        def up(t):
            return T.upload(t, device=ftw.tt_device) if DEVICE_UPLOAD else ftw._from_torch(t)

        # --- resident-z recurrence on device ---
        Lp = z.shape[1]
        pad = (-Lp) % E.PAD_MULTIPLE
        padz = lambda t: F.pad(t, (0, 0, 0, pad, 0, pad)) if pad else t
        mask = None
        if pad:
            real = torch.zeros(1, Lp + pad, Lp + pad)
            real[:, :Lp, :Lp] = 1.0
            mask = ftw._from_torch(real)
        zt = up(padz(z).float())
        # The injection is loop-invariant only when no loop redraws its LM dropout or MSA rows.
        per_loop = (use_lm and bool(lm_p)) or (msa_loops is not None and len(msa_loops) > 1)
        ipt = None if per_loop else up(
            padz(inject_proj(msa_loops[0] if msa_loops else None)).float())
        at = ftw._from_torch(a.reshape(1, 1, 1, -1).float())  # parcae a, broadcasts over L,L
        on_dev = (_DEVICE_INJECTION and per_loop and isinstance(lm_z, _DevPair)
                  and (msa_loops is None or msa_fixed is not None))
        if on_dev:
            # What no loop redraws, uploaded once: the MSA pair (or z_init), LN and b_mat.
            base = up(padz(msa_fixed if msa_fixed is not None else z_init).float())
            norm = self.parcae_input_norm
            ln_w, ln_b = (ttnn.from_torch(t.detach().float(), device=ftw.tt_device,
                                          layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                          for t in (norm.weight, norm.bias))
            b_t = ftw._from_torch(b_mat.t().contiguous().float())
            ck = ftw.compute_kernel_config
            draws = _KeepDraws(tuple(lm_z.t.shape), lm_p, total_steps)

        def inject_dev():
            x = _lm_dropout(lm_z, lm_p, draws.take())
            refined = _trunk_on_device(self.lm_encoder.m, x, to_host=False)
            ttnn.deallocate(x.t)
            s = ttnn.add(base, refined)
            ttnn.deallocate(refined)
            n = ttnn.layer_norm(s, weight=ln_w, bias=ln_b, epsilon=norm.eps, compute_kernel_config=ck)
            ttnn.deallocate(s)
            out = ttnn.linear(n, b_t, compute_kernel_config=ck)
            ttnn.deallocate(n)
            if pad:  # the host path pads the projection with zeros; LN(0) @ b is not zero
                real = ttnn.slice(out, [0, 0, 0, 0], [1, Lp, Lp, out.shape[-1]])
                ttnn.deallocate(out)
                out = ttnn.pad(real, [(0, 0), (0, pad), (0, pad), (0, 0)], value=0.0)
                ttnn.deallocate(real)
            return out

        for _step in range(total_steps):
            E.report_progress("trunk", _step, total_steps)
            if on_dev:
                ipt = inject_dev()
                INJECTION_STATS[0] += 1
            elif per_loop:
                ipt = up(padz(inject_proj(
                    msa_loops[min(_step, len(msa_loops) - 1)] if msa_loops else None)).float())
                INJECTION_STATS[1] += 1
            az = ttnn.multiply(zt, at)
            ttnn.deallocate(zt)
            znew = ttnn.add(az, ipt)
            ttnn.deallocate(az)
            if per_loop:
                ttnn.deallocate(ipt)
                ipt = None
            zt = ftw.module(znew, mask)  # folding trunk consumes znew, returns new z
        dram_peak("esmfold2/trunk-done")
        z_out = (T.download(zt) if DEVICE_UPLOAD else ftw._to_torch(zt))[:, :Lp, :Lp, :].to(z.dtype)
        for t in (zt, ipt, at, mask) + ((base, ln_w, ln_b, b_t) if on_dev else ()):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        return z_out

    model._run_one_loop = types.MethodType(_run_one_loop, model)


def patch_esmfold2(model, esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True,
                   fast: bool = False):
    """Replace every neural submodule of `model` with its ttnn implementation.

    After this, a normal `model.forward(...)` / input-builder fold runs the whole
    network on the TT device. Returns `model` for chaining.

    With ``persistent_lm=True`` (default) the ESMC-6B device weights stay
    resident across folds — so predicting many proteins in one process keeps all
    weights loaded (tt-bio style: pay the ~60 s ESMC load once on the first
    fold, then reuse). The trunk / encoders / structure-head weights are always
    resident. This fits the full supported range: a single Blackhole holds the
    resident 6B (~12.8 GB) plus the L=1024 trunk/diffusion activations with room
    to spare (validated: L=1024 folds in persistent mode without OOM).
    ``persistent_lm=False`` releases+reloads the 6B per fold as extra headroom
    for unusually large inputs, at the cost of an ESMC reload each fold.
    """
    sd = {k: v.float() for k, v in model.state_dict().items()}
    comps = _components(sd, model.config)

    model.inputs_embedder = comps["inputs_embedder"]
    model.rel_pos = comps["rel_pos"]
    model.language_model = comps["language_model"]
    model.lm_encoder = comps["lm_encoder"]
    model.folding_trunk = comps["folding_trunk"]
    model.parcae_coda = comps["parcae_coda"]
    if model.msa_encoder is not None:
        model.msa_encoder = comps["msa_encoder"]
    model.distogram_head = comps["distogram_head"]
    model.structure_head = comps["structure_head"]
    # Confidence head: its dominant cost is an internal 4-block pair trunk
    # (triangle-multiplicative updates, O(L^3)) — move that onto the device too.
    # `FoldingTrunk` returns the fully residual-updated pair, matching the
    # reference, so the head's `pair.add_(folding_trunk(pair))` is unchanged. The
    # reference head keeps the cheap O(L^2) glue (s->z products, pae/pde heads,
    # row pooling) and the logit->value post-processing (pLDDT / pAE / pTM) the
    # output builder needs. pLDDT/pAE/pTM do not affect the predicted structure.
    sub = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    n_conf_blocks = len(model.confidence_head.folding_trunk.blocks)
    conf_trunk = E.FoldingTrunk(n_conf_blocks)
    conf_trunk.load_state_dict(sub("confidence_head.folding_trunk."), strict=False)
    model.confidence_head.folding_trunk = _Adapter(conf_trunk)

    # Best-of-N: the confidence head replicates the pair state to B=N
    # (`pair = repeat_batch(z, N)` -> [N,L,L,c]) before its triangle-mult trunk,
    # so it OOMs / circular-buffer-clashes at the recommended N=32 past short
    # lengths — independently of the diffusion sampler. Chunk it over samples
    # the same way (same B·L² budget); it runs AFTER the shared LM+trunk, so
    # chunking here costs nothing extra on those. Every confidence output is
    # per-sample ([B,...]), so chunks concatenate cleanly.
    _orig_conf = model.confidence_head.forward

    def _chunked_confidence(*args, **kw):
        E.report_progress("confidence")  # confidence head runs an on-device pair trunk
        n = int(kw.get("num_diffusion_samples", 1))
        x_pred, z = kw.get("x_pred"), kw.get("z")
        if x_pred is None or z is None or n <= 1:
            return _orig_conf(*args, **kw)
        L = int(z.shape[1])
        budget = _diffusion_budget()
        chunk = max(1, min(n, budget // (L * L)))
        if chunk >= n:
            return _orig_conf(*args, **kw)
        outs, done = [], 0
        while done < n:
            k = min(chunk, n - done)
            kw2 = dict(kw, x_pred=x_pred[done:done + k], num_diffusion_samples=k)
            try:
                outs.append(_orig_conf(*args, **kw2))
                done += k
                dram_peak(f"esmfold2/confidence chunk [k={k}]")
            except RuntimeError as exc:
                if k == 1 or not _is_oom(exc):
                    raise
                chunk = max(1, k // 2)
        return {key: torch.cat([o[key] for o in outs], dim=0) for key in outs[0]}

    model.confidence_head.forward = _chunked_confidence

    # ttnn ESMC-6B language model. Loaded lazily on the first fold; with
    # persistent_lm it then stays resident for all subsequent folds.
    model._esmc = _ESMCAdapter(esmc_repo, persistent=persistent_lm, fast=fast)
    model._tt_fast = fast
    model._esmc_fp8 = False

    # Keep the pair state resident on-device across the trunk loop (hoist the
    # loop-invariant LM/MSA/injection work and run the parcae recurrence on
    # device), avoiding per-loop host<->device round-trips of the L²·256 pair.
    _install_resident_trunk_loop(model)
    _install_pair_state_draw(model)
    dram_peak("esmfold2/head-built")
    return model


def load_ttnn_esmfold2(esmfold2_repo: str = "biohub/ESMFold2",
                       esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True,
                       fast: bool = False):
    """Load + patch an ESMFold2 model for on-device inference, weights resident.

    Returns a patched model ready to fold many proteins without reloading. The
    24 GB CPU ESMC checkpoint is skipped (ttnn ESMC-6B is used instead).

    `fast` (the CLI `--fast` flag, off by default) keeps the ESMC-6B projection/FFN
    weights in block-fp8 (resident 12.8 GB -> 6.4 GB) and folds under LEVERS_FAST.
    Everything else is built and folded with the older bfp8 `_FAST_MODE` off, like
    Boltz-2 and OpenFold3: under it the trunk's triangle multiplications lost their
    fused in-projection and ran 8.4 s slower per 512-aa fold than in normal mode.
    """
    from tt_bio import tenstorrent
    # Normal precision does not fit a Wormhole chip's 12 GiB (the bf16 ESMC-6B alone is ~12.8 GB),
    # so ESMFold2 is fast-only there, for every caller and not just the CLI that says so.
    fast = fast or is_wormhole()
    # Reserve the diffusion step's trace region (esmfold2.StructureHead._step). A device some
    # caller already opened without one stays as it is, and the sampler runs eager.
    if E.DIFFUSION_TRACE:
        tenstorrent.get_device(trace="esmfold2")
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model

    # Pinned commit, not `main` — see weights.HF_REVISIONS for why an unpinned repo
    # took production down. A repo with no pin still tracks the hub default.
    from tt_bio.weights import hf_revision

    rev = hf_revision(esmfold2_repo)
    kwargs = {"revision": rev} if rev else {}
    model = ESMFold2Model.from_pretrained(esmfold2_repo, load_esmc=False, **kwargs).eval()
    # Built under the set it folds under: some levers are read when a module is built (kernel configs,
    # weight formats), the rest when it runs.
    with fold_scope(fast):
        return patch_esmfold2(model, esmc_repo=esmc_repo, persistent_lm=persistent_lm, fast=fast)


def _msa_from_csv(path, max_sequences):
    """Build an esm ``MSA`` from a Boltz-2 ``{hash}.csv`` cache.

    The CSV is ``key,sequence`` rows (query first) with a3m-style lowercase
    insertions; strip them so every row aligns to the query, matching what
    ``MSA.from_a3m`` does. Lets the esm path reuse a Boltz-2 server search.
    """
    from pathlib import Path

    from tt_bio._vendor.esm.utils.msa.msa import MSA

    seqs = []
    for line in Path(path).read_text().splitlines()[1:]:  # skip "key,sequence" header
        _, _, seq = line.partition(",")
        if seq:
            seqs.append(seq)
        if len(seqs) >= max_sequences:
            break
    return MSA.from_sequences(seqs, remove_insertions=True) if seqs else None


# The most alignment rows upstream ESMFold2 reads: `compute_msa_features(max_seqs=16384)`
# (_vendor/esm/models/esmfold2/prepare_input.py).
ESMFOLD2_MSA_ROWS = 16384


def pair_keyed_msa(msa, paired_a3m, max_sequences=ESMFOLD2_MSA_ROWS):
    """Put a chain's species-paired rows in front of its MSA, headed ``key=j``.

    That header is how upstream ESMFold2 pairs across chains
    (``esm/models/esmfold2/paired_msa.py``: rows sharing a ``key=N`` are paired, anything
    else goes to the chain's block-diagonal section). Row j of every chain's paired a3m is
    one genome, so its key is j. An all-gap row is a genome this chain has no homolog in:
    it is dropped and the partners keep their row j. The unpaired rows follow unkeyed.
    """
    import io

    from tt_bio._vendor.esm.utils.msa.msa import MSA
    from tt_bio._vendor.esm.utils.parsing import FastaEntry

    rows = MSA.from_a3m(io.StringIO(paired_a3m), remove_insertions=True).entries
    keyed = [FastaEntry(f"key={j}", seq) for j, (_h, seq) in enumerate(rows)
             if j and seq.strip("-")]
    query, unpaired = (msa.entries[0], msa.entries[1:]) if msa is not None else (rows[0], [])
    return MSA(([query] + keyed + unpaired)[:max_sequences])


def resolve_msa(msa_spec, sequence, msa_dir=None, max_sequences=ESMFOLD2_MSA_ROWS):
    """Resolve a chain's MSA to an esm ``MSA`` object (or None).

    Tries, in order: an explicit a3m path (``msa_spec``); a cached
    ``{sha256(seq)[:16]}.a3m``; then ``{hash}.csv`` (the Boltz-2 server cache)
    in the shared ``msa_dir``. The a3m and csv caches are written by either
    model, so a sequence searched once serves both. Returns None for
    single-sequence folding.
    """
    from pathlib import Path

    from tt_bio._vendor.esm.utils.msa.msa import MSA
    from tt_bio.cache import EMPTY_MSA, cached, seq_hash

    if msa_spec == EMPTY_MSA:
        return None
    candidates = []
    if msa_spec:
        candidates.append(Path(msa_spec).expanduser())
    if msa_dir:
        h = seq_hash(sequence)
        candidates.append(Path(msa_dir) / f"{h}.a3m")
        candidates.append(Path(msa_dir) / f"{h}.csv")
    for p in candidates:
        if cached(p):
            if p.suffix == ".csv":
                return _msa_from_csv(p, max_sequences)
            return MSA.from_a3m(str(p), max_sequences=max_sequences)
    return None


MOL_TYPES = ("protein", "dna", "rna", "ligand")


def build_spi(chains, bonds=None):
    """Canonical ``chains`` -> the vendored ``StructurePredictionInput``.

    One entry per chain: ``(chain_id, sequence, msa, mol_type, modifications)``, the
    5-tuple ``tt_bio.main._read_bio_chains`` emits — so the CLI, the platform worker and
    the parity harness all build the model's input through this one function instead of
    each assembling dataclasses of its own. Shorter protein-only tuples are accepted
    (``mol_type`` defaults to ``"protein"``, the trailing fields to None).

    * ``protein`` -- ``msa`` is an esm ``MSA`` (or None for single-sequence).
    * ``dna`` / ``rna`` -- nucleotide sequence, single-sequence only.
    * ``ligand`` -- the sequence slot carries the featurizer spec: ``CCD_<code>`` (or
      ``CCD_<code>,<code>`` for a multi-residue ligand chain such as a glycan) selects
      CCD ideal-conformer tokenization, anything else is read as SMILES and gets an
      RDKit conformer on the host. Same encoding Protenix and OpenFold3 already read.
    * ``modifications`` -- ``[{"position": N, "ccd": CODE}]``, 1-indexed (the Boltz YAML
      convention; the featurizer's ``Modification`` is 0-indexed). Polymer chains only.

    Every entry becomes one token-axis chain: a protein residue is one token, a ligand
    atom is one token, so a ligand extends the token axis by its heavy-atom count.
    """
    from tt_bio._vendor.esm.models.esmfold2 import (
        DNAInput, LigandInput, Modification, ProteinInput, RNAInput,
        StructurePredictionInput)

    _POLYMER = {"protein": ProteinInput, "dna": DNAInput, "rna": RNAInput}

    def _entry(c):
        cid, spec = c[0], c[1]
        msa = c[2] if len(c) > 2 else None
        mol_type = (c[3] or "protein") if len(c) > 3 else "protein"
        # The 4-tuple was (cid, seq, msa, modifications) before the readers unified on
        # mol_type; a list in that slot is a modifications list, not a molecule type.
        mods = c[4] if len(c) > 4 else None
        if isinstance(mol_type, (list, tuple)):
            mods, mol_type = mol_type, "protein"
        if mol_type not in MOL_TYPES:
            raise ValueError(f"chain {cid}: unknown molecule type {mol_type!r}, "
                             f"expected one of {MOL_TYPES}")
        if mol_type == "ligand":
            if str(spec).upper().startswith("CCD_"):
                codes = [x.strip().upper() for x in str(spec)[4:].split(",") if x.strip()]
                if not codes:
                    raise ValueError(f"ligand {cid}: empty CCD code")
                return LigandInput(id=cid, ccd=codes)
            return LigandInput(id=cid, smiles=str(spec).strip())
        # Normalise the sequence (strip whitespace, upper-case): otherwise those
        # characters tokenize to unknowns and crash the MSA-feature step. Matches
        # the Boltz-2 parser's behaviour.
        seq = "".join(str(spec).split()).upper()
        modifications = None
        if mods:
            modifications = []
            for mod in mods:
                pos = mod["position"]
                if not isinstance(pos, int) or not (1 <= pos <= len(seq)) or not mod.get("ccd"):
                    raise ValueError(
                        f"Modification {mod!r} on chain {cid} needs a 1-indexed position "
                        f"within the sequence (length {len(seq)}) and a ccd code.")
                modifications.append(Modification(position=pos - 1, ccd=str(mod["ccd"])))
        kwargs = {"id": cid, "sequence": seq, "modifications": modifications}
        if mol_type == "protein":
            kwargs["msa"] = msa
        return _POLYMER[mol_type](**kwargs)

    spi = StructurePredictionInput(sequences=[_entry(c) for c in chains])
    if bonds:
        spi.covalent_bonds = _covalent_bonds(spi, bonds)
    return spi


def _covalent_bonds(spi, bonds):
    """`bond` endpoints ``(chain, 1-indexed residue, atom name)`` as upstream's
    ``CovalentBond``, which names an atom by its index in that residue's atom list.

    The list is read off the featurizer's own tokenization of this input, with the bonded
    chains already marked covalent: a covalently bound CCD ligand drops its leaving atoms,
    so the indices depend on it. A SMILES atom takes the portable name (C1 = first carbon in
    the SMILES), resolved against the heavy atoms in SMILES order the tokenizer emits.
    """
    from collections import defaultdict

    from tt_bio._vendor.esm.models.esmfold2 import CovalentBond, LigandInput
    from tt_bio._vendor.esm.models.esmfold2.prepare_input import build_chains_from_input
    from tt_bio.data.parse import _resolve_bond_atom, portable_atom_names

    spi.covalent_bonds = [CovalentBond(a[0], 0, 0, b[0], 0, 0) for a, b in bonds]
    infos, tokens, atoms = build_chains_from_input(spi, seed=0)
    asym = {c.chain_id: c.asym_id for c in infos}
    smiles = {i for e in spi.sequences if isinstance(e, LigandInput) and e.smiles
              for i in ([e.id] if isinstance(e.id, str) else e.id)}
    residue = defaultdict(list)
    for a in atoms:
        if a.is_valid and a.token_index < len(tokens):
            t = tokens[a.token_index]
            residue[(t.asym_id, t.residue_index)].append(a)

    def end(cid, res, name, key):
        if cid not in asym:
            raise ValueError(f"bond constraint references chain '{cid}', which is not in the "
                             "input.")
        here = residue.get((asym[cid], res - 1))
        if not here:
            raise ValueError(f"bond constraint references residue {res} on chain '{cid}', "
                             "which it does not have.")
        names, shown, asked = [a.name for a in here], None, name
        if cid in smiles:
            portable = portable_atom_names((a.name, a.element) for a in here)
            name, shown = _resolve_bond_atom(portable, cid, name, key), list(portable)
        if name not in names:
            raise ValueError(f"bond constraint references atom '{asked}' on residue {res} of "
                             f"chain '{cid}', which it does not have. Its atoms: "
                             f"{', '.join(shown or names)}.")
        return res - 1, names.index(name)

    out = []
    for a1, a2 in bonds:
        r1, i1 = end(*a1, "atom1")
        r2, i2 = end(*a2, "atom2")
        out.append(CovalentBond(a1[0], r1, i1, a2[0], r2, i2))
    return out


# Largest token count measured to fold on a 12 GiB Wormhole chip with the ESMC-6B resident
# (single sequence, whglx 2026-09-23). Above it the LM is released after its forward.
WH_LM_RESIDENT_MAX_TOKENS = 1088


# The shared-kernel levers (tenstorrent.LEVERS) ESMFold2 folds under, per mode. Empty until a set is graded on
# ESMFold2 itself: the sets Protenix and Boltz-2 ship were graded on their folds, not on this one. A harness
# grades a candidate through TT_BIO_LEVERS.
# FAST: tenstorrent.NORMAL_LEVERS, graded on ESMFold2-Fast against the fast bar (11 complexes x 4 seeds, 44 paired
# folds vs none): CA-lDDT +0.0009 [-0.0019, +0.0044], pLDDT +0.0007, PASS; 2.6 s off ESMFold2 and 1.1 s off
# ESMFold2-Fast at 512 aa (p300c, 1350 MHz). The same set fails the normal bar on ESMFold2-Fast (TM -0.0051, CI
# excludes 0) and passes it on ESMFold2 (41 paired folds, CA-lDDT +0.0006, top-pose median 0.195 A vs 1.115 A floor).
# NORMAL: the same set without trunk_hifi3 (the trunk matmuls stay HiFi4) passes the normal bar on ESMFold2-Fast
# (44 paired folds vs none: CA-lDDT -0.0006 [-0.0022, +0.0006], TM +0.0007, docking 24/44 both, top-pose median
# 0.93 A vs 1.40 A floor) and folds the 11-set in 9 % less time than none (p300c, paired sums 2498 vs 2751 s).
# FAST adds trimul_b8in (the trimul in-projection writes bfp8 for the gated move), graded with the legacy SiLU gate
# against none on the fast bar, 11 complexes x 4 seeds on Blackhole: ESMFold2-Fast CA-lDDT +0.0012 [-0.0026, +0.0059],
# top-pose median 0.92 A vs 1.40 A floor, docking 24/44 both; ESMFold2 (41 paired) CA-lDDT +0.0021 [-0.0006,
# +0.0061], top-pose median 0.44 A vs 1.12 A floor, docking 30/41 both. ~1 s off ESMFold2 --fast at 512 aa
# (p300c, 1350 MHz: 27.80 vs 28.78 s; -Fast 18.71 vs 19.00).
LEVERS_NORMAL = "normal-trunk_hifi3"
LEVERS_FAST = "normal+trimul_b8in"


def fold_levers(fast: bool) -> str:
    """The lever set ESMFold2 builds and folds under: TT_BIO_LEVERS when a harness grades a set through the
    serving path (no user sets it), else the mode's."""
    return os.environ.get("TT_BIO_LEVERS") or (LEVERS_FAST if fast else LEVERS_NORMAL)


def fold_scope(fast: bool):
    """The context ESMFold2 builds and folds in. Blackhole: `lever_scope(fold_levers(fast))`, the older bfp8
    `_FAST_MODE` off. Wormhole keeps the path main shipped, `_FAST_MODE` = `fast` and no lever set (a harness's
    TT_BIO_LEVERS still applies): the Blackhole path's --fast threw a circular-buffer clash at 546-559 tokens on a
    Galaxy chip (program 217, L1 buffer at 188416 under a CB region ending at 300320) and three of four folds then
    hung, where main folds the same complexes. Neither path has been graded on Wormhole against the other."""
    from tt_bio import tenstorrent
    if is_wormhole():
        tenstorrent.set_fast_mode(fast)
        return tenstorrent.levers(os.environ.get("TT_BIO_LEVERS") or ())
    return tenstorrent.lever_scope(fold_levers(fast))


def fold_complex(model, chains, *, num_loops=3, num_sampling_steps=20,
                 num_diffusion_samples=1, seed=0, return_all=False, bonds=None):
    """Fold one (possibly multi-chain) complex on an already-patched model.

    `chains` is the canonical chain list :func:`build_spi` documents: protein, DNA, RNA
    and ligand (CCD code or SMILES) entries, with optional MSA and modified residues on
    the polymers. Non-protein chains are atom-tokenized by the vendored featurizer, the
    same machinery a modified residue already goes through. When an MSA is given the
    on-device MSA encoder runs. Returns the reference fold result (with `.complex`,
    `.plddt`, `.ptm`).

    With ``num_diffusion_samples > 1`` the diffusion head emits one structure per
    sample (distinct seeds); the reference ``fold`` returns them as a list. This
    is best-of-N folding, so we return the single highest-confidence sample,
    ranked by mean pLDDT (ESMFold's confidence metric) — not sample 0.

    ``return_all=True`` returns the whole list instead, ranked best-first by the
    same key. The samples are drawn either way; without this they are computed
    and dropped on the floor here, which is why esmfold2 could report "samples:
    16" while writing one structure and no per-sample confidences. Anything that
    needs the *distribution* over samples rather than its maximum — a
    sample-scaling curve, say — needs the list. The default is unchanged, so the
    single-result callers keep the exact object they had.
    """
    from tt_bio._vendor.esm.models.esmfold2 import ESMFold2InputBuilder

    spi = build_spi(chains, bonds)
    # A 12 GiB Wormhole chip cannot hold the resident block-fp8 ESMC-6B (6.29 GiB)
    # plus the MSA encoder's [1, L, M, d] activation, which is 1.0 GiB at 128 aa for
    # the default M=8192 and grows with L: every MSA fold died in the encoder with
    # ~70 MiB of contiguous DRAM left. Release the language model after its single
    # forward (it runs once per fold, outside the recycling loop) and pay one reload.
    # Bit-exact: same weights, same dtype, same order, only the device buffers move.
    # Blackhole (32 GB) always keeps the LM resident, and so does a Wormhole fold without an
    # MSA up to WH_LM_RESIDENT_MAX_TOKENS.
    #
    # The same holds without an MSA once the pair track is large: single-sequence esmfold2 folded
    # 1088 tokens on the Galaxy with the LM resident and died at 1152 in the MSA encoder's pair FFN
    # with 10.6 MiB per bank free, the LM holding ~537 of each bank's 1024 MiB. One fold per process
    # pays nothing for the release; a server folding many large targets pays one reload each.
    esmc = getattr(model, "_esmc", None)
    release_lm = (esmc is not None and getattr(esmc, "_persistent", False)
                  and (any(len(c) > 2 and c[2] is not None for c in chains)
                       or sum(len(c[1]) for c in chains) > WH_LM_RESIDENT_MAX_TOKENS)
                  and is_wormhole())
    if release_lm:
        esmc._persistent = False
    from tt_bio import tenstorrent
    try:
        with fold_scope(getattr(model, "_tt_fast", False)):
            res = ESMFold2InputBuilder().fold(
                model, spi, num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples, seed=seed)
    finally:
        if release_lm:
            esmc._persistent = True
    if isinstance(res, list):
        ranked = sorted(res, key=lambda r: float(r.plddt.mean()), reverse=True)
        return ranked if return_all else ranked[0]
    return [res] if return_all else res
