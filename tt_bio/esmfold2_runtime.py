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

    model = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False)
    patch_esmfold2(model)                      # replace nn modules with ttnn
    # then drive it through the normal input builder / .forward

`load_esmc=False` skips the 24 GB CPU ESMC checkpoint — the ttnn ESMC-6B is
loaded from its own sharded safetensors instead.
"""

from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F

from tt_bio import esmfold2 as E
from tt_bio.esmc import ESMCLanguageModel


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

    def __init__(self, repo: str, persistent: bool = True):
        self._repo = repo
        self._persistent = persistent
        self.lm = None  # loaded lazily on first fold

    def preload(self):
        """Load the ESMC-6B weights now (≈60 s) instead of lazily on the first
        fold — lets the CLI attribute that time to the 'loading' stage."""
        if self.lm is None:
            self.lm = ESMCLanguageModel.from_pretrained(self._repo)

    def __call__(self, input_ids, sequence_id=None, output_hidden_states=True, **_):
        attn_mask = None
        if sequence_id is not None:
            sid = sequence_id
            same = sid[:, :, None] == sid[:, None, :]          # [B,L,L]
            valid = sid[:, :, None] >= 0
            allow = same & valid & (sid[:, None, :] >= 0)
            attn_mask = torch.where(allow, 0.0, float("-inf"))  # [B,L,L]
        if self.lm is None:
            self.lm = ESMCLanguageModel.from_pretrained(self._repo)
        hs = self.lm(input_ids, attn_mask=attn_mask)            # [n_layers+1,B,L,D]
        if not self._persistent:
            # Memory-conservative mode: free the ~12.8 GB of 6B device weights
            # after the single LM forward (reloaded lazily next fold). Not needed
            # in practice — persistent mode fits the full L<=1024 range — but
            # available as extra headroom for unusually large inputs.
            self.lm.release()
            self.lm = None
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
        args = (_host_pair(z_trunk).float(), s_inputs.float(), relative_position_encoding.float(),
                ref_pos.float(), ref_charge.float(), ref_mask.float(), ref_element.float(),
                ref_atom_name_chars.float(), ref_space_uid, tok_idx)
        budget = _diffusion_budget()
        chunk = max(1, min(n, budget // (L * L)))
        out, done = [], 0
        while done < n:
            k = min(chunk, n - done)
            try:
                out.append(self.m.sample(*args, steps=steps, seed=seed + done, multiplicity=k))
                done += k
            except RuntimeError as exc:
                if k == 1 or not _is_oom(exc):
                    raise
                chunk = max(1, k // 2)  # too big for this length/card — halve and retry
        return {"sample_atom_coords": torch.cat(out, dim=0)}  # [N, n_atoms, 3]


# ---------------------------------------------------------------------------
# Device-resident pair handoffs
# ---------------------------------------------------------------------------
# Several places hand a [1,L,L,256] pair tensor from one ttnn module to the next
# through a host round trip. The device side is bf16 and the host side is fp32,
# so bf16 -> fp32 -> bf16 is lossless and dropping the round trip is bit-exact by
# construction. Two of them can go:
#
#   B  the LM shim -> LM encoder handoff inside the resident trunk loop
#   A  parcae_coda -> distogram head
#
# What cannot go: `parcae_input_norm` and `confidence_head.z_norm` are host fp32
# LayerNorms, and moving a reduction onto the device changes the arithmetic. The
# pair therefore still comes back to the host once; A only deletes a re-upload.

_DEVICE_LM_HANDOFF = bool(int(os.environ.get("TT_ESMFOLD2_DEVICE_LM_HANDOFF", "0")))
_DEVICE_Z = bool(int(os.environ.get("TT_ESMFOLD2_DEVICE_Z", "0")))

# [served, declined] per lever. A zero in `served` means the arm never fired and
# any A/B against it is vacuous.
LM_HANDOFF_STATS = [0, 0]
Z_STATS = [0, 0]


def set_device_lm_handoff(on: bool) -> None:
    global _DEVICE_LM_HANDOFF
    _DEVICE_LM_HANDOFF = bool(on)


def set_device_z(on: bool) -> None:
    global _DEVICE_Z
    _DEVICE_Z = bool(on)


class _DevPair:
    """A pair tensor left on the device, carried through code that only stores it."""

    __slots__ = ("t", "seq_len")

    def __init__(self, t, seq_len):
        self.t, self.seq_len = t, int(seq_len)

    @property
    def shape(self):
        return (1, self.seq_len, self.seq_len, self.t.shape[-1])


def _trunk_on_device(ftw, dev):
    """`E.FoldingTrunk.forward` on an already-resident input, minus the upload.

    The incoming bf16 device tensor is exactly what `_from_torch` would have
    produced from the host fp32 copy, so this is the same computation.
    """
    import ttnn

    seq_len = dev.seq_len
    pad = (-seq_len) % E.PAD_MULTIPLE
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


class _ZPair:
    """The trunk's final pair state, live on the device and lazily on the host.

    The confidence head needs it on the host, so the download stays. The
    distogram head re-uploads the same tensor plus its transpose; with this it
    reads the device copy instead and the symmetrisation runs on device.
    """

    def __init__(self, ftw, dev):
        self._ftw, self.dev, self._host = ftw, dev, None

    @property
    def host(self):
        if self._host is None:
            n = self.dev.seq_len
            self._host = self._ftw._to_torch(self.dev.t)[:, :n, :n, :]
        return self._host

    @property
    def shape(self):
        return self.dev.shape

    def float(self):
        return self

    def detach(self):
        return self

    def transpose(self, a, b):
        if sorted((a % 4, b % 4)) == [1, 2]:
            return _ZPairT(self)
        return self.host.transpose(a, b)

    def __add__(self, other):
        # The only add the reference does on this tensor is the distogram's
        # symmetrisation, `z + z.transpose(-2, -3)`.
        if isinstance(other, _ZPairT) and other.z is self:
            import ttnn

            zt = ttnn.permute(self.dev.t, (0, 2, 1, 3))
            out = ttnn.add(self.dev.t, zt)
            ttnn.deallocate(zt)
            return _DevPair(out, self.dev.seq_len)
        return self.host + other

    def __radd__(self, other):
        return other + self.host


class _ZPairT:
    """`z.transpose(-2, -3)` deferred, so the symmetrisation can run on device."""

    __slots__ = ("z",)

    def __init__(self, z):
        self.z = z

    def __radd__(self, other):
        return other + self.z.host.transpose(-2, -3)


def _host_pair(x):
    """Host fp32 view of a pair tensor, whichever side it currently lives on."""
    return x.host if isinstance(x, _ZPair) else x


class _CodaAdapter(_Adapter):
    """parcae_coda: keep the trunk's output on the device as well (lever A)."""

    def forward(self, *args, **kw):
        if not _DEVICE_Z:
            Z_STATS[1] += 1
            return super().forward(*args, **kw)
        m = self.m
        z = _to_t(args[0])
        seq_len = int(z.shape[1])
        pad = (-seq_len) % E.PAD_MULTIPLE
        mask = None
        if pad:
            z = F.pad(z, (0, 0, 0, pad, 0, pad))
            real = torch.zeros(1, z.shape[1], z.shape[1])
            real[:, :seq_len, :seq_len] = 1.0
            mask = m._from_torch(real)
        Z_STATS[0] += 1
        return _ZPair(m, _DevPair(m.module(m._from_torch(z), mask), seq_len))


class _DistogramAdapter(_Adapter):
    """Distogram head that accepts the already-symmetrised device pair (lever A)."""

    def forward(self, *args, **kw):
        a0 = args[0]
        if isinstance(a0, _DevPair):
            m = self.m
            n = a0.seq_len
            out = m.module(a0.t)
            try:
                import ttnn

                ttnn.deallocate(a0.t)
            except Exception:
                pass
            return m._to_torch(out)[:, :n, :n, :]
        return super().forward(*args, **kw)


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
        cls = {"language_model": _ShimAdapter, "parcae_coda": _CodaAdapter,
               "distogram_head": _DistogramAdapter}.get(name, cls)
        built[name] = cls(mod, *(argnames or ()))
    return built


def _install_resident_trunk_loop(model):
    """Replace the reference `_run_one_loop` with an on-device, resident-z version.

    Two wins over the per-module reference loop:
      * Deterministic inference (the per-loop lm_dropout's expectation is the
        identity) makes the LM-encoder, MSA-encoder and injection projection
        LOOP-INVARIANT — they are computed once instead of every iteration.
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

    def _run_one_loop(self, z, z_init, lm_z, _msa_kwargs, pair_mask, a, b_mat, total_steps):
        # --- loop-invariant injection, computed ONCE ---
        z_inject = z_init
        if self.msa_encoder is not None and _msa_kwargs is not None:
            # reference passes x_pair (the current pair state) separately from _msa_kwargs
            msa_pair = self.msa_encoder(x_pair=z_inject, **_msa_kwargs).to(z_init.dtype)
            z_inject = msa_pair if overwrite else (z_inject + msa_pair)
        if lm_z is not None and self.lm_encoder is not None:
            if isinstance(lm_z, _DevPair):
                refined = _trunk_on_device(self.lm_encoder.m, lm_z)
            else:
                refined = self.lm_encoder(lm_z.to(z_init.dtype), pair_attention_mask=pair_mask)
            z_inject = z_inject + refined.to(z_init.dtype)
        injected = self.parcae_input_norm(z_inject)
        inject_proj = F.linear(injected.to(z.dtype), b_mat)  # [1,L,L,256] (host)

        # --- resident-z recurrence on device ---
        Lp = z.shape[1]
        pad = (-Lp) % E.PAD_MULTIPLE
        padz = lambda t: F.pad(t, (0, 0, 0, pad, 0, pad)) if pad else t
        mask = None
        if pad:
            real = torch.zeros(1, Lp + pad, Lp + pad)
            real[:, :Lp, :Lp] = 1.0
            mask = ftw._from_torch(real)
        zt = ftw._from_torch(padz(z).float())
        ipt = ftw._from_torch(padz(inject_proj).float())
        at = ftw._from_torch(a.reshape(1, 1, 1, -1).float())  # parcae a, broadcasts over L,L
        for _step in range(total_steps):
            E.report_progress("trunk", _step, total_steps)
            az = ttnn.multiply(zt, at)
            ttnn.deallocate(zt)
            znew = ttnn.add(az, ipt)
            ttnn.deallocate(az)
            zt = ftw.module(znew, mask)  # folding trunk consumes znew, returns new z
        z_out = ftw._to_torch(zt)[:, :Lp, :Lp, :].to(z.dtype)
        for t in (zt, ipt, at, mask):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        return z_out

    model._run_one_loop = types.MethodType(_run_one_loop, model)


def patch_esmfold2(model, esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True):
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
        if isinstance(z, _ZPair):
            z = z.host
            kw = dict(kw, z=z)
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
            except RuntimeError as exc:
                if k == 1 or not _is_oom(exc):
                    raise
                chunk = max(1, k // 2)
        return {key: torch.cat([o[key] for o in outs], dim=0) for key in outs[0]}

    model.confidence_head.forward = _chunked_confidence

    # ttnn ESMC-6B language model. Loaded lazily on the first fold; with
    # persistent_lm it then stays resident for all subsequent folds.
    model._esmc = _ESMCAdapter(esmc_repo, persistent=persistent_lm)
    model._esmc_fp8 = False

    # Keep the pair state resident on-device across the trunk loop (hoist the
    # loop-invariant LM/MSA/injection work and run the parcae recurrence on
    # device), avoiding per-loop host<->device round-trips of the L²·256 pair.
    _install_resident_trunk_loop(model)
    return model


def load_ttnn_esmfold2(esmfold2_repo: str = "biohub/ESMFold2",
                       esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True,
                       fast: bool = False):
    """Load + patch an ESMFold2 model for on-device inference, weights resident.

    Returns a patched model ready to fold many proteins without reloading. The
    24 GB CPU ESMC checkpoint is skipped (ttnn ESMC-6B is used instead).

    `fast` (the CLI `--fast` flag, off by default — same opt-in semantics as the
    Boltz-2 path) runs the heavy matmuls in block-fp8 (bfloat8_b) for a faster
    fold at a slight precision cost:
      * folding-trunk triangle-multiplications (the dominant O(L^3) cost),
      * the ESMC-6B projection/FFN weights (qkv, out_proj, fc1, fc2) — which also
        halves the resident language-model size (~12.8 GB -> ~6.4 GB),
      * the pair-transition FFN (shared SwiGLU).
    The token-DiT attention stays fp32 and the diffusion coords stay bf16
    regardless, so the structure head's precision is unaffected. Default (off)
    is full bf16/fp32 precision.
    """
    from tt_bio import tenstorrent
    tenstorrent.set_fast_mode(fast)
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(esmfold2_repo, load_esmc=False).eval()
    return patch_esmfold2(model, esmc_repo=esmc_repo, persistent_lm=persistent_lm)


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


def resolve_msa(msa_spec, sequence, msa_dir=None, max_sequences=16384):
    """Resolve a chain's MSA to an esm ``MSA`` object (or None).

    Tries, in order: an explicit a3m path (``msa_spec``); a cached
    ``{sha256(seq)[:16]}.a3m``; then ``{hash}.csv`` (the Boltz-2 server cache)
    in the shared ``msa_dir``. The a3m and csv caches are written by either
    model, so a sequence searched once serves both. Returns None for
    single-sequence folding.
    """
    import hashlib
    from pathlib import Path

    from tt_bio._vendor.esm.utils.msa.msa import MSA

    candidates = []
    if msa_spec:
        candidates.append(Path(msa_spec).expanduser())
    if msa_dir:
        h = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        candidates.append(Path(msa_dir) / f"{h}.a3m")
        candidates.append(Path(msa_dir) / f"{h}.csv")
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            if p.suffix == ".csv":
                return _msa_from_csv(p, max_sequences)
            return MSA.from_a3m(str(p), max_sequences=max_sequences)
    return None


def fold_complex(model, chains, *, num_loops=3, num_sampling_steps=20,
                 num_diffusion_samples=1, seed=0, return_all=False):
    """Fold one (possibly multi-chain) protein complex on an already-patched model.

    `chains` is a list of ``(chain_id, sequence)`` or ``(chain_id, sequence,
    msa)`` where ``msa`` is an esm ``MSA`` object (or None for single-sequence).
    When an MSA is given the on-device MSA encoder runs. Returns the reference
    fold result (with `.complex`, `.plddt`, `.ptm`).

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
    from tt_bio._vendor.esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)

    def _protein(c):
        msa = c[2] if len(c) > 2 else None
        # Normalise the sequence (strip whitespace, upper-case): otherwise those
        # characters tokenize to unknowns and crash the MSA-feature step. Matches
        # the Boltz-2 parser's behaviour.
        return ProteinInput(id=c[0], sequence="".join(c[1].split()).upper(), msa=msa)

    spi = StructurePredictionInput(sequences=[_protein(c) for c in chains])
    res = ESMFold2InputBuilder().fold(
        model, spi, num_loops=num_loops, num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples, seed=seed)
    if isinstance(res, list):
        ranked = sorted(res, key=lambda r: float(r.plddt.mean()), reverse=True)
        return ranked if return_all else ranked[0]
    return [res] if return_all else res
