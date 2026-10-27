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
from tt_bio.envflags import env_flag
from tt_bio.token_axis import pad_amount
from tt_bio.esmc import ESMCLanguageModel, esmc6b_tile_dir
from tt_bio.weights import fetch
from tt_bio.tenstorrent import dram_peak, is_wormhole, pair_lookup_device


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

    def __init__(self, repo: str, persistent: bool = True, fast: bool = False, shim=None):
        self._repo = repo
        self._persistent = persistent
        self._fast = fast
        self.lm = None  # loaded lazily on first fold
        # Host tiles of the 6B's weights, kept on Wormhole where fold_complex may release the LM
        # after a fold: the reload uploads them (seconds) instead of re-reading and re-tiling the
        # checkpoint, which cost a warm 1024 aa ESMFold2 fold 96 of its 201 s. ~6.3 GB host RAM,
        # used only when the persistent tile cache is unavailable.
        self._tiles = None
        # Under LM_COMBINE the 6B returns this shim's layer-combined state, one layer instead of 81.
        self._shim = shim

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
                # The persistent tile cache makes a load seconds; without one (disk too full),
                # Wormhole keeps the host tiles in RAM for the reload after a release.
                tile_dir = esmc6b_tile_dir("esmc-6b", fetch("esmc-6b"))
                if self._tiles:
                    with tenstorrent.weight_cache(self._tiles, "load"):
                        self.lm = ESMCLanguageModel()
                        self.lm.load_state_dict({}, strict=False)
                elif tile_dir or not is_wormhole():
                    self.lm = ESMCLanguageModel.from_pretrained(self._repo, tile_dir=tile_dir or False)
                else:
                    self._tiles = {}
                    with tenstorrent.weight_cache(self._tiles, "dump"):
                        self.lm = ESMCLanguageModel.from_pretrained(self._repo, tile_dir=False)
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
        combine = self._shim.combine_lm() if E.gate(LM_COMBINE, "ef2_lm_combine") and self._shim is not None else None
        hs = self.lm(input_ids, attn_mask=attn_mask, combine=combine)  # [n_layers+1 or 1,B,L,D]
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
        if self.argnames:  # anything past `argnames` (e.g. to_host) is the wrapper's own keyword
            return self.m(*[_to_t(kw.pop(n)) for n in self.argnames], **kw)
        if isinstance(args[0], _DevPair):
            return _dev_forward(self.m, args[0])
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
        if isinstance(z_trunk, _DevPair):  # the device tail: prepare consumes (and frees) the device pair
            if _device_conf():  # the confidence head reads its own copy on the device
                import ttnn
                z_trunk.keep = ttnn.clone(z_trunk.t)
            else:
                z_trunk.host = self.m._to_torch(z_trunk.t)[:, :L, :L].float()  # for the confidence head
            z_trunk = z_trunk.t
        args = (z_trunk.float() if torch.is_tensor(z_trunk) else z_trunk, s_inputs.float(), relative_position_encoding.float(),
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
    """A pair tensor left on the device, carried through code that only stores it.

    The device tail (`_device_tail()`) also runs the vendored forward's own pair glue on it:
    `.float()` / `.detach()` are no-ops, `z + z.transpose(-2, -3)` is a device permute and add,
    and `host` keeps the one download the confidence head still needs."""

    __slots__ = ("t", "seq_len", "host", "keep", "_tmp")
    dtype = torch.float32

    def __init__(self, t, seq_len, tmp=False):
        self.t, self.seq_len, self.host, self.keep, self._tmp = t, int(seq_len), None, None, tmp

    @property
    def shape(self):
        return (1, self.seq_len, self.seq_len, self.t.shape[-1])

    def float(self):
        return self

    detach = float

    def to_host(self):
        if self.host is None:
            from tt_bio import tenstorrent as T
            self.host = T.download(self.t)[:, :self.seq_len, :self.seq_len].float()
        return self.host

    def transpose(self, a, b):
        import ttnn
        assert {a % 4, b % 4} == {1, 2}, (a, b)
        return _DevPair(ttnn.permute(self.t, (0, 2, 1, 3)), self.seq_len, tmp=True)

    def __add__(self, other):
        import ttnn
        out = _DevPair(ttnn.add(self.t, other.t), self.seq_len, tmp=True)
        for x in (self, other):
            if x._tmp and x is not out:
                ttnn.deallocate(x.t)
        return out


# The trunk's output stays on the device through parcae_readout (a device linear instead of the host
# fp32 one), parcae_coda, the distogram's symmetrisation and the diffusion conditioning; it is
# downloaded once, for the confidence head. Removes the trunk download and three pair uploads
# (~1.1 s of transfers at 512 aa on p300c). Not bit-exact (bf16 readout and add): graded PASS on Blackhole
# (ESMFold2 normal bar, ESMFold2-Fast fast bar, 11 complexes x 4 seeds) and on Wormhole (fast bar for ESMFold2,
# which Wormhole folds --fast, normal bar for -Fast; 4-6 s off a 5-sample fold there), so on by default.
def _device_tail() -> bool:
    return env_flag("TT_BIO_ESMFOLD2_DEVICE_TAIL", True)


DEVICE_TAIL_STATS = [0, 0]  # folds whose tail ran on the device, on the host


def _aliases(a, b) -> bool:
    """True when two device tensors share one buffer. Under `ws_resid` every op of a PairUpdateBlock writes into its
    input, so a trunk can return the very tensor it was given; whoever frees the input must not free the output."""
    return a is b or (a.is_allocated() and b.is_allocated() and a.buffer_address() == b.buffer_address())


def _unpad(t, seq_len):
    import ttnn
    if int(t.shape[1]) == seq_len:
        return t
    out = ttnn.slice(t, [0, 0, 0, 0], [1, seq_len, seq_len, int(t.shape[-1])])
    ttnn.deallocate(t)
    return out


def _dev_forward(m, x):
    """An _Adapter's wrapped module on a device-resident pair: the trunk returns one, the distogram host logits."""
    import ttnn
    if isinstance(m, E.DistogramHeadModel):
        out = m._to_torch(m.module(x.t))[:, :x.seq_len, :x.seq_len]
        if x._tmp:
            ttnn.deallocate(x.t)
        return out
    r = _trunk_on_device(m, x, to_host=False)
    if not _aliases(r, x.t):
        ttnn.deallocate(x.t)
    return _DevPair(_unpad(r, x.seq_len), x.seq_len)


# The confidence head's L^2 part (ConfidenceHead._pair_outputs) on the device: z_norm, the pair sum, the outer
# product projection, the trunk, row pooling and the PAE/PDE logits. The host keeps what is [L,c]-sized or a lookup
# (summed in fp32, one upload) and the row softmax. Host fp32 there costs ~1.1 s per 512-aa fold (xf2 profile).
# Not bit-exact (bf16 pair): with the device tail, graded PASS on Blackhole (ESMFold2 normal bar, ESMFold2-Fast fast
# bar, 11 complexes x 4 seeds), so on by default like the tail. p150a ESMFold2-Fast --fast: 15.23 vs 16.17 s.
# Wormhole (Blackhole fold path), 11 complexes x 4 seeds against the old WH path, fast and normal bar, PASS for both
# models (ESMFold2 CA-lDDT +0.0034, docking 33/44 as without it); folds clean to 1536 tokens, 3-5 % off the fold.
def _device_conf() -> bool:
    return env_flag("TT_BIO_ESMFOLD2_DEVICE_CONF", True)


DEVICE_CONF_STATS = [0, 0]  # heads on the device, on the host


def _conf_weights(head, ftw):
    w = getattr(head, "_tt_conf_w", None)
    if w is None:
        import ttnn
        vec = lambda t: ttnn.from_torch(t.detach().float(), device=ftw.tt_device, layout=ttnn.TILE_LAYOUT,
                                        dtype=ttnn.bfloat16)
        mat = lambda lin: ftw._from_torch(lin.weight.detach().t().contiguous().float())
        w = head._tt_conf_w = {
            "zn": (vec(head.z_norm.weight), vec(head.z_norm.bias)),
            "pae_ln": (vec(head.pae_ln.weight), vec(head.pae_ln.bias)),
            "pde_ln": (vec(head.pde_ln.weight), vec(head.pde_ln.bias)),
            "po": mat(head.s_to_z_prod_out), "ap": mat(head.row_attention_pooling.attn_proj),
            "pae": mat(head.pae_head), "pde": mat(head.pde_head)}
    return w


# PAE, PDE and pTM read only the expectations of their [L,L,64] logits, so the device reduces them: exp(logits - max)
# projected on [bin centres, TM term, ones] and divided by the ones column on the host. The division keeps the
# denominator exact (ttnn.softmax rows sum to ~0.977, see tenstorrent._accurate_softmax). Two [L,L,3] downloads
# replace two fp32 [L,L,64] ones and the host softmaxes over them: 68 vs 303 ms at 512 tokens on Wormhole, PAE within
# 0.03 A of float64 (perf/spd_gap_esmf/conf_proj.py).


def _expectations_dev(lg, vecs, ck):
    """[1,L,L,64] logits on the device -> [1,L,L,K] host: E[vecs[:, k]] under softmax(logits) per pair."""
    import ttnn
    x = ttnn.typecast(lg, ttnn.float32)
    m = ttnn.max(x, dim=-1, keepdim=True)
    d = ttnn.subtract(x, m)
    for t in (x, m):
        ttnn.deallocate(t)
    e = ttnn.exp(d)
    ttnn.deallocate(d)
    pj = ttnn.matmul(e, vecs, compute_kernel_config=ck)
    ttnn.deallocate(e)
    out = torch.Tensor(ttnn.to_torch(pj)).float()
    ttnn.deallocate(pj)
    return out[..., :-1] / out[..., -1:]


def _conf_pair_dev(head, ftw, z, s_n, relpos, bonds, bins, mask, n, tm_per_bin):
    """ConfidenceHead._pair_outputs with the pair on the device; `z` is a device tensor (not freed here) or a host one.

    The sample-independent part (z_norm, the [L,c] broadcasts, relpos, bonds, the outer product) is built once; each
    of the n samples adds its own distance-bin embedding and runs the trunk and heads (the host path batches the same
    n pairs as B=n).
    """
    import ttnn
    if not _device_conf():
        DEVICE_CONF_STATS[1] += 1
        if isinstance(z, ttnn.Tensor):
            z = ftw._to_torch(z).float()
        return type(head)._pair_outputs(head, z, s_n, _host(relpos), _host(bonds), bins, mask, n, tm_per_bin)
    DEVICE_CONF_STATS[0] += 1
    w, ck, L = _conf_weights(head, ftw), ftw.compute_kernel_config, int(bins.shape[-1])
    lin = lambda x, wt: ttnn.linear(x, wt, compute_kernel_config=ck)

    def ln(x, wb, mod):
        return ttnn.layer_norm(x, weight=wb[0], bias=wb[1], epsilon=mod.eps, compute_kernel_config=ck)

    def swap(old, new):
        ttnn.deallocate(old)
        return new

    zt = z if isinstance(z, ttnn.Tensor) else ftw._from_torch(z.float())
    p0 = ln(zt, w["zn"], head.z_norm)
    if zt is not z:
        ttnn.deallocate(zt)
    if E.gate(_PAIR_INIT, "ef2_pair_init"):  # the [L,c] broadcasts and the lookups added on the device, as z_init
        for t in (head.s_to_z(s_n).unsqueeze(2), head.s_to_z_transpose(s_n).unsqueeze(1), relpos, bonds):
            if t is not None:
                p0 = _add_pair(p0, t, ftw)
    else:
        host = head.s_to_z(s_n).unsqueeze(2) + head.s_to_z_transpose(s_n).unsqueeze(1)
        for t in (relpos, bonds):
            if t is not None:
                host = host + _host(t)
        p0 = _add_pair(p0, host, ftw)
    a = ftw._from_torch(head.s_to_z_prod_in1(s_n)[:, :, None, :].float())
    b = ftw._from_torch(head.s_to_z_prod_in2(s_n)[:, None, :, :].float())
    op = ttnn.multiply(a, b)  # [1,L,1,c] x [1,1,L,c] -> the outer product [1,L,L,c]
    for t in (a, b):
        ttnn.deallocate(t)
    prod = swap(op, lin(op, w["po"]))
    p0 = swap(p0, ttnn.add(p0, prod))
    ttnn.deallocate(prod)
    singles, pae, pde = [], [], []
    centres = torch.linspace(0.0, 32.0, tm_per_bin.shape[-1] + 1)
    centres = (centres[:-1] + centres[1:]) / 2  # _categorical_mean's bins
    up = lambda *cols: ttnn.from_torch(torch.stack(cols, -1).float().reshape(1, 1, -1, len(cols)),
                                       layout=ttnn.TILE_LAYOUT, device=ftw.tt_device, dtype=ttnn.float32)
    one = torch.ones_like(centres)
    v_pde = up(centres, one)
    for k in range(int(bins.shape[0])):
        et = pair_lookup_device(bins[k], head.dist_bin_pairwise_embed.weight.detach(), ftw.tt_device)
        p = ttnn.add(p0, et)
        ttnn.deallocate(et)
        x = ttnn.clone(p)  # the trunk consumes its input; the residual needs p
        delta = _unpad(_trunk_on_device(ftw, _DevPair(x, L), to_host=False), L)
        pair = ttnn.add(p, delta)
        for t in (x, p, delta):
            if t.is_allocated():
                ttnn.deallocate(t)
        sc = lin(pair, w["ap"])
        scores = ftw._to_torch(sc)[:, :L, :L, 0].float()
        ttnn.deallocate(sc)
        mk = mask[k:k + 1]
        scores = scores + torch.where(mk[:, None, :].bool(), torch.zeros_like(scores), torch.full_like(scores, -1e9))
        wt = ftw._from_torch(torch.softmax(scores, dim=-1).reshape(1, L, 1, L))
        pooled = ttnn.matmul(wt, pair, compute_kernel_config=ck)  # per row i: weights[i] @ pair[i]
        ttnn.deallocate(wt)
        singles.append(head.row_attention_pooling.out_proj(ftw._to_torch(pooled)[:, :L, 0, :].float()))
        ttnn.deallocate(pooled)
        v_pae = up(centres, tm_per_bin[k].float().cpu(), one)
        for norm, key, out, v in ((head.pae_ln, "pae", pae, v_pae), (head.pde_ln, "pde", pde, v_pde)):
            xn = ln(pair, w[key + "_ln"], norm)
            lg = swap(xn, lin(xn, w[key]))
            out.append(_expectations_dev(lg, v, ck)[:, :L, :L, :])
            ttnn.deallocate(lg)
        for t in (v_pae, pair):
            ttnn.deallocate(t)
    for t in (p0, v_pde):
        ttnn.deallocate(t)
    pae, pde = torch.cat(pae), torch.cat(pde)
    return torch.cat(singles), (None, None), (pae[..., 0], pde[..., 0], pae[..., 1])


class _ReadoutAdapter(torch.nn.Module):
    """parcae_readout: the host nn.Linear, or a device linear on a device-resident pair."""

    def __init__(self, lin, ftw):
        super().__init__()
        self.lin, self.ftw, self.w = lin, ftw, None

    def forward(self, z):
        if not isinstance(z, _DevPair):
            return self.lin(z)
        import ttnn
        if self.w is None:
            self.w = self.ftw._from_torch(self.lin.weight.detach().t().contiguous().float())
        out = ttnn.linear(z.t, self.w, compute_kernel_config=self.ftw.compute_kernel_config)
        ttnn.deallocate(z.t)
        return _DevPair(out, z.seq_len)


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
    for t in (z if pad and not _aliases(z, out) else None, mask):
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
        pre = m.is_precombined(hs)
        if pre:
            hs = m.fill_absent(hs)
        return _DevPair(m.module(m._from_torch(E.flat_layers(hs)), torch.softmax(m._combine, 0), precombined=pre),
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


# The device keep mask crosses PCIe as bits: 32 rows of the pair per int32 word (`_fold_words`), 8 MB a loop
# at 512 aa instead of the 128 MB bf16 mask (2.2 s of uploads per fold, p300c). Same mask, same output.
_PACKED_KEEP = True
_KEEP_BITS = 16
# A dyadic dropout p draws the packed words directly (`_draw_packed_keep`); off = the float draw.
# Unset follows the ef2_bit_draw lever (LEVERS_FAST), see esmfold2.gate.
_BIT_DRAW = env_flag("TT_BIO_ESMFOLD2_BIT_DRAW", None)



def _pack_keep(keep):
    """bool [B, L, ...] -> int32 [B, L / 16, ...]: row j * L/16 + g of `keep` is bit j of word row g."""
    k = keep.unflatten(1, (_KEEP_BITS, keep.shape[1] // _KEEP_BITS)).to(torch.int32)
    w = k[:, 0].clone()
    for j in range(1, _KEEP_BITS):
        w |= k[:, j] << j
    return w


def _dyadic(p, max_bits=_KEEP_BITS):
    """(k, m) with p == k / 2**m, 0 < p < 1, or None when p is not such a fraction of at most `max_bits` bits."""
    if not 0 < p < 1:
        return None
    for m in range(1, max_bits + 1):
        k = p * (1 << m)
        if k == int(k):
            return int(k), m
    return None


def _draw_packed_keep(shape, p, gen):
    """`_pack_keep(torch.rand(shape) >= p)` in law, drawn as the packed words themselves.

    For p = k / 2**m every mask bit is an m-bit uniform integer u, dropped when u < k; the m bits
    of 16 elements come from m random 16-bit words and the comparison runs bitwise, MSB first, on
    whole words. p = 0.25 (ESMFold2's LM dropout) is two words and two ops per 16 elements, where
    the float draw spent a 32-bit uniform, a compare and a bit-pack on each element: 0.54 s a loop
    at 512 aa against 0.03 s (p150a host), on the thread the device loop waits for.
    """
    k, m = _dyadic(p)
    w = (shape[0], shape[1] // _KEEP_BITS, *shape[2:])
    full = (1 << _KEEP_BITS) - 1
    lt = torch.zeros(w, dtype=torch.int32)
    eq = torch.full(w, full, dtype=torch.int32)
    for i in reversed(range(m)):
        r = torch.randint(0, 1 << _KEEP_BITS, w, dtype=torch.int32, generator=gen)
        if k >> i & 1:
            lt |= eq & ~r
            eq &= r
        else:
            eq &= ~r
    return ~lt & full


def _fold_words(w):
    """`_pack_keep`'s 16-bit words two to an int32: word row g holds rows g (low half) and g + n/2 (high half).

    The draw keeps its 16-bit stream, so the mask is the same; only half the bytes cross PCIe (0.88 s of
    uploads a fold at 1024 aa on Wormhole). An odd word count stays as it is.
    """
    if w.shape[1] % 2:
        return w
    h = w.shape[1] // 2
    return w[:, :h] | (w[:, h:] << _KEEP_BITS)


def _unpack_keep_dev(w, device, rows):
    """`_pack_keep`'s words (or `_fold_words`' pairs of them) to the bf16 0/1 mask of `rows` rows on the device."""
    import ttnn
    from tt_bio.tenstorrent import upload

    wt = upload(w, dtype=ttnn.int32, device=device)
    halves = rows // (_KEEP_BITS * w.shape[1])
    out = []
    for j in range(_KEEP_BITS):
        for k in range(halves):  # bit j of the low, then the high half: rows in order
            s = ttnn.bitwise_right_shift(wt, j + k * _KEEP_BITS) if j or k else wt
            b = ttnn.bitwise_and(s, 1)
            if j or k:
                ttnn.deallocate(s)
            out.append(ttnn.typecast(b, ttnn.bfloat16))
            ttnn.deallocate(b)
    ttnn.deallocate(wt)
    m = ttnn.concat(out, dim=1)
    for r in out:
        ttnn.deallocate(r)
    return m


def _packable(shape):
    return _PACKED_KEEP and len(shape) == 4 and shape[1] % _KEEP_BITS == 0


def _lm_dropout(lm_z, p, keep=None):
    """esm 3.4.1's per-loop `F.dropout(lm_z, p, training=True)` (`model.py::_run_one_loop`).

    The keep mask comes from the global torch RNG, which `fold()` seeds, so the draw follows the
    seed and not the card. A device-resident pair is masked on the device: the 0/1 mask is exact
    in bf16, the 1/(1-p) scale is applied after it rather than baked into the mask. `keep` is an
    already drawn host mask (`_KeepDraws`): int32 words from `_fold_words`, or bf16 0/1.
    """
    if not isinstance(lm_z, _DevPair):
        return lm_z * torch.empty(lm_z.shape).bernoulli_(1.0 - p).to(lm_z.dtype) / (1.0 - p)
    import ttnn

    t = lm_z.t
    if keep is None:
        keep = torch.empty(tuple(t.shape)).bernoulli_(1.0 - p).to(torch.bfloat16)
        if _packable(keep.shape):
            keep = _fold_words(_pack_keep(keep.bool()))
    if keep.dtype == torch.int32:
        m = _unpack_keep_dev(keep, t.device(), int(t.shape[1]))
    else:
        # Uploaded row-major and tiled on the device: a tile-layout from_torch tilizes the pair-sized
        # mask on the host, 0.24 s a loop at 512 aa (cProfile, p300c), with the device waiting.
        rm = ttnn.from_torch(keep, device=t.device(), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        m = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
    # In place on the mask: the dropout allocates no pair-sized buffer of its own. A fresh one did not fit the
    # fragmented DRAM of a warm 1536 aa ESMFold2 fold on Wormhole (largest free block 100.2 of 100.7 MB a bank).
    m = ttnn.multiply_(m, t)
    m = ttnn.multiply_(m, 1.0 / (1.0 - p))
    return _DevPair(m, lm_z.seq_len)


class _KeepDraws:
    """The fold's per-loop LM dropout masks, each drawn (and bit-packed, `_pack_keep`) on a host
    thread while the device runs the previous loop. Only torch runs on the thread; the upload
    stays on the caller's.

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
        if E.gate(_BIT_DRAW, "ef2_bit_draw") and _packable(self.shape) and _dyadic(self.p):
            return _fold_words(_draw_packed_keep(self.shape, self.p, self.gen))
        keep = torch.rand(self.shape, generator=self.gen) >= self.p
        return _fold_words(_pack_keep(keep)) if _packable(self.shape) else keep.to(torch.bfloat16)

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


# The pair state's truncated normal drawn by rejection: a standard normal, the 0.27 % outside +-3
# redrawn until none is left. Same distribution as torch's trunc_normal_ (inverse-CDF through erfinv),
# other values for a seed; 0.20 vs 1.23 s at 512 aa on qb2's host. Off = trunc_normal_ (A/B flag).
TRUNC_REJECT = True


def _trunc_normal_reject(shape, g):
    x = torch.randn(shape, generator=g)
    flat = x.view(-1)
    idx = (flat.abs() > 3.0).nonzero().squeeze(1)
    while idx.numel():
        flat[idx] = torch.randn(idx.numel(), generator=g)
        idx = idx[flat[idx].abs() > 3.0]
    return x


class _PairStateDraw:
    """The trunk's initial pair state, esm 3.4.1's `_init_pair_state`: a truncated normal over the
    [B, L, L, c_z] pair in float32, ~1 s of host at 512 aa that the device used to wait for. It is
    drawn on a host thread from the fold's first line, while the device runs the input embedding,
    ESMC-6B and the LM projection, from a copy of the global RNG taken there. `take` then moves the
    global RNG past the draw, so every later draw follows the seed as the inline one would. With
    TRUNC_REJECT the state's values differ from esm's trunc_normal_ (same distribution); off, they are
    bit for bit the inline ones. If anything drew from the global RNG in between, or no draw was started,
    `take` runs the same `_draw` on the spot, so a seed gives one state whichever way it was drawn.
    """

    def __init__(self):
        self.pool = self.key = self.nxt = self.rng = None

    @staticmethod
    def _draw(shape, rng):
        std = (2.0 / (5.0 * shape[-1])) ** 0.5
        g = torch.Generator()
        g.set_state(rng)
        if TRUNC_REJECT:
            state = _trunc_normal_reject(shape, g).mul_(std)
        else:
            state = torch.nn.init.trunc_normal_(torch.empty(shape, dtype=torch.float32), mean=0.0, std=std,
                                                a=-3 * std, b=3 * std, generator=g)
        return state, g.get_state()

    def start(self, shape):
        from concurrent.futures import ThreadPoolExecutor

        if self.pool is None:
            self.pool = ThreadPoolExecutor(1)
        self.key, self.rng = tuple(shape), torch.get_rng_state()
        self.nxt = self.pool.submit(self._draw, self.key, self.rng)

    def take(self, ref):
        nxt, key, rng, self.nxt = self.nxt, self.key, self.rng, None
        if nxt is None or key != tuple(ref.shape) or not torch.equal(torch.get_rng_state(), rng):
            state, after = self._draw(tuple(ref.shape), torch.get_rng_state())
        else:
            state, after = nxt.result()
        torch.set_rng_state(after)
        return state.to(ref.dtype)


# z_init = z_init_1(x)[i] + z_init_2(x)[j] + relpos[i,j] + bonds[i,j] summed on the device: the host built three
# fp32 [L,L,256] tensors (0.26 s at 512 tokens on qb1) and uploaded the sum. Relative positions stay a PairLookup
# (codes + rows) that the diffusion conditioning and the confidence head build on the device with one embedding,
# so the 134 MB bf16 relpos pair never crosses PCIe. Not bit-exact: the parts are rounded to bf16 before the sum.
_PAIR_INIT = env_flag("TT_BIO_ESMFOLD2_PAIR_INIT", None)  # unset: the ef2_pair_init lever
PAIR_INIT_STATS = [0, 0]  # folds whose pair init ran on the device, on the host


def _host(x):
    """The host tensor of a pair that may be device-resident (_DevPair) or a lookup (PairLookup)."""
    return x.to_host() if isinstance(x, _DevPair) else x.host() if isinstance(x, E.PairLookup) else x


def _install_pair_init(model):
    import types

    import ttnn

    host_init, ftw, rel = model._pair_init, model.folding_trunk.m, model.rel_pos.m

    def _pair_init(self, x, residue_index, asym_id, sym_id, entity_id, token_index, token_bonds):
        if not (E.gate(_PAIR_INIT, "ef2_pair_init") and E.gate(E.RELPOS_LOOKUP, "ef2_relpos_lookup") and x.shape[0] == 1):
            PAIR_INIT_STATS[1] += 1
            return host_init(x, residue_index, asym_id, sym_id, entity_id, token_index, token_bonds)
        PAIR_INIT_STATS[0] += 1
        relpos = rel.rows(residue_index, asym_id, sym_id, entity_id, token_index)
        bonds = self.token_bonds(token_bonds.float()) if bool(token_bonds.any()) else None
        zi, zj = (ftw._from_torch(t.float()) for t in (self.z_init_1(x).unsqueeze(2), self.z_init_2(x).unsqueeze(1)))
        z = ttnn.add(zi, zj)  # [1,L,1,c] + [1,1,L,c] -> [1,L,L,c]
        ttnn.deallocate(zi)
        ttnn.deallocate(zj)
        for t in (relpos, bonds):
            if t is not None:
                z = _add_pair(z, t, ftw)
        return _DevPair(z, x.shape[1]), relpos, bonds

    model._pair_init = types.MethodType(_pair_init, model)


def _add_pair(z, t, ftw):
    """`z + t` on the device for a host pair or a PairLookup; frees `z`'s old buffer and the upload."""
    import ttnn
    tt = t.device(ftw.tt_device) if isinstance(t, E.PairLookup) else ftw._from_torch(t.float())
    out = ttnn.add(z, tt)
    ttnn.deallocate(z)
    ttnn.deallocate(tt)
    return out


def _install_pair_state_draw(model):
    """Start `_PairStateDraw` at forward entry; `_init_pair_state` takes the result."""
    import functools
    import inspect
    import types

    draw = _PairStateDraw()
    fwd = model.forward
    sig = inspect.signature(fwd)
    c_z = model.config.d_pair

    @functools.wraps(fwd)
    def forward(*args, **kwargs):
        mask = sig.bind(*args, **kwargs).arguments["token_attention_mask"]
        if PAIR_STATE_THREAD:
            draw.start((mask.shape[0], mask.shape[1], mask.shape[1], c_z))
        return fwd(*args, **kwargs)

    model.forward = forward
    model._init_pair_state = types.MethodType(lambda self, ref: draw.take(ref), model)


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
            x_pair = z_init.t if isinstance(z_init, _DevPair) else z_init
            msa_pair = self.msa_encoder(x_pair=x_pair, **msa_kw).to(z_init.dtype)
            dram_peak("esmfold2/msa-encoder")
            return msa_pair if overwrite else (_host(z_init) + msa_pair)

        # The MSA encoder reads z_init and the loop's MSA draw. With one draw for every loop
        # (an MSA no deeper than max_depth) its output is the same in every loop, so it is
        # encoded once even when the LM dropout makes the rest of the injection per-loop.
        msa_fixed = msa_inject(msa_loops[0]) if msa_loops is not None and len(msa_loops) == 1 \
            else None
        if msa_fixed is not None and overwrite and isinstance(z_init, _DevPair):
            ttnn.deallocate(z_init.t)  # nothing reads z_init past the MSA encoder

        def inject_proj(msa_kw):
            refined = lm_refined() if use_lm else None
            z_inject = _host(z_init)
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
        # The injection is loop-invariant only when no loop redraws its LM dropout or MSA rows.
        per_loop = (use_lm and bool(lm_p)) or (msa_loops is not None and len(msa_loops) > 1)
        # A per-loop MSA draw is encoded on the device each loop (overwrite: the MSA pair IS the base).
        msa_dev = msa_loops is not None and msa_fixed is None
        on_dev = (_DEVICE_INJECTION and per_loop and isinstance(lm_z, _DevPair)
                  and (not msa_dev or overwrite))
        # The first keep mask draws on its thread while the uploads below run (no RNG use in between).
        draws = _KeepDraws(tuple(lm_z.t.shape), lm_p, total_steps) if on_dev else None
        zt = up(padz(z).float())
        ipt = None if per_loop else up(
            padz(inject_proj(msa_loops[0] if msa_loops else None)).float())
        at = ftw._from_torch(a.reshape(1, 1, 1, -1).float())  # parcae a, broadcasts over L,L
        if on_dev:
            # What no loop redraws, uploaded once: the MSA pair (or z_init), LN and b_mat.
            base = None if msa_dev else msa_fixed if msa_fixed is not None else z_init
            if base is None:
                pass
            elif isinstance(base, _DevPair) and pad:
                # Nothing reads the unpadded pair past this point (the caller drops z_init after the loop), and
                # keeping it beside the padded copy is a second [L,L,c] for the whole trunk: on a 12 GiB Wormhole
                # chip that OOMed ESMFold2-Fast --fast at ~800 tokens.
                src = base.t
                base = ttnn.pad(src, [(0, 0), (0, pad), (0, pad), (0, 0)], value=0.0)
                ttnn.deallocate(src)
            elif isinstance(base, _DevPair):
                base = base.t
            else:
                base = up(padz(base).float())
            norm = self.parcae_input_norm
            ln_w, ln_b = (ttnn.from_torch(t.detach().float(), device=ftw.tt_device,
                                          layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                          for t in (norm.weight, norm.bias))
            b_t = ftw._from_torch(b_mat.t().contiguous().float())
            ck = ftw.compute_kernel_config

        def msa_base(msa_kw):
            """This loop's MSA pair on the device, padded like `base`."""
            x_pair = z_init.t if isinstance(z_init, _DevPair) else z_init
            out = self.msa_encoder(x_pair=x_pair, to_host=False, **msa_kw)
            dram_peak("esmfold2/msa-encoder")
            if pad:
                src = out
                out = ttnn.pad(src, [(0, 0), (0, pad), (0, pad), (0, 0)], value=0.0)
                ttnn.deallocate(src)
            return out

        def inject_dev(step):
            x = _lm_dropout(lm_z, lm_p, draws.take())
            refined = _trunk_on_device(self.lm_encoder.m, x, to_host=False)
            if not _aliases(refined, x.t):
                ttnn.deallocate(x.t)
            b = base if base is not None else msa_base(msa_loops[min(step, len(msa_loops) - 1)])
            s = ttnn.add(b, refined)
            ttnn.deallocate(refined)
            if b is not base:
                ttnn.deallocate(b)
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
                ipt = inject_dev(_step)
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
        tail = _device_tail()
        DEVICE_TAIL_STATS[not tail] += 1
        if tail:
            z_out, zt = _DevPair(_unpad(zt, Lp), Lp), None
        else:
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
                   fast: bool = False, lm_b8: bool | None = None):
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
    ``lm_b8`` (default: ``fast``) builds the ESMC-6B weights in block-fp8, independent of the fold's mode.
    """
    sd = {k: v.float() for k, v in model.state_dict().items()}
    comps = _components(sd, model.config)

    model.inputs_embedder = comps["inputs_embedder"]
    model.rel_pos = comps["rel_pos"]
    model.language_model = comps["language_model"]
    model.lm_encoder = comps["lm_encoder"]
    model.folding_trunk = comps["folding_trunk"]
    model.parcae_coda = comps["parcae_coda"]
    model.parcae_readout = _ReadoutAdapter(model.parcae_readout, comps["parcae_coda"].m)
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
    head = model.confidence_head
    head._pair_outputs = lambda *a: _conf_pair_dev(head, conf_trunk, *a)

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
        keep = None
        if isinstance(kw.get("z"), _DevPair):
            z = kw["z"]
            keep, z.keep = z.keep, None
            kw = dict(kw, z=keep if keep is not None else z.host)
        try:
            return _confidence(*args, **kw)
        finally:
            if keep is not None:
                import ttnn
                ttnn.deallocate(keep)

    def _confidence(*args, **kw):
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
    model._esmc = _ESMCAdapter(esmc_repo, persistent=persistent_lm, fast=fast if lm_b8 is None else lm_b8,
                               shim=model.language_model.m)
    model._tt_fast = fast
    model._esmc_fp8 = False

    # Keep the pair state resident on-device across the trunk loop (hoist the
    # loop-invariant LM/MSA/injection work and run the parcae recurrence on
    # device), avoiding per-loop host<->device round-trips of the L²·256 pair.
    _install_resident_trunk_loop(model)
    _install_pair_state_draw(model)
    _install_pair_init(model)
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

    On Wormhole the ESMC-6B is block-fp8 in both modes: in bf16 it alone (~12.8 GB) does not fit
    a chip's 12 GiB. The fold's lever set still follows `fast` (fold_levers).
    """
    from tt_bio import tenstorrent
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
        return patch_esmfold2(model, esmc_repo=esmc_repo, persistent_lm=persistent_lm, fast=fast,
                              lm_b8=fast or is_wormhole())


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
# An MSA fold costs tokens x drawn rows x 1 KiB (the encoder activation; a deeper MSA is subsampled to max_depth
# rows per loop) plus tokens x depth x 256 B for the full alignment. The cap is the deepest point measured with the
# LM resident on a 12 GiB Wormhole chip: 768 tokens x 4096 rows (1.5 GiB by this count) peaked at 9.78 GiB, --fast
# and normal, 256 and 512 tokens x 1024 and 4096 rows below it (lmr, 2026-10-20). The depth slope measured
# 170-256 B per token and row.
WH_LM_RESIDENT_MSA_BYTES = 3 << 29
WH_LM_RESIDENT_MSA_MAX_TOKENS = 768  # an MSA fold above it releases at any depth (pair track + encoder headroom)


def wh_releases_lm(chains, max_depth=None) -> bool:
    """Whether a Wormhole fold of `chains` releases the ESMC-6B after its forward (see fold_complex)."""
    tokens = sum(len(c[1]) for c in chains)
    depth = max((c[2].depth for c in chains if len(c) > 2 and c[2] is not None), default=0)
    if depth:
        drawn = min(depth, max_depth or depth)
        return (tokens > WH_LM_RESIDENT_MSA_MAX_TOKENS
                or tokens * (drawn * 1024 + depth * 256) > WH_LM_RESIDENT_MSA_BYTES)
    return tokens > WH_LM_RESIDENT_MAX_TOKENS


# The ESMC-6B sums the shim's weighted per-layer LayerNorms on the device and hands over one state instead of
# all 81 (each a blocking copy that drains the dispatch pipeline). Same math as the shim, summed before its
# bias-free linear instead of after, in fp32. Unset: the ef2_lm_combine lever.
LM_COMBINE = env_flag("TT_BIO_ESMFOLD2_LM_COMBINE", None)

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
# trimul_gin (in-projection, both gates and both channel moves in one kernel, as channel slices at c_z 256) passes
# the normal bar on ESMFold2, 11 complexes x 4 seeds, 44 paired folds against the set without it (p150a + p300c,
# 1350 MHz): CA-lDDT -0.0017 [-0.0068, +0.0025], top-pose median 0.24 A vs 1.22 A floor, docking 31 -> 30 of 44.
# Warm fold time at 1350 MHz: 256 and 512 aa unchanged (49.4 vs 49.8 s, 131.1 vs 131.5 s), 1024 aa 436 -> 419 s,
# 1536 aa 1028 -> 939 s. On ESMFold2-Fast the 11-set folds 8.4 % faster (normal bar PASS, 44 paired folds).
# trimul_t2g (output layer norm, p projection, g_out gate and residual add in one kernel) takes only the calls
# the F1 tail would serve with the residual folded in, and is bit-exact against it there (torch.equal, both sets,
# p150a; CIF byte-identical in same-card folds at 544, 576 and 1024 aa). p150a 1350 MHz, ESMFold2's call at
# 512: 9.79 -> 9.39 ms (normal set), 9.49 -> 8.66 ms (fast set); --fast fold warm, 544 aa 30.5 -> 29.1 s, 1024 aa
# 98.7 -> 94.0 s. Wormhole (1000 MHz) folds the same CIF in the same time (544 aa 94.4 vs 94.5 s).
# trimul_gin_hifi2 (gin's kernel at HiFi2) passes the normal bar on ESMFold2 against the same set without it,
# 11 complexes x 4 seeds, 44 paired folds, base and test from one tree (p150a, 1350 MHz): CA-lDDT -0.0025
# [-0.0070, +0.0002], pLDDT -0.0008, top-pose median 0.67 A vs 1.16 A floor, docking 32 -> 30 of 44. Warm fold at
# 1350 MHz: 544 aa 26.8 -> 25.8 s, 1024 aa 89.7 -> 87.2 s.
# Four ef2_* levers (bit draw, device pair init, relpos lookup, atom FFN split) pass ESMFold2's normal bar as a stack
# against the set without them, 11 complexes x 4 seeds, 44 paired folds (p300c, 1350 MHz): CA-lDDT -0.0001
# [-0.0011, +0.0008], TM +0.0092, top-pose median 0.33 A vs 1.15 A seed floor, docking 31 -> 31 of 44. The other
# ef2_lm_combine fails it alone (DockQ, L-RMSD and I-RMSD CIs exclude 0) and stays fast-only.
# trimul_bres (the trimul einsum with b[c] resident in L1, bmm_res) is a pure speed lever in both modes on both archs:
# coords are byte-identical with and without it in ESMFold2 folds under LEVERS_NORMAL and LEVERS_FAST (9DBP, 9LLG x 2
# seeds, p300c, spd-gap-whesmf wbxbh), the Wormhole fast set (11 complexes x 4 seeds x 2 models, p150a, wbgbh) and the
# Wormhole normal set (Galaxy chip, wbxe). It declines above 704 tokens (bf16 b) and the stock matmul runs. 512 aa
# --fast, same chip: p150a ESMFold2 14.47 -> 14.12 s, ESMFold2-Fast 9.06 -> 8.72 s (1350 MHz, wbe3bh); Wormhole
# ESMFold2 30.36 -> 29.24 s, ESMFold2-Fast 18.25 -> 17.65 s (1000 MHz, wbx).
LEVERS_NORMAL = ("normal-trunk_hifi3+trimul_gin+trimul_t2g+trimul_gin_hifi2"
                 "+ef2_bit_draw+ef2_pair_init+ef2_relpos_lookup+ef2_atom_ffn_split+trimul_bres+transition_ws+ws_resid"
                 "+trimul_t1+trimul_t1_c256+trimul_t1w8+trimul_t1_2g+trimul_t1_psig+trimul_t2+trimul_t2cm+ws_ln_resid+dit_qkvf")
# ws_ln_resid (described above _BH_ONLY) passes the normal bar on Blackhole against the set without it, 11 complexes x
# 4 seeds, 44 paired folds per model (qb2, bwlng): ESMFold2 CA-lDDT +0.0034 [+0.0006, +0.0065], top-pose median 0.41 A
# vs 1.17 A seed floor, docking 31 -> 31 of 44; ESMFold2-Fast +0.0011 [+0.0002, +0.0022], 0.89 A vs 1.50 A, 24 -> 24.
# In-process A/B on the Blackhole normal set, repeat 5 (bwln): ESMFold2 512 aa -0.20 s, ESMFold2-Fast 1024 aa -0.45 s.
# dit_qkvf (described above _WH_FAST_EXTRA) passes the normal bar on Blackhole against the set without it, 11 complexes
# x 4 seeds, 44 paired folds per model (qb2, bqng): ESMFold2 CA-lDDT +0.0007 [-0.0002, +0.0016], top-pose median 0.59 A
# vs 1.23 A seed floor, docking 31 -> 31 of 44; ESMFold2-Fast -0.0004 [-0.0011, +0.0003], 0.63 A vs 1.66 A, 24 -> 24.
# In-process A/B on the Blackhole normal set, repeat 5, p150a at 1256-1350 MHz (bqn): ESMFold2 512 aa 12.47 -> 12.05 s,
# ESMFold2-Fast 512 aa 7.79 -> 7.32 s. Wormhole's normal set drops it (its ESMFold2 L-RMSD reading, see there).
# Blackhole's normal mode takes the --fast trimul route (T1 front with bfp8 weights, T2 tail; Wormhole drops it,
# _BH_ONLY). Normal bar against the set without it, 11 complexes x 4 seeds, 44 paired folds per model (qb1/qb2, bt1ng):
# ESMFold2 CA-lDDT -0.0009 [-0.0051, +0.0019], top-pose median 0.54 A vs 1.85 A seed floor, docking 32 -> 31 of 44;
# ESMFold2-Fast -0.0014 [-0.0033, +0.0002], 0.58 A vs 1.64 A, docking 24 -> 24. In-process A/B, repeat 4, 1318-1350 MHz
# (bt1n): ESMFold2 512 aa 14.46 -> 12.78 s, ESMFold2-Fast 512 aa 8.86 -> 7.90 s, 1024 aa 31.75 -> 29.43 s. T1 with the
# gin-era tail loses 0.5-2.9 s; the T2 tail is what makes T1 pay.
# Confirmed on fresh seeds 105-108 (bt1nc, 11 x 4): ESMFold2 CA-lDDT -0.0011 [-0.0049, +0.0022], docking 29 -> 30 of 44;
# ESMFold2-Fast -0.0004 [-0.0014, +0.0005], 24 -> 24 of 43.
# Normal mode runs the pair SwiGLU kernel (transition_ws + ws_resid) on both archs. Without it the pair FFN runs
# row-blocked, ~25k ttnn.linear calls a 512 aa ESMFold2 fold on Blackhole. Normal bar on Blackhole against the set
# without both, 11 complexes x 4 seeds, 44 paired folds per model (p300c, 1350 MHz, bnwg): ESMFold2 CA-lDDT -0.0018
# [-0.0048, +0.0015], top-pose median 0.84 A vs 1.20 A seed floor, docking 31 -> 32 of 44; ESMFold2-Fast CA-lDDT
# +0.0003 [-0.0011, +0.0018], 0.77 A vs 1.73 A, docking 24 -> 24. Same card, in-process A/B, warm, repeat 4 (bnwr):
# ESMFold2 512 aa 18.82 -> 14.76 s, ESMFold2-Fast 512 aa 10.92 -> 8.91 s, 1024 aa 43.5 -> 33.1 s.
# FAST also takes transition_ws (fast grade PASS on both ESMFold2 models, both archs; tenstorrent.FAST_LEVERS)
# and trimul_gin: ESMFold2 fast bar against the set without it, 11 complexes x 4 seeds, 44 paired folds (p300c,
# 1350 MHz): CA-lDDT +0.0026 [-0.0022, +0.0086], pLDDT +0.0005, top-pose median 0.35 A vs 1.30 A floor, docking
# 31 -> 30 of 44. Same-card A/B at 544 and 576 tokens (p150a, 1350 MHz): 34.2 -> 33.1 s and 40.7 -> 38.4 s warm.
# The ef2_* stack (see esmfold2.gate) is graded on the fast bar: ESMFold2-Fast --fast vs the set without it, 11 complexes
# x 4 seeds, 44 paired folds (p300c, 1350 MHz): CA-lDDT -0.0000 [-0.0003, +0.0002], TM +0.0013, docking 24 -> 24;
# ESMFold2 normal-mode folds of the stack also clear the fast bar (CA-lDDT -0.0019). 512 aa --fast on the stack:
# ESMFold2 16.92 s, ESMFold2-Fast 10.14 s (p300c, 1350 MHz). On Wormhole the stack without ef2_pair_init passes the
# same fast bar (ESMFold2-Fast, 44 paired folds, 1000 MHz): CA-lDDT -0.0000 [-0.0025, +0.0019], docking 24 -> 24 of
# 44, and folds 772, 778 and 1024 tokens in both modes on both models. ef2_pair_init joined Wormhole's fast set later
# (see _BH_ONLY).
# ws_resid (the pair SwiGLU adds into z in fc3's fp32 dest) passes the fast bar on ESMFold2-Fast against the shipped
# --fast set, 11 complexes x 4 seeds, 44 paired folds (Blackhole, 1350 MHz): CA-lDDT +0.0008 [-0.0006, +0.0032],
# TM +0.0073, top-pose median 0.55 A vs 1.65 A seed floor, docking 23 -> 24 of 44. p150a 512 aa: ESMFold2 -0.44 s,
# ESMFold2-Fast -0.19 s. On Wormhole the fast set keeps ws_resid and adds trimul_gin_hifi2, graded together on the fast
# bar against the set without both, 11 complexes x 4 seeds, 44 paired folds per model (Galaxy chip, 1000 MHz):
# ESMFold2 CA-lDDT -0.0004 [-0.0032, +0.0025], top-pose median 0.94 A vs 1.84 A seed floor, docking 31 -> 31 of 44;
# ESMFold2-Fast CA-lDDT -0.0002 [-0.0015, +0.0008], docking 24 -> 24. Same-chip A/B, ESMFold2 512 aa --fast: ws_resid
# -1.1 s, trimul_gin_hifi2 -0.8 s.
# ws_lofi (the pair SwiGLU at LoFi) and trimul_gin_hifi2 pass the fast bar as a stack on Blackhole against the set
# without both, 11 complexes x 4 seeds, 44 paired folds per model (p150a/p300c, 1350 MHz): ESMFold2 CA-lDDT +0.0003
# [-0.0012, +0.0023], DockQ +0.015, top-pose median 0.67 A vs 1.39 A seed floor, docking 33 -> 33 of 44; ESMFold2-Fast
# CA-lDDT +0.0016 [+0.0001, +0.0034], docking 24 -> 24. p150a 512 aa --fast warm at 1350 MHz: ws_lofi ESMFold2 16.80 ->
# 15.80 s, ESMFold2-Fast 10.41 -> 10.17 s; trimul_gin_hifi2 -0.28 / -0.20 s. Wormhole drops ws_lofi (not graded there).
# Wormhole used to add trimul_t2m (T2 reads the einsum's last chunk channel-major, so the permute-back pass never
# writes; same digests). It won 1.0-1.5 s before trimul_bres; with bres feeding the tail it costs time: the T2m tail
# runs 7.2 ms a call against 3.9 ms for plain T2 plus 2.2 ms for the permute it saves. 512 aa --fast, D M D M on one
# Galaxy chip at 1000 MHz (spd-gap-whesmf wtm): ESMFold2 31.10 -> 29.49 s, ESMFold2-Fast 18.93 -> 18.15 s, pLDDT equal.
# Blackhole --fast also runs the trimul as T1 (in-projection, c_z 256 plan included) and T2 with the channel-major
# einsum read (trimul_t1 + t1_c256 + t2 + t2cm), graded on the fast bar against the set without them, 11 complexes x
# 4 seeds per model (p300c, 1350 MHz): ESMFold2 CA-lDDT +0.0006 [-0.0001, +0.0014], ipTM +0.0051, top-pose median
# 0.14 A vs 1.17 A seed floor, docking 33 -> 33 of 44; ESMFold2-Fast CA-lDDT -0.0009 [-0.0021, +0.0000], top-pose
# median 0.17 A vs 1.36 A, docking 24 -> 24. 512 aa --fast warm at 1350 MHz, A C A C: ESMFold2 15.75 -> 15.49 s,
# ESMFold2-Fast 9.56 -> 9.31 s. Wormhole loses 2.4 s with them, so it drops them (_BH_ONLY).
LEVERS_FAST = ("normal+trimul_b8in+transition_ws+trimul_gin+trimul_t2g+trimul_gin_hifi2+ws_resid+ws_lofi"
               "+ef2_bit_draw+ef2_pair_init+ef2_lm_combine+ef2_relpos_lookup+ef2_atom_ffn_split"
               "+trimul_t1+trimul_t1_c256+trimul_t2+trimul_t2cm+trimul_bres+trimul_t1w8+trimul_t1_2g+trimul_t1_psig+dit_qkvf+ws_ln_resid")
# ws_lofi passes the fast bar on Wormhole too, 11 complexes x 4 seeds, 44 paired folds per model (Galaxy chip, 1000
# MHz, spd-gap-esmf wlg): ESMFold2 CA-lDDT -0.0004 [-0.0024, +0.0015], DockQ +0.017, docking 31 -> 32 of 44;
# ESMFold2-Fast CA-lDDT -0.0001 [-0.0021, +0.0023], docking 24 -> 24. Same chip, 512 aa, 1000 MHz: ESMFold2 33.60 ->
# 33.10 s, ESMFold2-Fast 20.76 -> 20.38 s (spd-gap-whesmf wm, two runs per arm). It costs mean plDDT: ESMFold2-Fast
# cdk2x2_512 at 10 recycles x 100 steps, seeds 0,1 against the fp32 reference, reads a same-seed gap of 0.0167 with it
# and 0.0018 without (one Galaxy chip, 1000 MHz, spd-gap-whesmf wpl; ef2_pair_init moves it 0.0004). pTM 0.792 vs 0.807,
# Kabsch inside the seed floor both ways. Within the fast bar (0.03), which is what the release leg grades --fast at.
# ef2_pair_init passes the fast bar on Wormhole as well, against the WH fast set without it, 11 complexes x 4 seeds, 44
# paired folds per model (Galaxy chip, 1000 MHz, spd-gap-whesmf wpg): ESMFold2 CA-lDDT +0.0015 [-0.0004, +0.0043],
# top-pose median 0.55 A vs 1.73 A seed floor, docking 32 -> 31 of 44; ESMFold2-Fast CA-lDDT +0.0002 [-0.0005, +0.0009],
# 1.00 A vs 1.03 A, docking 24 -> 24. The 1024-token confidence-head DRAM overflow that kept it off no longer reproduces:
# both models fold 1024 and ESMFold2 1536 with it. Same chip, 512 aa, 1000 MHz: ESMFold2 34.0 -> 32.5 s, ESMFold2-Fast
# 21.8 -> 20.1 s (wpi). Wormhole's normal set was graded without it (wng), so it stays fast-only there.
# trimul_t1w8 holds T1's resident weights bfp8, so c_z 256 runs in one launch and reads the pair once instead of per
# 128-channel half; trimul_t1_2g lets T1 hold 2 GiB of outputs, so it serves c_z 256 up to ~1152 tokens instead of
# declining above ~900. In-process A/B on the Blackhole --fast set, repeat 4, p150a at 1350 MHz, T1 served in both arms
# (bt1w): t1w8 ESMFold2 512 aa 13.63 -> 12.59 s, ESMFold2-Fast 512 aa 8.75 -> 8.00 s (4 of 4 rounds faster each);
# t1_2g ESMFold2-Fast 1024 aa 30.06 -> 29.54 s, no-op at 512.
# Fast-bar grade bt1g (11 complexes x 4 seeds, qb1/qb2): ESMFold2 CA-lDDT -0.0006 [-0.0032, +0.0025], docking 31 -> 31
# of 44; ESMFold2-Fast -0.0002 [-0.0007, +0.0001], 24 -> 24. At 1024 tokens t1_2g's same-seed CA RMSD sits inside the
# seed spread on both models (bt2s).
# dit_qkvf (its Wormhole grade is in the notes above _WH_FAST_EXTRA) on Blackhole --fast, fresh processes A B A B, p150a at 1350
# MHz (bqkd): diffusion draw ESMFold2-Fast 512 aa 1.98 / 2.04 -> 1.44 / 1.50 s, ESMFold2 1.91 / 2.03 -> 1.47 / 1.67 s.
# Fast-bar grade bqkg (11 complexes x 4 seeds, qb1/qb2, shipped Blackhole --fast set vs + dit_qkvf): ESMFold2 CA-lDDT
# +0.0011 [+0.0001, +0.0021], docking 31 -> 32 of 44; ESMFold2-Fast -0.0002 [-0.0008, +0.0004], 23 -> 23 of 43.
# trimul_t1_psig runs T1's gate sigmoid on the packer thread, overlapping the FPU's next matmul block. Same
# sigmoid_poly on the same values, so a, b and g_out are bit-identical (Protenix-v2 took it on the same c_z 256 T1).
# p150a 1350 MHz, CIFs identical on every fold: T1 3.31 -> 3.06 ms a call at 512; paired fold median ESMFold2 512
# --fast -0.23 s, normal -0.34 s; -Fast 512 --fast -0.02 s, normal +0.02 s, 256 --fast -0.01 s (repeat 9, noise).
# ws_ln_resid moves the pair transition's input LN into the SwiGLU kernel's prologue, where ws_resid adds into the same
# z. Blackhole only: Wormhole's plan with the prologue is over its L1 budget, so the lever declines there. In-process
# A/B on the Blackhole --fast set, repeat 5, p150a at 1275-1350 MHz (bwlr2): ESMFold2 512 aa -0.26 s, ESMFold2-Fast
# 512 aa -0.13 s, 1024 aa -0.52 s. Fast-bar grade bwlg (11 complexes x 4 seeds, qb2, the Blackhole --fast set vs +
# ws_ln_resid): ESMFold2 CA-lDDT +0.0009 [-0.0019, +0.0038], docking 32 -> 31 of 44; ESMFold2-Fast -0.0005 [-0.0029,
# +0.0018], top-pose median 0.98 A vs 1.42 A seed floor, docking 24 -> 24.
_BH_ONLY = ("trimul_t1", "trimul_t1_c256", "trimul_t2", "trimul_t2cm", "trimul_t1w8", "trimul_t1_2g", "trimul_t1_psig", "ws_ln_resid")
# Wormhole's normal mode folds with the LM in bfp8 (load_ttnn_esmfold2) and adds ef2_lm_combine to LEVERS_NORMAL.
# The pair SwiGLU kernel was Wormhole-only in normal mode until bnwg (above); its Wormhole grade:
# Without transition_ws ESMFold2's pair FFN runs row-blocked, 65 ms a call against 10 ms, and a 512 aa fold takes
# 50.9 s instead of 34.7 (Galaxy chip, 1000 MHz, spd-gap-whesmf wl). Normal bar on Wormhole, against LEVERS_NORMAL with the LM
# in bfp8: ESMFold2 CA-lDDT +0.0013 [-0.0006, +0.0039], top-pose median 0.73 A vs 1.73 A floor (44 paired folds, wng);
# ESMFold2-Fast CA-lDDT -0.0004 [-0.0015, +0.0009], 1.24 A vs 1.33 A, docking 48 -> 48 of 88 (88 paired folds over
# two seed sets, wng + wng3). Galaxy chip, 1000 MHz.
# lm_qk_chain keeps the 6B's attention off qk_rope_heads (esmc._qkrh, on for Wormhole elsewhere): ESMFold2 normal with
# it against without, 11 complexes x 4 seeds, 44 paired folds (Galaxy chip, 1000 MHz, spd-gap-whlm qrn) fails the normal
# bar, L-RMSD +1.77 A [+0.10, +3.92] and I-RMSD +0.30 A [+0.02, +0.74], though the top-pose median is 0.56 A against a
# 1.20 A seed floor and docking 31 -> 31 of 44. ESMFold2-Fast --fast passes the fast bar with it (qrf, 44 paired folds:
# CA-lDDT -0.0007 [-0.0031, +0.0011], top-pose median 0.79 A vs 1.26 A floor, docking 24 -> 24), so --fast keeps it.
# Normal mode on Wormhole also takes the --fast trimul pair (trimul_t2fold, trimul_gin_w8). With dit_shoist and
# trunk_hifi3 on top the stack failed ESMFold2's normal bar on DockQ (wnsg, -0.011 [-0.022, -0.002]); each half alone
# passed it (wnsp: gin_w8 +0.014 [-0.008, +0.045], t2fold +0.018 [+0.001, +0.044], docking 30 -> 32 of 44), and the
# pair is graded as the stack it ships as (wnp2). The pair carries the speed (wnx3, 1000 MHz, 512 aa): ESMFold2
# 26.44 -> 25.27 s, ESMFold2-Fast 16.33 -> 15.64 s for the four-lever stack.
_WH_NORMAL_EXTRA = "+ef2_lm_combine+lm_qk_chain+trimul_t2fold+trimul_gin_w8"
# lm_gate_ng keeps the 6B's SwiGLU gate on binary_ng in both modes (esmc._lm_gate_legacy, on for Wormhole elsewhere):
# the legacy multiply is graded on ESMC's embeddings (6B pcc vs fp32 -0.00002 / +0.00001), not yet on a fold.
# ef2_atom_rope is exact (rotate_half as a +-1 permutation matmul): ESMFold2-Fast 512 aa --fast CIF identical with and
# without it, warm fold 16.53 -> 16.26 s mean of 3 pairs (Galaxy chip, 1000 MHz, spd-gap-whesmf wrp2).
# ws_psilu runs the pair SwiGLU's silu on the packer thread, overlapping the matmuls: CIFs identical with and without
# it. In-process A/B, 512 aa --fast, repeat 3, Galaxy chip at 1000 MHz (wps6): ESMFold2 25.32 -> 24.54 s, ESMFold2-Fast
# 15.07 -> 14.57 s, -Fast 256 aa 5.32 -> 5.22 s; normal mode (wpsn) ESMFold2 25.78 -> 25.07 s, -Fast 15.25 -> 14.81 s.
# Wormhole only: on Blackhole -Fast 512 aa --fast reads +0.15 s at 1350 MHz.
# atom_l1 writes the denoise step's linears to L1 where `_atom_l1` admits the output (placement only, so CIFs are
# identical with and without it). In-process A/B, repeat 3, Galaxy chip at 1000 MHz on every sample (wl1), warm pairs:
# ESMFold2-Fast 512 aa --fast -0.30 / -0.47 s, normal -0.17 / -0.13 s, 1024 aa --fast -0.38 / -0.40 s; ESMFold2 512 aa
# normal -0.21 / -0.16 s.
_WH_EXTRA = "+lm_gate_ng+ef2_atom_rope+ws_psilu+atom_l1"
# dit_qkvf runs the token DiT's attention as one `_sdpa32_fused` program off a folded q|k|v projection. Fast bar on
# Wormhole against the WH fast set without it, 11 complexes x 4 seeds, 44 paired folds per model (Galaxy chip, 1000
# MHz, spd-bet-ditpersist dsg): ESMFold2-Fast CA-lDDT -0.0002 [-0.0006, +0.0003], top-pose median 0.55 A vs 1.46 A
# seed floor, docking 24 -> 24 of 44; ESMFold2 --fast CA-lDDT +0.0016 [+0.0009, +0.0022], 0.69 A vs 1.33 A, docking
# 32 -> 32 (its L-RMSD reads +2.5 A [+0.4, +5.4], so Wormhole's normal set does not take it). Same chip, -Fast, 1000 MHz:
# diffusion draw 2.52 -> 1.97 s at 384 aa, 3.04 -> 2.15 s at 512 (dst).
# dit_shoist (the DiT conditioning as three wide linears a step) on top, fast bar on Wormhole against that set, 11
# complexes x 4 seeds per model (.107, 1000 MHz, dsq2): ESMFold2-Fast CA-lDDT +0.0005 [-0.0003, +0.0016], top-pose
# median 0.51 A vs 1.32 A seed floor, docking 24 -> 24 of 44, fold -0.73 s median at 515-791 aa (10 of 11 faster);
# ESMFold2 --fast (40 folds) CA-lDDT -0.0004 [-0.0008, -0.0000], 0.39 A vs 1.43 A, docking 32 -> 32.
# trunk_hifi2 on top, fast bar on Wormhole against that set, 11 complexes x 4 seeds per model (.107, 1000 MHz, dh2):
# ESMFold2-Fast CA-lDDT -0.0006 [-0.0027, +0.0013], top-pose median 0.82 A vs 1.19 A seed floor, docking 24 -> 24 of
# 44; ESMFold2 --fast CA-lDDT -0.0004 [-0.0024, +0.0010], 0.56 A vs 1.58 A, docking 32 -> 33. 512 aa: -0.13 / -0.15 s.
# trimul_t2fold on top (the GATE-form tail folds its LN's centring and gamma into Wp), fast bar on Wormhole against
# that set (.107, 1000 MHz, wt2fg2): ESMFold2-Fast 11 complexes x 4 seeds, CA-lDDT +0.0010 [-0.0004, +0.0029], top-pose
# median 0.71 A vs 1.19 A seed floor, docking 24 -> 24 of 44, fold -0.4 % median at 515-791 aa (-3 % at 772+);
# ESMFold2 --fast 9 complexes x 4 seeds, CA-lDDT +0.0009 [-0.0009, +0.0035], 0.53 A vs 1.46 A, docking 27 -> 27 of 36,
# fold -3.2 % median (9 of 9 faster). In-process A/B (wt2f): ESMFold2 512 -0.47 s, -Fast 1024 -1.16 s.
# trimul_bres_b8 on top holds the trimul einsum's b in L1 as bfp8 where bf16 would split the 704+ token einsum, so a is
# read once; below 704 tokens nothing changes. Fast bar on Wormhole against the set without trimul_t2fold (.107, 1000 MHz,
# wbfb4, 9DBP/9PCQ/9W89/9W8A x 4 seeds): ESMFold2-Fast CA-lDDT -0.0003 [-0.0009, +0.0001], docking 8 -> 8 of 16;
# ESMFold2 --fast CA-lDDT +0.0004 [-0.0086, +0.0098], docking 10 -> 10 of 16. In-process A/B (wb8f3): -Fast 1024
# -1.83 s (-3.3 %), 768 -1.38 s, ESMFold2 1024 -2.91 s.
# trimul_gin_w8 holds gin_moved's resident weights bfp8, so c_z 256 runs in one launch and reads the normed pair once
# instead of per 128-channel slice. In-process A/B on the WH --fast set, repeat 3, Galaxy chip at 1000 MHz (wgw8):
# ESMFold2 512 aa 25.11 -> 24.63 s, -Fast 512 14.74 -> 14.53 s, -Fast 256 5.30 -> 5.17 s, -Fast 1024 52.62 -> 52.47 s.
# Fast-bar grade wgw8g (same chip, WH --fast set vs + this, 4 seeds x 5 samples): -Fast 11 complexes CA-lDDT +0.0008
# [-0.0002, +0.0022], docking 23 -> 24 of 44; ESMFold2 10 complexes +0.0028 [-0.0000, +0.0073], docking 28 -> 27 of 40.
_WH_FAST_EXTRA = "+dit_shoist+trunk_hifi2+trimul_t2fold+trimul_bres_b8+trimul_gin_w8"
# lm_sdpa_stock keeps the LM's stock SDPA over the full mask on both arches (esmc._key_lens, shsdpa elsewhere);
# lm_ln_stock its ttnn.layer_norm where ESMC runs ln_split (esmc._ln).
_LM_STOCK = "+lm_sdpa_stock+lm_ln_stock"
# The ESMC language model on Blackhole as ESMFold2 folded it before ESMC took the Wormhole LM path there: graded on
# ESMC's embeddings only, not on a fold yet.
_BH_LM_KEEP = "+lm_qk_chain+lm_gate_ng+lm_k64+lm_linear"


def fold_levers(fast: bool) -> str:
    """The lever set ESMFold2 builds and folds under: TT_BIO_LEVERS when a harness grades a set through the
    serving path (no user sets it), else the mode's."""
    if os.environ.get("TT_BIO_LEVERS"):
        return os.environ["TT_BIO_LEVERS"]
    levers = LEVERS_FAST if fast else LEVERS_NORMAL
    if is_wormhole():
        # ws_lofi too, since 2026-10-20: LoFi hangs Wormhole chips (tenstorrent.lofi_allowed)
        drop = _BH_ONLY + ("ws_lofi",) if fast else _BH_ONLY + ("ef2_pair_init",)
        levers = "+".join(n for n in levers.split("+") if n not in drop)
        levers += (_WH_FAST_EXTRA if fast else _WH_NORMAL_EXTRA) + _WH_EXTRA
    else:
        levers += _BH_LM_KEEP
    return levers + _LM_STOCK


def fold_scope(fast: bool):
    """The context ESMFold2 builds and folds in: `lever_scope(fold_levers(fast))`, the older bfp8 `_FAST_MODE` off,
    on both archs. Wormhole used to keep `_FAST_MODE` (LEVERS_NORMAL + transition_ws) after the Blackhole path's
    --fast threw a circular-buffer clash at 546-559 tokens (program 217); that clash no longer reproduces (552 folds
    clean, 2026-10-13). On a Wormhole Galaxy chip at 1000 MHz the Blackhole path folds 512 aa in 43.4 s vs 69.6 s
    (ESMFold2) and 30.5 vs 46.9 s (ESMFold2-Fast). Graded against the old path, 11 complexes x 4 seeds, fast and
    normal bar: ESMFold2-Fast CA-lDDT +0.0003 [-0.0010, +0.0023], pLDDT -0.002, top-pose median 1.08 A vs 1.36 A
    seed floor, docking 24/44 both. ESMFold2: CA-lDDT +0.0034 [+0.0004, +0.0078], pLDDT +0.0005, top-pose median
    0.82 A vs 1.71 A seed floor, docking 30 -> 33 of 44. Both models fold clean at 768, 1088 and 1536 tokens on the
    Blackhole path, 24-33 % faster than the old one at each size."""
    from tt_bio import tenstorrent
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
    # M=8192 and grows with L and M: those MSA folds died in the encoder with ~70 MiB of
    # contiguous DRAM left. Release the language model after its single forward (it runs
    # once per fold, outside the recycling loop) and pay one reload from host tiles (_ESMCAdapter._tiles).
    # Bit-exact: same weights, same dtype, same order, only the device buffers move.
    # Blackhole (32 GB) always keeps the LM resident, and so does a Wormhole fold without an
    # MSA up to WH_LM_RESIDENT_MAX_TOKENS, or with one up to WH_LM_RESIDENT_MSA_MAX_TOKENS whose
    # footprint stays within WH_LM_RESIDENT_MSA_BYTES. Since the encoder draws max_depth rows a loop, that covers
    # JapanFold's typical deep MSAs up to 768 aa, so they no longer pay the ~4 s reload.
    #
    # The same holds without an MSA once the pair track is large: single-sequence esmfold2 folded
    # 1088 tokens on the Galaxy with the LM resident and died at 1152 in the MSA encoder's pair FFN
    # with 10.6 MiB per bank free, the LM holding ~537 of each bank's 1024 MiB. One fold per process
    # pays nothing for the release; a server folding many large targets pays one reload each.
    esmc = getattr(model, "_esmc", None)
    release_lm = (esmc is not None and getattr(esmc, "_persistent", False)
                  and is_wormhole()
                  and wh_releases_lm(chains, getattr(getattr(model.config, "msa_encoder", None), "max_depth", None)))
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
