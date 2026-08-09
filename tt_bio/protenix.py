"""Protenix-v2 (ByteDance AF3 reproduction) modules on Tenstorrent.

Protenix-v2 is the same AF3 family as Boltz-2 (already ported in boltz2.py) and
shares tt_bio.tenstorrent primitives (AttentionPairBias, AdaLN,
ConditionedTransitionBlock, PairformerLayer, Transition). This module adds the
genuinely-new v2 pieces, built component-by-component and validated against the
real v2 reference (see scripts/protenix_*.py and tests/test_protenix.py).

Status (all on-device, validated vs real v2 golden; see tests/test_protenix_*.py):
- AtomFeaturization (c_l, p_lm)                          PCC > 0.9999
- AtomTransformer (3-block windowed atom attention)      PCC 0.999998
- AtomAttentionEncoder -> s_inputs (full InputFeatureEmbedder atom encoder) PCC 0.999999
- TrunkInput -> s_init, z_init                           PCC 0.999997
- 48-block Pairformer stack vs real trunk I/O            PCC s 0.993 / z 0.980
- full 10-cycle trunk (assembled)                        PCC s 0.991 / z 0.990
- DiffusionConditioning (pair/single)                    PCC 1.0 / 0.99999
- diffusion atom encoder(has_coords)                     PCC 0.99999
- 24-block token DiT (per-block)                         PCC 1.0 (torch) / 0.997 (bf16)
- diffusion atom decoder                                 PCC 0.99992
- ConfidenceHead (pae/pde ; plddt/resolved)              PCC 1.0 ; 0.93/0.77
EVERY v2 compute module validated on-device. ASSEMBLED into the top-level Protenix
class (load_from_checkpoint + fold): full on-device pipeline (atom encoder -> diffusion
atom cache -> 10-cycle Trunk -> diffusion conditioning -> EDM sampler) produces valid
structures within sample variance of the reference (scripts/protenix_fold_e2e.py,
scripts/protenix_predict.py -> PDB). Remaining (packaging): data-pipeline vendoring
(sequence/CCD -> feats dict), worker/CLI --model protenix-v2, unified README.
"""
import os
import torch
import ttnn

from . import protenix_weights as PW
from .protenix_weights import remap_adaln  # single source of all v2->tt-bio weight remaps
from .tenstorrent import (Module, CORE_GRID_MAIN, get_device, dram_peak,
                          MSA_CHUNK_SIZE)


# How many diffusion samples a single batched denoise carries by default. The batched
# denoise holds a (chunk, n_heads, NT, NT) attention score tensor per DiT block, so the
# per-step footprint grows linearly in the chunk -- 80 MB per sample per block at
# NT=1095 in fp32. Defaulting an omitted chunk size to "all n_sample at once" made a
# 50-sample fold of a 1095-token target ask for 4 GB in one allocation. Callers that
# know their headroom can raise it; `tt-bio predict --max_parallel_samples` exposes it.
DEFAULT_MAX_PARALLEL_SAMPLES = 5

# Row-chunk the trunk's MSA representation once it exceeds this many bytes.
#
# Set so that no target is left near the whole path's OOM boundary, because that boundary cannot
# be predicted from this tensor's size. Measured on AbAg-XM:
#     9lwc  0.618 GiB, 285 tokens  -> folds on the whole path
#     9d72  1.783 GiB, 682 tokens  -> refused 1909293056 B on the whole path
# The whole path's peak is driven by PairWeightedAveraging's intermediates, which scale with
# tokens^2; `m_feat` only correlates with that. So a size threshold near the boundary is tuning a
# variable that is not the cause. 0.25 GiB is below the panel's smallest target (0.618 GiB), which
# means every AbAg-XM target chunks and none of them sits near the edge.
#
# Deciding this from *free DRAM at the start of the block* was tried and REVERTED -- see
# _msa_take_whole_path. Chunking m_feat is bit-exact either way (Trunk._msa.update_msa, verified
# by the 9lwc md5 check), so this threshold trades only speed, never numbers.
#
# Cost, measured, and NOT a flat multiplier -- the chunked path's fixed cost scales with the
# target: 9lwc 194 s, 9d72 836 s. 125 of 164 targets already exceeded the old 1.0 GiB gate, and
# the remaining 39 (0.618-0.999 GiB) now chunk too. Chunking all 164 also makes the dataset
# uniform: one code path for every target rather than a mix differing by one bf16 step.
#
# NOTE this threshold does NOT make every target fit. 9lof (2.603 GiB, 754 tokens) chunks and
# still OOMs, in OuterProductMean rather than here -- see OuterProductMean in tenstorrent.py.
#
# A Blackhole part has 32 GiB and has never needed this, but deep-MSA targets there will now
# chunk too. That is numerically inert; it is NOT perf-measured on Blackhole yet.
MSA_ROW_CHUNK_BUDGET_BYTES = 1 << 28      # 0.25 GiB

# Above this size the pristine m_feat lives on the HOST between recycling cycles and is
# streamed up one depth-chunk at a time (see Trunk.__call__). A 1 GiB pristine held for all
# 10 cycles is affordable; the 2.5-3.2 GiB ones the WH-DRAM-blocked targets carry are not --
# the per-cycle peak pays pristine + updated copy at once. 117/298-aa targets are far below
# the gate, so their path (and perf) is untouched.
MSA_HOST_OFFLOAD_MIN_BYTES = 1 << 30      # 1 GiB


def _msa_host_offload_min_bytes():
    """Offload gate, with a test-only env override (same rationale as the chunk budget):
    forcing it to 0 makes a SMALL target take the host-streamed path so its output can be
    compared byte-for-byte against the same target folded device-resident."""
    v = os.environ.get("TT_BIO_MSA_HOST_OFFLOAD_MIN_BYTES")
    return int(v) if v else MSA_HOST_OFFLOAD_MIN_BYTES


def _msa_take_whole_path(nbytes):
    """Can the full-depth MSA path afford `nbytes`, or must it row-chunk?

    A plain threshold, deliberately, after the alternative was measured and failed.

    Deciding from real free DRAM -- `get_memory_view(...).total_bytes_free_per_bank * num_banks`
    at the top of the block, requiring some multiple of `nbytes` to be free -- looks strictly
    better and is not. The check point and the allocation point are far apart: PWA allocates its
    own projections and per-head weights in between, so free-at-check-time overstates
    free-at-allocation-time by more than the margin. Measured, with a 3x requirement: 9d72's
    1.778 GiB representation saw ~10 GiB free, passed the test, took the whole path, and died on
    the next 1.778 GiB request -- a fold that the fixed 1.0 GiB threshold had been completing in
    882 s. The free-DRAM gate made a working target fail.

    Predicting the peak correctly would mean modelling every intervening allocation, which is a
    model that rots the moment PWA changes. A measured constant does not.

    The env override still wins, because it is the acceptance test for the chunked path: forcing
    the budget to 0 makes a SMALL target chunk so its output can be compared byte-for-byte
    against the same target folded whole. `predict` runs the fold in a spawned worker, so this
    has to be an env var -- a value set on the parent's module object would not reach it.
    """
    v = os.environ.get("TT_BIO_MSA_ROW_CHUNK_BUDGET_BYTES")
    return nbytes <= (int(v) if v else MSA_ROW_CHUNK_BUDGET_BYTES)


_TRUNK_TAP_CALL = 0     # one increment per Trunk._msa call, i.e. per recycling cycle
_HOST_TAP_SEEN = {}     # tag prefix -> how many host taps written, to bound a 200-step sampler


def trunk_tap(tag, t, always=False):
    """Fingerprint a trunk tensor, when TT_BIO_TRUNK_TAP names a file to append to.

    Six probes each proved a component of the chunked MSA path bit-exact while the full fold kept
    diverging, which is the signature of a difference that only exists in the composition. Guessing
    at components cannot find that; comparing the trunk's own intermediates between two runs can.

    Fields are pipe-delimited on purpose. The first version wrote `shape=(1, 8722, 285, 128)` in a
    space-separated line, a reader split on whitespace and indexed a fixed field, and so compared
    shape fragments instead of hashes -- reporting vacuous "identical" for two entire passes. A
    delimiter that cannot occur inside a value removes that failure mode at the source.

    Records a sha256 of the tensor's exact bytes rather than the tensor itself: m_feat is ~0.7 GiB
    for a small target, so dumping it per block per cycle would be hundreds of GB, while a hash
    localises the FIRST differing tensor just as precisely. Costs one device->host transfer per tap,
    so it is gated behind the env var and further limited by TT_BIO_TRUNK_TAP_CYCLES (default 2).

    Enable it in BOTH arms of a comparison: then whatever it perturbs, it perturbs identically, and
    the comparison stays honest (see the observer effect that invalidated an earlier measurement)."""
    path = os.environ.get("TT_BIO_TRUNK_TAP")
    if not path:
        return
    # `always` taps ignore the per-cycle budget: the trunk's exit and the first diffusion steps are
    # a handful of tensors, and they are the ones that say whether a divergence that is absent from
    # cycle 0's MSA blocks has appeared by the time diffusion starts.
    if not always and _TRUNK_TAP_CALL >= int(os.environ.get("TT_BIO_TRUNK_TAP_CYCLES", "2")):
        return
    import hashlib
    # The MSA representation is a LIST of depth chunks on the chunked path. Hash the chunks in
    # order, which gives the same fingerprint the concatenated tensor would.
    parts = t if isinstance(t, list) else [t]
    shape = (f"list[{len(parts)}]{tuple(parts[0].shape)}" if isinstance(t, list)
             else tuple(t.shape))
    # Upcast before hashing: to_torch hands back bfloat16 for a bf16 device tensor and numpy has no
    # bfloat16 dtype, so .numpy() raises. float32 is a lossless widening of bf16, so the hash is
    # still an exact fingerprint of the device bytes.
    _h = hashlib.sha256()
    for _p in parts:
        _h.update(ttnn.to_torch(_p).to(torch.float32).contiguous().numpy().tobytes())
    h = _h.hexdigest()[:16]
    try:
        with open(path, "a") as fp:
            fp.write(f"cycle{_TRUNK_TAP_CALL}|{tag}|{shape}|{h}\n")
    except OSError:
        pass


def trunk_tap_host(tag, t, limit=2):
    """Same fingerprint as trunk_tap, for a HOST torch tensor.

    The EDM sampler keeps its coordinate stream on the host between steps (denoise takes and returns
    host tensors), so the diffusion side cannot be tapped with the device helper. Bounded to the
    first `limit` calls per tag prefix so a 200-step sampler does not write 200 lines."""
    path = os.environ.get("TT_BIO_TRUNK_TAP")
    if not path:
        return
    import hashlib
    n = _HOST_TAP_SEEN.get(tag.split("[")[0], 0)
    if n >= limit:
        return
    _HOST_TAP_SEEN[tag.split("[")[0]] = n + 1
    h = hashlib.sha256(t.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]
    try:
        with open(path, "a") as fp:
            fp.write(f"host|{tag}|{tuple(t.shape)}|{h}\n")
    except OSError:
        pass


def _msa_row_chunk_size():
    """Chunk width, with a test-only env override (same rationale as the budget above).

    Set it larger than the MSA depth to force the chunked branch to emit exactly ONE chunk.
    That is the bisect that separates "chunking granularity changes the numbers" from "the
    chunk-path plumbing changes the numbers": with one chunk the arithmetic is the whole
    path's, so any remaining difference is the slice/reshape/concat wrapper, not the split."""
    v = os.environ.get("TT_BIO_MSA_ROW_CHUNK_SIZE")
    return int(v) if v else MSA_CHUNK_SIZE

def _window_q(x, N, NP, nq=32):
    """Window the query axis into local blocks: (N,C)|(1,N,C) -> (NP//nq, nq, C), right-padded
    to NP. Shared by the atom-encoder and diffusion atom-cache windowing."""
    x = ttnn.reshape(x, (1, N, x.shape[-1])) if len(x.shape) == 2 else x
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)
    return ttnn.to_layout(ttnn.reshape(x, (NP // nq, nq, x.shape[-1])), ttnn.TILE_LAYOUT)


_WIN_KV_IDX = {}  # (NP,nq,nk) -> precomputed (1, nb*nk) uint32 gather index (device tensor)


def _window_kv(x, N, NP, nq=32, nk=128, pad_left=48):
    """Window the key/value axis into overlapping local blocks: (N,C)|(1,N,C) -> (NP//nq, nk, C)
    with a left pad of (nk-nq)/2. A SINGLE ttnn.embedding gather (window i, key j <- padded row
    i*nq+j). Replaces an nb-element ttnn.slice loop + nb-way ttnn.concat that DEADLOCKS the device
    at large nb (e.g. Wormhole, seq>~64 atoms): nb slices + concat -> ~2*nb dispatched ops that
    never complete. Same gather the AtomTransformer KV-windowing uses; bit-identical semantics."""
    x = ttnn.reshape(x, (1, N, x.shape[-1])) if len(x.shape) == 2 else x
    C = x.shape[-1]; nb = NP // nq; Lp = pad_left + NP + nk
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.pad(x, [[0, 0], [pad_left, Lp - pad_left - N], [0, 0]], 0.0)
    if x.dtype == ttnn.float32:
        # embedding only accepts bf16 tables. Build the same overlapping windows from
        # four block-aligned slices so the full-fp32 diffusion path stays on device
        # without quantizing the atom stream back to bf16.
        blocks_per_window = nk // nq
        x = ttnn.slice(x, [0, 0, 0], [1, NP + nk - nq, C])
        x = ttnn.reshape(x, (nb + blocks_per_window - 1, nq, C))
        x = ttnn.concat([
            ttnn.slice(x, [i, 0, 0], [i + nb, nq, C])
            for i in range(blocks_per_window)
        ], dim=1)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = ttnn.reshape(x, (Lp, C))                                   # gather table (Lp, C)
    idx = _WIN_KV_IDX.get((NP, nq, nk))
    if idx is None:
        ii = (torch.arange(nb).reshape(nb, 1) * nq + torch.arange(nk).reshape(1, nk)).reshape(1, nb * nk)
        idx = ttnn.from_torch(ii.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=get_device(), dtype=ttnn.uint32)
        _WIN_KV_IDX[(NP, nq, nk)] = idx
    x = ttnn.embedding(idx, x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, (nb, nk, C))                               # (nb, nk, C)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------
# M-aware (multiplicity-batched) windowing helpers.
# Carries the multiplicity M as a real leading batch dim through the windowed
# attention, mirroring boltz2.AtomDiffusion.sample. Selected by
# DiffusionModule.supports_multiplicity; the M=1 path above is bit-exact and
# untouched. Design proven in pure
# torch by tests/test_windowing_multiplicity.py + tests/test_windowing_m_shapes.py:
# fold M into the windowed-attention block dim B = M*nb via a LEADING-M 3D pad
# (per-sample right-pad on dim 1 of (M,N,C), then reshape to (M*nb, ...)). Sample
# k's nb windows are blocks k*nb..k*nb+nb-1 (no cross-sample bleed). Sample-
# invariant tensors (cond, biases) keep their leading dim of 1 and are BROADCAST
# over M by the consuming add; only the per-atom/per-token single embeddings, which
# AdaLN consumes elementwise, carry a real M dim. Replicating the pair biases
# instead (ttnn.concat of M copies) is numerically identical but costs M times the
# bias in DRAM, which is what made the batched path OOM on large targets.
# ---------------------------------------------------------------------------


def _window_q_m(x, M, N, NP, nq=32):
    """M-aware query windowing: (M,N,C) -> (M*nb, nq, C), per-sample right-pad to NP
    via a leading-M 3D pad on dim 1, then reshape to fold M into the block dim."""
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)            # (M, NP, C), per-sample
    return ttnn.to_layout(ttnn.reshape(x, (M * (NP // nq), nq, x.shape[-1])), ttnn.TILE_LAYOUT)



def _window_kv_m(x, M, N, NP, nq=32, nk=128, pad_left=48):
    """M-aware KV windowing: (M,N,C) -> (M*nb, nk, C). Loops the verified M=1
    _window_kv gather per sample and concats along dim 0 (correct-by-construction:
    the ttnn.embedding gather's per-row semantics don't extend cleanly to a 3D
    (M,Lp,C) table, so reuse the M=1 gather M times -- the gather is cheap relative
    to attention). Sample k's nb windows land at blocks k*nb..k*nb+nb-1, matching
    _window_q_m's layout (no cross-sample bleed)."""
    cols = [_window_kv(x[k:k + 1], N, NP, nq, nk, pad_left) for k in range(M)]  # each (nb, nk, C)
    return ttnn.to_layout(ttnn.concat(cols, dim=0), ttnn.TILE_LAYOUT)          # (M*nb, nk, C)


class _KeyedWeights:
    """Mixin: cached weight-upload-by-key + linear / layernorm by key, reading a flat
    {name: torch.Tensor} dict ``self._w``. Weights upload ONCE (TILE/bf16) and are cached
    in ``self._wc`` -- reused across recycle cycles, sampling steps, and folds (the model
    stays resident). The protenix submodules all share this instead of each re-implementing
    the cache + ttnn.linear/layer_norm wrappers. Requires self._w + self.compute_kernel_config."""

    def _w_tt(self, key, transpose=True):
        cache = self.__dict__.setdefault("_wc", {})
        v = cache.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=get_device(),
                                dtype=getattr(self, "dtype", ttnn.bfloat16))
            cache[(key, transpose)] = v
        return v

    def _up(self, t):
        """Upload an activation/host tensor (per call, not cached)."""
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=get_device(),
                               dtype=getattr(self, "dtype", ttnn.bfloat16))

    def _lin(self, x, wkey, bkey=None, activation=None):
        return ttnn.linear(x, self._w_tt(wkey), bias=(self._w_tt(bkey, False) if bkey else None),
                           activation=activation, compute_kernel_config=self.compute_kernel_config,
                           dtype=getattr(self, "dtype", ttnn.bfloat16), core_grid=CORE_GRID_MAIN)

    def _ln(self, x, wkey, bkey=None):
        return ttnn.layer_norm(x, weight=self._w_tt(wkey, False),
                               bias=(self._w_tt(bkey, False) if bkey else None),
                               epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)


class AtomTransformer(_KeyedWeights, Module):
    """Protenix AtomTransformer = DiffusionTransformer(cross_attention_mode=True),
    3 blocks, local windowed attention (n_queries=32, n_keys=128). Fully on-device.

    Each block: AttentionPairBias(double AdaLN q/kv, windowed attn w/ pair bias +
    mask_trunked validity, per-head linear_g gate + output sigmoid(linear_a_last(s))
    gate) -> residual -> ConditionedTransitionBlock -> residual. Validated vs the
    real v2 golden_qout (PCC>0.9999). Reference: transformer.py AtomTransformer.
    """
    N_HEADS = 4
    HEAD_DIM = 32
    N_QUERIES = 32
    N_KEYS = 128
    PAD_LEFT = 48  # (n_keys - n_queries) // 2

    def __init__(self, n_blocks, state_dict, compute_kernel_config, dtype=ttnn.bfloat16):
        super().__init__(state_dict, compute_kernel_config)
        self.dtype = dtype
        self.n_blocks = n_blocks
        self._w = {k: v for k, v in self.weights.data.items()}
        self._kv_widx = {}  # cached KV-window gather indices, keyed by NP

    def _adaln(self, a, s, pre):
        # Cache the AdaLN module per prefix: constructing it re-uploads its 4 weights
        # (ttnn.from_torch) every call, which for the diffusion enc/decoder means thousands
        # of redundant device writes per fold (2 AdaLN x 3 blocks x 200 steps). Building once
        # and replaying is bit-identical and cuts that dispatch (and is a prerequisite for
        # trace capture, which forbids writes). Keyed by prefix.
        from .tenstorrent import AdaLN
        cache = self.__dict__.setdefault("_adaln_cache", {})
        ada = cache.get(pre)
        if ada is None:
            sub = {k[len(pre):]: v for k, v in self._w.items() if k.startswith(pre)}
            ada = AdaLN(False, remap_adaln(sub), self.compute_kernel_config, dtype=self.dtype)
            cache[pre] = ada
        return ada(a, s)

    def _windows_q(self, x, N, NP):
        H, dh = self.N_HEADS, self.HEAD_DIM
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)
        x = ttnn.reshape(x, (NP // self.N_QUERIES, self.N_QUERIES, H, dh))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _kv_window_idx(self, nb, nq, nk, NP):
        """Constant gather indices for the sliding KV windows: window i, key j sources
        padded row i*nq+j. Built once per NP (replaces the per-window slice loop)."""
        idx = self._kv_widx.get(NP)
        if idx is None:
            import torch
            ii = (torch.arange(nb).reshape(nb, 1) * nq + torch.arange(nk).reshape(1, nk)).reshape(1, nb * nk)
            idx = ttnn.from_torch(ii.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=self.device, dtype=ttnn.uint32)
            self._kv_widx[NP] = idx
        return idx

    def _windows_kv(self, x, N, NP):
        # Sliding KV windows via a SINGLE gather (ttnn.embedding) instead of nb slice ops +
        # concat: window i, key j = padded row i*nq+j (indices precomputed, fixed per NP).
        H, dh, nq, nk = self.N_HEADS, self.HEAD_DIM, self.N_QUERIES, self.N_KEYS
        nb = NP // nq
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        Lp = self.PAD_LEFT + NP + nk
        x = ttnn.pad(x, [[0, 0], [self.PAD_LEFT, Lp - self.PAD_LEFT - N], [0, 0]], 0.0)
        if x.dtype == ttnn.float32:
            blocks_per_window = nk // nq
            x = ttnn.slice(x, [0, 0, 0], [1, NP + nk - nq, H * dh])
            x = ttnn.reshape(x, (nb + blocks_per_window - 1, nq, H * dh))
            x = ttnn.concat([
                ttnn.slice(x, [i, 0, 0], [i + nb, nq, H * dh])
                for i in range(blocks_per_window)
            ], dim=1)
        else:
            x = ttnn.reshape(x, (Lp, H * dh))                      # gather table (Lp, H*dh)
            idx = self._kv_window_idx(nb, nq, nk, NP)              # (1, nb*nk) uint32
            x = ttnn.embedding(idx, x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(x, (nb, nk, H, dh))
        x = ttnn.reshape(x, (nb, nk, H, dh))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _pair_bias(self, p, apb):
        """Atom-pair attention bias: LayerNorm(p, weight only) -> linear_nobias_z -> permute
        to (nb,H,nq,nk). Pure function of p; in the diffusion enc/decoder p is constant across
        sampling steps, so this is precomputed once per fold (see _precompute_biases)."""
        z = ttnn.layer_norm(p, weight=self._w_tt(apb + "layernorm_z.weight", False), epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        z = self._lin(z, apb + "linear_nobias_z.weight")          # (nb,nq,nk,H)
        return ttnn.permute(z, (0, 3, 1, 2))                       # (nb,H,nq,nk)

    def _bmm(self, a, b, in0_block_w=1):
        """Batched matmul that puts the batch on the core grid instead of a serial loop.

        ttnn's default batched matmul parallelises only within ONE batch element's output tile
        grid and walks the batch serially. The windowed atom attention is 75 windows x 4 heads
        of a single-tile problem, so at most 4 of the 110 cores hold work at any instant and the
        op costs a flat ~1 us per window-head however many cores are idle: measured 1.067 us at
        nb=1 and at nb=256 alike. An explicit MatmulMultiCoreReuseProgramConfig maps
        batch x M-blocks x N-blocks onto the grid, so the batch is chunked to at most one block
        per core and the two matmuls run 2.8x / 4.6x faster (fp32) with torch.equal parity.

        in0_block_w has to equal the K blocking the default picks (2 tiles for the 4-tile K of
        A@V), or the accumulation order changes and the result stops being bit-identical.
        Anything that does not fit the config falls back to the plain matmul.
        """
        ncores = CORE_GRID_MAIN.x * CORE_GRID_MAIN.y
        nb, h = a.shape[0], a.shape[1]
        m_t, k_t, n_t = a.shape[-2] // 32, a.shape[-1] // 32, b.shape[-1] // 32
        if (h > ncores or k_t % in0_block_w or a.shape[-2] % 32 or a.shape[-1] % 32
                or b.shape[-1] % 32 or b.shape[0] != nb or b.shape[1] != h):
            return ttnn.matmul(a, b, compute_kernel_config=self.compute_kernel_config)
        per = max(1, ncores // h)
        g = (nb + per - 1) // per
        per = (nb + g - 1) // g
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(CORE_GRID_MAIN.x, CORE_GRID_MAIN.y),
            in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
            per_core_M=m_t, per_core_N=n_t)
        if g == 1:
            return ttnn.matmul(a, b, program_config=pc,
                               compute_kernel_config=self.compute_kernel_config)
        outs = []
        for c in range(0, nb, per):
            e = min(c + per, nb)
            outs.append(ttnn.matmul(
                ttnn.slice(a, [c, 0, 0, 0], [e, h, a.shape[2], a.shape[3]]),
                ttnn.slice(b, [c, 0, 0, 0], [e, h, b.shape[2], b.shape[3]]),
                program_config=pc, compute_kernel_config=self.compute_kernel_config))
        return ttnn.concat(outs, dim=0)

    def _attention(self, q_norm, kv_norm, p, apb, N, NP, pad_bias, z_pre=None):
        H, dh = self.N_HEADS, self.HEAD_DIM
        Q = self._lin(q_norm, apb + "attention.linear_q.weight", apb + "attention.linear_q.bias")
        K = self._lin(kv_norm, apb + "attention.linear_k.weight")
        V = self._lin(kv_norm, apb + "attention.linear_v.weight")
        Qb = self._windows_q(Q, N, NP); Kb = self._windows_kv(K, N, NP); Vb = self._windows_kv(V, N, NP)
        z = z_pre if z_pre is not None else self._pair_bias(p, apb)   # precomputed (fixed p) or inline
        sc = self._bmm(Qb, ttnn.permute(Kb, (0, 1, 3, 2)))
        sc = ttnn.multiply(sc, dh ** -0.5)
        sc = ttnn.add(ttnn.add(sc, z), pad_bias)
        o = self._bmm(ttnn.softmax(sc, dim=-1), Vb, in0_block_w=2)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (NP, H * dh))
        o = ttnn.slice(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [0, 0], [N, H * dh])
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    def _block(self, a, s, p, b, N, NP, pad_bias, z_pre=None):
        P = f"diffusion_transformer.blocks.{b}."; apb = P + "attention_pair_bias."
        q_norm = self._adaln(a, s, apb + "layernorm_a.")
        kv_norm = self._adaln(q_norm, s, apb + "layernorm_kv.")
        o = self._attention(q_norm, kv_norm, p, apb, N, NP, pad_bias, z_pre=z_pre)
        g = ttnn.linear(q_norm, self._w_tt(apb + "attention.linear_g.weight"),
                        compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        attn = self._lin(o, apb + "attention.linear_o.weight")
        gate = self._lin(s, apb + "linear_a_last.weight", apb + "linear_a_last.bias")
        attn = ttnn.multiply(attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        a1 = ttnn.add(attn, a)
        ctb = P + "conditioned_transition_block."
        an = self._adaln(a1, s, ctb + "adaln.")
        b1 = self._lin(an, ctb + "linear_nobias_a1.weight", activation="silu")
        b2 = self._lin(an, ctb + "linear_nobias_a2.weight")
        out = self._lin(ttnn.multiply(b1, b2), ctb + "linear_nobias_b.weight")
        cg = self._lin(s, ctb + "linear_s.weight", ctb + "linear_s.bias")
        out = ttnn.multiply(out, cg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.add(out, a1)

    def _make_pad_bias(self, mask_trunked):
        pad = torch.where(mask_trunked < 0.5, torch.full_like(mask_trunked, -1e9),
                          torch.zeros_like(mask_trunked)).unsqueeze(1)  # (nb,1,nq,nk)
        return ttnn.from_torch(pad, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype)

    def precompute_biases(self, p, mask_trunked):
        """Per-block atom-pair bias + the pad bias -- both pure functions of (p, mask_trunked),
        which are FIXED across diffusion sampling steps. Compute once per fold; replay via
        __call__(bias_cache=...) to avoid recomputing them every step (24->1 per fold)."""
        z_pre = [self._pair_bias(p, f"diffusion_transformer.blocks.{b}.attention_pair_bias.")
                 for b in range(self.n_blocks)]
        return (z_pre, self._make_pad_bias(mask_trunked))

    def __call__(self, a, s, p, mask_trunked, bias_cache=None):
        """a,s: (1,N,c_atom); p: (nb,nq,nk,c_atompair); mask_trunked: (nb,nq,nk) host
        tensor of per-window key validity. bias_cache = optional (per-block z_pre, pad_bias)
        from precompute_biases() (when p/mask are fixed across calls). Returns (1,N,c_atom)."""
        N = a.shape[1]
        NP = ((N + self.N_QUERIES - 1) // self.N_QUERIES) * self.N_QUERIES
        z_pre, pad_bias = bias_cache if bias_cache is not None else (None, self._make_pad_bias(mask_trunked))
        x = a
        for b in range(self.n_blocks):
            x = self._block(x, s, p, b, N, NP, pad_bias, z_pre=(z_pre[b] if z_pre is not None else None))
        return x

    # --- M-aware (multiplicity-batched) path --------------------------------------
    # Mirrors the M=1 _block/_attention but carries M as the leading batch dim. The
    # sample-invariant s (c_la) is replicated (AdaLN needs it elementwise); the
    # precomputed pair bias is NOT -- it stays (nb,H,nq,nk) and is broadcast over M in
    # _attention_m. Q windowing is batched (leading-M 3D pad -> (M*nb,...)); KV windowing loops
    # the verified M=1 gather per sample and concats (correct-by-construction). The
    # attention output recovers per-sample atoms via reshape (M,NP,...) -> slice dim 1 :N.
    # See module-level _window_q_m / _window_kv_m docstrings for the no-bleed proof.

    def _windows_q_m(self, x, M, N, NP):
        H, dh = self.N_HEADS, self.HEAD_DIM
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)            # (M, NP, H*dh)
        x = ttnn.reshape(x, (M * (NP // self.N_QUERIES), self.N_QUERIES, H, dh))
        x = ttnn.permute(x, (0, 2, 1, 3))                              # (M*nb, H, nq, dh)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _windows_kv_m(self, x, M, N, NP):
        # Loop the verified M=1 _windows_kv gather per sample; concat along dim 0.
        cols = [self._windows_kv(x[k:k + 1], N, NP) for k in range(M)]  # each (nb, H, nk, dh)
        return ttnn.to_layout(ttnn.concat(cols, dim=0), ttnn.TILE_LAYOUT)      # (M*nb, H, nk, dh)

    def _attention_m(self, q_norm, kv_norm, p, apb, N, NP, M, pad_bias, z_pre=None):
        H, dh = self.N_HEADS, self.HEAD_DIM
        Q = self._lin(q_norm, apb + "attention.linear_q.weight", apb + "attention.linear_q.bias")
        K = self._lin(kv_norm, apb + "attention.linear_k.weight")
        V = self._lin(kv_norm, apb + "attention.linear_v.weight")
        Qb = self._windows_q_m(Q, M, N, NP); Kb = self._windows_kv_m(K, M, N, NP); Vb = self._windows_kv_m(V, M, N, NP)
        z = z_pre if z_pre is not None else self._pair_bias(p, apb)
        sc = self._bmm(Qb, ttnn.permute(Kb, (0, 1, 3, 2)))
        sc = ttnn.multiply(sc, dh ** -0.5)
        if z.shape[0] != sc.shape[0]:
            # z is the sample-INVARIANT precomputed bias, still (nb,H,nq,nk). Fold the
            # (M*nb, H) leading dims into (M, nb*H) so the add broadcasts it over M --
            # the last two dims are untouched, so both reshapes are free in tile layout.
            # Replicating z instead cost M copies of the whole per-block bias in DRAM.
            nb, g = sc.shape[0] // M, sc.shape[1] * (sc.shape[0] // M)
            sc = ttnn.reshape(ttnn.add(ttnn.reshape(sc, (M, g, sc.shape[2], sc.shape[3])),
                                       ttnn.reshape(z, (1, g, z.shape[2], z.shape[3]))),
                              (M * nb, H, sc.shape[2], sc.shape[3]))
        else:
            sc = ttnn.add(sc, z)
        sc = ttnn.add(sc, pad_bias)
        o = self._bmm(ttnn.softmax(sc, dim=-1), Vb, in0_block_w=2)
        o = ttnn.permute(o, (0, 2, 1, 3))                       # (M*nb, nq, H, dh)
        o = ttnn.reshape(o, (M, NP, H * dh))                    # (M, NP, H*dh)
        o = ttnn.slice(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0], [M, N, H * dh])  # (M, N, H*dh)
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    def _block_m(self, a, s, p, b, N, NP, M, pad_bias, z_pre=None):
        P = f"diffusion_transformer.blocks.{b}."; apb = P + "attention_pair_bias."
        q_norm = self._adaln(a, s, apb + "layernorm_a.")
        kv_norm = self._adaln(q_norm, s, apb + "layernorm_kv.")
        o = self._attention_m(q_norm, kv_norm, p, apb, N, NP, M, pad_bias, z_pre=z_pre)
        g = ttnn.linear(q_norm, self._w_tt(apb + "attention.linear_g.weight"),
                        compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        attn = self._lin(o, apb + "attention.linear_o.weight")
        gate = self._lin(s, apb + "linear_a_last.weight", apb + "linear_a_last.bias")
        attn = ttnn.multiply(attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        a1 = ttnn.add(attn, a)
        ctb = P + "conditioned_transition_block."
        an = self._adaln(a1, s, ctb + "adaln.")
        b1 = self._lin(an, ctb + "linear_nobias_a1.weight", activation="silu")
        b2 = self._lin(an, ctb + "linear_nobias_a2.weight")
        out = self._lin(ttnn.multiply(b1, b2), ctb + "linear_nobias_b.weight")
        cg = self._lin(s, ctb + "linear_s.weight", ctb + "linear_s.bias")
        out = ttnn.multiply(out, cg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.add(out, a1)

    def __call__(self, a, s, p, mask_trunked, bias_cache=None, multiplicity=1):
        """a,s: (1,N,c_atom) at M=1, (M,N,c_atom) at M>1; p: (nb,nq,nk,c_atompair) shared;
        mask_trunked: (nb,nq,nk) host. bias_cache = optional (per-block z_pre, pad_bias)
        from precompute_biases() (shared across samples). multiplicity (M): when >1 and
        supports_multiplicity is on, run the M-aware batched path (one batched forward per
        block); else the M=1 per-sample path. Returns (1,N,c_atom) at M=1, (M,N,c_atom) at M>1."""
        N = a.shape[1] if multiplicity == 1 else a.shape[1]
        NP = ((N + self.N_QUERIES - 1) // self.N_QUERIES) * self.N_QUERIES
        if multiplicity == 1:
            z_pre, pad_bias = bias_cache if bias_cache is not None else (None, self._make_pad_bias(mask_trunked))
            x = a
            for b in range(self.n_blocks):
                x = self._block(x, s, p, b, N, NP, pad_bias, z_pre=(z_pre[b] if z_pre is not None else None))
            return x
        # M-aware. Only the per-atom single s needs a real M dim (AdaLN is elementwise);
        # the pair bias z_pre is broadcast over M in _attention_m rather than replicated,
        # and p is read ONLY when there is no precomputed z_pre, so replicate it lazily.
        z_pre, pad_bias = bias_cache if bias_cache is not None else (None, self._make_pad_bias(mask_trunked))
        # Sample batching (one target, M samples) hands in a shared s/p/bias with a leading
        # dim of 1 and they are replicated here. Prediction batching (M different targets,
        # protenix._merge_conds) hands in tensors that already carry the M members, so pass
        # those straight through -- same shapes, no copy.
        nb_ = NP // self.N_QUERIES
        s_m = s if s.shape[0] == multiplicity else ttnn.to_layout(
            ttnn.concat([s] * multiplicity, dim=0), ttnn.TILE_LAYOUT)                         # (M,N,c_atom)
        p_m = p if (z_pre is not None or p.shape[0] == multiplicity * nb_) else ttnn.to_layout(
            ttnn.concat([p] * multiplicity, dim=0), ttnn.TILE_LAYOUT)                         # (M*nb,nq,nk,c_ap)
        # pad_bias is (nb,1,nq,nk) -- H-broadcast, so ~H times smaller than z_pre; keep the
        # replication here so the (sc + z) + pad add order matches the M=1 path exactly.
        if pad_bias.shape[0] != multiplicity * nb_:
            pad_bias = ttnn.to_layout(ttnn.concat([pad_bias] * multiplicity, dim=0), ttnn.TILE_LAYOUT)
        x = a
        for b in range(self.n_blocks):
            x = self._block_m(x, s_m, p_m, b, N, NP, multiplicity, pad_bias,
                                z_pre=(z_pre[b] if z_pre is not None else None))
        return x


class AtomFeaturization(Module):
    """Protenix AtomAttentionEncoder.prepare_cache (has_coords=False path).

    Builds the per-atom single embedding c_l and the windowed atom-pair embedding
    p_lm from reference features. Pure linears + arcsinh + elementwise — no
    attention. Reference: protenix/model/modules/transformer.py prepare_cache.

      c_l = W_pos(ref_pos) + W_charge(arcsinh(ref_charge)) + W_f([mask|elem|name])
      c_l *= ref_mask
      p_lm = W_d(d_lm)*v_lm*mask_trunked + W_invd(1/(1+sum d_lm^2))*v_lm + W_v(v_lm)
    """

    def __init__(self, state_dict, compute_kernel_config, dtype=ttnn.bfloat16):
        super().__init__(state_dict, compute_kernel_config)
        self.dtype = dtype
        # nn.Linear weights are (out, in); _lin/ttnn.linear want (in, out).
        self.w_ref_pos = self.torch_to_tt("linear_no_bias_ref_pos.weight", dtype=dtype)
        self.w_ref_charge = self.torch_to_tt("linear_no_bias_ref_charge.weight", dtype=dtype)
        self.w_f = self.torch_to_tt("linear_no_bias_f.weight", dtype=dtype)
        self.w_d = self.torch_to_tt("linear_no_bias_d.weight", dtype=dtype)
        self.w_invd = self.torch_to_tt("linear_no_bias_invd.weight", dtype=dtype)
        self.w_v = self.torch_to_tt("linear_no_bias_v.weight", dtype=dtype)

    def _lin_nb(self, x, w):
        return ttnn.linear(
            x, w, compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype, core_grid=CORE_GRID_MAIN,
        )

    def c_l(self, ref_pos, ref_charge_asinh, ref_mask, f_in):
        """All inputs are device tensors. ref_charge_asinh is arcsinh(charge)[...,1],
        ref_mask is [...,1], f_in is cat([mask|element|name_chars]) -> [...,449]."""
        c = ttnn.add(self._lin_nb(ref_pos, self.w_ref_pos),
                     self._lin_nb(ref_charge_asinh, self.w_ref_charge))
        c = ttnn.add(c, self._lin_nb(f_in, self.w_f))
        return ttnn.mul(c, ref_mask)

    def p_lm(self, d_lm, v_lm, invd, mask_trunked):
        """Windowed atom-pair embedding. d_lm/v_lm/invd/mask_trunked are flattened
        to [n_blocks*n_queries*n_keys, *] device tensors (last-dim linears)."""
        p = ttnn.mul(ttnn.mul(self._lin_nb(d_lm, self.w_d), v_lm), mask_trunked)
        p = ttnn.add(p, ttnn.mul(self._lin_nb(invd, self.w_invd), v_lm))
        p = ttnn.add(p, self._lin_nb(v_lm, self.w_v))
        return p


class AtomAttentionEncoder(_KeyedWeights, Module):
    """Protenix InputFeatureEmbedder atom encoder (has_coords=False) -> s_inputs.

    featurization (AtomFeaturization) -> p_lm augmentation (windowed c_l projections
    + small_mlp) -> AtomTransformer -> relu(linear_q) + mean atom->token aggregate
    -> a; then s_inputs = cat([a, restype, profile, deletion_mean]) (c_s_inputs=449).
    Validated vs the real v2 golden s_inputs. Reference: transformer.py
    AtomAttentionEncoder.forward + embedders.py InputFeatureEmbedder.forward.
    """
    NQ, NK, PAD_LEFT, C_ATOMPAIR = 32, 128, 48, 16

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.feat = AtomFeaturization(self.weights, compute_kernel_config)
        self.atx = AtomTransformer(3, self.scope("atom_transformer"), compute_kernel_config)
        self._w = {k: v for k, v in self.weights.data.items()}

    def _win_q(self, x, N, NP):
        return _window_q(x, N, NP, self.NQ)

    def _win_kv(self, x, N, NP):
        return _window_kv(x, N, NP, self.NQ, self.NK, self.PAD_LEFT)

    def _augment_plm(self, p, c_l, N, NP):
        # p: (nb,nq,nk,16); add windowed c_l projections + small_mlp. c_l: (1,N,128).
        clq = ttnn.relu(self._win_q(c_l, N, NP))            # (nb,nq,128)
        clk = ttnn.relu(self._win_kv(c_l, N, NP))           # (nb,nk,128)
        cl = ttnn.unsqueeze(self._lin(clq, "linear_no_bias_cl.weight"), 2)   # (nb,nq,1,16)
        cm = ttnn.unsqueeze(self._lin(clk, "linear_no_bias_cm.weight"), 1)   # (nb,1,nk,16)
        p = ttnn.add(ttnn.add(p, cl), cm)
        m = self._lin(ttnn.relu(p), "small_mlp.1.weight")
        m = self._lin(ttnn.relu(m), "small_mlp.3.weight")
        m = self._lin(ttnn.relu(m), "small_mlp.5.weight")
        return ttnn.add(p, m)

    def __call__(self, ref_pos, ref_charge_asinh, ref_mask, f_in, d_lm, v_lm, invd,
                 mask_trunked, atom_to_token_mean, restype, profile, deletion_mean):
        """All tensors on device except mask_trunked (host, for the attn pad bias) and
        atom_to_token_mean ((N_token,N) host averaging matrix). p_lm built in windowed
        flat form then reshaped to (nb,nq,nk,16)."""
        N = ref_pos.shape[1] if len(ref_pos.shape) == 3 else ref_pos.shape[0]
        NP = ((N + self.NQ - 1) // self.NQ) * self.NQ
        nb = NP // self.NQ
        c_l = self.feat.c_l(ref_pos, ref_charge_asinh, ref_mask, f_in)        # (1,N,128) or (N,128)
        if len(c_l.shape) == 2:
            c_l = ttnn.reshape(c_l, (1, c_l.shape[0], c_l.shape[1]))
        mt_dev = ttnn.from_torch(mask_trunked.reshape(-1, 1), layout=ttnn.TILE_LAYOUT,
                                 device=self.device, dtype=ttnn.bfloat16)
        p_flat = self.feat.p_lm(d_lm, v_lm, invd, mt_dev)                    # (nb*nq*nk,16)
        p = ttnn.reshape(p_flat, (nb, self.NQ, self.NK, self.C_ATOMPAIR))
        p = self._augment_plm(p, c_l, N, NP)
        q_out = self.atx(c_l, c_l, p, mask_trunked.reshape(nb, self.NQ, self.NK))  # (1,N,128)
        q = ttnn.relu(self._lin(q_out, "linear_no_bias_q.weight"))           # (1,N,384)
        q = ttnn.reshape(q, (N, q.shape[-1]))
        a = ttnn.matmul(atom_to_token_mean, q, compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)                            # (N_token,384)
        return ttnn.concat([a, restype, profile, deletion_mean], dim=-1)     # (N_token,449)


class DiffusionModule(_KeyedWeights):
    """Protenix-v2 diffusion denoiser (one EDM-preconditioned step).

    denoise(x_noisy, t_hat, cond) -> denoised coords, where cond holds the fixed
    trunk conditioning (s_trunk, s_inputs, pair_z, c_l, p_lm, atom->token matrix S,
    mask_trunked). Composition (validated end-to-end vs the real v2 reference across
    the full sigma schedule, PCC 0.99961..1.0; scripts/protenix_traj_replay.py,
    tests/test_protenix_traj.py):

      single cond  : LN(cat[s_trunk,s_inputs])->W_s + LN(fourier(log(t/sd)/4))->W_n,
                     + transition_s1 + transition_s2  -> s_single
      atom encoder : c_la = c_l + S @ W_s(LN(s_trunk)); q = c_la + W_r(x/sqrt(sd^2+t^2));
                     p = p_lm + windowed(W_cl(relu c_la)) + windowed(W_cm(...)) + small_mlp;
                     AtomTransformer; a_tok = meanpool_atom->token(relu W_q(q_out))
      a_tok += W_s(LN(s_single))   [diffusion_module.linear_no_bias_s]
      token DiT    : 24-block AttentionPairBias(token-level, per-block pair bias from
                     LN(pair_z)) + s-gate sigmoid(linear_a_last(s)) + ConditionedTransition
      a = LN(a)    [diffusion_module.layernorm_a]
      atom decoder : q = S @ W_a(a) + q_skip; AtomTransformer; r = W_out(LN(q))
      EDM precond  : denoised = x/(1+sr^2) + t/sqrt(1+sr^2)*r,  sr = t/sigma_data(16)

    The 24-block token DiT runs ON-DEVICE (device_dit=True): each block's pair bias is
    a pure function of the trunk pair_z, so the 24 biases are precomputed once per fold
    (_dit_block_biases) and replayed every sampling step (AttentionPairBias bias_precomputed)
    rather than recomputed; the atom-encoder conditioning and atom-transformer pair/pad
    biases are likewise hoisted once per fold. A host fp32 DiT fallback is kept (device_dit
    =False) for the strict per-step parity test, which builds no precomputed device bias.
    Everything runs on-device (ttnn, HiFi4). The coordinate-sensitive diffusion uses fp32
    by default to match the reference boundary across the conditioning cache, atom encoder and
    transformers, token DiT, and atom decoder. Set PROTENIX_DIFFUSION_FP32_DEVICE=0 to A/B bf16.
    The --fast mode still applies only to the trunk (see Protenix.__init__)."""

    SIGMA_DATA = 16.0
    NQ, NK, PAD_LEFT = 32, 128, 48
    DIT_BLOCKS, DIT_HEAD_DIM, DIT_N_HEADS = 24, 48, 16
    supports_multiplicity = True   # ON: verified on qb2 card0

    def __init__(self, diffusion_state_dict, device, compute_kernel_config, diffusion_fp32=None):
        """diffusion_state_dict: {key: tensor} for diffusion_module.* (prefix stripped).
        diffusion_fp32: explicit override; None falls back to PROTENIX_DIFFUSION_FP32_DEVICE."""
        import torch.nn.functional as F  # noqa: F401  (used in DiT)
        self._w = dict(diffusion_state_dict)
        self.dev = device
        self.compute_kernel_config = compute_kernel_config
        self._diffusion_fp32 = (os.environ.get("PROTENIX_DIFFUSION_FP32_DEVICE", "1") == "1"
                                 if diffusion_fp32 is None else diffusion_fp32)
        self.dtype = ttnn.float32 if self._diffusion_fp32 else ttnn.bfloat16
        self.atxE = AtomTransformer(3, {k[len("atom_attention_encoder.atom_transformer."):]: v
                                        for k, v in self._w.items()
                                        if k.startswith("atom_attention_encoder.atom_transformer.")},
                                    compute_kernel_config, dtype=self.dtype)
        self.atxD = AtomTransformer(3, {k[len("atom_attention_decoder.atom_transformer."):]: v
                                        for k, v in self._w.items()
                                        if k.startswith("atom_attention_decoder.atom_transformer.")},
                                    compute_kernel_config, dtype=self.dtype)
        self._wc = {}  # device-weight cache (upload once; reused across all sampling steps)
        from .tenstorrent import AdaLN, AttentionPairBias, Transition
        C = "diffusion_conditioning."
        self._cond_transitions = [
            Transition(PW.remap_transition({k[len(C + nm + "."):]: v for k, v in self._w.items()
                                                if k.startswith(C + nm + ".")}), compute_kernel_config,
                       dtype=self.dtype)
            for nm in ("transition_s1", "transition_s2")]
        # On-device token DiT: per-block AdaLN + AttentionPairBias (compute_pair_bias=False,
        # fed the precomputed UNSCALED bias as the SDPA mask -> matches the host math exactly;
        # these primitives handle the head_dim=48 tile padding). s-gate + conditioned-transition
        # are raw ttnn (protenix's ctb differs from tt-bio's ConditionedTransitionBlock).
        # Full on-device fp32 diffusion. The GPU reference disables autocast for the
        # entire sampling score model, while its trunk remains under bf16 autocast. Match that
        # exact boundary: the conditioning cache, atom encoder/transformer, token DiT, and atom
        # decoder all use ttnn float32. This stays on the device and keeps the existing traced
        # device_dit path. The paired HSA accuracy and performance gates passed; bf16 remains
        # available as an explicit A/B opt-out through PROTENIX_DIFFUSION_FP32_DEVICE=0.
        self._dit_fp32 = self._diffusion_fp32
        self._dit_dtype = self.dtype
        self._dit_ckc = compute_kernel_config   # HiFi4 + fp32_dest_acc_en: dtype is on the tensors
        self.device_dit = True
        DT = "diffusion_transformer."
        sub = lambda pfx: {k[len(pfx):]: v for k, v in self._w.items() if k.startswith(pfx)}
        self._dit = []
        for b in range(self.DIT_BLOCKS):
            A = DT + f"blocks.{b}.attention_pair_bias."
            Cc = DT + f"blocks.{b}.conditioned_transition_block."
            self._dit.append((
                AdaLN(False, remap_adaln(sub(A + "layernorm_a.")), self._dit_ckc, dtype=self._dit_dtype),
                AttentionPairBias(self.DIT_HEAD_DIM, self.DIT_N_HEADS, True, False,
                                  PW.remap_attention_pair_bias(sub(A)), self._dit_ckc,
                                  dtype=self._dit_dtype, fp32_raw_matmul_attention=True),
                AdaLN(False, remap_adaln(sub(Cc + "adaln.")), self._dit_ckc, dtype=self._dit_dtype),
                A, Cc))


    def _up_dit(self, t):
        """Upload an activation/host tensor at the DiT dtype (fp32 when the gate is on)."""
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=self._dit_dtype)

    def _w_tt_dit(self, key, transpose=True):
        """Dedicated DiT weight-upload cache at the active diffusion dtype."""
        cache = self.__dict__.setdefault("_wc_dit", {})
        v = cache.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=self._dit_dtype)
            cache[(key, transpose)] = v
        return v

    def _ln_dit(self, x, wkey, bkey=None):
        """Layer norm at the DiT dtype (used for the DiT-output layernorm_a when fp32)."""
        return ttnn.layer_norm(x, weight=self._w_tt_dit(wkey, False),
                               bias=(self._w_tt_dit(bkey, False) if bkey else None),
                               epsilon=1e-5, compute_kernel_config=self._dit_ckc)

    def _atom_cond(self, cond):
        """Hoist the t-INDEPENDENT diffusion conditioning out of the per-step denoise.
        The single-conditioning base, the atom-encoder single c_la, and the windowed
        atom-pair p (+ small_mlp) depend ONLY on the trunk outputs -- not on x_noisy or
        t_hat -- yet were recomputed every sampling step (200x). They are the same kind
        of fixed conditioning already hoisted for the DiT (dit_z / pair bias). Compute
        once per fold; store resident device tensors in cond. Idempotent."""
        import torch
        if "c_la_dev" in cond:
            return
        s_trunk = cond["s_trunk"].float(); s_inputs = cond["s_inputs"].float()
        c_l = cond["c_l"].float(); p_lm = cond["p_lm"].float(); S = cond["S"].float()
        N = c_l.shape[0]; NT = s_inputs.shape[0]
        NP = ((N + self.NQ - 1) // self.NQ) * self.NQ
        T = self._up
        E = "atom_attention_encoder."
        # single-conditioning base (the t-dependent fourier term + transitions stay per-step)
        cond["ss_base"] = self._lin(self._ln(T(torch.cat([s_trunk, s_inputs], -1)),
                                    "diffusion_conditioning.layernorm_s.weight"),
                                    "diffusion_conditioning.linear_no_bias_s.weight")
        # atom-encoder single c_la and windowed atom-pair p (+ small_mlp)
        sp = self._lin(self._ln(T(s_trunk), E + "layernorm_s.weight"), E + "linear_no_bias_s.weight")
        c_la = ttnn.add(T(c_l), ttnn.matmul(T(S), sp, compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN))
        clq = ttnn.relu(self._winq(c_la, N, NP)); clk = ttnn.relu(self._winkv(c_la, N, NP))
        p = ttnn.add(ttnn.add(T(p_lm), ttnn.unsqueeze(self._lin(clq, E + "linear_no_bias_cl.weight"), 2)),
                     ttnn.unsqueeze(self._lin(clk, E + "linear_no_bias_cm.weight"), 1))
        mm = self._lin(ttnn.relu(p), E + "small_mlp.1.weight")
        mm = self._lin(ttnn.relu(mm), E + "small_mlp.3.weight")
        mm = self._lin(ttnn.relu(mm), E + "small_mlp.5.weight")
        cond["c_la_dev"] = c_la
        cond["p_dev"] = ttnn.add(p, mm)
        # atom->token mean-pool matrix and S onehot, resident (re-uploaded every step otherwise)
        Smean = S.t().contiguous() / (S.sum(0, keepdim=True).t() + 1e-6)
        cond["Smean_dev"] = T(Smean)
        cond["S_dev"] = T(S)
        # atom-transformer (enc + dec) pair/pad biases: pure functions of p (=p_dev, fixed) and
        # the window mask -> precompute once; replayed every step via bias_cache.
        mtf = cond["mask_trunked"].float()
        cond["atxE_bias"] = self.atxE.precompute_biases(cond["p_dev"], mtf)
        cond["atxD_bias"] = self.atxD.precompute_biases(cond["p_dev"], mtf)

    def denoise(self, x_noisy, t_hat, cond):
        """x_noisy (1,N,3) host at M=1, (M,N,3) at M>1; t_hat scalar host tensor (1,); cond dict with host
        tensors s_trunk (NT,c_s), s_inputs (NT,449), pair_z (NT,NT,c_z), c_l (N,128),
        p_lm (nb,nq,nk,16), S (N,NT) atom->token onehot, mask_trunked (nb,nq,nk).
        Returns denoised coords (1,N,3) at M=1, (M,N,3) at M>1 host tensor."""
        import torch.nn.functional as F
        _M = x_noisy.shape[0]
        if _M > 1 and getattr(self, "supports_multiplicity", False):
            return self._denoise_multiplicity(x_noisy, t_hat, cond)
        self._atom_cond(cond)   # idempotent: t-independent conditioning, computed once per fold
        s_inputs = cond["s_inputs"]
        sd = self.SIGMA_DATA
        N = cond["c_l"].shape[0]; NT = s_inputs.shape[0]
        T = self._up
        E = "atom_attention_encoder."
        mt = cond["mask_trunked"].float()
        c_la = cond["c_la_dev"]; p = cond["p_dev"]   # hoisted (resident across all steps)

        # 1) single conditioning: cached base + per-step fourier(t_hat), then transitions
        wf = self._w["diffusion_conditioning.fourier_embedding.w"]; bf = self._w["diffusion_conditioning.fourier_embedding.b"]
        tp = torch.log(t_hat / sd) / 4
        fou = torch.cos(2 * torch.pi * (tp.unsqueeze(-1) * wf + bf))
        nn_ = self._lin(self._ln(T(fou), "diffusion_conditioning.layernorm_n.weight"),
                        "diffusion_conditioning.linear_no_bias_n.weight")
        ss = ttnn.reshape(ttnn.add(cond["ss_base"], nn_), (1, NT, cond["ss_base"].shape[-1]))
        for t in self._cond_transitions:   # prebuilt once (weights resident)
            ss = ttnn.add(ss, ttnn.reshape(t(ss), tuple(ss.shape)))
        s_single = ss

        # 2) atom encoder: only the coordinate-dependent path (c_la / p come from cond)
        r_noisy = x_noisy / torch.sqrt(torch.tensor(sd ** 2) + t_hat ** 2).reshape(-1, 1, 1)
        q_l = ttnn.add(c_la, self._lin(T(r_noisy[0]), E + "linear_no_bias_r.weight"))
        q_out = self.atxE(ttnn.reshape(q_l, (1, N, 128)), ttnn.reshape(c_la, (1, N, 128)), p, mt,
                          bias_cache=cond.get("atxE_bias"))
        a_tok = ttnn.matmul(cond["Smean_dev"], ttnn.reshape(ttnn.relu(self._lin(q_out, E + "linear_no_bias_q.weight")), (N, 768)),
                            compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        q_skip = q_out; c_skip = c_la; p_skip = p
        a_tok = ttnn.add(a_tok, ttnn.reshape(
            self._lin(self._ln(ttnn.reshape(s_single, (NT, s_single.shape[-1])), "layernorm_s.weight"),
                      "linear_no_bias_s.weight"), (NT, 768)))

        # 3) token DiT (fp32 host; precision-limited on-device, see class docstring)
        # per-block pair bias depends only on pair_z (fixed across steps) -> precomputed once
        if self.device_dit and cond.get("dit_z") is not None:
            if "dit_block_biases" not in cond:   # precompute per-block pair biases ONCE per fold
                cond["dit_block_biases"] = self._dit_block_biases(
                    cond["dit_z"], cond.get("structural_pair_attn_bias"))
            a_t = self._token_dit_device(ttnn.reshape(a_tok, (1, NT, 768)), s_single,
                                         cond["dit_block_biases"], NT)
            if self._dit_fp32:
                a_t = self._ln_dit(a_t, "layernorm_a.weight")
            else:
                a_t = self._ln(a_t, "layernorm_a.weight")
        else:  # host fp32 fallback (max fidelity / no precomputed device bias)
            a_h = torch.Tensor(ttnn.to_torch(ttnn.reshape(a_tok, (1, NT, 768)))).float().reshape(NT, 768)
            s_h = torch.Tensor(ttnn.to_torch(s_single)).float().reshape(NT, s_single.shape[-1])
            biases = cond.get("dit_biases") or self._dit_pair_biases(
                cond["pair_z"].float(), cond.get("structural_pair_attn_bias"))
            a_h = self._token_dit(a_h, s_h, biases, NT)
            a_t = self._ln(T(a_h.reshape(1, NT, 768)), "layernorm_a.weight")

        # 4) atom decoder
        DE = "atom_attention_decoder."
        q = ttnn.add(ttnn.matmul(cond["S_dev"], self._lin(ttnn.reshape(a_t, (NT, 768)), DE + "linear_no_bias_a.weight"),
                                 compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN),
                     ttnn.reshape(q_skip, (N, 128)))
        qd = self.atxD(ttnn.reshape(q, (1, N, 128)), ttnn.reshape(c_skip, (1, N, 128)), p_skip, mt,
                       bias_cache=cond.get("atxD_bias"))
        qn = self._ln(qd, DE + "layernorm_q.weight")
        r_update = torch.Tensor(ttnn.to_torch(self._lin(qn, DE + "linear_no_bias_out.weight"))).float().reshape(1, N, 3)[:, :N]

        # EDM preconditioning
        sr = (t_hat / sd).reshape(-1, 1, 1)
        return (1.0 / (1.0 + sr ** 2)) * x_noisy[:, :N] + (t_hat.reshape(-1, 1, 1) / torch.sqrt(1.0 + sr ** 2)) * r_update

    # ---------------------------------------------------------------------------
    # ttnn TRACE of the denoise device stream (opt-in via fold(trace=True)).
    # Protenix diffusion warm is per-step DISPATCH-bound (~400 op launches/step,
    # L-independent): capturing the device stream once per fold and replaying it
    # collapses the per-step host dispatch. The two per-step-varying host inputs
    # (fourier(t_hat) and the scaled coords) are staged into fixed device buffers;
    # the fold-fixed conditioning (cond) stays resident. device_dit path only.
    # ---------------------------------------------------------------------------
    def _denoise_device(self, r_noisy_dev, fou_dev, cond):
        """Pure on-device denoise (device_dit path). r_noisy_dev (N,3) and fou_dev
        (1,fdim) are device tensors (the per-step host inputs, already uploaded);
        returns r_update (1,N,3) on device (pre EDM-precond). No host round-trips."""
        s_inputs = cond["s_inputs"]
        N = cond["c_l"].shape[0]; NT = s_inputs.shape[0]
        E = "atom_attention_encoder."
        mt = cond["mask_trunked"].float()
        c_la = cond["c_la_dev"]; p = cond["p_dev"]
        nn_ = self._lin(self._ln(fou_dev, "diffusion_conditioning.layernorm_n.weight"),
                        "diffusion_conditioning.linear_no_bias_n.weight")
        ss = ttnn.reshape(ttnn.add(cond["ss_base"], nn_), (1, NT, cond["ss_base"].shape[-1]))
        for t in self._cond_transitions:
            ss = ttnn.add(ss, ttnn.reshape(t(ss), tuple(ss.shape)))
        s_single = ss
        q_l = ttnn.add(c_la, self._lin(r_noisy_dev, E + "linear_no_bias_r.weight"))
        q_out = self.atxE(ttnn.reshape(q_l, (1, N, 128)), ttnn.reshape(c_la, (1, N, 128)), p, mt,
                          bias_cache=cond.get("atxE_bias"))
        a_tok = ttnn.matmul(cond["Smean_dev"], ttnn.reshape(ttnn.relu(self._lin(q_out, E + "linear_no_bias_q.weight")), (N, 768)),
                            compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        q_skip = q_out; c_skip = c_la; p_skip = p
        a_tok = ttnn.add(a_tok, ttnn.reshape(
            self._lin(self._ln(ttnn.reshape(s_single, (NT, s_single.shape[-1])), "layernorm_s.weight"),
                      "linear_no_bias_s.weight"), (NT, 768)))
        if "dit_block_biases" not in cond:
            cond["dit_block_biases"] = self._dit_block_biases(
                cond["dit_z"], cond.get("structural_pair_attn_bias"))
        a_t = self._token_dit_device(ttnn.reshape(a_tok, (1, NT, 768)), s_single, cond["dit_block_biases"], NT)
        if self._dit_fp32:
            a_t = self._ln_dit(a_t, "layernorm_a.weight")
        else:
            a_t = self._ln(a_t, "layernorm_a.weight")
        DE = "atom_attention_decoder."
        q = ttnn.add(ttnn.matmul(cond["S_dev"], self._lin(ttnn.reshape(a_t, (NT, 768)), DE + "linear_no_bias_a.weight"),
                                 compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN),
                     ttnn.reshape(q_skip, (N, 128)))
        qd = self.atxD(ttnn.reshape(q, (1, N, 128)), ttnn.reshape(c_skip, (1, N, 128)), p_skip, mt,
                       bias_cache=cond.get("atxD_bias"))
        qn = self._ln(qd, DE + "layernorm_q.weight")
        return ttnn.reshape(self._lin(qn, DE + "linear_no_bias_out.weight"), (1, N, 3))

    def _host_tt(self, x):
        """Host-resident ttnn tensor (no device) for copy_host_to_device_tensor staging."""
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, dtype=self.dtype)

    def _release_trace(self):
        tr = getattr(self, "_trace", None)
        if tr is not None:
            try:
                ttnn.release_trace(self.dev, tr["tid"])
            except Exception:
                pass
            self._trace = None

    def _capture_trace(self, fou, r_noisy, cond, N):
        fou_dev = self._up(fou); r_dev = self._up(r_noisy)   # persistent input buffers
        _ = self._denoise_device(r_dev, fou_dev, cond)       # warmup / compile
        _ = self._denoise_device(r_dev, fou_dev, cond)       # 2nd warmup: populate any lazy caches
        ttnn.synchronize_device(self.dev)
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        out = self._denoise_device(r_dev, fou_dev, cond)     # record
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        self._trace = {"N": N, "tid": tid, "in_fou": fou_dev, "in_r": r_dev, "out": out}
        return self._trace

    def denoise_traced(self, x_noisy, t_hat, cond):
        """Traced equivalent of denoise (device_dit path). Falls back to denoise when the
        device_dit precomputed bias path is unavailable."""
        import torch
        self._atom_cond(cond)
        if not (self.device_dit and cond.get("dit_z") is not None):
            return self.denoise(x_noisy, t_hat, cond)
        sd = self.SIGMA_DATA; N = cond["c_l"].shape[0]
        wf = self._w["diffusion_conditioning.fourier_embedding.w"]; bf = self._w["diffusion_conditioning.fourier_embedding.b"]
        tp = torch.log(t_hat / sd) / 4
        fou = torch.cos(2 * torch.pi * (tp.unsqueeze(-1) * wf + bf)).contiguous()          # (1,fdim)
        r_noisy = (x_noisy / torch.sqrt(torch.tensor(sd ** 2) + t_hat ** 2).reshape(-1, 1, 1))[0].contiguous()  # (N,3)
        tr = getattr(self, "_trace", None)
        if tr is None or tr["N"] != N:
            if tr is not None:
                self._release_trace()
            tr = self._capture_trace(fou, r_noisy, cond, N)
        ttnn.copy_host_to_device_tensor(self._host_tt(fou), tr["in_fou"])
        ttnn.copy_host_to_device_tensor(self._host_tt(r_noisy), tr["in_r"])
        ttnn.execute_trace(self.dev, tr["tid"], cq_id=0, blocking=False)
        r_update = torch.Tensor(ttnn.to_torch(tr["out"])).float().reshape(1, N, 3)[:, :N]
        sr = (t_hat / sd).reshape(-1, 1, 1)
        return (1.0 / (1.0 + sr ** 2)) * x_noisy[:, :N] + (t_hat.reshape(-1, 1, 1) / torch.sqrt(1.0 + sr ** 2)) * r_update

    # --- windowing helpers (atom encoder p augmentation) ---
    def _winq(self, x, N, NP):
        return _window_q(x, N, NP, self.NQ)

    def _winkv(self, x, N, NP):
        return _window_kv(x, N, NP, self.NQ, self.NK, self.PAD_LEFT)

    def _dit_pair_biases(self, pair_z, extra_attn_bias=None):
        """Per-block DiT attention pair bias linear_z(LN(LN(pair_z))). Depends only on the
        trunk pair_z (fixed across all sampling steps), so it is computed ONCE per fold and
        reused every diffusion step -- the dominant host cost otherwise. Returns 24 tensors
        of shape (n_heads, NT, NT)."""
        import torch.nn.functional as F
        gP = lambda k: self._w["diffusion_transformer." + k].float()
        z_h = F.layer_norm(pair_z, (pair_z.shape[-1],))
        biases = []
        for b in range(self.DIT_BLOCKS):
            A = f"blocks.{b}.attention_pair_bias."
            zb = F.layer_norm(z_h, (z_h.shape[-1],)) * gP(A + "layernorm_z.weight")
            bias = F.linear(zb, gP(A + "linear_nobias_z.weight")).permute(2, 0, 1)
            if extra_attn_bias is not None:
                bias = bias + extra_attn_bias.float().unsqueeze(0)
            biases.append(bias)
        return biases

    def _token_dit(self, a_h, s_h, biases, NT):
        import torch.nn.functional as F
        nbk, hd, nh = self.DIT_BLOCKS, self.DIT_HEAD_DIM, self.DIT_N_HEADS
        gP = lambda k: self._w["diffusion_transformer." + k].float()
        def adaln(a, s, pre):
            an = F.layer_norm(a, (a.shape[-1],)); sn = F.layer_norm(s, (s.shape[-1],)) * gP(pre + "layernorm_s.weight")
            return torch.sigmoid(F.linear(sn, gP(pre + "linear_s.weight"), gP(pre + "linear_s.bias"))) * an + F.linear(sn, gP(pre + "linear_nobias_s.weight"))
        for b in range(nbk):
            A = f"blocks.{b}.attention_pair_bias."; Cc = f"blocks.{b}.conditioned_transition_block."
            an = adaln(a_h, s_h, A + "layernorm_a.")
            bias = biases[b]
            Q = F.linear(an, gP(A + "attention.linear_q.weight"), gP(A + "attention.linear_q.bias")).reshape(NT, nh, hd).permute(1, 0, 2)
            K = F.linear(an, gP(A + "attention.linear_k.weight")).reshape(NT, nh, hd).permute(1, 0, 2)
            V = F.linear(an, gP(A + "attention.linear_v.weight")).reshape(NT, nh, hd).permute(1, 0, 2)
            o = torch.einsum("hij,hjd->hid", torch.softmax(torch.einsum("hid,hjd->hij", Q, K) / (hd ** 0.5) + bias, -1), V).permute(1, 0, 2).reshape(NT, nh * hd)
            o = o * torch.sigmoid(F.linear(an, gP(A + "attention.linear_g.weight"))); attn = F.linear(o, gP(A + "attention.linear_o.weight"))
            attn = torch.sigmoid(F.linear(s_h, gP(A + "linear_a_last.weight"), gP(A + "linear_a_last.bias"))) * attn; ao = attn + a_h
            an2 = adaln(ao, s_h, Cc + "adaln."); bb = F.silu(F.linear(an2, gP(Cc + "linear_nobias_a1.weight"))) * F.linear(an2, gP(Cc + "linear_nobias_a2.weight"))
            a_h = torch.sigmoid(F.linear(s_h, gP(Cc + "linear_s.weight"), gP(Cc + "linear_s.bias"))) * F.linear(bb, gP(Cc + "linear_nobias_b.weight")) + ao
        return a_h

    def _dit_z_device(self, pair_z):
        """Upload LN(pair_z) once per fold as (1,NT,NT,c_z) for the on-device DiT; each block's
        AttentionPairBias (compute_pair_bias=True) derives its own pair bias from it (matching
        the validated trunk-pairformer convention, incl. the head-dim scaling)."""
        import torch.nn.functional as F
        z_h = F.layer_norm(pair_z, (pair_z.shape[-1],)).unsqueeze(0).contiguous()
        return ttnn.from_torch(z_h, layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=self._dit_dtype)

    def _dit_block_biases(self, z_dev, extra_attn_bias=None):
        """Per-block DiT attention pair biases, computed ONCE per fold from z_dev=LN(pair_z).
        Each block's bias is a pure function of pair_z (fixed across all sampling steps), so
        recomputing the NxNxc_z layer_norm+linear every step (24 blocks x n_step) was the
        dominant diffusion cost. Precompute here; replay via AttentionPairBias(bias_precomputed
        =True). Mirrors the host _dit_pair_biases / the _atom_cond hoist."""
        extra = None
        if extra_attn_bias is not None:
            # AttentionPairBias scales projected masks by sqrt(head_dim) to
            # compensate ttnn SDPA's mask scaling. Match it for this direct bias.
            extra = self._up_dit(extra_attn_bias.float().reshape(
                1, 1, extra_attn_bias.shape[-2], extra_attn_bias.shape[-1])
                * self.DIT_HEAD_DIM ** 0.5)
        return [ttnn.add(apb.compute_bias(z_dev), extra) if extra is not None
                else apb.compute_bias(z_dev) for (_, apb, _, _, _) in self._dit]

    def _token_dit_device(self, a_t, s_t, biases, NT):
        """On-device 24-block token DiT (ttnn). a_t (1,NT,768), s_t (1,NT,384); biases = list
        of per-block precomputed (1,n_heads,NT,NT) pair biases (from _dit_block_biases, fixed
        across steps). Mirrors host _token_dit; reuses AdaLN + AttentionPairBias. When
        PROTENIX_DIFFUSION_FP32_DEVICE=1 the full diffusion stack, including this DiT,
        runs in ttnn fp32."""
        ckc = self._dit_ckc
        if self._dit_fp32:   # inputs are already fp32; typecast is trace-safe and idempotent
            a_t = ttnn.typecast(a_t, self._dit_dtype)
            s_t = ttnn.typecast(s_t, self._dit_dtype)
        wtt = self._w_tt_dit if self._dit_fp32 else self._w_tt

        def linb(x, wk, bk=None, act=None):
            return ttnn.linear(x, wtt(wk), bias=(wtt(bk, False) if bk else None), activation=act,
                               compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        for _bi, ((adaln_a, apb, ctb_adaln, A, Cc), bias) in enumerate(zip(self._dit, biases)):
            b = adaln_a(a_t, s_t)
            attn = apb(b, bias, bias_precomputed=True)
            dram_peak(f"dit[M={a_t.shape[0]}] block {_bi}")
            sg = ttnn.sigmoid(linb(s_t, A + "linear_a_last.weight", A + "linear_a_last.bias"))
            ao = ttnn.add(ttnn.multiply(attn, sg), a_t)
            an2 = ctb_adaln(ao, s_t)
            bb = ttnn.multiply(linb(an2, Cc + "linear_nobias_a1.weight", act="silu"),
                               linb(an2, Cc + "linear_nobias_a2.weight"))
            cs = ttnn.sigmoid(linb(s_t, Cc + "linear_s.weight", Cc + "linear_s.bias"))
            a_t = ttnn.add(ttnn.multiply(cs, linb(bb, Cc + "linear_nobias_b.weight")), ao)
        return a_t


    def _token_dit_device_m(self, a_t, s_t, biases, NT):
        """M-aware on-device 24-block token DiT. a_t (M,NT,768), s_t (M,NT,384); biases = list
        of (1,n_heads,NT,NT) precomputed pair biases (broadcast over M in the QK-scale add).
        Mirrors _token_dit_device; AdaLN + AttentionPairBias handle a leading batch dim when
        s_t is replicated to (M,NT,384). Used only by _denoise_multiplicity (gated)."""
        ckc = self._dit_ckc
        if self._dit_fp32:
            a_t = ttnn.typecast(a_t, self._dit_dtype)
            s_t = ttnn.typecast(s_t, self._dit_dtype)
        wtt = self._w_tt_dit if self._dit_fp32 else self._w_tt

        def linb(x, wk, bk=None, act=None):
            return ttnn.linear(x, wtt(wk), bias=(wtt(bk, False) if bk else None), activation=act,
                               compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        for _bi, ((adaln_a, apb, ctb_adaln, A, Cc), bias) in enumerate(zip(self._dit, biases)):
            b = adaln_a(a_t, s_t)
            attn = apb(b, bias, bias_precomputed=True)
            dram_peak(f"dit[M={a_t.shape[0]}] block {_bi}")
            sg = ttnn.sigmoid(linb(s_t, A + "linear_a_last.weight", A + "linear_a_last.bias"))
            ao = ttnn.add(ttnn.multiply(attn, sg), a_t)
            an2 = ctb_adaln(ao, s_t)
            bb = ttnn.multiply(linb(an2, Cc + "linear_nobias_a1.weight", act="silu"),
                               linb(an2, Cc + "linear_nobias_a2.weight"))
            cs = ttnn.sigmoid(linb(s_t, Cc + "linear_s.weight", Cc + "linear_s.bias"))
            a_t = ttnn.add(ttnn.multiply(cs, linb(bb, Cc + "linear_nobias_b.weight")), ao)
        return a_t

    def _denoise_multiplicity(self, x_noisy, t_hat, cond):
        """M-aware batched denoise (gated by supports_multiplicity). Mirrors denoise() but
        carries the leading M batch dim through the atom encoder, token DiT, and atom
        decoder. Shared conditioning (cond: c_la_dev, p_dev, Smean_dev, S_dev, atxE_bias,
        atxD_bias, dit_block_biases) is sample-invariant, so it keeps its leading dim of 1
        and is broadcast over M; only the coordinate stream x_noisy carries M. The two
        atom<->token pooling matrices are the exception -- ttnn matmul has no batch
        broadcast -- so those are replicated (see Smean_m / S_m below).
        Returns denoised coords (M,N,3) host."""
        import torch.nn.functional as F
        self._atom_cond(cond)
        M = x_noisy.shape[0]
        s_inputs = cond["s_inputs"]
        sd = self.SIGMA_DATA
        N = cond["c_l"].shape[0]; NT = s_inputs.shape[0]
        T = self._up
        E = "atom_attention_encoder."
        mt = cond["mask_trunked"].float()
        c_la = cond["c_la_dev"]; p = cond["p_dev"]
        # A merged cond (protenix._merge_conds) already carries one entry per member, so
        # every replication below is a pass-through there; sample batching keeps its
        # leading dim of 1 and gets the copy.
        rep = lambda t: t if t.shape[0] == M else ttnn.to_layout(
            ttnn.concat([t] * M, dim=0), ttnn.TILE_LAYOUT)
        # c_la_dev is 2D (N,128) (see _atom_cond) and sample-invariant, so it is kept at
        # (1,N,128) and broadcast over M. Merged it is already (M,N,128).
        c_la_1 = c_la if len(c_la.shape) == 3 else ttnn.reshape(c_la, (1, N, 128))

        # 1) single conditioning: shared (t-independent base + per-step fourier(t_hat)).
        wf = self._w["diffusion_conditioning.fourier_embedding.w"]
        bf = self._w["diffusion_conditioning.fourier_embedding.b"]
        tp = torch.log(t_hat / sd) / 4
        fou = torch.cos(2 * torch.pi * (tp.unsqueeze(-1) * wf + bf))
        nn_ = self._lin(self._ln(T(fou), "diffusion_conditioning.layernorm_n.weight"),
                        "diffusion_conditioning.linear_no_bias_n.weight")
        _ssb = cond["ss_base"]
        _cs = _ssb.shape[-1]
        ss = (ttnn.add(_ssb, ttnn.reshape(nn_, (1, 1, _cs))) if len(_ssb.shape) == 3
              else ttnn.reshape(ttnn.add(_ssb, nn_), (1, NT, _cs)))
        for t in self._cond_transitions:
            ss = ttnn.add(ss, ttnn.reshape(t(ss), tuple(ss.shape)))
        s_single = ss   # (1, NT, c) shared across M

        # 2) atom encoder: coordinate-dependent path with M-leading q_l.
        r_noisy = x_noisy / torch.sqrt(torch.tensor(sd ** 2) + t_hat ** 2).reshape(-1, 1, 1)
        q_l = ttnn.add(self._lin(T(r_noisy), E + "linear_no_bias_r.weight"), c_la_1)  # (M,N,128)
        q_out = self.atxE(q_l, c_la_1, p, mt,
                          bias_cache=cond.get("atxE_bias"), multiplicity=M)   # (M,N,128)
        qo_lin = ttnn.relu(self._lin(q_out, E + "linear_no_bias_q.weight"))   # (M,N,768)
        _smean = cond["Smean_dev"]
        Smean_m = _smean if len(_smean.shape) == 3 else ttnn.to_layout(
            ttnn.concat([ttnn.reshape(_smean, (1, NT, N))] * M, dim=0), ttnn.TILE_LAYOUT)
        a_tok = ttnn.matmul(Smean_m, qo_lin, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)                          # (M,NT,768)
        _Ms = s_single.shape[0]                       # 1 when shared, M when merged
        s_bias = ttnn.reshape(
            self._lin(self._ln(ttnn.reshape(s_single, (_Ms * NT, s_single.shape[-1])),
                               "layernorm_s.weight"),
                      "linear_no_bias_s.weight"), (_Ms, NT, 768))
        a_tok = ttnn.add(a_tok, s_bias)                                       # (M,NT,768), s_bias bcast

        # 3) token DiT (device_dit path): M-leading a_t/s_t; per-block biases broadcast.
        if self.device_dit and cond.get("dit_z") is not None:
            if "dit_block_biases" not in cond:
                cond["dit_block_biases"] = self._dit_block_biases(
                    cond["dit_z"], cond.get("structural_pair_attn_bias"))
            # The per-block pair bias is sample-invariant, so it is passed UNREPLICATED and
            # broadcast over M in the QK-scale add. Replicating it (M copies of 24 x
            # (n_heads,NT,NT)) was the multiplicity path's dominant allocation: 1.9 GB per
            # copy at NT=1095 in fp32, i.e. ~9.6 GB at M=5.
            a_t = self._token_dit_device_m(ttnn.reshape(a_tok, (M, NT, 768)), rep(s_single),
                                           cond["dit_block_biases"], NT)
            a_t = (self._ln_dit if self._dit_fp32 else self._ln)(a_t, "layernorm_a.weight")
        else:
            biases = cond.get("dit_biases") or self._dit_pair_biases(
                cond["pair_z"].float(), cond.get("structural_pair_attn_bias"))
            a_h = torch.stack([self._token_dit(
                torch.Tensor(ttnn.to_torch(ttnn.reshape(a_tok[m:m + 1], (1, NT, 768)))).float().reshape(NT, 768),
                torch.Tensor(ttnn.to_torch(s_single)).float().reshape(NT, s_single.shape[-1]),
                biases, NT) for m in range(M)], 0)
            a_t = self._ln(T(a_h.reshape(M, NT, 768)), "layernorm_a.weight")

        # 4) atom decoder: q = S_dev @ lin(a_t) + q_out.
        DE = "atom_attention_decoder."
        a_lin = self._lin(ttnn.reshape(a_t, (M, NT, 768)), DE + "linear_no_bias_a.weight")
        _sdev = cond["S_dev"]
        S_m = _sdev if len(_sdev.shape) == 3 else ttnn.to_layout(
            ttnn.concat([ttnn.reshape(_sdev, (1, N, NT))] * M, dim=0), ttnn.TILE_LAYOUT)
        q = ttnn.add(ttnn.matmul(S_m, a_lin, compute_kernel_config=self.compute_kernel_config,
                                 core_grid=CORE_GRID_MAIN), q_out)            # (M,N,128)
        qd = self.atxD(q, c_la_1, p, mt,
                       bias_cache=cond.get("atxD_bias"), multiplicity=M)      # (M,N,128)
        qn = self._ln(qd, DE + "layernorm_q.weight")
        r_update = torch.Tensor(ttnn.to_torch(self._lin(qn, DE + "linear_no_bias_out.weight"))
                                ).float().reshape(M, N, 3)[:, :N]

        # EDM preconditioning (t_hat broadcasts over M).
        sr = (t_hat / sd).reshape(-1, 1, 1)
        return (1.0 / (1.0 + sr ** 2)) * x_noisy[:, :N] + (t_hat.reshape(-1, 1, 1) / torch.sqrt(1.0 + sr ** 2)) * r_update


class ConfidenceHead:
    """Protenix-v2 ConfidenceHead -> per-atom pLDDT (and pae/pde logits).

    z = z_trunk + s1(s_inputs)[:,None] + s2(s_inputs)[None] + distance-embed(coords);
    4-block confidence Pairformer (on-device) -> s_single, z; heads (host linears):
    plddt = LN(s_single[atom->token]) . plddt_weight[atom_to_tokatom_idx]. Validated vs
    the real v2 reference (pae/pde PCC 1.0; plddt PCC ~0.93). Reference confidence_head."""

    def __init__(self, conf_state_dict, device, compute_kernel_config):
        import re
        from .tenstorrent import Pairformer
        self._w = dict(conf_state_dict)
        self.dev = device
        self.compute_kernel_config = compute_kernel_config
        nb = 1 + max(int(re.search(r"pairformer_stack\.blocks\.(\d+)\.", k).group(1))
                     for k in self._w if k.startswith("pairformer_stack.blocks."))
        comb = {}
        for i in range(nb):
            bsd = {k[len(f"pairformer_stack.blocks.{i}."):]: v for k, v in self._w.items()
                   if k.startswith(f"pairformer_stack.blocks.{i}.")}
            for kk, vv in PW.remap_pairformer_block(bsd).items():
                comb[f"layers.{i}.{kk}"] = vv
        b0 = "pairformer_stack.blocks.0."
        nhp = self._w[b0 + "tri_att_start.linear.weight"].shape[0]
        chpa = self._w[b0 + "tri_att_start.mha.linear_q.weight"].shape[0] // nhp
        apb_nh = self._w[b0 + "attention_pair_bias.linear_nobias_z.weight"].shape[0]
        self.pf = Pairformer(nb, chpa, nhp, 384 // apb_nh, apb_nh, True, comb, compute_kernel_config)

    def _g(self, k):
        return self._w[k].float()

    def _bias(self, k):
        return self._w[k].float() if k in self._w else 0.0

    def confidence(self, s_inputs, s_trunk, z_trunk, coords, feats):
        """Full confidence forward -> dict with per-atom pLDDT, mean pLDDT, and the token-token
        PAE / PDE matrices (Angstrom). All inputs host tensors; coords (N_atom,3). Recipe
        validated vs the real v2 reference (pae/pde PCC 1.0, plddt ~0.93;
        scripts/protenix_confidence_parity.py)."""
        import torch
        import torch.nn.functional as F
        N = s_trunk.shape[0]
        s_t = F.layer_norm(torch.clamp(s_trunk, -512, 512), (384,)) * self._g("input_strunk_ln.weight") + self._bias("input_strunk_ln.bias")
        z = (z_trunk + F.linear(s_inputs, self._g("linear_no_bias_s1.weight")).unsqueeze(1)
             + F.linear(s_inputs, self._g("linear_no_bias_s2.weight")).unsqueeze(0))
        mask = feats["distogram_rep_atom_mask"].bool()
        xr = coords.reshape(-1, 3)[mask]
        d = torch.cdist(xr, xr)
        oh = ((d.unsqueeze(-1) >= self._g("lower_bins")) & (d.unsqueeze(-1) < self._g("upper_bins"))).float()
        z = z + F.linear(oh, self._g("linear_no_bias_d.weight")) + F.linear(d.unsqueeze(-1), self._g("linear_no_bias_d_wo_onehot.weight"))
        T = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
        so, zo = self.pf(T(s_t.unsqueeze(0)), T(z.unsqueeze(0)))
        s_single = torch.Tensor(ttnn.to_torch(so)).float().reshape(N, 384)
        zf = torch.Tensor(ttnn.to_torch(zo)).float().reshape(N, N, -1)

        pae_logits = F.linear(F.layer_norm(zf, (zf.shape[-1],)) * self._g("pae_ln.weight") + self._bias("pae_ln.bias"),
                              self._g("linear_no_bias_pae.weight"))                          # (N,N,n_bins)
        pde_logits = F.linear(F.layer_norm(zf + zf.transpose(0, 1), (zf.shape[-1],)) * self._g("pde_ln.weight") + self._bias("pde_ln.bias"),
                              self._g("linear_no_bias_pde.weight"))                          # (N,N,n_bins)
        a2t = feats["atom_to_token_idx"].long(); a2ta = feats["atom_to_tokatom_idx"].long()
        a = s_single[a2t]
        aln = F.layer_norm(a, (384,)) * self._g("plddt_ln.weight") + self._bias("plddt_ln.bias")
        plddt_logits = torch.einsum("nc,ncb->nb", aln, self._g("plddt_weight")[a2ta])        # (N_atom, n_bins)
        return self._postprocess(pae_logits, pde_logits, plddt_logits, feats)

    # ---------------------------------------------------------------------------
    # Device-resident confidence path (opt-in). The host path (confidence())
    # builds z on host, UPLOADS (1,N,N,256) to the device Pairformer, then
    # DOWNLOADS (s_single, zf) and runs the pae/pde/plddt heads on host -- a
    # full (N,N,256) device<->host round-trip every sample. On large N the host
    # side dominates confidence (measured 53% of 355 ms @N=256 on BH 'pc'; the
    # device Pairformer is only 116 ms). This path keeps z_base resident: the
    # sample-invariant z_base = z_trunk + s1(s_inputs)[:,None] + s2(s_inputs)
    # [None,:] is computed ONCE on device, and per-sample only the (N,3) coords
    # are uploaded; the distance-embed + Pairformer + pae/pde/plddt heads all
    # run on device, and only the small final logits (pae/pde (N,N,64), plddt
    # (N_atom,50)) are downloaded. Feature-detected + gated behind
    # TT_PROTENIX_CONF_DEVICE=1; off by default until PCC is verified (plddt is
    # precision-sensitive: the host path is already PCC ~0.93 vs the reference,
    # and moving the einsum to bf16 can regress it -- see the parity harness).
    # ---------------------------------------------------------------------------
    @staticmethod
    def device_confidence_enabled():
        """True only if the user opted in (TT_PROTENIX_CONF_DEVICE=1) AND the
        installed ttnn exposes every op the device path needs. Off otherwise
        (the host-heads path in confidence() is the default)."""
        import os
        if os.environ.get("TT_PROTENIX_CONF_DEVICE", "0") not in ("1", "true", "True"):
            return False
        import ttnn
        need = ("clamp", "ge", "lt", "sqrt", "embedding", "layer_norm", "linear",
                "matmul", "softmax", "permute")
        return all(hasattr(ttnn, k) for k in need)

    def _wtt(self, key, transpose=True):
        """Upload a confidence weight once (TILE/bf16), cached on the instance."""
        cache = self.__dict__.setdefault("_wtt_cache", {})
        v = cache.get(key)
        if v is None:
            import ttnn
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
            cache[key] = v
        return v

    def _dev_lin(self, x, wkey, bias=False):
        import ttnn
        b = None
        if bias:
            b = self._wtt(wkey.replace(".weight", ".bias"), False) if (wkey.replace(".weight", ".bias") in self._w) else None
        return ttnn.linear(x, self._wtt(wkey), bias=b,
                           compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)

    def _device_resident(self, s_inputs, s_trunk, z_base_dev, feats):
        """Build the sample-invariant device tensors ONCE per fold: s_t, the
        coords gather index (which atoms pass distogram_rep_atom_mask), and the
        plddt per-atom-type weight table. z_base itself (z_trunk + s1 + s2) is
        passed in already-uploaded as a RESIDENT bf16 device tensor (computed
        fp32 on host once via z_base_device -- bf16-accumulating it on device
        regresses the pairformer input at small N; see the parity harness).
        Idempotent via a cached tag on z_base_dev."""
        import torch, torch.nn.functional as F, ttnn
        cache = self.__dict__.setdefault("_dev_res", {})
        tag = id(z_base_dev)
        if cache.get("tag") == tag:
            return cache
        T = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
        N = s_trunk.shape[0]
        s_t_h = F.layer_norm(torch.clamp(s_trunk, -512, 512), (384,)) * self._g("input_strunk_ln.weight") + self._bias("input_strunk_ln.bias")
        s_t = T(s_t_h.unsqueeze(0))                                       # (1,N,384)
        # coords gather: which atoms pass distogram_rep_atom_mask (== N tokens)
        mask = feats["distogram_rep_atom_mask"].bool()
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1).to(torch.int32)   # (N,)
        idx_dev = ttnn.from_torch(idx.reshape(1, N), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, dtype=ttnn.uint32)
        # plddt per-atom-type weight table (24, 384, 50) -> flat (24, 384*50) for embedding gather
        pw = self._g("plddt_weight")                                       # (n_tokatom, 384, 50)
        n_ta, c, nb = pw.shape
        pw_dev = ttnn.from_torch(pw.reshape(n_ta, c * nb).contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self.dev, dtype=ttnn.bfloat16)
        a2ta = feats["atom_to_tokatom_idx"].long().to(torch.int32).reshape(-1, 1)  # (N_atom,1)
        a2ta_dev = ttnn.from_torch(a2ta, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, dtype=ttnn.uint32)
        a2t = feats["atom_to_token_idx"].long().to(torch.int32).reshape(-1, 1)     # (N_atom,1) -> s_single gather
        a2t_dev = ttnn.from_torch(a2t, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, dtype=ttnn.uint32)
        cache.update(tag=tag, s_t=s_t, z_base=z_base_dev, idx_dev=idx_dev, N=N,
                     pw_dev=pw_dev, a2ta_dev=a2ta_dev, a2t_dev=a2t_dev,
                     pw_shape=(n_ta, c, nb))
        return cache

    def z_base_device(self, s_inputs, s_trunk, z_trunk):
        """Build the sample-invariant z_base = z_trunk + s1(s_inputs)[:,None] +
        s2(s_inputs)[None,:] in fp32 on host (precision-safe -- bf16-accumulating
        it on device regresses the pairformer input at small N), then upload as
        bf16 ONCE. Returned as a resident (1,N,N,256) device tensor; reuse across
        samples so the (N,N,256) upload is paid once per fold, not per sample."""
        import torch, torch.nn.functional as F, ttnn
        z_base = (z_trunk + F.linear(s_inputs, self._g("linear_no_bias_s1.weight")).unsqueeze(1)
                  + F.linear(s_inputs, self._g("linear_no_bias_s2.weight")).unsqueeze(0)).unsqueeze(0).contiguous()
        return ttnn.from_torch(z_base.float(), layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)

    def confidence_device(self, s_inputs, s_trunk, z_base_dev, coords, feats):
        """Device-resident confidence forward. z_base_dev is the RESIDENT bf16
        device tensor from z_base_device() (z_trunk + s1 + s2, fp32-computed once
        and uploaded once per fold). coords is a host (N_atom,3) tensor
        (per-sample). Returns the same dict as confidence(). Mirrors confidence()
        exactly except the per-sample distance-embed + Pairformer + pae/pde/
        plddt heads run on device (bf16) and only the final logits are
        downloaded -- the (N,N,256) z never round-trips per sample."""
        import torch, ttnn
        # Tapped because the trunk and the diffusion stream are, and this was not: three AbAg-XM
        # targets fail ~1300 s in, long after the trunk, at refused sizes that match no trunk
        # tensor. Without a tag here the last thing TT_BIO_DRAM_PEAK reports is the final edm
        # step, which cannot distinguish "died in confidence" from "died after it".
        dram_peak("confidence: enter")
        rc = self._device_resident(s_inputs, s_trunk, z_base_dev, feats)
        N = rc["N"]
        dram_peak(f"confidence: resident built [N={N}]")
        # ---- per-sample distance-embed on device ----
        coords_tbl = ttnn.from_torch(coords.float().reshape(coords.shape[0], 3), layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=self.dev, dtype=ttnn.bfloat16)        # (N_atom,3) gather table
        xr = ttnn.embedding(rc["idx_dev"], coords_tbl, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)               # (1,N,3)
        xr = ttnn.to_layout(ttnn.reshape(xr, (1, N, 3)), ttnn.TILE_LAYOUT)
        # squared dist = |xr|^2 + |xr|^2 - 2 xr.xr^T ; d = sqrt(clamp(d2, 0))
        sq = ttnn.pow(xr, 2.0)
        srow = ttnn.sum(sq, dim=-1)                                          # (1,N)
        d2 = ttnn.add(ttnn.add(srow, ttnn.reshape(srow, (1, 1, N))),
                      ttnn.multiply(ttnn.matmul(xr, ttnn.permute(xr, (0, 2, 1)),
                                                compute_kernel_config=self.compute_kernel_config,
                                                core_grid=CORE_GRID_MAIN), -2.0))
        d2 = ttnn.clamp(d2, 0.0, None)
        d = ttnn.sqrt(d2)                                                    # (1,N,N)
        d3 = ttnn.unsqueeze(d, -1)                                           # (1,N,N,1)
        lb = self._wtt("lower_bins", False); ub = self._wtt("upper_bins", False)
        lb4 = ttnn.reshape(lb, (1, 1, 1, lb.shape[-1])); ub4 = ttnn.reshape(ub, (1, 1, 1, ub.shape[-1]))
        ge = ttnn.ge(d3, lb4)                                                # (1,N,N,39) bool->bf16
        lt = ttnn.lt(d3, ub4)
        oh = ttnn.multiply(ge, lt)                                           # AND
        oh = ttnn.to_layout(oh, ttnn.TILE_LAYOUT) if oh.layout != ttnn.TILE_LAYOUT else oh
        z = ttnn.add(rc["z_base"], self._dev_lin(oh, "linear_no_bias_d.weight"))
        z = ttnn.add(z, self._dev_lin(d3, "linear_no_bias_d_wo_onehot.weight"))
        # ---- confidence Pairformer (device, z stays resident) ----
        so, zo = self.pf(rc["s_t"], z)                                       # (1,N,384),(1,N,N,256)
        # ---- heads on device ----
        zof = ttnn.reshape(zo, (1, N, N, 256))
        pae_ln = ttnn.layer_norm(zof, weight=self._wtt("pae_ln.weight", False),
                                 bias=(self._wtt("pae_ln.bias", False) if "pae_ln.bias" in self._w else None),
                                 epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        pae_logits = self._dev_lin(pae_ln, "linear_no_bias_pae.weight")      # (1,N,N,64)
        zot = ttnn.permute(zof, (0, 2, 1, 3))                                # transpose token axes
        zsym = ttnn.add(zof, zot)
        pde_ln = ttnn.layer_norm(zsym, weight=self._wtt("pde_ln.weight", False),
                                 bias=(self._wtt("pde_ln.bias", False) if "pde_ln.bias" in self._w else None),
                                 epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        pde_logits = self._dev_lin(pde_ln, "linear_no_bias_pde.weight")      # (1,N,N,64)
        # plddt: a = s_single[a2t]; aln = LN(a)*w+b; logits = einsum('nc,ncb->nb', aln, pw[a2ta])
        s_single = ttnn.reshape(so, (N, 384))
        a = ttnn.embedding(rc["a2t_dev"], ttnn.to_layout(s_single, ttnn.ROW_MAJOR_LAYOUT),
                           layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # (N_atom,1,384)
        a = ttnn.to_layout(ttnn.reshape(a, (a.shape[0], 384)), ttnn.TILE_LAYOUT)
        aln = ttnn.layer_norm(a, weight=self._wtt("plddt_ln.weight", False),
                              bias=(self._wtt("plddt_ln.bias", False) if "plddt_ln.bias" in self._w else None),
                              epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        n_ta, c, nb = rc["pw_shape"]
        pw_g = ttnn.embedding(rc["a2ta_dev"], rc["pw_dev"], layout=ttnn.ROW_MAJOR_LAYOUT,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)          # (N_atom,1, c*nb)
        pw_g = ttnn.reshape(pw_g, (a.shape[0], c, nb))                       # (N_atom, 384, 50)
        pw_g = ttnn.to_layout(pw_g, ttnn.TILE_LAYOUT)
        # einsum nc,ncb->nb  ==  batched (N_atom,1,384) @ (N_atom,384,50) -> (N_atom,1,50)
        aln_b = ttnn.reshape(aln, (a.shape[0], 1, c))
        plddt_logits = ttnn.matmul(aln_b, pw_g, compute_kernel_config=self.compute_kernel_config)  # (N_atom,1,50)
        dram_peak("confidence: heads done, before download")
        # ---- download the small finals; post-process on host (small, exact) ----
        pae_h = torch.Tensor(ttnn.to_torch(pae_logits)).float().reshape(N, N, -1)
        pde_h = torch.Tensor(ttnn.to_torch(pde_logits)).float().reshape(N, N, -1)
        plddt_h = torch.Tensor(ttnn.to_torch(plddt_logits)).float().reshape(a.shape[0], -1)  # (N_atom,50)
        return self._postprocess(pae_h, pde_h, plddt_h, feats)

    def _postprocess(self, pae_logits, pde_logits, plddt_logits, feats):
        """Shared host-side post-processing: softmax over bins -> expected
        distance (pae/pde) and expected plddt; pTM/ipTM from the pae logits.
        Identical to the tail of confidence() so device/host paths share it."""
        import torch

        def _expected(logits, max_a=32.0):
            nb = logits.shape[-1]
            centers = (torch.arange(nb, dtype=torch.float32) + 0.5) * (max_a / nb)
            return (torch.softmax(logits, -1) * centers).sum(-1)
        pae = _expected(pae_logits)
        pde = _expected(pde_logits)
        ptm, iptm = self._ptm_iptm(pae_logits, feats.get("asym_id"))
        nb = plddt_logits.shape[-1]
        plddt_atom = (torch.softmax(plddt_logits, -1) * ((torch.arange(nb, dtype=torch.float32) + 0.5) / nb)).sum(-1)
        return {"plddt": float(plddt_atom.mean()), "plddt_atom": plddt_atom, "pae": pae, "pde": pde,
                "ptm": ptm, "iptm": iptm}

    @staticmethod
    def _ptm_iptm(pae_logits, asym_id, max_a: float = 32.0):
        """Predicted TM-score (pTM) and interface pTM (ipTM) from the PAE bin logits,
        the standard AlphaFold formula. pTM = max over alignment frame i of the mean
        predicted TM to all tokens j; ipTM restricts j to *other* chains (via asym_id).
        Returns (ptm, iptm); iptm is 0.0 for single-chain inputs."""
        import torch

        N, _, nb = pae_logits.shape
        centers = (torch.arange(nb, dtype=torch.float32) + 0.5) * (max_a / nb)
        probs = torch.softmax(pae_logits.float(), -1)                       # (N,N,nb)
        n = max(N, 19)
        d0 = 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8
        tm_per_bin = 1.0 / (1.0 + (centers / d0) ** 2)                      # (nb,)
        e_tm = (probs * tm_per_bin).sum(-1)                                 # (N,N) E[TM] per pair
        ptm = float(e_tm.mean(dim=-1).max())
        iptm = 0.0
        if asym_id is not None:
            a = asym_id.long().reshape(-1)
            if a.numel() == N and int(a.unique().numel()) > 1:
                cross = a[:, None] != a[None, :]                           # (N,N) different-chain
                denom = cross.sum(dim=-1).clamp(min=1)
                row = (e_tm * cross.float()).sum(dim=-1) / denom
                valid = cross.any(dim=-1)
                iptm = float(row[valid].max()) if bool(valid.any()) else 0.0
        return round(ptm, 6), round(iptm, 6)

    def plddt(self, s_inputs, s_trunk, z_trunk, coords, feats):
        """Mean pLDDT in [0,1] (back-compat thin wrapper over confidence())."""
        return self.confidence(s_inputs, s_trunk, z_trunk, coords, feats)["plddt"]


# Byte gate for the row-blocked DiffusionConditioning pair chain in _diffusion_pair_cond.
# The unblocked chain holds the LN'd z ([Ns,Ns,c_z]) and the channel concat ([Ns,Ns,c_z'+128])
# full-size; on a churned Wormhole heap those are the requests that get refused (measured:
# 9i3p's [1902,1920,256] bf16 = 1.87 GiB concat refused at 7.5 GiB used / 4.5 free). 9ivj
# (2.27 GiB z) folds unblocked with 2.4 GiB headroom; 9i3p (2.61) does not, so the gate sits
# at 2.5 GiB and only the two largest structural-scale cells take the blocked path. Every op
# in the chain is row-local (layer_norm/linear per position, channel concat), so the blocks
# are in the same tile-replan equivalence class as the chunked pair track.
PAIRCOND_BLOCK_BYTES = int(os.environ.get("TT_BIO_PAIRCOND_BLOCK_BYTES", 5 * 2 ** 28))  # 2.5 GiB


class Protenix:
    """Top-level Protenix-v2 structure predictor on Tenstorrent (inference-only).

    fold(feats) composes the validated submodules into the full forward:
      InputFeatureEmbedder atom encoder      -> s_inputs (per-token, c_s_inputs=449)
      diffusion atom-cache (AtomFeaturization) -> c_l, p_lm  (t-independent)
      Trunk (10-cycle recycling)             -> s_trunk, z_trunk
      EDM ancestral sampler (edm_sample, DiffusionModule denoiser) -> atom coords

    Every submodule is validated on-device vs the real v2 reference in
    tests/test_protenix*.py; the full diffusion is
    validated end-to-end (sampler draws structures within the reference's sample
    variance). feats is a dict of model-ready tensors (from the v2 data pipeline)."""

    def __init__(self, model_state_dict, compute_kernel_config, device=None, c_z=None,
                 msa_update_first=False, diffusion_fp32=None):
        """diffusion_fp32: explicit per-instantiation override for the coordinate-sensitive
        diffusion stack's precision; None falls back to PROTENIX_DIFFUSION_FP32_DEVICE (default
        fp32). Callers that share this class across models (e.g. OpenDDE) should pass this
        explicitly rather than rely on the env-var default, which is scoped to Protenix-v2."""
        from .tenstorrent import get_device
        import tt_bio.tenstorrent as _TT
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = device or get_device()
        self._c_z = c_z
        def under(pfx):
            return {k[len(pfx):]: v for k, v in self._w.items() if k.startswith(pfx)}
        resolved_diffusion_fp32 = (os.environ.get("PROTENIX_DIFFUSION_FP32_DEVICE", "1") == "1"
                                   if diffusion_fp32 is None else diffusion_fp32)
        # --fast for Protenix changes only the trunk to bf8. The trunk tolerates bf8
        # (s/z PCC 0.99), but bf8 in the coordinate-sensitive diffusion collapses the
        # structure (Rg 4.7 vs 22). Capture the --fast intent, then build each stage at its
        # own precision; fold() re-applies the per-stage flag (the trunk's triangle/transition
        # ops read _dtype() at RUNTIME, so the global flag must match the weights per stage).
        self._fast = _TT._FAST_MODE
        _TT.set_fast_mode(False)   # input embedder stays bf16; diffusion precision is gate-controlled
        self.input_aae = AtomAttentionEncoder(under("input_embedder.atom_attention_encoder."), compute_kernel_config)
        diffusion_dtype = ttnn.float32 if resolved_diffusion_fp32 else ttnn.bfloat16
        self.diff_feat = AtomFeaturization(under("diffusion_module.atom_attention_encoder."),
                                            compute_kernel_config, dtype=diffusion_dtype)
        _TT.set_fast_mode(self._fast)   # trunk: bf8 when --fast
        self.trunk = Trunk(model_state_dict, compute_kernel_config, c_z=self._c_z,
                           msa_update_first=msa_update_first)
        _TT.set_fast_mode(False)   # diffusion uses its gate-controlled dtype; confidence stays bf16
        self.diffusion = DiffusionModule(under("diffusion_module."), self.dev, compute_kernel_config,
                                          diffusion_fp32=resolved_diffusion_fp32)
        self.confidence_head = ConfidenceHead(under("confidence_head."), self.dev, compute_kernel_config)

    @classmethod
    def load_from_checkpoint(cls, path, compute_kernel_config=None, device=None):
        """Load a v2 checkpoint (.pt) and build the model. Untrusted weights are read
        with weights_only=True."""
        import torch
        import ttnn
        from .tenstorrent import get_device
        dev = device or get_device()
        ckc = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        ck = ck.get("model", ck)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
        return cls(sd, ckc, dev)

    def _tt(self, x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)

    @staticmethod
    def _to_host(t, shape=None):
        import torch
        h = torch.Tensor(ttnn.to_torch(t)).float()
        return h.reshape(shape) if shape is not None else h

    @staticmethod
    def _generate_relp(feats, r_max=32, s_max=2):
        """RelativePositionEncoder feature (reference embedders.generate_relp): one-hot of
        clipped residue/token/chain offsets + same-entity. dims 2(r_max+1)+2(r_max+1)+1+
        2(s_max+1) = 139. Model-side; lets the data pipeline emit only the index features."""
        import torch
        import torch.nn.functional as F
        asym = feats["asym_id"].long(); res = feats["residue_index"].long()
        ent = feats["entity_id"].long(); tok = feats["token_index"].long(); sym = feats["sym_id"].long()
        sc = (asym[:, None] == asym[None, :]).long()
        sr = (res[:, None] == res[None, :]).long()
        se = (ent[:, None] == ent[None, :]).long()
        d_res = torch.clip(res[:, None] - res[None, :] + r_max, 0, 2 * r_max) * sc + (1 - sc) * (2 * r_max + 1)
        d_tok = torch.clip(tok[:, None] - tok[None, :] + r_max, 0, 2 * r_max) * sc * sr + (1 - sc * sr) * (2 * r_max + 1)
        d_ch = torch.clip(sym[:, None] - sym[None, :] + s_max, 0, 2 * s_max) * se + (1 - se) * (2 * s_max + 1)
        return torch.cat([F.one_hot(d_res, 2 * (r_max + 1)), F.one_hot(d_tok, 2 * (r_max + 1)),
                          se[..., None], F.one_hot(d_ch, 2 * (s_max + 1))], dim=-1).float()

    @staticmethod
    def _atom_pair_feats(ref_pos, ref_space_uid):
        """Algorithm 5 lines 1-3 (reference update_input_feature_dict): windowed atom-pair
        feats from ref_pos + ref_space_uid (NQ=32, NK=128, pad_left=48). Validated vs the
        reference (d_lm exact, v_lm/mask exact). Returns d_lm (nb,NQ,NK,3), v_lm (nb,NQ,NK,1),
        mask_trunked (nb,NQ,NK)."""
        import torch
        import torch.nn.functional as F
        N = ref_pos.shape[0]; NQ, NK, PADL = 32, 128, 48
        nb = (N + NQ - 1) // NQ; NP = nb * NQ; qpad = NP - N
        ruid = ref_space_uid.long()
        qpos = F.pad(ref_pos.float(), (0, 0, 0, qpad)).reshape(nb, NQ, 3)
        quid = F.pad(ruid, (0, qpad), value=0).reshape(nb, NQ)              # pad value 0 (reference)
        pad_right = int((nb - 0.5) * NQ + NK / 2 - N + 0.5)
        kpos_p = F.pad(ref_pos.float(), (0, 0, PADL, pad_right))
        kuid_p = F.pad(ruid, (PADL, pad_right), value=0)
        kpos = torch.stack([kpos_p[b * NQ:b * NQ + NK] for b in range(nb)], 0)   # (nb,NK,3)
        kuid = torch.stack([kuid_p[b * NQ:b * NQ + NK] for b in range(nb)], 0)   # (nb,NK)
        d_lm = qpos[:, :, None, :] - kpos[:, None, :, :]                         # (nb,NQ,NK,3)
        v_lm = (quid[:, :, None] == kuid[:, None, :]).float().unsqueeze(-1)      # (nb,NQ,NK,1)
        qidx = torch.arange(NP).reshape(nb, NQ); qval = (qidx < N).float()
        kglob = torch.stack([torch.arange(b * NQ - PADL, b * NQ - PADL + NK) for b in range(nb)], 0)
        kval = ((kglob >= 0) & (kglob < N)).float()
        mask_trunked = qval[:, :, None] * kval[:, None, :]                      # (nb,NQ,NK)
        return d_lm, v_lm, mask_trunked

    def _atom_feat_inputs(self, feats):
        """Build the per-atom feature tensors shared by both atom encoders. Accepts the
        canonical protenix input_feature_dict: d_lm/v_lm/mask_trunked are computed from
        ref_pos + ref_space_uid (model-side, Algorithm 5) when not already provided."""
        import torch
        N = feats["ref_pos"].shape[0]
        f_in = torch.cat([feats["ref_mask"].reshape(N, 1), feats["ref_element"].reshape(N, 128),
                          feats["ref_atom_name_chars"].reshape(N, 256)], dim=-1)
        if "d_lm" in feats and "v_lm" in feats:
            d_lm, v_lm = feats["d_lm"], feats["v_lm"]
            mt = feats.get("mask_trunked")
            if mt is None:
                mt = feats["pad_info"]["mask_trunked"]
        else:
            d_lm, v_lm, mt = self._atom_pair_feats(feats["ref_pos"], feats["ref_space_uid"])
        nb, nq, nk, _ = d_lm.shape
        M = nb * nq * nk
        d = d_lm.reshape(M, 3); v = v_lm.reshape(M, 1)
        invd = (1.0 / (1.0 + (d_lm ** 2).sum(-1, keepdim=True))).reshape(M, 1)
        a2t = feats["atom_to_token_idx"].long(); NT = int(a2t.max()) + 1
        S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0
        return dict(N=N, NT=NT, nb=nb, nq=nq, nk=nk, f_in=f_in, d=d, v=v, invd=invd,
                    mt=mt.float(), a2t=a2t, S=S, ref_charge_asinh=torch.arcsinh(feats["ref_charge"]).reshape(N, 1))

    def _diffusion_pair_cond(self, z_trunk_tt, relp):
        """DiffusionConditioning pair branch (computed once; t-independent):
        zc = LN(concat[z_trunk, relpe(relp)]); pz = linear_z(zc); pz += transition_z1 +
        transition_z2. Reference diffusion_module.diffusion_conditioning. Validated
        PCC ~1.0 (scripts/protenix_diffcond_parity.py). Returns conditioned pair_z host.

        When c_z_pair_diffusion < c_z (OpenDDE: pair-diffusion channel compressed to 128 vs
        the shared Trunk's c_z=384; Protenix-v2 keeps them equal, 256==256, no compression),
        the reference (DiffusionConditioning.prepare_cache / compress_pair_z) first LN+projects
        z_trunk down to c_z_pair_diffusion via layernorm_z_trunk/linear_no_bias_z_trunk, BEFORE
        concatenating with relpe -- gated on those keys' presence so Protenix-v2 is unchanged."""
        from .tenstorrent import Transition
        C = "diffusion_module.diffusion_conditioning."
        T = self.diffusion._up
        relpe = ttnn.linear(T(relp), T(self._w[C + "relpe.linear_no_bias.weight"].t().contiguous()),
                            compute_kernel_config=self.compute_kernel_config, dtype=self.diffusion.dtype,
                            core_grid=CORE_GRID_MAIN)
        if self.diffusion._diffusion_fp32:
            z_trunk_tt = ttnn.typecast(z_trunk_tt, self.diffusion.dtype)
        z_trunk_tt = ttnn.reshape(z_trunk_tt, (relpe.shape[0], relpe.shape[1], -1))
        N, W, cz = (int(d) for d in z_trunk_tt.shape)
        if N * W * cz * 2 > PAIRCOND_BLOCK_BYTES and self.diffusion.dtype == ttnn.bfloat16:
            # Row-blocked chain (see PAIRCOND_BLOCK_BYTES): no full-size LN'd z or channel
            # concat ever materializes; pz assembles on the host (bf16 round trip is
            # bit-preserving) and uploads once at [Ns,Ns,128].
            rb = max(32, (256 * 2 ** 20) // (W * cz * 2) // 32 * 32)
            blocks = []
            for s in range(0, N, rb):
                e = min(s + rb, N)
                zt = z_trunk_tt[s:e]
                if C + "linear_no_bias_z_trunk.weight" in self._w:
                    zn = ttnn.layer_norm(zt, weight=T(self._w[C + "layernorm_z_trunk.weight"]),
                                         epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
                    zt = ttnn.linear(zn, T(self._w[C + "linear_no_bias_z_trunk.weight"].t().contiguous()),
                                     compute_kernel_config=self.compute_kernel_config,
                                     dtype=self.diffusion.dtype, core_grid=CORE_GRID_MAIN)
                    ttnn.deallocate(zn)
                zc = ttnn.concat([zt, relpe[s:e]], dim=-1)
                ttnn.deallocate(zt)
                zc = ttnn.layer_norm(zc, weight=T(self._w[C + "layernorm_z.weight"]), epsilon=1e-5,
                                     compute_kernel_config=self.compute_kernel_config)
                pb = ttnn.linear(zc, T(self._w[C + "linear_no_bias_z.weight"].t().contiguous()),
                                 compute_kernel_config=self.compute_kernel_config,
                                 dtype=self.diffusion.dtype, core_grid=CORE_GRID_MAIN)
                ttnn.deallocate(zc)
                blocks.append(torch.Tensor(ttnn.to_torch(pb)))
                ttnn.deallocate(pb)
            pz = ttnn.from_torch(torch.cat(blocks, dim=0), layout=ttnn.TILE_LAYOUT,
                                 device=get_device(), dtype=self.diffusion.dtype)
            pz = ttnn.reshape(pz, (1, N, N, pz.shape[-1]))
            for nm in ("transition_z1", "transition_z2"):
                sub = {k[len(C + nm + "."):]: v for k, v in self._w.items() if k.startswith(C + nm + ".")}
                t = Transition(PW.remap_transition(sub), self.compute_kernel_config,
                               dtype=self.diffusion.dtype)
                pz = ttnn.add(pz, t(pz))
            return self._to_host(pz)
        if C + "linear_no_bias_z_trunk.weight" in self._w:
            zt = ttnn.layer_norm(z_trunk_tt, weight=T(self._w[C + "layernorm_z_trunk.weight"]),
                                 epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
            z_trunk_tt = ttnn.linear(zt, T(self._w[C + "linear_no_bias_z_trunk.weight"].t().contiguous()),
                                     compute_kernel_config=self.compute_kernel_config,
                                     dtype=self.diffusion.dtype, core_grid=CORE_GRID_MAIN)
        zc = ttnn.concat([z_trunk_tt, relpe], dim=-1)
        zc = ttnn.layer_norm(zc, weight=T(self._w[C + "layernorm_z.weight"]), epsilon=1e-5,
                             compute_kernel_config=self.compute_kernel_config)
        pz = ttnn.linear(zc, T(self._w[C + "linear_no_bias_z.weight"].t().contiguous()),
                         compute_kernel_config=self.compute_kernel_config, dtype=self.diffusion.dtype,
                         core_grid=CORE_GRID_MAIN)
        # keep the pair tensor 4D (1,N,N,c) so Transition uses its chunked H/W path
        # (the 3D path doesn't chunk pair tensors -> OOM at large N).
        N = relpe.shape[0]
        pz = ttnn.reshape(pz, (1, N, N, pz.shape[-1]))
        for nm in ("transition_z1", "transition_z2"):
            sub = {k[len(C + nm + "."):]: v for k, v in self._w.items() if k.startswith(C + nm + ".")}
            t = Transition(PW.remap_transition(sub), self.compute_kernel_config,
                           dtype=self.diffusion.dtype)
            pz = ttnn.add(pz, t(pz))
        return self._to_host(pz)

    def _plm_z_term(self, pair_z, a2t, nb, nq, nk):
        """broadcast_token_to_local_atom_pair: W_z(LN_z(z_trunk)) gathered into windowed
        atom-pair blocks (nb,nq,nk,16). The diffusion atom-encoder's p_lm cache adds this
        trunk-pair-z term (reference transformer.py prepare_cache, r_l path)."""
        import torch
        import torch.nn.functional as F
        E = "diffusion_module.atom_attention_encoder."
        lnz = F.layer_norm(pair_z, (pair_z.shape[-1],)) * self._w[E + "layernorm_z.weight"]
        ztok = F.linear(lnz, self._w[E + "linear_no_bias_z.weight"])     # (NT,NT,16)
        N = a2t.shape[0]; NQ, NK, PADL = 32, 128, 48; NP = nb * NQ
        aq = torch.cat([a2t, torch.zeros(NP - N, dtype=torch.long)]).reshape(nb, NQ)
        ak_src = torch.cat([torch.zeros(PADL, dtype=torch.long), a2t,
                            torch.zeros(PADL + NP + NK, dtype=torch.long)])
        ak = torch.stack([ak_src[b * NQ:b * NQ + NK] for b in range(nb)], 0)   # (nb,nk)
        return torch.stack([ztok[aq[b][:, None].expand(NQ, NK), ak[b][None, :].expand(NQ, NK)]
                            for b in range(nb)], 0)                            # (nb,nq,nk,16)

    def fold_many(self, feats_list, *, n_step=200, seed=None, progress_fn=None,
                  return_confidence=False, n_cycles=None):
        """Fold B targets with ONE batched diffusion trajectory. Returns a list of B coord
        tensors (1,N,3), or (coords, confs) when return_confidence.

        The trunk runs per target: its pair ops already fill the grid at production N, so a
        batch dimension buys nothing there (measured: per-prediction cost at B=8 is 0.93x for
        the pair transition and 1.07x for tri-attention). The diffusion is the opposite -- its
        tensors are atom- and token-sized and leave the grid idle -- so the B conditionings are
        stacked on the sampler's multiplicity axis and one denoise per step covers all B.

        All targets must share the atom count and the token count; bucket before calling.
        Each member draws its own host RNG stream, so member b sees exactly the noise
        fold(feats_list[b], seed=seed) would have drawn alone.
        """
        B = len(feats_list)
        if B == 1:
            out = self.fold(feats_list[0], n_step=n_step, n_sample=1, seed=seed,
                            progress_fn=progress_fn, return_confidence=return_confidence,
                            n_cycles=n_cycles)
            return ([out[0]], [out[1]]) if return_confidence else [out]
        conds, auxs = [], []
        for feats in feats_list:
            cond, aux = self._trunk_cond(feats, progress_fn=progress_fn, n_cycles=n_cycles)
            conds.append(cond)
            auxs.append(aux)
        merged = merge_conds(self.diffusion, conds)
        N = auxs[0]["N"]
        # One batched forward per step: the merged conditioning covers all B members, so the
        # per-step chunking knob (max_parallel_samples) does not apply here -- a chunk would
        # slice the coordinate stream but not the conditioning. DRAM is the limit on B.
        coords = edm_sample(self.diffusion, merged, N, n_step=n_step, multiplicity=B,
                            max_parallel_samples=B, member_seeds=[seed] * B,
                            progress_fn=progress_fn)
        out = [coords[b:b + 1] for b in range(B)]
        if not return_confidence:
            return out
        confs = [self._confidence_for(auxs[b], feats_list[b], out[b][0]) for b in range(B)]
        return out, confs

    def _confidence_for(self, aux, feats, coords):
        """Confidence head for one prediction's coords (N,3). Same device/host split fold()
        uses: the device path needs NT>=128, below which bf16 distance-embed rounding moves
        the pLDDT head."""
        s_inputs, s_trunk, z_trunk = aux["s_inputs"], aux["s_trunk"], aux["z_trunk"]
        if self.confidence_head.device_confidence_enabled() and aux["NT"] >= 128:
            z_base_dev = self.confidence_head.z_base_device(s_inputs, s_trunk, z_trunk)
            return self.confidence_head.confidence_device(s_inputs, s_trunk, z_base_dev, coords, feats)
        return self.confidence_head.confidence(s_inputs, s_trunk, z_trunk, coords, feats)

    def _trunk_cond(self, feats, *, progress_fn=None, n_cycles=None):
        """Trunk plus the t-independent diffusion conditioning for one target: everything
        fold() does before the sampler. Returns (cond, aux); aux carries what the confidence
        head needs. fold() and fold_many() share this so a batched fold runs the same trunk
        as a single one."""
        fi = self._atom_feat_inputs(feats)
        N, NT, nb, nq, nk = fi["N"], fi["NT"], fi["nb"], fi["nq"], fi["nk"]
        mt = fi["mt"]; S = fi["S"]
        tt = self._tt
        # 1) s_inputs (input embedder atom encoder)
        Mmat = (S.t() / (S.t().sum(-1, keepdim=True) + 1e-6))
        dm = feats["deletion_mean"]; dm = dm.reshape(-1, 1) if dm.dim() == 1 else dm
        s_inputs_tt = self.input_aae(
            tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]), tt(feats["ref_mask"].reshape(N, 1)),
            tt(fi["f_in"]), tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt, tt(Mmat),
            tt(feats["restype"]), tt(feats["profile"]), tt(dm))
        s_inputs = self._to_host(s_inputs_tt)[:NT]
        # 2) diffusion atom cache (c_l, p_lm) -- t-independent and gate-controlled
        dtt = self.diffusion._up
        mt_dev = dtt(mt.reshape(-1, 1).float())
        c_l = self._to_host(self.diff_feat.c_l(dtt(feats["ref_pos"]), dtt(fi["ref_charge_asinh"]),
                                               dtt(feats["ref_mask"].reshape(N, 1)), dtt(fi["f_in"])), (N, 128))
        p_lm = self._to_host(self.diff_feat.p_lm(dtt(fi["d"]), dtt(fi["v"]), dtt(fi["invd"]), mt_dev),
                             (nb, nq, nk, 16))
        # 3) trunk (bf8 under --fast: toggle the global flag ON so the trunk's triangle/
        #    transition runtime _dtype() matches its bf8 weights, then restore bf16 for the
        #    coordinate-sensitive diffusion. Trunk tolerates bf8 (s/z PCC 0.99); diffusion does not.)
        import tt_bio.tenstorrent as _TT
        relp = feats["relp"] if "relp" in feats else self._generate_relp(feats)
        if self._fast:
            _TT.set_fast_mode(True)
        s_trunk_tt, z_tt = self.trunk(feats, s_inputs, relp, feats["token_bonds"],
                                      progress_fn=progress_fn, n_cycles=n_cycles)
        if self._fast:
            _TT.set_fast_mode(False)
        s_trunk = self._to_host(s_trunk_tt, (NT, s_trunk_tt.shape[-1]))
        z_trunk = self._to_host(z_tt, (NT, NT, self.trunk.C_Z))   # raw trunk z (for confidence)
        # diffusion pair conditioning (once, t-independent): conditioned pair_z
        pair_z = self._diffusion_pair_cond(z_tt, relp).reshape(NT, NT, self.trunk.C_Z)
        # diffusion p_lm cache also carries the (conditioned) pair-z broadcast to atom pairs
        p_lm = p_lm + self._plm_z_term(pair_z, fi["a2t"], nb, nq, nk)
        # 4) EDM sampler
        cond = {"s_trunk": s_trunk, "s_inputs": s_inputs, "pair_z": pair_z, "c_l": c_l,
                "p_lm": p_lm, "S": S, "mask_trunked": mt.float()}
        # DiT pair input is t-independent -> upload LN(pair_z) once (on-device DiT derives the
        # per-block bias), or precompute the host biases for the fp32 fallback.
        if self.diffusion.device_dit:
            cond["dit_z"] = self.diffusion._dit_z_device(pair_z)
        else:
            cond["dit_biases"] = self.diffusion._dit_pair_biases(pair_z)
        return cond, dict(N=N, NT=NT, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk)

    def fold(self, feats, *, n_step=200, n_sample=1, seed=None, progress_fn=None,
             return_confidence=False, n_cycles=None, trace=False,
             max_parallel_samples=None):
        """Run the full pipeline. feats: model-ready tensor dict. n_cycles = trunk recycling
        iterations (default 10, protenix-v2's spec; fewer trades accuracy for speed). Returns
        coords (n_sample, N, 3) host tensor; if return_confidence, returns (coords, conf) where
        conf is a dict {plddt (mean, float), plddt_atom (N_atom,), pae (N,N), pde (N,N),
        ptm, iptm} for n_sample==1, or a list of such dicts (one per sample) for n_sample>1.
        trace=True replays a captured ttnn trace of the denoise stream (lossless; faster on
        dispatch-bound diffusion, e.g. -22% warm at L256). Requires the device to have been
        opened with a trace region: get_device(trace_region_size=1 << 30)."""
        import torch
        if trace:
            import tt_bio.tenstorrent as _TTd
            if _TTd.trace_region_size() <= 0:
                raise ValueError(
                    "fold(trace=True) needs a device opened with a trace region; "
                    "call get_device(trace_region_size=1 << 30) before folding.")
        cond, _aux = self._trunk_cond(feats, progress_fn=progress_fn, n_cycles=n_cycles)
        N, NT = _aux["N"], _aux["NT"]
        s_inputs, s_trunk, z_trunk = _aux["s_inputs"], _aux["s_trunk"], _aux["z_trunk"]
        import os as _os, time as _time
        if _os.environ.get("TT_PROTENIX_DBG_COND"):
            self._dbg_cond = cond
        _prof = _os.environ.get("TT_PROTENIX_PROFILE")
        # Multiplicity batching: when the device denoise carries the multiplicity dim
        # (DiffusionModule.supports_multiplicity, flipped on once the batched denoise is
        # parity-verified), draw all n_sample samples in ONE batched trajectory
        # (boltz2.AtomDiffusion.sample pattern). Until then, fall back to the per-sample
        # loop (bit-exact with the prior path). max_parallel_samples caps the per-step
        # batched denoise chunk (OOM safety); None = n_sample (one batched forward).
        if n_sample > 1 and getattr(self.diffusion, "supports_multiplicity", False):
            _mps = DEFAULT_MAX_PARALLEL_SAMPLES if max_parallel_samples is None else max_parallel_samples
            if _prof:
                import ttnn as _tn; _tn.synchronize_device(self.diffusion.dev); _ts = _time.time()
            coords = edm_sample(self.diffusion, cond, N, n_step=n_step, multiplicity=n_sample,
                                 max_parallel_samples=_mps, seed=seed, trace=trace,
                                 progress_fn=progress_fn)
            if _prof:
                import ttnn as _tn; _tn.synchronize_device(self.diffusion.dev); print(f"[PROF] edm_sample[batch={n_sample}] {_time.time()-_ts:.3f}s", flush=True)
        else:
            coords = []
            for k in range(n_sample):
                sd_seed = None if seed is None else seed + k
                if _prof:
                    import ttnn as _tn; _tn.synchronize_device(self.diffusion.dev); _ts = _time.time()
                coords.append(edm_sample(self.diffusion, cond, N, n_step=n_step, seed=sd_seed,
                                         trace=trace, progress_fn=progress_fn)[0])
                if _prof:
                    import ttnn as _tn; _tn.synchronize_device(self.diffusion.dev); print(f"[PROF] edm_sample[{k}] {_time.time()-_ts:.3f}s", flush=True)
            coords = torch.stack(coords, 0)
        if return_confidence:
            # Per-sample confidence so callers can rank samples (best-of-N) and
            # report pTM/ipTM/pLDDT per sample. n_sample==1 returns a single dict
            # (back-compat); n_sample>1 returns a list aligned with coords.
            # Device-resident path (opt-in, TT_PROTENIX_CONF_DEVICE=1): keep z_base
            # on device across samples -- pass the raw trunk z device tensor
            # straight in, skipping the (N,N,256) host round-trip the host-heads
            # path takes. Falls back to the host-heads path otherwise.
            if self.confidence_head.device_confidence_enabled() and NT >= 128:
                # z_base (z_trunk + s1 + s2) is sample-invariant: build it ONCE in
                # fp32 on host and upload as a resident bf16 device tensor, then
                # run the per-sample distance-embed + Pairformer + heads on device
                # -- the (N,N,256) z never round-trips per sample. Restricted to
                # NT>=128: at small N the per-sample bf16 dist-embed rounding
                # diverges from the host path's fp32-then-round (amplified by the
                # Pairformer into the precision-sensitive plddt head), so the host
                # path is kept there (it is only ~23 ms at NT=38 anyway).
                z_base_dev = self.confidence_head.z_base_device(s_inputs, s_trunk, z_trunk)
                confs = [self.confidence_head.confidence_device(
                            s_inputs, s_trunk, z_base_dev, coords[k], feats)
                         for k in range(n_sample)]
            else:
                confs = [self.confidence_head.confidence(s_inputs, s_trunk, z_trunk, coords[k], feats)
                         for k in range(n_sample)]
            return coords, (confs[0] if n_sample == 1 else confs)
        return coords


class Trunk(_KeyedWeights):
    """Protenix-v2 trunk: s_inputs -> (s_trunk, z_trunk) over 10 recycling cycles.

    Each cycle: z = z_init + linear_z_cycle(LN(z)); z += template_embedder(z);
    z = msa_module(z, m); s = s_init + linear_s_cycle(LN(s)); (s,z) = pairformer48(s,z).
    Composes TrunkInput + 48-block Pairformer + template embedder (nt templates x 2
    pair-only blocks) + 4-block MSA module, all reusing tt_bio.tenstorrent primitives
    with v2 weights (tt_bio.protenix_weights remaps). Validated vs the real v2 reference
    (PCC s 0.991 / z 0.990; scripts/protenix_trunk_assembly.py). Reference:
    protenix/model/protenix.py get_pairformer_output."""

    N_CYCLES = 10
    C_Z = 256          # Protenix-v2 default; instances override via __init__(c_z=...)
    TRI_HEAD_DIM = 32  # constant across c_z variants (Protenix-v2 256/8 heads, OpenDDE 384/12)

    def __init__(self, model_state_dict, compute_kernel_config, c_z=None,
                 msa_update_first=False):
        """model_state_dict: full v2-family model dict with the 'module.' prefix STRIPPED.
        c_z: pair channel width (default 256, Protenix-v2's; OpenDDE's shared Trunk subtree
        is c_z=384 -- same architecture, wider pair, head_dim fixed at 32 so n_tri_heads
        scales as c_z // 32)."""
        import re
        from .tenstorrent import (get_device, Pairformer, PairformerLayer,
                                   OuterProductMean, PairWeightedAveraging, Transition)
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = get_device()
        self.C_Z = c_z or self.C_Z
        self._msa_update_first = msa_update_first
        n_tri_heads = self.C_Z // self.TRI_HEAD_DIM
        self._wc = {}  # cached device weights (upload once; reused every recycle cycle)
        ti_keys = ("linear_no_bias_sinit", "linear_no_bias_zinit1", "linear_no_bias_zinit2",
                   "linear_no_bias_token_bond", "relative_position_encoding")
        ti_sd = {k: v for k, v in self._w.items() if any(k.startswith(p) for p in ti_keys)}
        self.trunk_input = TrunkInput(ti_sd, compute_kernel_config)
        # 48-block pairformer
        nb_pf = 1 + max(int(re.search(r"pairformer_stack\.blocks\.(\d+)\.", k).group(1))
                        for k in self._w if "pairformer_stack.blocks." in k)
        comb = {}
        for i in range(nb_pf):
            blk = {k[len(f"pairformer_stack.blocks.{i}."):]: v for k, v in self._w.items()
                   if k.startswith(f"pairformer_stack.blocks.{i}.")}
            for k, v in PW.remap_pairformer_block(blk).items():
                comb[f"layers.{i}.{k}"] = v
        self.PF = Pairformer(nb_pf, self.TRI_HEAD_DIM, n_tri_heads, 384 // 16, 16, True, comb, compute_kernel_config)
        # template embedder: 2 pair-only PairformerLayers
        tpl = {k[len(f"template_embedder.pairformer_stack.blocks.{b}."):]: v for b in range(2)
               for k, v in self._w.items()
               if k.startswith(f"template_embedder.pairformer_stack.blocks.{b}.")}
        self.TPL = [PairformerLayer(32, 2, None, None, False,
                    PW.remap_msa_pair_stack({k[len(f"template_embedder.pairformer_stack.blocks.{b}."):]: v
                                             for k, v in self._w.items()
                                             if k.startswith(f"template_embedder.pairformer_stack.blocks.{b}.")}),
                    compute_kernel_config) for b in range(2)]
        # 4-block MSA module
        self.MSA = []
        nb_msa = 4
        for i in range(nb_msa):
            P = f"msa_module.blocks.{i}."
            sub = lambda pp: {k[len(pp):]: v for k, v in self._w.items() if k.startswith(pp)}
            opm = OuterProductMean(PW.remap_outer_product_mean(sub(P + "outer_product_mean_msa.")), compute_kernel_config)
            pl = PairformerLayer(self.TRI_HEAD_DIM, n_tri_heads, None, None, False, PW.remap_msa_pair_stack(sub(P + "pair_stack.")), compute_kernel_config)
            has = any(k.startswith(P + "msa_stack.") for k in self._w)
            pwa = tm = None
            if has:
                pwa = PairWeightedAveraging(8, 8, PW.remap_pair_weighted_averaging(sub(P + "msa_stack.msa_pair_weighted_averaging.")), compute_kernel_config)
                tm = Transition(PW.remap_transition(sub(P + "msa_stack.transition_m.")), compute_kernel_config)
            self.MSA.append((opm, pwa, tm, pl))

    def _template(self, z3, tpl_a, N, nt):
        zn = self._ln(z3, "template_embedder.layernorm_z.weight", "template_embedder.layernorm_z.bias")
        u = None
        for t in range(nt):
            v = ttnn.add(tpl_a[t],
                         self._lin(zn, "template_embedder.linear_no_bias_z.weight"))
            for pl in self.TPL:
                v = pl(None, v)[1]
            v = self._ln(v, "template_embedder.layernorm_v.weight", "template_embedder.layernorm_v.bias")
            u = v if u is None else ttnn.add(u, v)
        u = ttnn.multiply(u, 1.0 / (1e-7 + nt))
        return self._lin(ttnn.relu(u), "template_embedder.linear_no_bias_u.weight")

    def _msa(self, z3, m_feat):
        def update_msa(m, z, pwa, transition):
            if pwa is None:
                return m
            if isinstance(m, list):
                # Already chunked: transform each chunk in place of the list, freeing the old
                # chunk as soon as its replacement exists, so the peak stays ~2 chunks over the
                # list rather than a second full copy.
                zc = ttnn.clone(z)
                out = []
                for mc in m:
                    t1 = ttnn.add(mc, ttnn.reshape(pwa(mc, zc), tuple(mc.shape)))
                    ttnn.deallocate(mc)
                    t2 = ttnn.add(t1, ttnn.reshape(transition(t1), tuple(t1.shape)))
                    ttnn.deallocate(t1)
                    out.append(t2)
                    dram_peak("trunk msa block: after pwa add (list)")
                ttnn.deallocate(zc)
                return out
            D = m.shape[1]                                  # MSA depth
            # Row-chunk the MSA depth axis when the representation is too big to hold several
            # copies of. Unchunked, PairWeightedAveraging's FIRST op is an out-of-place
            # layer_norm over the whole tensor, so a second full copy has to exist alongside
            # the input; measured on a 12 GiB Wormhole part, that allocation is exactly what
            # a deep-MSA target dies on (9qqf refused 2.494 GiB with m_feat = 2.494 GiB).
            #
            # This is bit-exact, not an approximation: nothing in this path reduces along the
            # depth axis. PWA's weights come from `z` alone (linear(z,...) then softmax), its
            # matmul contracts the TOKEN axis, its head accumulation is over heads, and
            # layer_norm/Transition are per-row over channels. So each output row depends only
            # on its own input row and on z, and partitioning the rows cannot change any
            # accumulation order. (OuterProductMean is the opposite case -- its reduction IS
            # over depth -- which is why it stays full-depth below and must not be chunked.)
            if _msa_take_whole_path(D * m.shape[2] * m.shape[3] * 2):
                m_up = self._up(m) if torch.is_tensor(m) else m   # host offload + forced whole
                m2 = ttnn.add(m_up, ttnn.reshape(
                    pwa(m_up, ttnn.clone(z)), tuple(m_up.shape)))
                dram_peak("trunk msa block: after pwa add")
                trunk_tap("whole:after_pwa", m2)
                out = ttnn.add(
                    m2, ttnn.reshape(transition(m2), tuple(m2.shape)))
                trunk_tap("whole:after_transition", out)
                if m_up is not m:
                    ttnn.deallocate(m_up)
                return out
            # One clone of z for the whole loop, not one per chunk: PWA only reads z (it
            # rebinds its own local through reshape/layer_norm and never deallocates it), and
            # at c_z=384 a per-chunk clone would cost ~0.9 GiB each for a 1120-token target.
            zc = ttnn.clone(z)
            parts = []
            cw = _msa_row_chunk_size()
            host_m = torch.is_tensor(m)     # host-resident pristine (deep-MSA offload)
            for s in range(0, D, cw):
                if host_m:
                    mc = self._up(m[:, s:min(s + cw, D)])         # stream one chunk from host
                else:
                    mc = m[:, s:min(s + cw, D), :, :]             # slice => private copy
                # Deliberately the SAME out-of-place ttnn.add as the unchunked branch above.
                # An in-place add_ here would be safe for aliasing (mc is a private copy) and
                # would save a chunk-sized buffer, but it is a second change riding along with
                # the chunking, and the acceptance test cannot then attribute a difference to
                # one or the other. Keep the arithmetic identical to the whole path; the memory
                # win comes from the chunk being small, not from mutating it.
                mc = ttnn.add(mc, ttnn.reshape(pwa(mc, zc), tuple(mc.shape)))
                mc = ttnn.add(mc, ttnn.reshape(transition(mc), tuple(mc.shape)))
                parts.append(mc)
                dram_peak("trunk msa block: after pwa add")
            ttnn.deallocate(zc)
            # Return the CHUNKS, not a concatenation of them. The terminal concat used to need a
            # third full-size buffer (source + parts + destination) and that 3x peak is what made
            # 9d72 OOM at 1.78 GiB m_feat even with chunking on. Every downstream consumer is
            # chunk-wise -- PWA and Transition are per-row, and OuterProductMean now takes the
            # list and joins only its c=32 projections -- so the contiguous tensor is never needed.
            return parts

        # DO NOT make update_msa's first add in-place on the tensor passed in. `m_feat` is
        # built ONCE outside the recycling loop in __call__ and this method is re-entered for
        # every cycle, so the caller's buffer has to survive intact; mutating it would corrupt
        # cycles 2..n_cycles. That is a numerical change, not an allocation change, and it
        # would not show up as an OOM -- only as wrong coordinates. The chunked path above is
        # safe because it slices (a copy) and mutates only the slice.
        global _TRUNK_TAP_CALL
        if torch.is_tensor(m_feat):     # host-resident pristine (deep-MSA offload)
            trunk_tap_host("msa_enter:m_feat", m_feat)
        else:
            trunk_tap("msa_enter:m_feat", m_feat)
        trunk_tap("msa_enter:z3", z3)
        for _bi, (opm, pwa, tm, pl) in enumerate(self.MSA):
            dram_peak(f"trunk msa block {_bi} enter")
            # OpenDDE refreshes the MSA before OPM. Protenix-v2 retains the
            # ordering its checkpoint was trained with.
            if self._msa_update_first:
                m_feat = update_msa(m_feat, z3, pwa, tm)
            if torch.is_tensor(m_feat):
                # Protenix-order block 0 reads the pristine m with OPM. OPM chunk-gates itself,
                # so a single transient upload is enough; update_msa below streams from host.
                m_dev = self._up(m_feat)
                z3 = ttnn.add(z3, opm(m_dev, None, None))
                ttnn.deallocate(m_dev)
            else:
                z3 = ttnn.add(z3, opm(m_feat, None, None))
            dram_peak("trunk msa block: after opm")
            if not self._msa_update_first:
                m_feat = update_msa(m_feat, z3, pwa, tm)
            z3 = pl(None, z3)[1]
            dram_peak(f"trunk msa block {_bi} done")
            if not torch.is_tensor(m_feat):
                trunk_tap(f"block{_bi}:m_feat", m_feat)
            trunk_tap(f"block{_bi}:z3", z3)
        # The per-cycle chunk list is dead once the blocks are done: __call__ keeps the original
        # contiguous m_feat and re-derives the chunks next cycle (§34.1 -- the caller's copy must
        # stay pristine, so it is never freed here). A host-resident pristine needs no freeing.
        if isinstance(m_feat, list):
            for _c in m_feat:
                ttnn.deallocate(_c)
        _TRUNK_TAP_CALL += 1
        return z3

    def __call__(self, feat, s_inputs, relp, token_bonds, progress_fn=None, n_cycles=None):
        """feat: dict with template_* / msa / has_deletion / deletion_value / asym_id (host
        tensors). s_inputs (N,449), relp (N,N,139), token_bonds (N,N) host. n_cycles is the
        number of recycling iterations (default N_CYCLES=10, protenix-v2's spec). Returns
        (s_trunk (N,384), z_trunk (1,N,N,256)) as ttnn tensors."""
        import torch
        import torch.nn.functional as F
        N = s_inputs.shape[0]
        s_init, z_init = self.trunk_input(self._up(s_inputs), self._up(relp), self._up(token_bonds.unsqueeze(-1)))
        # template feature concat (per template). Offline (no-template) inference omits
        # template_* entirely -> nt=0, template embedder skipped (the reference's
        # use_template=False path carries all-zero template geometry, a negligible update).
        asym = feat["asym_id"]; mc = (asym[:, None] == asym[None, :]).float(); pm = torch.ones(N, N)
        nt = feat["template_aatype"].shape[0] if "template_aatype" in feat else 0
        te_at = []
        for t in range(nt):
            dg = feat["template_distogram"][t] * mc[..., None] * pm[..., None]
            pb = (feat["template_pseudo_beta_mask"][t] * mc * pm).unsqueeze(-1)
            aa = F.one_hot(feat["template_aatype"][t].long(), 32).float()
            aai = aa[None].expand(N, N, 32); aaj = aa[:, None].expand(N, N, 32)
            uv = feat["template_unit_vector"][t] * mc[..., None] * pm[..., None]
            bb = (feat["template_backbone_frame_mask"][t] * mc * pm).unsqueeze(-1)
            te_at.append(torch.cat([dg, pb, aai, aaj, uv, bb], -1))
        # te_at is cycle-invariant: upload + project it once here instead of re-uploading
        # ~57 MB x nt on every recycle cycle inside _template. Per-fold local on purpose:
        # N varies between targets, so never cache this on self.
        tpl_a = [self._lin(self._up(t.unsqueeze(0)), "template_embedder.linear_no_bias_a.weight")
                 for t in te_at]
        # msa feature
        msa = F.one_hot(feat["msa"].long(), 32).float()
        ms = torch.cat([msa, feat["has_deletion"].unsqueeze(-1), feat["deletion_value"].unsqueeze(-1)], -1).unsqueeze(0)
        # The MSA representation is this trunk's DRAM limiter on a 12 GiB part: it scales as
        # depth * tokens, and a deep-MSA target OOMs right here. Tag the upload and the
        # projection SEPARATELY, with shape and dtype, because the two are the same number of
        # bytes under different readings (34->64 padded channels in fp32 vs c_m channels in
        # bf16) and only the tensor's own dtype tells them apart. No-op without TT_BIO_DRAM_PEAK.
        # Tag WITHOUT binding the intermediates to locals. Naming them measurably changes what
        # is measured: ttnn frees a buffer when its last Python reference drops, so hoisting
        # `_up(ms)` and the projection into locals kept ~3.7 GiB alive past the point the
        # original expression frees it, and moved the OOM to a different allocation.
        dram_peak(f"trunk msa pre-upload [ms={tuple(ms.shape)} {ms.dtype}"
                  f" -> dtype={getattr(self, 'dtype', ttnn.bfloat16)}]")
        m_feat = ttnn.add(self._lin(self._up(ms), "msa_module.linear_no_bias_m.weight"),
                          self._lin(self._up(s_inputs), "msa_module.linear_no_bias_s.weight"))
        dram_peak(f"trunk m_feat built [{tuple(m_feat.shape)} {m_feat.dtype}]")
        # Deep-MSA offload: every read of the pristine m_feat is row-local (PWA/Transition per
        # row, OPM per chunk), so past 1 GiB it is kept on the host between recycling cycles and
        # streamed up one depth-chunk at a time in update_msa. That removes a full-size device
        # copy from the per-cycle peak (pristine + updated list used to coexist). Bit-exact:
        # to_torch preserves the bf16 bytes and _up re-tilizes the same values, so each chunk
        # holds exactly the bytes a device slice would have held. bf16 only: a bfloat8_b tensor
        # is block-floating-point, so a host round-trip would re-quantise the tile scales.
        if (m_feat.dtype == ttnn.bfloat16
                and m_feat.shape[1] * m_feat.shape[2] * m_feat.shape[3] * 2 > _msa_host_offload_min_bytes()):
            _m_host = ttnn.to_torch(m_feat)
            ttnn.deallocate(m_feat)
            m_feat = _m_host
            dram_peak(f"trunk m_feat host-offloaded [{tuple(m_feat.shape)} torch]")
        z3 = ttnn.reshape(ttnn.mul(z_init, 0.0), (1, N, N, self.C_Z))
        s = ttnn.mul(s_init, 0.0)
        n_cycles = self.N_CYCLES if n_cycles is None else n_cycles
        for cyc in range(n_cycles):
            if progress_fn:
                progress_fn("trunk", step=cyc, total=n_cycles)
            zc = self._lin(self._ln(z3, "layernorm_z_cycle.weight", "layernorm_z_cycle.bias"), "linear_no_bias_z_cycle.weight")
            z3 = ttnn.add(ttnn.reshape(z_init, (1, N, N, self.C_Z)), zc)
            if nt > 0:
                z3 = ttnn.add(z3, self._template(z3, tpl_a, N, nt))
            z3 = self._msa(z3, m_feat)
            sc = self._lin(self._ln(s, "layernorm_s.weight", "layernorm_s.bias"), "linear_no_bias_s.weight")
            s = ttnn.add(s_init, sc)
            s, z3 = self.PF(ttnn.reshape(s, (1, N, 384)), z3)
            s = ttnn.reshape(s, (N, 384))
            # The one region cycle 0's MSA taps did not cover: the s update and the trunk-level
            # Pairformer. Cheap to tap every cycle -- z3 is ~62 MB here, not m_feat's ~0.7 GiB --
            # so this both tests cycle 0 and, if cycle 0 matches, names the first cycle that does not.
            trunk_tap(f"cyc{cyc}_after_PF:s", s, always=True)
            trunk_tap(f"cyc{cyc}_after_PF:z3", z3, always=True)
        for t in tpl_a:
            ttnn.deallocate(t)
        # The trunk's final conditioning. If cycle 0's blocks matched but this differs, the
        # divergence is born in a later cycle; if this matches too, the trunk is fully exonerated
        # and only the diffusion path (and the allocator state it inherits) can explain the fold.
        trunk_tap("trunk_exit:s_trunk", s, always=True)
        trunk_tap("trunk_exit:z_trunk", z3, always=True)
        return s, z3


def merge_conds(diffusion_module, conds):
    """Stack B per-target diffusion conditionings into one M=B conditioning.

    The M-aware denoise already builds exactly these shapes when it replicates a shared
    conditioning over M samples, so nothing downstream changes: per-member tensors gain a
    leading B (c_la, ss_base, the atom<->token pooling matrices, the DiT pair biases) and
    the windowed atom-pair biases concatenate along their block dim to B*nb. Every target
    must share the atom count N and the token count NT.

    Returns a cond dict the batched denoise consumes directly. Host entries (shapes only,
    on this path) come from member 0.
    """
    B = len(conds)
    for c in conds:
        diffusion_module._atom_cond(c)
        if diffusion_module.device_dit and c.get("dit_z") is not None and "dit_block_biases" not in c:
            c["dit_block_biases"] = diffusion_module._dit_block_biases(
                c["dit_z"], c.get("structural_pair_attn_bias"))
    N = conds[0]["c_l"].shape[0]
    NT = conds[0]["s_inputs"].shape[0]
    for i, c in enumerate(conds[1:], 1):
        if c["c_l"].shape[0] != N or c["s_inputs"].shape[0] != NT:
            raise ValueError(
                f"batch member {i} has (atoms, tokens) "
                f"({c['c_l'].shape[0]}, {c['s_inputs'].shape[0]}), member 0 has ({N}, {NT}); "
                "bucket targets by shape before batching")
    cat = lambda ts: ttnn.to_layout(ttnn.concat(ts, dim=0), ttnn.TILE_LAYOUT)
    m = dict(conds[0])
    m["c_la_dev"] = cat([ttnn.reshape(c["c_la_dev"], (1, N, 128)) for c in conds])
    m["ss_base"] = cat([ttnn.reshape(c["ss_base"], (1, NT, c["ss_base"].shape[-1])) for c in conds])
    m["Smean_dev"] = cat([ttnn.reshape(c["Smean_dev"], (1, NT, N)) for c in conds])
    m["S_dev"] = cat([ttnn.reshape(c["S_dev"], (1, N, NT)) for c in conds])
    m["p_dev"] = cat([c["p_dev"] for c in conds])
    for key in ("atxE_bias", "atxD_bias"):
        z_pre = [c[key][0] for c in conds]
        m[key] = ([cat([z[b] for z in z_pre]) for b in range(len(z_pre[0]))],
                  cat([c[key][1] for c in conds]))
    if conds[0].get("dit_block_biases") is not None:
        m["dit_block_biases"] = [cat([c["dit_block_biases"][b] for c in conds])
                                 for b in range(len(conds[0]["dit_block_biases"]))]
    m["_members"] = B
    return m


def edm_sample(diffusion_module, cond, n_atoms, *, multiplicity=1, max_parallel_samples=None,
               member_seeds=None, n_step=200, gamma0=0.8, gamma_min=1.0,
               noise_scale=1.003, step_scale=1.5, sigma_data=16.0, s_max=160.0, s_min=4e-4,
               rho=7.0, seed=None, trace=False, progress_fn=None, dump_fn=None):
    """AF3 EDM ancestral sampler for Protenix-v2 (same family as Boltz-2's
    AtomDiffusion.sample; reuses tt_bio.boltz2.compute_random_augmentation). Produces
    atom coords by iteratively denoising from noise with diffusion_module.denoise.

    The v2 noise schedule uses denominator N_step (i/N), verified to reproduce the real
    v2 reference t_hat sequence to 4 sig figs (4608, 2490, ... 0.1264 for N_step=10):
        sigma[i] = sigma_data * (s_max^(1/rho) + (i/N_step)*(s_min^(1/rho)-s_max^(1/rho)))^rho
    then a final sigma=0; gammas[i] = gamma0 if sigma[i] > gamma_min else 0; per step
    (sigma_tm=sigmas[k], sigma_t=sigmas[k+1], gamma=gammas[k+1]); t_hat=sigma_tm*(1+gamma).
    cond is the fixed trunk conditioning dict passed to DiffusionModule.denoise.

    multiplicity (M): number of independent samples drawn in ONE batched trajectory.
    The conditioning is sample-invariant (same target -> same s_trunk/s_inputs/pair_z/
    c_l/p_lm/S/mask), so only the coordinate stream x_noisy carries the M dim -- mirrors
    boltz2.AtomDiffusion.sample's multiplicity pattern (repeat_interleave would be a no-op
    here since cond is shared, not per-sample). max_parallel_samples caps the per-step
    batched denoise chunk; the per-step device footprint is linear in the chunk, so this
    is the memory knob (None = M, one batched forward -- Protenix.fold and
    OpenDDE.fold substitute DEFAULT_MAX_PARALLEL_SAMPLES rather than pass None through).

    Parity: at multiplicity=1 (default) this is bit-exact with the prior per-sample loop
    -- same single seed, same (1,N,3) draws, one _denoise((1,N,3),...) call per step.
    At multiplicity>1 the sampler draws all M samples from one RNG stream (seeded once
    with `seed`), so each sample's noise differs from the old per-sample loop's seed+k
    draw but is statistically equivalent; the fold lands in the same basin within the
    seed-to-seed noise floor (PCC parity, the established diffusion-leg bar). This matches
    boltz2.AtomDiffusion.sample, which also uses one stream for its multiplicity batch.

    trace=True replays a captured ttnn trace of the denoise device stream (lossless;
    collapses per-step dispatch on dispatch-bound diffusion). The captured trace is fixed
    at (1,N,3), so trace=True with multiplicity>1 falls back to the untraced denoise
    (correctness first; a batched trace would need re-capture per (N,M)). Requires the
    device to have been opened with a trace region (get_device(trace_region_size=...))."""
    import torch
    from .boltz2 import compute_random_augmentation
    M = max(1, int(multiplicity))
    if max_parallel_samples is None or max_parallel_samples > M:
        max_parallel_samples = M
    max_parallel_samples = max(1, int(max_parallel_samples))
    # trace is captured at (1,N,3): keep it only for the unbatched path; fall back to
    # the untraced (but batch-aware) denoise for M>1 so the device forward is correct.
    _denoise = (diffusion_module.denoise_traced if trace and M == 1 else diffusion_module.denoise)
    if seed is not None:
        torch.manual_seed(seed)
    inv_rho = 1.0 / rho
    i = torch.arange(n_step, dtype=torch.float64)
    sig = sigma_data * (s_max ** inv_rho + (i / n_step) * (s_min ** inv_rho - s_max ** inv_rho)) ** rho
    sigmas = torch.cat([sig, torch.zeros(1, dtype=torch.float64)]).float()      # (n_step+1,)
    gammas = torch.where(sigmas > gamma_min, torch.tensor(gamma0), torch.tensor(0.0))
    shape = (M, n_atoms, 3)
    # Shared-draws hook for the integration-parity gate (see tt_bio/boltz2.py
    # AtomDiffusion.sample for the full rationale). The ttnn and torch trunks
    # consume the global RNG differently between worker.py's single up-front
    # torch.manual_seed and this sampler, so device and CPU-reference folds would
    # otherwise draw DIFFERENT diffusion noise and the envelope numerator would
    # compare different basins, not arithmetic. edm_sample already re-seeds to
    # `seed` at its top, but that only holds when the caller threads a non-None
    # seed identically on both paths; TT_BIO_SHARED_DRAW_SEED forces an identical
    # global-RNG state immediately before the first draw regardless. Both protenix
    # and opendde route their initial noise through this one function, so this hook
    # covers both. Unset in production, so normal folds are untouched.
    import os as _os
    _sds = _os.environ.get("TT_BIO_SHARED_DRAW_SEED")
    if _sds:
        torch.manual_seed(int(_sds))
    # member_seeds gives each batch member its own host RNG stream, advanced in the order a
    # standalone fold advances it: initial noise, then per step the augmentation and eps. A
    # batch of DIFFERENT targets then draws exactly what each target would have drawn alone,
    # so a batched member is comparable to its own single fold rather than to a different
    # noise stream (memory diffusion-port-parity-shared-draws). Samples of one target keep
    # the single-stream behaviour (member_seeds=None).
    _per_member = None
    if member_seeds is not None:
        assert len(member_seeds) == M, f"member_seeds {len(member_seeds)} != multiplicity {M}"
        _states = []
        for _sd in member_seeds:
            torch.manual_seed(int(_sds) if _sds else (0 if _sd is None else int(_sd)))
            _states.append(torch.get_rng_state())

        def _per_member(fn):
            outs = []
            for _b in range(M):
                torch.set_rng_state(_states[_b])
                outs.append(fn())
                _states[_b] = torch.get_rng_state()
            return outs
        x = sigmas[0] * torch.cat(_per_member(lambda: torch.randn((1, n_atoms, 3))), 0)
    else:
        x = sigmas[0] * torch.randn(shape)
    if dump_fn is not None:                          # optional trajectory dump (default off)
        for _m in range(M):                          # preserve the (1,N,3) per-sample dump contract
            dump_fn(-1, x[_m:_m + 1].detach().cpu())  # step -1 == initial noise frame
    # Per-step chunked batched denoise (boltz2 max_parallel_samples pattern): one batched
    # _denoise call per chunk per step instead of one per sample per step. The chunking
    # is over the multiplicity dim only -- cond is shared (sample-invariant), so it is
    # NOT replicated; the device denoise is responsible for carrying the chunk's leading
    # dim through the atom encoder / DiT / decoder (see DiffusionModule.denoise).
    sample_ids = torch.arange(M)
    n_chunks = max(1, (M + max_parallel_samples - 1) // max_parallel_samples)
    chunks = [c for c in sample_ids.chunk(n_chunks) if c.numel() > 0]
    for k in range(n_step):
        if progress_fn:
            progress_fn("diffusion", step=k, total=n_step)
        sigma_tm, sigma_t, gamma = sigmas[k].item(), sigmas[k + 1].item(), gammas[k + 1].item()
        if _per_member is not None:
            _aug = _per_member(lambda: compute_random_augmentation(1, device=x.device, dtype=x.dtype))
            R = torch.cat([a[0] for a in _aug], 0)
            tr = torch.cat([a[1] for a in _aug], 0)
        else:
            R, tr = compute_random_augmentation(M, device=x.device, dtype=x.dtype)
        x = x - x.mean(dim=-2, keepdim=True)
        x = torch.einsum("bmd,bds->bms", x, R) + tr
        t_hat = sigma_tm * (1 + gamma)
        noise_var = noise_scale ** 2 * (t_hat ** 2 - sigma_tm ** 2)
        if noise_var <= 0:
            eps = torch.zeros(shape)
        elif _per_member is not None:
            eps = (noise_var ** 0.5) * torch.cat(
                _per_member(lambda: torch.randn((1, n_atoms, 3))), 0)
        else:
            eps = (noise_var ** 0.5) * torch.randn(shape)
        x_noisy = x + eps
        denoised = torch.zeros_like(x_noisy)
        for _chunk in chunks:
            denoised[_chunk] = _denoise(x_noisy[_chunk], torch.tensor([t_hat], dtype=torch.float32), cond)
        dram_peak(f"edm step {k}")
        trunk_tap_host(f"edm_denoised[step{k}]", denoised)
        d = (x_noisy - denoised) / t_hat
        x = x_noisy + step_scale * (sigma_t - t_hat) * d
        trunk_tap_host(f"edm_x[step{k}]", x)
        if dump_fn is not None:                      # per-step coords (noise -> structure)
            for _m in range(M):
                dump_fn(k, x[_m:_m+1].detach().cpu())
    return x


class TrunkInput(_KeyedWeights, Module):
    """Protenix trunk input construction: s_inputs -> s_init, z_init.
    s_init = linear_sinit(s_inputs); z_init = zinit1(s_init)[:,None] + zinit2(s_init)[None]
    + relative_position_encoding(relp) + token_bond(token_bonds). All LinearNoBias.
    Reference: protenix/model/protenix.py get_pairformer_output (lines 208-226).
    Validated vs real v2 golden (PCC 0.999997). (Constraint embedder omitted: the
    inference feat carries no active constraints for plain folding.)"""

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self._w = {k: v for k, v in self.weights.data.items()}

    def __call__(self, s_inputs, relp, token_bonds):
        """s_inputs (N,449); relp (N,N,139); token_bonds (N,N,1). Returns (s_init (N,c_s),
        z_init (N,N,c_z))."""
        N = s_inputs.shape[0]
        s_init = self._lin(s_inputs, "linear_no_bias_sinit.weight")
        cz = self._w["linear_no_bias_zinit1.weight"].shape[0]
        z1 = ttnn.reshape(self._lin(s_init, "linear_no_bias_zinit1.weight"), (N, 1, cz))
        z2 = ttnn.reshape(self._lin(s_init, "linear_no_bias_zinit2.weight"), (1, N, cz))
        z = ttnn.add(z1, z2)
        z = ttnn.add(z, self._lin(relp, "relative_position_encoding.linear_no_bias.weight"))
        z = ttnn.add(z, self._lin(token_bonds, "linear_no_bias_token_bond.weight"))
        return s_init, z
