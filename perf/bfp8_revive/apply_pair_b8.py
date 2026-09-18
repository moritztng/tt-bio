#!/usr/bin/env python3
"""Port wk/b2x-bfp8-pair-track's TT_BIO_PAIR_B8 storage flag onto current main.

The original (96baedf0b, 2026-09-11) no longer cherry-picks: tenstorrent.py has moved under it in
four of the eight hunks. This re-applies the SAME SEMANTICS by exact-string replacement so every
site is named and a silent miss is impossible -- each replacement asserts it matched exactly once.

Semantics, unchanged from the original:
  * the PAIR-SCALE activations (z and everything trimul/transition derive from it) are stored in
    bfloat8_b; weights, projections and the token track stay bf16.
  * the channel move widens on the CHUNK dtype rather than on _FAST_MODE, because
    reblock_permute.eligible* reject a non-bf16 operand and always have.
  * z is cast once entering the pairformer stack and once leaving it, not per block.
"""
from pathlib import Path

F = Path(__file__).resolve().parents[2] / "tt_bio" / "tenstorrent.py"
t = F.read_text()


def sub(old, new, n=1, tag=""):
    global t
    c = t.count(old)
    assert c == n, f"{tag}: expected {n} match(es), found {c} for {old[:70]!r}"
    t = t.replace(old, new)


# ---- 1. the flag and the pair-scale dtype helper, right after _dtype -----------------------
sub(
    """    return ttnn.bfloat8_b if _FAST_MODE else ttnn.bfloat16


def _no_host_pad""",
    '''    return ttnn.bfloat8_b if _FAST_MODE else ttnn.bfloat16


# Store the PAIR TRACK in bfloat8_b while weights, projections and the token track stay bf16. A
# bf16 tile is 2048 B and a bfloat8_b tile is 1088 B (1024 mantissa bytes plus a 64-byte shared
# exponent), so this is 0.531x the bytes at the same transaction count. Opt-in and off by default:
# it moves the structure, and under the bfp8 accuracy policy (state/bfp8-accuracy-policy.md) that
# is a reported cost rather than a gate, but only for a user who asked for it.
_PAIR_B8 = env_flag("TT_BIO_PAIR_B8", False)


def _pair_dtype(default=None):
    """Storage dtype for a PAIR-SCALE activation: z and everything the trimul derives from it.

    Distinct from `_dtype()`, which also governs weights: `_FAST_MODE` demoting stored weights to
    bfloat8_b regressed esmfold2 confidence to NaN on Wormhole, so the two questions stay
    separable. An explicit `_DTYPE_OVERRIDE` (the fp32 affinity trunk) still wins.
    """
    if _PAIR_B8 and _DTYPE_OVERRIDE is None:
        return ttnn.bfloat8_b
    return _dtype(default)


def _no_host_pad''',
    tag="flag+helper",
)

# ---- 2. the trimul in-projection, both kernels ---------------------------------------------
sub(
    """        out = DN.in_proj(x, w, ckc, _dtype(), memory_config, split)""",
    """        out = DN.in_proj(x, w, ckc, _pair_dtype(), memory_config, split)""",
    tag="in_proj dualnoc",
)
sub(
    """    return ttnn.experimental.minimal_matmul(
        x, w, bias_tensor=bias, memory_config=memory_config, dtype=_dtype(),
        compute_kernel_config=ckc)""",
    """    return ttnn.experimental.minimal_matmul(
        x, w, bias_tensor=bias, memory_config=memory_config, dtype=_pair_dtype(),
        compute_kernel_config=ckc)""",
    tag="in_proj minimal_matmul",
)

# ---- 3. the trimul output projection, both kernels -----------------------------------------
sub(
    """            x, weight, bias_tensor=bias, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=_dtype(),
            compute_kernel_config=ckc,""",
    """            x, weight, bias_tensor=bias, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=_pair_dtype(), compute_kernel_config=ckc,""",
    tag="out_proj minimal_matmul",
)
sub(
    """    return _pair_proj_linear(x, weight, ckc, _dtype(), l1_out=True, bias=bias)""",
    """    return _pair_proj_linear(x, weight, ckc, _pair_dtype(), l1_out=True, bias=bias)""",
    tag="out_proj linear",
)

# ---- 4. the channel move widens on the chunk dtype, not on the mode ------------------------
sub(
    """        ops = [(ttnn.typecast, ttnn.bfloat16)] if _FAST_MODE else []""",
    """        # The channel move has no block-format reader: all three `reblock_permute.eligible*`
        # gates reject a non-bf16 operand outright, so `_FAST_MODE`'s bfloat8_b chunk has always
        # been widened either side of it. Read that off the CHUNK rather than off the mode, so
        # `_PAIR_B8`'s bfloat8_b chunk takes the same widening and is restored to the dtype it
        # arrived in. Byte-identical for every previously reachable config: under `_FAST_MODE` the
        # chunk is bfloat8_b on every path that reaches here (the fp32 affinity trunk has its own
        # `Fp32TriangleMultiplication`), which is the one case the old spelling handled.
        src_dtype = chunk.dtype
        widen = src_dtype != ttnn.bfloat16
        ops = [(ttnn.typecast, ttnn.bfloat16)] if widen else []""",
    tag="transform_chunk widen in",
)
sub(
    """        if _FAST_MODE:
            ops.append((ttnn.typecast, ttnn.bfloat8_b))""",
    """        if widen:
            ops.append((ttnn.typecast, src_dtype))""",
    tag="transform_chunk widen out",
)

# ---- 5. the fused gated move: name the dtype instead of the mode ---------------------------
sub(
    """                            and (mask is None or mask_moved_ok)
                            and not _FAST_MODE
                            and not _TRIMUL_RAW_CHANNEL_MOVES""",
    """                            and (mask is None or mask_moved_ok)
                            # `eligible_gated` rejects a non-bf16 operand anyway; naming the dtype
                            # here keeps `_FAST_MODE` and `_PAIR_B8` on one rule instead of two.
                            and gp_in_fused.dtype == ttnn.bfloat16
                            and not _TRIMUL_RAW_CHANNEL_MOVES""",
    tag="gated move dtype",
)

# ---- 6. the per-channel trimul matmul writes the pair dtype --------------------------------
sub(
    """                        program_config=program_config,
                        dtype=ttnn.bfloat16,
                        transpose_a=defer_a,""",
    """                        program_config=program_config,
                        dtype=_pair_dtype(ttnn.bfloat16),
                        transpose_a=defer_a,""",
    tag="trimul matmul dtype",
)

# ---- 7. the PAIR transition (4-D input); the token transition is three orders smaller ------
sub(
    """        def swiglu(x):
            dtype = self.dtype if self.dtype is not None else _dtype()""",
    """        # A 4-D input is the PAIR transition ([1, S, S, c_z]); 3-D is the token transition, whose
        # activations are three orders smaller and so have nothing to win from a narrower store.
        pair_scale = len(x.shape) >= 4

        def swiglu(x):
            dtype = self.dtype if self.dtype is not None else (
                _pair_dtype() if pair_scale else _dtype())""",
    tag="pair transition",
)

# ---- 8. cast z once in and once out of the whole stack, not per block ----------------------
sub(
    """        dram_peak(f"pairformer enter [z={'x'.join(str(d) for d in z.shape)}]")
        for i, block in enumerate(self.blocks):
            s, z = block(s, z, mask, attn_mask_start, attn_mask_end, extra_attn_bias)
            dram_peak(f"pairformer block {i} done")
        return s, z""",
    """        dram_peak(f"pairformer enter [z={'x'.join(str(d) for d in z.shape)}]")
        # `_PAIR_B8` stores the pair track in bfloat8_b for the whole stack. Cast once on the way
        # in and once on the way out rather than per block, so the residual `add_`s write into a
        # bfloat8_b z and every consumer of it reads 0.531x the bytes, while the caller still gets
        # the bf16 tensor it has always been handed.
        z_in_dtype = z.dtype
        if _PAIR_B8 and z.dtype == ttnn.bfloat16:
            z = ttnn.typecast(z, ttnn.bfloat8_b)
        for i, block in enumerate(self.blocks):
            s, z = block(s, z, mask, attn_mask_start, attn_mask_end, extra_attn_bias)
            dram_peak(f"pairformer block {i} done")
        if z.dtype != z_in_dtype:
            out = ttnn.typecast(z, z_in_dtype)
            ttnn.deallocate(z)
            z = out
        return s, z""",
    tag="pairformer z cast",
)

F.write_text(t)
print("applied 8 hunks to", F)
