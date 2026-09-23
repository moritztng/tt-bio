"""The diffusion sample axis, denoised in chunks of one width that narrows when DRAM refuses it.

Every sampler that batches several samples through one device denoise call goes through
``denoise_in_chunks``: Boltz-2's ``AtomDiffusion.sample`` and Protenix's ``edm_sample``, which
Protenix-v1/v2, OpenDDE, OpenDDE-AbAg and PXDesign all call. The chunk width is
``--max_parallel_samples`` (5 by default) at every size, and device memory grows with it: at
1728 tokens a 5-wide Boltz-2 chunk asks 1.19 GiB at the first step on a chip the trunk has
already filled. Before this, that refusal ended the fold after the whole trunk had run.

A refusal now halves the width and reruns the refused chunk from its saved input, down to one
sample. Nothing else about a sample changes: the samplers draw every sample's initial noise,
augmentation and step noise over the whole sample axis on the host before any chunking, so a
sample gets the same draws at every width. A fold that fits at the default width never enters
the fallback and keeps its exact numbers.
"""

from __future__ import annotations

import gc


def resolve_sample_chunk_width(multiplicity, max_parallel_samples):
    """The one chunk width a trajectory is denoised at, given the cap.

    ``max_parallel_samples`` is a cap: omit it (None) and the whole multiplicity runs as one
    chunk. Given the number of chunks the cap forces, the widest chunk only needs to be
    ``ceil(m / n_chunks)``, so the width is rebalanced to that: less padding and less peak memory
    at the same chunk count. This is also exactly the partition ``torch.chunk(n_chunks)`` makes,
    which is what Protenix's sampler used before it shared this function.
    """
    m = max(1, int(multiplicity))
    width = m if max_parallel_samples is None else min(m, max(1, int(max_parallel_samples)))
    n_chunks = -(-m // width)
    return -(-m // n_chunks)


def denoise_in_chunks(x, width, run, *, reset=None, narrowest=1, tag="diffusion"):
    """One denoising step over the sample axis of ``x``, ``width`` samples per device call.

    ``run(chunk, width)`` denoises ``chunk`` (at most ``width`` rows; the tail may be shorter) and
    returns a tensor with the same leading dim. When DRAM refuses a chunk, the width halves
    (rebalanced as ``resolve_sample_chunk_width`` does), ``reset()`` drops any device state the
    caller keyed on the old width, and the refused chunk reruns from its input. The chunks
    already denoised in this step stand. Below ``narrowest`` the refusal is raised: at one sample
    the width is not what is too big, and a batch whose conditioning is concatenated per member
    (Protenix's multi-target batch) cannot be split at all.

    Returns ``(denoised, width)``. The caller starts its next step at the returned width, so a
    trajectory pays for one refused allocation and not one per step. Anything that is not an
    allocator refusal propagates untouched.
    """
    from tt_bio.size_limits import is_alloc_refusal

    m = x.shape[0]
    out = None
    start = 0
    while start < m:
        chunk = x[start:start + width]
        try:
            y = run(chunk, width)
        except Exception as exc:  # noqa: BLE001 -- anything else is re-raised below
            if width <= narrowest or not is_alloc_refusal(exc):
                raise
            narrower = resolve_sample_chunk_width(m, max(narrowest, width // 2))
            print(f"[{tag}] DRAM refused a {width}-sample chunk; denoising {narrower} at a "
                  f"time from here on. The tt-metal 'Out of Memory' line above is expected "
                  f"and handled.", flush=True)
            width = narrower
        else:
            if out is None:
                out = x.new_zeros((m, *y.shape[1:]))
            out[start:start + chunk.shape[0]] = y
            start += chunk.shape[0]
            continue
        # Outside the except block, so the refused call's frames (and every device tensor they
        # hold) are gone before the narrower chunk allocates.
        y = chunk = None
        gc.collect()
        if reset is not None:
            reset()
    return out, width
