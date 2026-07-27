"""CPU-only test for the diffusion conditioning cache under ragged sample chunks.

No Tenstorrent device needed: the device calls are stubbed, so this covers the cache
invalidation control flow, not device arithmetic.

Background: ``DiffusionModule`` hoists the per-step-invariant conditioning onto the
device once, and several of those tensors carry the sample (``r``) batch -- ``q``/``c``
in the runtime cache, plus ``_c_reshaped`` and the per-layer ``s_o`` inside the module.
``Tensor.chunk`` equalises chunk sizes rather than capping them at
``max_parallel_samples``, so a sample count that mps does not divide ends in a short
final chunk, which then meets a cache built for the wider batch. On device that is a
broadcasting TT_FATAL ~68 s into a boltz2 fold. The cache now invalidates wholesale when
the sample batch changes.

Verifies:
  1. A changed sample batch triggers ``reset_static_cache``.
  2. An unchanged sample batch does not -- the per-step fast path stays untouched.
  3. ``reset_static_cache`` clears every batch-dependent cache, module-internal ones
     included. Enumerating them at the call site is what the first attempt at this fix
     got wrong: it refreshed q/c only and the fold still crashed on ``_c_reshaped``.
  4. The chunk enumeration this guards against: which (samples, mps) pairs go ragged.

Run: python3 tests/test_diffusion_cache_ragged_chunk.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_bio import tenstorrent

SEQ, N_ATOMS, ATOM_DIM, TOKEN_DIM = 64, 96, 128, 384
CACHE_KEYS = (
    "s_inputs", "s_trunk", "q", "c", "keys_indexing", "bias_encoder", "bias_token",
    "bias_decoder", "atom_to_token", "atom_to_token_normed", "atom_pad", "r_batch",
)


class _ResetCalled(Exception):
    """Raised by the stubbed reset so the test stops before any device work."""


def _conditioning():
    """The un-batched conditioning tensors, shaped as boltz2 hands them over."""
    nw = N_ATOMS // tenstorrent.ATOM_WINDOW
    return dict(
        s_inputs=torch.randn(1, SEQ, TOKEN_DIM),
        s_trunk=torch.randn(1, SEQ, TOKEN_DIM),
        q=torch.randn(1, N_ATOMS, ATOM_DIM),
        c=torch.randn(1, N_ATOMS, ATOM_DIM),
        bias_encoder=torch.randn(1, nw, tenstorrent.ATOM_WINDOW, 128, 12),
        bias_token=torch.randn(1, SEQ, SEQ, TOKEN_DIM),
        bias_decoder=torch.randn(1, nw, tenstorrent.ATOM_WINDOW, 128, 12),
        keys_indexing=torch.zeros(2 * nw, 8 * nw),
        mask=torch.ones(1, N_ATOMS),
        atom_to_token=torch.ones(1, N_ATOMS, SEQ),
    )


def _warm_module(r_batch):
    """A DiffusionModule that believes it is already warm for ``r_batch`` samples."""
    m = tenstorrent.DiffusionModule.__new__(tenstorrent.DiffusionModule)
    m._runtime_cache = {k: ("dev", k) for k in CACHE_KEYS}
    m._runtime_cache["atom_pad"] = 0
    m._runtime_cache["r_batch"] = r_batch
    m._first_forward_pass = False
    m.resets = 0

    def _reset():
        m.resets += 1
        raise _ResetCalled

    m.reset_static_cache = _reset
    return m


def test_changed_sample_batch_invalidates():
    m = _warm_module(5)
    try:
        m._populate_diffusion_cache(2, **_conditioning())
    except _ResetCalled:
        pass
    else:
        raise AssertionError("a short final chunk did not invalidate the cache")
    assert m.resets == 1
    print("[PASS] sample batch 5 -> 2 invalidates the conditioning cache")


def test_same_sample_batch_is_a_no_op():
    m = _warm_module(5)
    seq_len, n, n_padded = m._populate_diffusion_cache(5, **_conditioning())
    assert m.resets == 0, "the per-step fast path must not invalidate anything"
    assert (seq_len, n) == (SEQ, N_ATOMS), (seq_len, n, n_padded)
    print("[PASS] unchanged sample batch touches nothing")


def test_reset_clears_the_module_internal_caches():
    layers = [types.SimpleNamespace(s_o=("dev", "s_o")) for _ in range(4)]
    m = tenstorrent.DiffusionModule.__new__(tenstorrent.DiffusionModule)
    m._runtime_cache = {k: ("dev", k) for k in CACHE_KEYS}
    m._first_forward_pass = False
    m._diff_trace = None
    m.module = types.SimpleNamespace(
        _s_conditioned=("dev", "s"),
        _c_reshaped=("dev", "c"),
        encoder=types.SimpleNamespace(layers=layers[:2]),
        decoder=types.SimpleNamespace(layers=layers[2:]),
    )
    m.reset_static_cache()
    assert m._runtime_cache == {}, m._runtime_cache
    assert m._first_forward_pass is True
    assert m.module._c_reshaped is None and m.module._s_conditioned is None
    assert all(layer.s_o is None for layer in layers), "per-layer s_o survived the reset"
    print("[PASS] reset clears the runtime cache, _s_conditioned/_c_reshaped and every s_o")


def test_ragged_enumeration():
    """The safe set is the one an operator can rely on: mps dividing the sample count."""
    def sizes(multiplicity, mps):
        n_chunks = (multiplicity + mps - 1) // mps
        return {c.numel() for c in torch.arange(multiplicity).chunk(n_chunks)}

    ragged = {mps for mps in range(1, 17) if len(sizes(50, mps)) > 1}
    assert ragged == {3, 4, 6, 7, 8, 9, 13, 14, 15, 16}, ragged
    assert not 50 % 5, "mps=5 must divide 50"
    print(f"[PASS] at 50 samples, mps in {sorted(ragged)} produce ragged chunks")


if __name__ == "__main__":
    test_changed_sample_batch_invalidates()
    test_same_sample_batch_is_a_no_op()
    test_reset_clears_the_module_internal_caches()
    test_ragged_enumeration()
    print("\nALL DIFFUSION CACHE RAGGED-CHUNK TESTS PASS")
