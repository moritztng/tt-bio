"""CPU-only test for the diffusion conditioning cache under ragged sample chunks.

No Tenstorrent device needed: ``_from_torch`` / ``_deallocate_tensor_like`` are stubbed
so this exercises the cache control flow and the tensor shapes it produces, not device
arithmetic.

Background: ``DiffusionModule`` hoists the per-step-invariant conditioning onto the
device once. Two of those tensors, ``q`` and ``c``, arrive un-batched and are expanded
to the sample batch of ``r``. When ``max_parallel_samples`` does not divide
``diffusion_samples``, ``Tensor.chunk`` hands the sampler a short final chunk, and a
cache built for the first chunk's batch then mismatches it -- on device that surfaces as
a broadcasting TT_FATAL deep inside the DiT. The cache now refreshes q/c when the sample
batch changes.

Verifies:
  1. The uniform-chunk case uploads q/c exactly once -- the fast path is untouched.
  2. A short final chunk refreshes q/c to the new batch and frees the old tensors.
  3. Already-batched q/c are not re-expanded.
  4. The chunk enumeration this guards against: which (samples, mps) pairs go ragged.

Run: python3 tests/test_diffusion_cache_ragged_chunk.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_bio import tenstorrent

N_ATOMS, ATOM_DIM = 96, 128


def _stub_module():
    """A DiffusionModule with the device calls replaced by shape bookkeeping."""
    m = tenstorrent.DiffusionModule.__new__(tenstorrent.DiffusionModule)
    m._runtime_cache = {}
    m.uploads, m.frees = [], []
    m._from_torch = lambda t, **kw: ("dev", tuple(t.shape))
    m._deallocate_tensor_like = lambda v: m.frees.append(v)

    def _set(key, value):
        m.uploads.append(key)
        m._runtime_cache[key] = value
        return value

    m._cache_set = _set
    m._cache_get = lambda key, default=None: m._runtime_cache.get(key, default)
    return m


def _chunk_sizes(multiplicity, max_parallel_samples):
    n_chunks = (multiplicity + max_parallel_samples - 1) // max_parallel_samples
    return [c.numel() for c in torch.arange(multiplicity).chunk(n_chunks)]


def test_uniform_chunks_upload_once():
    m = _stub_module()
    q, c = torch.randn(1, N_ATOMS, ATOM_DIM), torch.randn(1, N_ATOMS, ATOM_DIM)
    for size in _chunk_sizes(50, 5):
        if m._cache_get("r_batch") != size:
            m._cache_sample_conditioning(q, c, size, 0)
    assert m.uploads.count("q") == 1, f"q re-uploaded on the uniform path: {m.uploads}"
    assert m.frees == [None, None], f"uniform path freed something: {m.frees}"
    print("[PASS] uniform chunks (50 samples, mps=5): q/c uploaded once, nothing freed")


def test_short_final_chunk_refreshes():
    m = _stub_module()
    q, c = torch.randn(1, N_ATOMS, ATOM_DIM), torch.randn(1, N_ATOMS, ATOM_DIM)
    atom_pad, seen = 32, []
    for size in _chunk_sizes(50, 3):
        if m._cache_get("r_batch") != size:
            m._cache_sample_conditioning(q, c, size, atom_pad)
        seen.append(m._cache_get("q"))
        assert m._cache_get("q") == ("dev", (size, N_ATOMS + atom_pad, ATOM_DIM)), (
            f"cached q {m._cache_get('q')} does not match chunk batch {size}"
        )
        assert m._cache_get("c") == m._cache_get("q"), "q and c batches diverged"
    assert len(set(seen)) == 2, f"expected two distinct cached batches, got {set(seen)}"
    # first refresh frees the two stale tensors; the first upload frees nothing
    assert m.frees.count(None) == 2, f"unexpected frees: {m.frees}"
    assert ("dev", (3, N_ATOMS + atom_pad, ATOM_DIM)) in m.frees, m.frees
    print("[PASS] short final chunk (50 samples, mps=3): q/c refreshed to batch 2, stale freed")


def test_prebatched_conditioning_not_expanded():
    m = _stub_module()
    q, c = torch.randn(4, N_ATOMS, ATOM_DIM), torch.randn(4, N_ATOMS, ATOM_DIM)
    m._cache_sample_conditioning(q, c, 4, 0)
    assert m._cache_get("q") == ("dev", (4, N_ATOMS, ATOM_DIM)), m._cache_get("q")
    print("[PASS] q/c already at the sample batch are used as-is")


def test_ragged_enumeration():
    """The safe set is the one an operator can rely on: mps dividing the sample count."""
    ragged = {mps for mps in range(1, 17) if len(set(_chunk_sizes(50, mps))) > 1}
    assert ragged == {3, 4, 6, 7, 8, 9, 13, 14, 15, 16}, ragged
    assert not (50 % 5), "mps=5 must divide 50"
    print(f"[PASS] at 50 samples, mps in {sorted(ragged)} produce ragged chunks")


if __name__ == "__main__":
    test_uniform_chunks_upload_once()
    test_short_final_chunk_refreshes()
    test_prebatched_conditioning_not_expanded()
    test_ragged_enumeration()
    print("\nALL DIFFUSION CACHE RAGGED-CHUNK TESTS PASS")
