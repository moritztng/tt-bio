"""The tape's gather backward, against float64 torch rather than another device op.

`ttnn.embedding` had no tape entry, so any model that broadcasts a per-token tensor onto
atoms stopped a taped training forward dead (OF3's diffusion module does it five times).
The entry scatter-adds the cotangent back into the table via `ttnn.embedding_bw`.

Two things are worth a test rather than a reading of the code. A gather whose indices
REPEAT has a backward that accumulates, and a closure that assigned instead would be
exactly right on a permutation and silently wrong on the real one -- so the duplicate case
is the test, and the permutation case is there only to show the two differ. And the
cotangent must come back to the parent in the parent's own LAYOUT: `to_layout`'s backward
used to hand it on unchanged, which put a row-major gradient into a matmul several ops
upstream and failed there with "Inputs to matmul must be tilized", nowhere near the cause.
"""
import os
import pytest
import torch
import ttnn

pytestmark = [pytest.mark.device]

# bf16 through a bf16 kernel. ttnn.embedding is bf16-only in the forward and
# ttnn.embedding_bw refuses fp32 outright, so every shipped call site downcasts the table
# before the gather and the gradient carries no rounding the forward did not already have.
# Measured headroom at OF3's shapes: 1.03e-2 at [96,384] from 608 rows, well inside the
# 5.0e-2 per-tensor bar. It grows with rows per table entry, so the bar here is set from
# the same reasoning and not from the number that came out.
BAR = 3.0e-2


def _rel(got, ref):
    return float((got - ref).norm() / (ref.norm() + 1e-30))


@pytest.mark.parametrize("V,C,N,dup", [(96, 384, 608, True), (96, 384, 608, False),
                                       (9216, 128, 4096, True)])
def test_embedding_backward_is_a_scatter_add(V, C, N, dup):
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    torch.manual_seed(0)
    idx = torch.randint(0, V, (N,)) if dup else (torch.arange(N) % V)
    table = torch.randn(V, C)
    cot = torch.randn(N, C)

    ref = torch.zeros(V, C, dtype=torch.float64)
    ref.index_add_(0, idx, cot.double())

    idx_t = ttnn.from_torch(idx.reshape(1, N).to(torch.int32),
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    w = ag.Tensor(ttnn.from_torch(table, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                  dtype=ttnn.bfloat16), requires_grad=True)
    g = ttnn.from_torch(cot, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    with ag.tape():
        out = ttnn.embedding(idx_t, w, layout=ttnn.ROW_MAJOR_LAYOUT)
        ag.backward([out], [g])

    got = torch.Tensor(ttnn.to_torch(w.grad)).double().reshape(V, C)
    rel = _rel(got, ref)
    assert rel < BAR, f"embedding backward rel={rel:.3e} over {BAR:.1e} (V={V} N={N} dup={dup})"


def test_embedding_backward_accumulates_rather_than_assigns():
    """The duplicate case must differ from an assigning backward, or the test above is
    passing on a coincidence: with 8 rows per entry an assign is 8x too small."""
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    torch.manual_seed(1)
    V, C, N = 16, 32, 128
    idx = torch.arange(N) % V                       # exactly 8 rows per table entry
    cot = torch.ones(N, C)

    idx_t = ttnn.from_torch(idx.reshape(1, N).to(torch.int32),
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    w = ag.Tensor(ttnn.from_torch(torch.zeros(V, C), layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=dev, dtype=ttnn.bfloat16), requires_grad=True)
    g = ttnn.from_torch(cot, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    with ag.tape():
        ag.backward([ttnn.embedding(idx_t, w, layout=ttnn.ROW_MAJOR_LAYOUT)], [g])

    got = torch.Tensor(ttnn.to_torch(w.grad)).double()
    assert abs(float(got.mean()) - 8.0) < 0.1, (
        f"expected 8 accumulated ones per entry, got {float(got.mean()):.4f} -- "
        "1.0 would mean the backward assigns")


def test_to_layout_backward_returns_the_parents_layout():
    """A tiled parent behind a row-major view gets its cotangent tiled back.

    Without this the gradient leaves `to_layout` row-major and the first matmul upstream
    raises, which is how the OF3 diffusion backward first failed.
    """
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    x = ag.Tensor(ttnn.from_torch(torch.randn(1, 64, 32), layout=ttnn.TILE_LAYOUT,
                                  device=dev, dtype=ttnn.bfloat16), requires_grad=True)
    g = ttnn.from_torch(torch.ones(1, 64, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    with ag.tape():
        ag.backward([ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)], [g])
    assert x.grad.layout == ttnn.TILE_LAYOUT, (
        f"gradient came back {x.grad.layout}, parent is {ttnn.TILE_LAYOUT}")
