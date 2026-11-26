"""The chunk's cost per edge is linear and measurable, so the chunk size is a division.

`dfd-device` owes this row one number, `edge_chunk` -- the largest edge slice that fits beside
everything else in the layer. Only a card knows the budget, but the *cost* is a property of
the irreps, and this file pins it down on CPU: the per-edge cost does not move with the chunk
size (so dividing a budget by it is legitimate), it is dominated by the tensor-product weight
block, and a chunked call is arithmetically the unchunked one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import graph as G  # noqa: E402
from tt_bio.diffdock.chunking import chunk_for_budget, per_edge, report  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer, tp_scatter  # noqa: E402
from tt_bio.diffdock.trace import assert_shape_stable, record_ops  # noqa: E402

IN_DIM, N_EDGE_FEATURES = 16, 24
CHUNKS = (64, 128, 256)


def _layer_and_inputs(n_edges=512, n_nodes=251, seed=0):
    # n_nodes is deliberately coprime with every chunk size: a node tensor whose leading axis
    # happened to equal the edge count would be counted as a per-edge cost and inflate it.
    torch.manual_seed(seed)
    layer = TensorProductConvLayer(
        in_irreps=f"{IN_DIM}x0e", sh_irreps="1x0e+1x1o+1x2e",
        out_irreps=f"{IN_DIM}x0e+4x1o", n_edge_features=N_EDGE_FEATURES,
        hidden_features=32, batch_norm=False, residual=False).eval()
    node_attr = torch.randn(n_nodes, IN_DIM) * 0.1
    edge_index = torch.stack([torch.randint(0, n_nodes, (n_edges,)),
                              torch.randint(0, n_nodes, (n_edges,))])
    edge_attr = torch.randn(n_edges, N_EDGE_FEATURES)
    edge_sh = torch.randn(n_edges, 9)
    mask = (torch.rand(n_edges) > 0.3).float()
    return layer, node_attr, edge_index, edge_attr, edge_sh, mask


def _cost(n_edges):
    layer, node_attr, edge_index, edge_attr, edge_sh, mask = _layer_and_inputs(n_edges)
    with torch.no_grad():
        return per_edge(lambda: tp_scatter(layer.tp, layer.fc, node_attr, edge_index,
                                           edge_attr, edge_sh, out_nodes=node_attr.shape[0],
                                           edge_mask=mask, edge_chunk=n_edges), n_edges)


def test_the_per_edge_cost_does_not_move_with_the_chunk_size():
    """The claim that makes `budget / cost` a legal way to pick a chunk."""
    costs = [_cost(n) for n in CHUNKS]
    totals = [c.total_bytes for c in costs]
    spread = (max(totals) - min(totals)) / min(totals)
    assert spread < 1e-9, dict(zip(CHUNKS, totals))


def test_the_weight_block_is_what_the_chunk_is_bounding():
    """If the widest per-edge tensor were not the TP weights, chunking edges would be the
    wrong lever -- so this is checked rather than asserted in a comment."""
    cost = _cost(128)
    layer, *_ = _layer_and_inputs(128)
    w_elems = layer.tp.weight_numel
    assert cost.peak_elems == pytest.approx(w_elems, rel=1e-9), (cost.peak_elems, w_elems)
    # Not asserted: that the weight block dominates the *sum*. At this toy signature it is
    # 320 of 1961 elements per edge -- the FC hidden activations and the einsum intermediates
    # are bigger together -- and `dfd-device`'s form does not build the block at all
    # (DEVICE.md 1: a ~6,000-wide term axis instead). Which tensor is widest is a property of
    # the implementation, so `per_edge` measures it rather than assuming it.


def test_a_budget_divides_into_a_tile_aligned_chunk():
    cost = _cost(128)
    for gib in (0.5, 2.0, 8.0):
        n = chunk_for_budget(cost, gib * 2 ** 30, slack=0.5)
        assert n % 32 == 0 and n >= 32
        assert n * cost.total_bytes <= gib * 2 ** 30 * 0.5
        # and it is the largest such chunk
        assert (n + 32) * cost.total_bytes > gib * 2 ** 30 * 0.5
    assert chunk_for_budget(cost, 1.0, slack=1.0) == 32, "never returns a chunk below a tile"
    with pytest.raises(ValueError):
        chunk_for_budget(cost, 2 ** 30, slack=0.0)


def test_chunking_changes_the_answer_only_by_summation_order():
    """The port's statement that `edge_chunk` is free of accuracy consequences."""
    layer, node_attr, edge_index, edge_attr, edge_sh, mask = _layer_and_inputs(512)
    with torch.no_grad():
        whole = tp_scatter(layer.tp, layer.fc, node_attr, edge_index, edge_attr, edge_sh,
                           out_nodes=node_attr.shape[0], edge_mask=mask)
        for n in CHUNKS:
            got = tp_scatter(layer.tp, layer.fc, node_attr, edge_index, edge_attr, edge_sh,
                             out_nodes=node_attr.shape[0], edge_mask=mask, edge_chunk=n)
            rel = ((got - whole).abs().max() / whole.abs().max()).item()
            assert rel < 1e-6, f"chunk {n}: {rel:.2e}"


def test_a_chunked_call_still_issues_one_fixed_program():
    """Chunking must not cost the capture: every chunk is the same shape, so the slices are
    an unrolled fixed loop. Checked on an edge count that divides, and one that does not."""
    for n_edges, chunk in ((512, 128), (480, 128)):
        layer, node_attr, edge_index, edge_attr, edge_sh, mask = _layer_and_inputs(n_edges)
        logs = []
        with torch.no_grad():   # the warm-up pass: first use casts the Wigner 3j tables
            tp_scatter(layer.tp, layer.fc, node_attr, edge_index, edge_attr, edge_sh,
                       out_nodes=node_attr.shape[0], edge_mask=mask, edge_chunk=chunk)
        for seed in range(3):
            torch.manual_seed(seed)
            pos_attr = torch.randn_like(edge_attr)
            with torch.no_grad():
                _, log = record_ops(lambda: tp_scatter(
                    layer.tp, layer.fc, node_attr, edge_index, pos_attr, edge_sh,
                    out_nodes=node_attr.shape[0], edge_mask=mask, edge_chunk=chunk))
            logs.append(log)
        assert_shape_stable(logs)
        if n_edges % chunk:
            # the ragged tail is a different shape from the full chunks, which is exactly why
            # the edge axis is padded to a multiple of 32 by `StaticGraphs` in the first place
            assert G.pad_count(n_edges) % 32 == 0


def test_report_is_readable():
    text = report(_cost(128), budgets_gib=(1.0,), slack=0.5)
    assert "edge_chunk" in text and "B/edge" in text
