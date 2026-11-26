"""The convolution layer: scatter semantics, batch-norm convention, and the 32-bucket proof.

The grade against upstream's own ``TensorProductConvLayer`` lives in
``scripts/diffdock_port/conv_parity.py`` and needs e3nn, torch_scatter and the upstream clone.
Measured there over all 7 distinct conv signatures the released score network builds: worst
max relative error 1.95e-07, worst PCC 0.999999999999995. The bar is fp32 rounding, not a
flat PCC: these layers are a dot product over ~4k weights followed by a mean over up to a few
hundred edges, so ~1e-7 relative is the floor and anything above ~1e-5 is a real difference.

What is tested here runs anywhere and catches the mistakes that survive a PCC:

* that the mean is over all edge groups jointly, not per group;
* that an isolated node yields zeros rather than NaN;
* that batch norm treats ``0o`` as a non-scalar (no mean, no bias) -- the parity-carrying
  track, where a subtracted mean would break equivariance;
* that 32-bucketing the node and edge axes leaves every real row bit-identical.
"""

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.conv import (IrrepBatchNorm, TensorProductConvLayer, bucket, fc_block,
                                  pad_edges, pad_nodes)

SH = "1x0e + 1x1o + 1x2e"
IN = "16x0e + 4x1o + 4x1e + 4x0o"
OUT = "16x0e + 4x1o + 4x1e + 4x0o"
NEF = 24


def _layer(**kw):
    torch.manual_seed(3)
    layer = TensorProductConvLayer(IN, SH, OUT, NEF, **kw).eval()
    if layer.batch_norm is not None:
        with torch.no_grad():
            layer.batch_norm.running_mean.normal_(0.0, 0.5)
            layer.batch_norm.running_var.uniform_(0.3, 2.0)
            layer.batch_norm.weight.normal_(1.0, 0.2)
            layer.batch_norm.bias.normal_(0.0, 0.2)
    return layer


def _graph(n_nodes, n_edges, dim, seed=5, max_src=None):
    g = torch.Generator().manual_seed(seed)
    node_attr = torch.randn(n_nodes, dim, generator=g)
    src = torch.randint(0, max_src or n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    edge_index = torch.stack([src, dst])
    v = torch.randn(n_edges, 3, generator=g)
    n = v / v.norm(dim=-1, keepdim=True)
    y, z, x = n[:, 0], n[:, 1], n[:, 2]
    import math
    s3, s15, s5 = math.sqrt(3.0), math.sqrt(15.0), math.sqrt(5.0)
    edge_sh = torch.cat([
        torch.ones(n_edges, 1),
        s3 * torch.stack([y, z, x], dim=-1),
        torch.stack([s15 * x * y, s15 * y * z, (s5 / 2) * (3 * z * z - 1),
                     s15 * x * z, (s15 / 2) * (x * x - y * y)], dim=-1),
    ], dim=1)
    edge_attr = torch.randn(n_edges, NEF, generator=g)
    return node_attr, edge_index, edge_attr, edge_sh


def test_fc_block_keeps_upstreams_module_indices():
    # Upstream's FCBlock is [Linear, ReLU, Dropout, Linear], so the second linear is index 3.
    # Dropping the dropout instead of replacing it with Identity would renumber it to 2 and
    # the checkpoint's `fc.3.weight` would have nowhere to load.
    block = fc_block(8, 16, 4, layers=2)
    assert isinstance(block[0], torch.nn.Linear) and isinstance(block[3], torch.nn.Linear)
    assert sorted(k.split(".")[0] for k in block.state_dict()) == ["0", "0", "3", "3"]
    deep = fc_block(8, 16, 4, layers=3)
    assert isinstance(deep[3], torch.nn.Linear) and isinstance(deep[6], torch.nn.Linear)


def test_the_mean_is_over_all_edge_groups_jointly():
    """Splitting the same edges into groups must not change the answer.

    The scatter sums each group and divides once at the end. Dividing inside the loop would
    give each group's contribution weight 1/n_i instead of 1/sum(n_i), which is a plausible
    reading of the code and wrong by a factor of the group count.
    """
    layer = _layer(edge_groups=1)
    node_attr, edge_index, edge_attr, edge_sh = _graph(40, 300, layer.in_irreps.dim)
    with torch.no_grad():
        one = layer(node_attr, edge_index, [edge_attr], edge_sh)
        split = layer(node_attr, edge_index,
                      [edge_attr[:120], edge_attr[120:260], edge_attr[260:]], edge_sh)
    assert torch.allclose(one, split, atol=1e-6, rtol=0)


def test_an_isolated_node_is_zero_not_nan():
    layer = _layer(edge_groups=1, residual=False, batch_norm=False)
    # every edge sources from node 0..9, so nodes 10.. have no incoming contribution
    node_attr, edge_index, edge_attr, edge_sh = _graph(32, 128, layer.in_irreps.dim, max_src=10)
    with torch.no_grad():
        out = layer(node_attr, edge_index, [edge_attr], edge_sh)
    assert torch.isfinite(out).all()
    assert torch.equal(out[10:], torch.zeros_like(out[10:]))


def test_batch_norm_leaves_the_pseudoscalar_track_unshifted():
    """``0o`` is not a scalar: no mean subtracted, no bias added.

    e3nn's ``is_scalar()`` is ``l == 0 and p == 1``, so the ``10x0o`` track DiffDock uses to
    carry parity is normalised by its second moment only. Treating it as a scalar would add a
    parity-even constant to a parity-odd feature, which is invisible in a PCC against a
    checkpoint whose bias happens to be small, and fatal to equivariance.
    """
    bn = IrrepBatchNorm("3x0e + 2x0o").eval()
    with torch.no_grad():
        bn.running_mean.fill_(2.0)     # 3 entries: the 0e track only
        bn.bias.fill_(7.0)
        bn.running_var.fill_(1.0)
        bn.weight.fill_(1.0)
    assert bn.running_mean.numel() == 3 and bn.bias.numel() == 3
    x = torch.zeros(4, 5)
    out = bn(x)
    k = (1.0 + bn.eps) ** -0.5
    assert torch.allclose(out[:, :3], torch.full((4, 3), (0.0 - 2.0) * k + 7.0), atol=1e-6)
    assert torch.equal(out[:, 3:], torch.zeros(4, 2))


def test_batch_norm_is_inference_only():
    bn = IrrepBatchNorm("3x0e").train()
    with pytest.raises(RuntimeError, match="inference port"):
        bn(torch.zeros(2, 3))


def test_bucket():
    assert [bucket(n) for n in (0, 1, 32, 33, 640, 641)] == [0, 32, 32, 64, 640, 672]
    assert bucket(5, 8) == 8


def test_padding_does_not_change_a_real_row():
    """The 32-bucketing proof: real rows come out bit-identical.

    Nodes are padded with zero rows at the end, so a real node keeps its index. Edges are
    padded with ``pad_node -> pad_node`` edges, so every padded edge's tensor product lands in
    the pad row and, just as importantly, increments only the pad row's divisor. The scatter
    is a mean, so an extra edge counted against a real node would change that node's output
    even if the edge contributed zero -- which is exactly the failure a zero-attribute pad
    edge pointing at a real node would produce, and why :func:`pad_edges` sets both ends.
    """
    layer = _layer(edge_groups=2)
    n_real, e_real = 37, 101
    node_attr, edge_index, edge_attr, edge_sh = _graph(n_real, e_real, layer.in_irreps.dim)
    groups = [edge_attr[:60], edge_attr[60:]]
    idx = [edge_index[:, :60], edge_index[:, 60:]]
    sh = [edge_sh[:60], edge_sh[60:]]

    with torch.no_grad():
        plain = layer(node_attr, edge_index, groups, edge_sh)

    n_pad = bucket(n_real + 1)          # +1: the pad node must exist even if n_real % 32 == 0
    pad_node = n_real
    p_idx, p_attr, p_sh = [], [], []
    for i in range(2):
        a, b, c = pad_edges(idx[i], groups[i], sh[i], bucket(idx[i].shape[1]), pad_node)
        p_idx.append(a); p_attr.append(b); p_sh.append(c)
    with torch.no_grad():
        padded = layer(pad_nodes(node_attr, n_pad), torch.cat(p_idx, dim=1), p_attr,
                       torch.cat(p_sh, dim=0), out_nodes=n_pad)

    assert padded.shape[0] == n_pad and n_pad % 32 == 0
    assert torch.equal(padded[:n_real], plain), \
        f"real rows moved by {(padded[:n_real] - plain).abs().max():.3e}"


def test_a_pad_edge_aimed_at_a_real_node_is_caught_by_the_same_check():
    """The negative control for the test above: the hazard is real, not hypothetical.

    Aiming the pad edges at node 0 instead of the pad node changes node 0's output, because
    the divisor counts them. If this ever stops failing, the scatter has stopped taking a
    mean and the proof above is vacuous.
    """
    layer = _layer(edge_groups=1, batch_norm=False, residual=False)
    n_real, e_real = 37, 101
    node_attr, edge_index, edge_attr, edge_sh = _graph(n_real, e_real, layer.in_irreps.dim)
    with torch.no_grad():
        plain = layer(node_attr, edge_index, [edge_attr], edge_sh)
    n_pad = bucket(n_real + 1)
    bad_idx, bad_attr, bad_sh = pad_edges(edge_index, edge_attr, edge_sh,
                                          bucket(e_real), pad_node=0)
    with torch.no_grad():
        bad = layer(pad_nodes(node_attr, n_pad), bad_idx, [bad_attr], bad_sh, out_nodes=n_pad)
    assert not torch.allclose(bad[0], plain[0], atol=1e-6)
    assert torch.allclose(bad[1:n_real], plain[1:n_real], atol=1e-6)


def test_residual_order_differs_between_the_two_layer_classes():
    """Score net: conv -> batch norm -> residual. Confidence net: conv -> residual -> norm.

    Same parameter shapes, same state-dict keys, different numbers. Nothing in a checkpoint
    records which one it was trained with, so it is read off the model class in the config.
    """
    a = _layer(edge_groups=1, residual_first=False)
    b = _layer(edge_groups=1, residual_first=True)
    na, ei, ea, es = _graph(40, 300, a.in_irreps.dim)
    with torch.no_grad():
        assert not torch.allclose(a(na, ei, [ea], es), b(na, ei, [ea], es), atol=1e-4)


def test_a_conv_with_zero_edges_returns_zeros_at_the_right_width():
    """D-dfd-orch-10: upstream raises here (``np.array_split(..., 0)``); the port must not.

    A step whose cross cutoff keeps no receptor inside it, or a ligand with no rotatable
    bond, hands a layer an empty edge set. The answer is the empty mean -- zeros -- and it
    must have ``out_nodes`` rows when the layer scatters to a different node set than it
    reads from, which is exactly the torsion head's bond conv.
    """
    layer = _layer(edge_groups=1, residual=False, batch_norm=True)
    node_attr, _, edge_attr, edge_sh = _graph(32, 4, layer.in_irreps.dim)
    empty = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        out = layer(node_attr, empty, [edge_attr[:0]], edge_sh[:0])
        few = layer(node_attr, empty, [edge_attr[:0]], edge_sh[:0], out_nodes=7)
    assert out.shape == (32, layer.out_size) and torch.equal(out, torch.zeros_like(out))
    assert few.shape == (7, layer.out_size) and torch.equal(few, torch.zeros_like(few))


def test_an_all_masked_edge_set_is_zero_not_nan():
    """The divisor is the mask's sum, so masking every edge divides 0 by 0 unless clamped."""
    layer = _layer(edge_groups=1, residual=False, batch_norm=False)
    node_attr, edge_index, edge_attr, edge_sh = _graph(32, 128, layer.in_irreps.dim)
    mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    with torch.no_grad():
        out = layer(node_attr, edge_index, [edge_attr], edge_sh, edge_mask=mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))
