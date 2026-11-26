"""The convolution layer: FC block -> tensor product -> scatter -> batch norm -> residual.

This is the body of the DiffDock score network. Upstream is
``models/tensor_layers.TensorProductConvLayer`` together with ``tp_scatter_simple`` and
``tp_scatter_multigroup``; this is the same arithmetic in plain torch, with no e3nn and no
``torch_scatter``, and with the node and edge axes bucketed to a multiple of 32 so the same
code runs on a device without a recompile per graph size.

Three things here are easy to get subtly wrong and are pinned by
``tests/test_diffdock_conv.py``:

* **Mean, computed as sum-then-divide.** Upstream never materialises the concatenated edge
  set: it scatters each edge group with ``reduce="sum"``, accumulates
  ``bincount(edge_src)`` per group, and divides once at the end, clamping the divisor at
  ``finfo(dtype).eps`` so an isolated node yields 0 rather than NaN. Dividing per group
  instead would weight the groups by their sizes.
* **Batch norm is over irreps, not over channels,** and it runs *before* the residual in this
  layer. (``OldTensorProductConvLayer``, which the confidence model uses, applies them the
  other way round -- see :class:`IrrepBatchNorm` and ``residual_first``.) Only true scalars
  (``0e``) have a mean subtracted and a bias added; the pseudoscalar track ``0o`` is
  normalised by its second moment alone, because subtracting a mean from it would break
  parity equivariance.
* **The residual is zero-padded, not projected.** ``out + pad(node_attr)`` lines the input
  irreps up with the first columns of the output irreps. That is only meaningful because the
  irrep sequence is a prefix chain (``48x0e`` -> ``+10x1o`` -> ``+10x1e`` -> ``+10x0o``), and
  it is why the layer plan's ``residual`` flag is false exactly where the widths break.

**What the padding does to the scatter.** Padded node rows are appended after the real ones,
so a real node keeps its index and its divisor. Padded *edges* must point at a padded node at
both ends: ``edge_src`` chooses the output row, so a pad edge whose source is a real node
would add a term to that node's sum *and* increment its divisor, corrupting a real row in two
ways at once. :func:`pad_edges` enforces that by construction, and
``test_padding_does_not_change_a_real_row`` checks the real rows come out bit-identical.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tt_bio.diffdock.irreps import Irreps
from tt_bio.diffdock.tensor_product import FullyConnectedTensorProduct

ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU}


def fc_block(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2,
             activation: str = "relu") -> nn.Sequential:
    """Upstream's ``FCBlock`` without the dropout, but with its module indices.

    Dropout is identity at inference. It is replaced by :class:`torch.nn.Identity` rather than
    dropped so the module indices still match the checkpoint's keys (``fc.0.weight``,
    ``fc.3.weight``); removing it would renumber every linear layer and the weights would load
    into the wrong places, or not at all.
    """
    if layers < 2:
        raise ValueError(f"FCBlock needs at least 2 layers, got {layers}")
    act = ACTIVATIONS[activation]
    seq: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act(), nn.Identity()]
    for _ in range(layers - 2):
        seq += [nn.Linear(hidden_dim, hidden_dim), act(), nn.Identity()]
    seq += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*seq)


class IrrepBatchNorm(nn.Module):
    """``e3nn.nn.BatchNorm(irreps)`` at inference, with ``normalization='component'``.

    Inference only: the running statistics come from the checkpoint and are never updated.
    Per irrep field of multiplicity ``mul`` and dimension ``d = 2l+1``:

        scalars (``0e``) only:  x <- x - running_mean
        all:                    x <- x * weight / sqrt(running_var + eps)
        scalars (``0e``) only:  x <- x + bias

    ``running_var`` holds the mean over the batch of ``mean_i(x_i^2)`` -- the *component*
    normalisation, which is the norm squared divided by ``d``. Using the norm instead (e3nn's
    other option, which upstream does not select) would rescale every ``l>0`` field by
    ``sqrt(d)`` and nothing would fail loudly.
    """

    def __init__(self, irreps, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.irreps = Irreps.parse(irreps)
        self.eps = eps
        self.affine = affine
        num_scalar = sum(t.mul for t in self.irreps if t.l == 0 and t.p == 1)
        num_features = sum(t.mul for t in self.irreps)
        self.register_buffer("running_mean", torch.zeros(num_scalar))
        self.register_buffer("running_var", torch.ones(num_features))
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            raise RuntimeError(
                "IrrepBatchNorm is an inference port: it has no batch statistics path. "
                "Call .eval() on the model."
            )
        lead = x.shape[:-1]
        out = torch.empty_like(x)
        im = iv = 0
        for term, sl in zip(self.irreps, self.irreps.slices()):
            d = 2 * term.l + 1
            field = x[..., sl].reshape(*lead, term.mul, d)
            is_scalar = term.l == 0 and term.p == 1
            if is_scalar:
                field = field - self.running_mean[im:im + term.mul].reshape(term.mul, 1)
            norm = (self.running_var[iv:iv + term.mul] + self.eps).pow(-0.5)
            if self.affine:
                norm = norm * self.weight[iv:iv + term.mul]
            field = field * norm.reshape(term.mul, 1)
            if self.affine and is_scalar:
                field = field + self.bias[im:im + term.mul].reshape(term.mul, 1)
            if is_scalar:
                im += term.mul
            iv += term.mul
            out[..., sl] = field.reshape(*lead, term.dim)
        return out

    def extra_repr(self) -> str:
        return f"{self.irreps}, eps={self.eps}"


def bucket(n: int, multiple: int = 32) -> int:
    """The next multiple of *multiple* at or above *n*. PLAYBOOKS 2b: tile-aligned axes."""
    return ((n + multiple - 1) // multiple) * multiple


def pad_nodes(x: Tensor, n_padded: int) -> Tensor:
    """Append zero rows so the node axis is *n_padded* long."""
    if n_padded < x.shape[0]:
        raise ValueError(f"cannot pad {x.shape[0]} nodes down to {n_padded}")
    return torch.nn.functional.pad(x, (0, 0, 0, n_padded - x.shape[0]))


def pad_edges(edge_index: Tensor, edge_attr: Tensor, edge_sh: Tensor, n_padded: int,
              pad_node: int) -> tuple[Tensor, Tensor, Tensor]:
    """Append dead edges so the edge axis is *n_padded* long.

    Every padded edge is ``pad_node -> pad_node``, with zero attributes and zero harmonics.
    *pad_node* must be an index in the padded node range that no real edge touches; the
    scatter then writes every padded edge's contribution into that one row and leaves every
    real row's sum and divisor untouched. Pointing a pad edge at a real node would silently
    change that node's output -- it is the one way 32-bucketing can corrupt this layer.
    """
    n = edge_index.shape[1]
    if n_padded < n:
        raise ValueError(f"cannot pad {n} edges down to {n_padded}")
    extra = n_padded - n
    if extra == 0:
        return edge_index, edge_attr, edge_sh
    tail = torch.full((2, extra), pad_node, dtype=edge_index.dtype, device=edge_index.device)
    return (
        torch.cat([edge_index, tail], dim=1),
        torch.nn.functional.pad(edge_attr, (0, 0, 0, extra)),
        torch.nn.functional.pad(edge_sh, (0, 0, 0, extra)),
    )


def tp_scatter(tp: FullyConnectedTensorProduct, fc, node_attr: Tensor, edge_index: Tensor,
               edge_attr_groups, edge_sh: Tensor, out_nodes: int | None = None,
               reduce: str = "mean", edge_weight=1.0) -> Tensor:
    """Tensor product per edge, then scatter to the source node. Upstream's multigroup form.

    *edge_attr_groups* is a list of per-group edge-feature blocks whose row counts partition
    ``edge_index``'s columns in order; each group has its own FC block when *fc* is a
    ``ModuleList``. The sum and the divisor are accumulated across groups and the division
    happens once, which is what makes the result independent of how the edges were grouped.
    """
    if reduce not in ("mean", "sum"):
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")
    if isinstance(edge_attr_groups, Tensor):
        edge_attr_groups = [edge_attr_groups]
    lengths = [g.shape[0] for g in edge_attr_groups]
    if sum(lengths) != edge_index.shape[1]:
        raise ValueError(
            f"edge groups cover {sum(lengths)} edges but edge_index has {edge_index.shape[1]}")
    if isinstance(fc, nn.ModuleList) and len(fc) != len(edge_attr_groups):
        raise ValueError(f"{len(fc)} FC blocks for {len(edge_attr_groups)} edge groups")

    dtype, device = node_attr.dtype, node_attr.device
    edge_src, edge_dst = edge_index
    out_nodes = out_nodes or node_attr.shape[0]
    out = torch.zeros((out_nodes, tp.out_irreps.dim), dtype=dtype, device=device)
    div = torch.zeros(out_nodes, dtype=dtype, device=device)

    start = 0
    for i, group in enumerate(edge_attr_groups):
        end = start + lengths[i]
        rng = slice(start, end)
        src = edge_src[rng]
        block = fc[i] if isinstance(fc, nn.ModuleList) else fc
        w = block(group).to(device=device, dtype=dtype)
        if isinstance(edge_weight, Tensor):
            w = w * edge_weight[rng]
        elif edge_weight != 1.0:
            w = w * edge_weight
        summand = tp(node_attr[edge_dst[rng], :], edge_sh[rng, :], w)
        out.index_add_(0, src, summand)
        div += torch.bincount(src, minlength=out_nodes).to(dtype)
        start = end

    if reduce == "mean":
        div = torch.clamp(div, torch.finfo(dtype).eps)
        out = out / div[:, None]
    return out


class TensorProductConvLayer(nn.Module):
    """One convolution: ``(node_attr, edges) -> node_attr'``.

    *residual_first* exists because the two upstream layer classes disagree:
    ``TensorProductConvLayer`` (the score network) batch-norms the convolution output and then
    adds the residual, while ``OldTensorProductConvLayer`` (the confidence model) adds the
    residual first and batch-norms the sum. The weights are the same shape either way, so
    picking the wrong one loads cleanly and is only visible in the numbers.
    """

    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, *, residual=True,
                 batch_norm=True, hidden_features=None, edge_groups=1, tp_weights_layers=2,
                 activation="relu", residual_first=False):
        super().__init__()
        self.in_irreps = Irreps.parse(in_irreps)
        self.sh_irreps = Irreps.parse(sh_irreps)
        self.out_irreps = Irreps.parse(out_irreps)
        self.residual = residual
        self.residual_first = residual_first
        self.edge_groups = edge_groups
        self.out_size = self.out_irreps.dim
        hidden_features = hidden_features or n_edge_features

        self.tp = FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps)
        make = lambda: fc_block(n_edge_features, hidden_features, self.tp.weight_numel,
                                tp_weights_layers, activation)
        self.fc = make() if edge_groups == 1 else nn.ModuleList([make() for _ in range(edge_groups)])
        self.batch_norm = IrrepBatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None,
                reduce="mean", edge_weight=1.0):
        if edge_index.shape[1] == 0 and node_attr.shape[0] == 0:
            raise ValueError("no edges and no nodes")
        if edge_index.shape[1] == 0:
            out = torch.zeros((node_attr.shape[0], self.out_size), dtype=node_attr.dtype,
                              device=node_attr.device)
        else:
            out = tp_scatter(self.tp, self.fc, node_attr, edge_index, edge_attr, edge_sh,
                             out_nodes, reduce, edge_weight)
            if self.batch_norm is not None and not self.residual_first:
                out = self.batch_norm(out)
        if self.residual:
            out = out + torch.nn.functional.pad(
                node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
        if self.batch_norm is not None and self.residual_first and edge_index.shape[1] != 0:
            out = self.batch_norm(out)
        return out.to(node_attr.dtype)
