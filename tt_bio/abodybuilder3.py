"""ABodyBuilder3 on the device: Alg. 22's invariant point attention and the block around it.

Inference module. It is written entirely in `abodybuilder3_ops`, so with no tape installed it is
the production forward and with `tt_bio.train.abodybuilder3_grad.install()` it is differentiable,
and the two compute the same bytes because a taped op's value comes from the shipped call.

**The layout is the design, and it was chosen by measurement.** Three facts from
`scripts/abb3_port/precision_probe.py` and `relayout_probe.py`, all on qb1 card 3:

* A matmul on fp32 operands keeps ~11 mantissa bits (1.25e-03 relative); eltwise is fp32-exact.
* Reorienting the last two axes rounds at 7.5e-04 through every call there is, while a permute of
  leading axes and a reshape that adds a 1 axis are bit-exact.
* A matmul transposing its own operands costs nothing over a plain one.

Together they say: never reorient after the fact, produce each projection in the orientation its
consumer needs, and never write an algebraic identity that cancels. Hence

* the point term is a two-sided broadcast subtract, `[B, 144, N, 1]` against `[B, 144, 1, N]`, not
  `|q|^2 + |k|^2 - 2 q.k` -- which was measured at 0.70 logit error and refused;
* the key points come out of `matmul(w, s, transpose_a=True, transpose_b=True)`, which is `(s @ w)^T`
  and lands residues on the width for free;
* the coordinate index of a point is folded into the HEAD axis, so the rotation -- which mixes
  coordinates -- mixes slices of an untiled axis rather than sub-tile channel ranges;
* every head's channel block is padded to 32 in the host weight, which is `TriangleAttention`'s own
  sub-tile path (`tenstorrent.py:7100-7120`), and the zeros contribute nothing to any product;
* the output projection is six matmuls against six host-sliced weight blocks, summed, so the
  2112-wide concatenation never exists.

The frame is carried as component tensors, one `[B, N]` per rotation entry and translation
component, reshaped to `[B, 1, N, 1]` or `[B, 1, 1, N]` at each use. That is what makes applying a
frame nine broadcast multiplies instead of a batched 3x3 matmul on a 3-wide axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from . import abodybuilder3_ops as ops
from .abodybuilder3_reference import ABB3Config
from .tenstorrent import get_device

TILE = 32


def _pad_heads(w: torch.Tensor, n_heads: int, per_head: int) -> torch.Tensor:
    """Pad each head's channel block out to a tile, in the host weight.

    `w` is upstream's `(out, in)` with `out = n_heads * per_head`. The result is
    `(n_heads * 32, in)` with zeros in the padding rows, so the device reshape that splits heads
    stays tile-legal and the padded channels contribute nothing to any product they enter.
    """
    out, cin = w.shape
    assert out == n_heads * per_head, (out, n_heads, per_head)
    padded = w.new_zeros(n_heads, TILE, cin)
    padded[:, :per_head] = w.reshape(n_heads, per_head, cin)
    return padded.reshape(n_heads * TILE, cin)


def _point_weight(w: torch.Tensor, n_heads: int, n_points: int, *, pad: bool) -> torch.Tensor:
    """Reorder a point projection's rows from upstream's layout to `(coord, head, point)`.

    Upstream builds points by `split(3, dim=-1)` then `stack`, so its row index is
    `coord * (heads * points) + head * points + point` -- coordinate-major already. What changes
    here is only the per-group padding: `pad` widens each `(coord, head)` group to a tile, which is
    what the value points need because the inverse rotation reads them back per head.

    Coordinate-major is the load-bearing half. It puts the three coordinate blocks at
    `[0:heads]`, `[heads:2*heads]`, `[2*heads:3*heads]` of the device head axis, so the rotation
    slices an untiled axis.
    """
    out, cin = w.shape
    assert out == 3 * n_heads * n_points, (out, n_heads, n_points)
    blocks = w.reshape(3, n_heads, n_points, cin)
    if not pad:
        return blocks.reshape(out, cin)
    padded = w.new_zeros(3, n_heads, TILE, cin)
    padded[:, :, :n_points] = blocks
    return padded.reshape(3 * n_heads * TILE, cin)


@dataclass
class Frame:
    """One rigid frame per residue, as twelve `[B, N]` component tensors.

    `rot` is row-major `(i, j)` so that `rot[i][j]` is the matrix entry, and applying the frame to a
    point is `sum_j rot[i][j] * p_j + trans[i]`. Inverting it reads the same entries transposed,
    which is why the components are kept separate rather than assembled into a 3x3.
    """

    rot: list[list]      # 3x3 of device tensors, each [B, N]
    trans: list          # 3 device tensors, each [B, N]

    def column(self, t):
        """`[B, N] -> [B, 1, N, 1]`: indexed by the QUERY residue, broadcast over heads."""
        b, n = [int(d) for d in t.shape]
        return ops.reshape(t, [b, 1, n, 1])

    def row(self, t):
        """`[B, N] -> [B, 1, 1, N]`: indexed by the KEY residue, broadcast over heads."""
        b, n = [int(d) for d in t.shape]
        return ops.reshape(t, [b, 1, 1, n])


def _apply(frame: Frame, pts, orient, *, invert: bool):
    """Apply the frame, or its inverse, to three coordinate blocks of points.

    `pts` is `(x, y, z)`, each `[B, heads * k, ...]`, and `orient` is `Frame.column` or `Frame.row`
    depending on whether the points are indexed by the query or the key residue. Nine broadcast
    multiplies and six adds; the inverse subtracts the translation first and reads the rotation
    transposed, which is `QuatAffine.invert_point`.
    """
    if invert:
        pts = [ops.sub(p, orient(t)) for p, t in zip(pts, frame.trans)]
    out = []
    for i in range(3):
        acc = None
        for j in range(3):
            r = orient(frame.rot[j][i] if invert else frame.rot[i][j])
            term = ops.mul(pts[j], r)
            acc = term if acc is None else ops.add(acc, term)
        if not invert:
            acc = ops.add(acc, orient(frame.trans[i]))
        out.append(acc)
    return out


class DeviceIPA:
    """Alg. 22 at ABodyBuilder3's dims, on the device.

    `head_weights` is the one parameter kept on the HOST: it is 12 scalars, `ttnn.softplus` carries
    3.37e-03 relative (`op_gradcheck.py`), and what the device needs is
    `softplus(w) * sqrt(1 / (3 * 4.5 * P))` as a `[1, H, 1, 1]` constant. Computing it on the host in
    fp64 and uploading 12 floats is exact where the kernel is not, and its gradient comes back
    through one reduction.
    """

    def __init__(self, weights: dict, cfg: ABB3Config, *, to_device):
        self.cfg = cfg
        h, c = cfg.no_heads_ipa, cfg.c_ipa
        pq, pv = cfg.no_qk_points, cfg.no_v_points
        self.w_q = to_device(_pad_heads(weights["linear_q.weight"], h, c).t())
        self.b_q = to_device(_pad_heads(weights["linear_q.bias"].unsqueeze(-1), h, c).squeeze(-1))
        kv = weights["linear_kv.weight"].reshape(h, 2 * c, -1)
        kv_b = weights["linear_kv.bias"].reshape(h, 2 * c, 1)
        self.w_k = to_device(_pad_heads(kv[:, :c].reshape(h * c, -1), h, c).t())
        self.b_k = to_device(_pad_heads(kv_b[:, :c].reshape(h * c, 1), h, c).squeeze(-1))
        self.w_v = to_device(_pad_heads(kv[:, c:].reshape(h * c, -1), h, c).t())
        self.b_v = to_device(_pad_heads(kv_b[:, c:].reshape(h * c, 1), h, c).squeeze(-1))

        self.w_qp = to_device(_point_weight(weights["linear_q_points.weight"], h, pq, pad=False).t())
        self.b_qp = to_device(_point_weight(weights["linear_q_points.bias"].unsqueeze(-1), h, pq,
                                            pad=False).squeeze(-1))
        kvp = weights["linear_kv_points.weight"].reshape(3, h, pq + pv, -1)
        kvp_b = weights["linear_kv_points.bias"].reshape(3, h, pq + pv, 1)
        # Kept in upstream's (out, in) orientation rather than transposed like the others,
        # because `_points_row` wants it that way and a weight transpose on the host is free.
        self.w_kp = to_device(_point_weight(kvp[:, :, :pq].reshape(-1, kvp.shape[-1]), h, pq,
                                            pad=False))
        self.b_kp = to_device(_point_weight(kvp_b[:, :, :pq].reshape(-1, 1), h, pq,
                                            pad=False).squeeze(-1))
        # The value points are the one projection whose consumer is a matmul over residues
        # (`attn @ v_pts`), so they need the points on the WIDTH and residues on the height -- the
        # opposite of what the query and key points need. Three separate projections, one per
        # coordinate, each head-padded, so `_heads` can split them exactly.
        vp = _point_weight(kvp[:, :, pq:].reshape(-1, kvp.shape[-1]), h, pv, pad=True)
        vp_b = _point_weight(kvp_b[:, :, pq:].reshape(-1, 1), h, pv, pad=True).squeeze(-1)
        stride = h * TILE
        self.w_vp = [to_device(vp[i * stride:(i + 1) * stride].t()) for i in range(3)]
        self.b_vp = [to_device(vp_b[i * stride:(i + 1) * stride]) for i in range(3)]

        # The pair-bias projection, one column per head. 12 matmuls of one output column rather than
        # one matmul of 12: the 12-wide result would have to be reoriented to [B, H, N, N] and
        # moving the last axis rounds at 7.5e-04. Each column's [B, N, N, 1] result reshapes to
        # [B, 1, N, N] for free, and a leading-axis concat assembles them exactly.
        self.w_b = [to_device(weights["linear_b.weight"][i:i + 1].t()) for i in range(h)]
        self.b_b = [to_device(weights["linear_b.bias"][i:i + 1]) for i in range(h)]

        # The output projection, split into the six blocks its input is a concatenation of, each
        # padded to the device layout of the piece that feeds it.
        out_w = weights["linear_out.weight"]              # (c_s, H*(c_z + c + 4*pv))
        widths = [h * c, h * pv, h * pv, h * pv, h * pv, h * cfg.embed_dim]
        per_head = [c, pv, pv, pv, pv, None]
        blocks, at = [], 0
        for width, ph in zip(widths, per_head):
            block = out_w[:, at:at + width]
            at += width
            if ph is not None:
                block = _pad_heads(block.t().contiguous(), h, ph).t().contiguous()
            blocks.append(to_device(block.t().contiguous()))
        assert at == out_w.shape[1], (at, out_w.shape)
        self.w_out = blocks
        self.b_out = to_device(weights["linear_out.bias"])
        self.head_weight_scale = (1.0 / (3 * (pq * 9.0 / 2))) ** 0.5
        self.head_weights_host = weights["head_weights"].detach().clone()
        self.head_weight = to_device(
            (torch.nn.functional.softplus(self.head_weights_host.double())
             * self.head_weight_scale).reshape(1, h, 1, 1).float())

    def head_weights_grad(self, device_grad: torch.Tensor) -> torch.Tensor:
        """Map the gradient of the uploaded constant back onto the host parameter.

        The device carries `softplus(w) * scale`, so `dL/dw = dL/d(uploaded) * scale * sigmoid(w)`.
        Both factors are computed here in float64 on 12 numbers, which is the point of keeping this
        parameter on the host: `ttnn.softplus` carries 3.37e-03 relative, and 12 scalars per block
        are not worth a kernel's error.
        """
        g = device_grad.double().reshape(-1)
        return (g * self.head_weight_scale
                * torch.sigmoid(self.head_weights_host.double())).to(self.head_weights_host.dtype)

    def _heads(self, x, n):
        """`[B, N, H*32] -> [B, H, N, 32]`, both steps bit-exact."""
        b, tok = [int(d) for d in x.shape][:2]
        return ops.permute(ops.reshape(x, [b, tok, n, TILE]), [0, 2, 1, 3])

    def _points_column(self, s, w, bias, total):
        """Point projection with residues on the HEIGHT and a width of 1: `[B, total/3, N, 1]` per
        coordinate.

        A width of 1 is what the pairwise subtract wants on the query side, and getting there is two
        bit-exact moves: `reshape` to add the trailing 1 axis, then a leading-axis `permute` to put
        residues on the height. Each coordinate block is contiguous on the resulting head axis
        because the weight is coordinate-major.
        """
        b, tok = [int(d) for d in s.shape][:2]
        flat = ops.linear(s, w, bias)
        pts = ops.permute(ops.reshape(flat, [b, tok, total, 1]), [0, 2, 1, 3])
        per = total // 3
        return [ops.slice_dim(pts, 1, i * per, (i + 1) * per) for i in range(3)]

    def _points_row(self, s, w, bias, total):
        """Point projection with residues on the WIDTH: `[B, total/3, 1, N]` per coordinate.

        The key points are the only tensor in the block that needs residues on the width, and the
        orientation has to come out of the matmul because moving it afterwards rounds at 7.5e-04 --
        which on a global coordinate in Angstrom is 0.02 A, and puts a squared distance 1.6 A^2 out.

        `matmul(w, s, transpose_b=True)` is `(s @ w)^T` and needs no transpose of the weight at all,
        since the host keeps it in `(out, in)` order. What it does need is the sample axis folded
        into the width: ttnn refuses a rank-2 operand against a batched one unless the batched
        side's front dims are 1 ("front dimensions need to be 1"), so `s` enters as
        `[1, B*N, c_s]`. Unfolding afterwards is a reshape, a leading-axis permute and a reshape,
        measured at exactly 0.0 added error (`scripts/abb3_port/relayout_probe.py` covers the class;
        this chain was checked directly when it was written).

        The per-sample alternative -- B small matmuls and a leading-axis concat -- measures the same
        1.43e-03, which is the matmul's own floor, and costs B launches per block instead of one.
        """
        b, tok, c_in = [int(d) for d in s.shape]
        folded = ops.reshape(s, [1, b * tok, c_in])
        out = ops.matmul(w, folded, transpose_b=True)          # [1, total, B*N]
        out = ops.permute(ops.reshape(out, [total, b, tok]), [1, 0, 2])
        out = ops.reshape(out, [b, total, 1, tok])
        out = ops.add(out, ops.reshape(bias, [1, total, 1, 1]))
        per = total // 3
        return [ops.slice_dim(out, 1, i * per, (i + 1) * per) for i in range(3)]

    def _point_term(self, q_pts, k_pts, n_tok, batch):
        """`sum_p |q_ip - k_jp|^2` per head, as `[B, H, N, N]`.

        One coordinate at a time, and that is the memory decision: the `[B, H*P, N, N]` difference
        is 100 MB at micro-batch 8 and 256 tokens where all three coordinates at once would be 302,
        and the three partial sums add afterwards for nothing. Each coordinate's channel order is
        `(head, point)`, so the reshape that groups the points inside a head is free and the
        reduction runs over an axis that is already grouped.

        The subtract is two-sided broadcast, `[B, H*P, N, 1]` against `[B, H*P, 1, N]`, which is
        every residue pair in one program and is fp32-exact.
        """
        cfg = self.cfg
        h, pq = cfg.no_heads_ipa, cfg.no_qk_points
        acc = None
        for q_d, k_d in zip(q_pts, k_pts):
            diff = ops.sub(q_d, k_d)
            sq = ops.mul(diff, diff)
            per_head = ops.sum_dim(ops.reshape(sq, [batch * h, pq, n_tok, n_tok]), 1)
            term = ops.reshape(per_head, [batch, h, n_tok, n_tok])
            acc = term if acc is None else ops.add(acc, term)
        return acc

    def __call__(self, s, z, frame: Frame, square_mask):
        """`s` is `[B, N, c_s]`, `z` is `[B, N, N, c_z]`, `square_mask` a raw `[B, 1, N, N]`."""
        cfg = self.cfg
        h, c = cfg.no_heads_ipa, cfg.c_ipa
        pq, pv = cfg.no_qk_points, cfg.no_v_points
        b, n_tok = [int(d) for d in s.shape][:2]

        q = self._heads(ops.linear(s, self.w_q, self.b_q), h)
        k = self._heads(ops.linear(s, self.w_k, self.b_k), h)
        v = self._heads(ops.linear(s, self.w_v, self.b_v), h)

        # Three logit terms of equal variance. The scalar term is scaled after the matmul, as
        # upstream scales it, so the operands enter the matmul unscaled.
        logits = ops.scale(ops.matmul(q, k, transpose_b=True), (1.0 / (3 * c)) ** 0.5)

        bias_cols = []
        for i in range(h):
            col = ops.linear(z, self.w_b[i], self.b_b[i])
            bias_cols.append(ops.reshape(col, [b, 1, n_tok, n_tok]))
        logits = ops.add(logits, ops.scale(ops.concat(bias_cols, dim=1), (1.0 / 3) ** 0.5))

        q_pts = _apply(frame, self._points_column(s, self.w_qp, self.b_qp, 3 * h * pq),
                       frame.column, invert=False)
        k_pts = _apply(frame, self._points_row(s, self.w_kp, self.b_kp, 3 * h * pq),
                       frame.row, invert=False)
        d2 = self._point_term(q_pts, k_pts, n_tok, b)
        logits = ops.sub(logits, ops.scale(ops.mul(d2, self.head_weight), 0.5))
        logits = ops.add(logits, square_mask)
        attn = ops.softmax(logits)

        o = ops.reshape(ops.permute(ops.matmul(attn, v), [0, 2, 1, 3]), [b, n_tok, h * TILE])

        # The value points are rotated in the KEY residue's frame, because the attention contracts
        # over the key; the result is then brought back into the QUERY residue's frame, which is why
        # the inverse rotation happens after the attention and not before.
        v_pts = _apply(frame, [self._heads(ops.linear(s, w, bias), h)
                               for w, bias in zip(self.w_vp, self.b_vp)],
                       frame.column, invert=False)
        o_pt = _apply(frame, [ops.matmul(attn, p) for p in v_pts], frame.column, invert=True)
        o_pt_norm = None
        for p in o_pt:
            term = ops.mul(p, p)
            o_pt_norm = term if o_pt_norm is None else ops.add(o_pt_norm, term)
        o_pt_norm = ops.sqrt_plus(o_pt_norm, cfg.epsilon)
        flat = [ops.reshape(ops.permute(p, [0, 2, 1, 3]), [b, n_tok, h * TILE])
                for p in (*o_pt, o_pt_norm)]

        o_pair = ops.matmul(ops.permute(attn, [0, 2, 1, 3]), z)
        o_pair = ops.reshape(o_pair, [b, n_tok, h * int(z.shape[-1])])

        pieces = [o, *flat, o_pair]
        out = None
        for piece, weight in zip(pieces, self.w_out):
            term = ops.linear(piece, weight)
            out = term if out is None else ops.add(out, term)
        return ops.add(out, ops.reshape(self.b_out, [1, 1, cfg.embed_dim]))


def to_device_fp32(t: torch.Tensor):
    """Upload a weight as fp32 in tile layout. One place, so the port has one weight dtype."""
    return ttnn.from_torch(t.contiguous().float(), layout=ttnn.TILE_LAYOUT, device=get_device(),
                           dtype=ttnn.float32)


def frame_from_quaternion(quat: torch.Tensor, trans: torch.Tensor, *, upload=to_device_fp32
                          ) -> Frame:
    """Build a `Frame` on device from a host quaternion and translation, for a parity harness.

    The training loop builds its frames on device from the backbone update, so this exists for the
    gate: it takes the reference's own `QuatAffine` state and uploads the components it derives.
    """
    from .af2_reference import QuatAffine
    affine = QuatAffine(quat, trans)
    rot = affine.rotation
    return Frame(rot=[[upload(rot[..., i, j]) for j in range(3)] for i in range(3)],
                 trans=[upload(affine.translation[..., i]) for i in range(3)])
