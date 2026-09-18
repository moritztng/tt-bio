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
class Rigid:
    """The running frame as ABodyBuilder3 carries it: a quaternion and a translation, per residue.

    The quaternion is the state and the rotation matrix is derived, which is upstream's own choice
    (`Rigid.identity(..., fmt="quat")` and `compose_q_update_vec`) and not an implementation detail:
    the backbone update composes onto the quaternion and renormalises it, so keeping the matrix as
    state would drift off SO(3) over 8 blocks.
    """

    quat: list          # 4 device tensors, each [B, N]
    trans: list         # 3 device tensors, each [B, N]

    @classmethod
    def identity(cls, batch: int, n_tok: int, *, upload) -> "Rigid":
        one = upload(torch.ones(batch, n_tok))
        zero = upload(torch.zeros(batch, n_tok))
        return cls(quat=[one, zero, zero, zero], trans=[zero, zero, zero])

    def frame(self) -> "Frame":
        """The rotation matrix for this quaternion, as nine broadcastable components."""
        a, b, c, d = self.quat
        aa, bb, cc, dd = (ops.mul(x, x) for x in (a, b, c, d))
        ab, ac, ad = ops.mul(a, b), ops.mul(a, c), ops.mul(a, d)
        bc, bd, cd = ops.mul(b, c), ops.mul(b, d), ops.mul(c, d)
        two = lambda p, q, sign: ops.scale(ops.add(p, q) if sign > 0 else ops.sub(p, q), 2.0)
        rot = [
            [ops.sub(ops.add(aa, bb), ops.add(cc, dd)), two(bc, ad, -1), two(bd, ac, +1)],
            [two(bc, ad, +1), ops.sub(ops.add(aa, cc), ops.add(bb, dd)), two(cd, ab, -1)],
            [two(bd, ac, -1), two(cd, ab, +1), ops.sub(ops.add(aa, dd), ops.add(bb, cc))],
        ]
        return Frame(rot=rot, trans=list(self.trans))

    def compose_update(self, update: list) -> "Rigid":
        """`Rigid.compose_q_update_vec`: a pure-vector quaternion update, then a local translation.

        The quaternion product is written out rather than taken from a table, and it is the standard
        Hamilton product of `(a,b,c,d)` with `(0, v)`. The translation update is rotated by the OLD
        rotation before it is added, which is upstream's order and AlphaFold's
        (`Rigid.compose_q_update_vec:1018` applies `self._rots`, not the new ones).
        """
        a, b, c, d = self.quat
        v1, v2, v3 = update[:3]
        da = ops.scale(ops.add(ops.add(ops.mul(b, v1), ops.mul(c, v2)), ops.mul(d, v3)), -1.0)
        db = ops.sub(ops.add(ops.mul(a, v1), ops.mul(c, v3)), ops.mul(d, v2))
        dc = ops.sub(ops.add(ops.mul(a, v2), ops.mul(d, v1)), ops.mul(b, v3))
        dd_ = ops.sub(ops.add(ops.mul(a, v3), ops.mul(b, v2)), ops.mul(c, v1))
        raw = [ops.add(a, da), ops.add(b, db), ops.add(c, dc), ops.add(d, dd_)]
        norm_sq = None
        for q in raw:
            term = ops.mul(q, q)
            norm_sq = term if norm_sq is None else ops.add(norm_sq, term)
        norm = ops.sqrt_plus(norm_sq, 0.0)
        quat = [ops.div(q, norm) for q in raw]

        rot = self.frame().rot
        trans = []
        for i in range(3):
            acc = None
            for j in range(3):
                term = ops.mul(rot[i][j], update[3 + j])
                acc = term if acc is None else ops.add(acc, term)
            trans.append(ops.add(self.trans[i], acc))
        return Rigid(quat=quat, trans=trans)


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

        # The pair-bias projection: ONE matmul of 12 output columns, then a reorientation that
        # rounds at 7.5e-04. The exact alternative -- 12 matmuls of one column each, whose
        # [B, N, N, 1] results reshape to [B, 1, N, N] for free -- was built first and measured, and
        # it is what makes this the right trade rather than a concession:
        #
        #   * memory. A [B, N, N, 1] tile tensor pads its width-1 axis to 32, so each column costs
        #     67 MB instead of 2 at micro-batch 8 and 256 tokens, and 12 of them retained per block
        #     for the backward is 805 MB per block, 6.4 GB over 8 blocks. That is what put the step
        #     measurement into OOM on a 34 GB card.
        #   * bandwidth. Each column matmul reads the whole pair tensor, 268 MB, so the 12 of them
        #     move 3.2 GB per block where one matmul moves 268 MB.
        #
        # What it costs is 7.5e-04 relative on the bias term, against the ~1e-03 the logits already
        # carry from the matmul itself, so it does not become the dominant error. The model gate
        # quotes the whole-model number with this in place rather than trusting that argument.
        self.w_b = to_device(weights["linear_b.weight"].t())
        self.b_b = to_device(weights["linear_b.bias"])

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
        every residue pair in one program and is fp32-exact, and it carries the square in its own
        packer -- so the 50 MB difference is never written, and its backward recomputes it rather
        than holding 150 MB per block for the whole backward pass.
        """
        cfg = self.cfg
        h, pq = cfg.no_heads_ipa, cfg.no_qk_points
        acc = None
        for q_d, k_d in zip(q_pts, k_pts):
            sq = ops.sub_square(q_d, k_d)
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

        bias = ops.permute(ops.linear(z, self.w_b, self.b_b), [0, 3, 1, 2])
        logits = ops.add(logits, ops.scale(bias, (1.0 / 3) ** 0.5))

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


class DeviceTransition:
    """`StructureModuleTransition` at `no_transition_layers=1`: a 3-linear ReLU residual, then LN."""

    def __init__(self, weights: dict, cfg: ABB3Config, *, to_device):
        s = "layers.0."
        self.w = [to_device(weights[f"{s}linear_{i}.weight"].t()) for i in (1, 2, 3)]
        self.b = [to_device(weights[f"{s}linear_{i}.bias"]) for i in (1, 2, 3)]
        self.ln_w = to_device(weights["layer_norm.weight"])
        self.ln_b = to_device(weights["layer_norm.bias"])

    def __call__(self, s, *, dropout=None):
        h = ops.relu(ops.linear(s, self.w[0], self.b[0]))
        h = ops.relu(ops.linear(h, self.w[1], self.b[1]))
        s = ops.add(s, ops.linear(h, self.w[2], self.b[2]))
        if dropout is not None:
            s = dropout(s)
        return ops.layer_norm(s, self.ln_w, self.ln_b)


class DeviceAngleResnet:
    """Alg. 20 lines 11-14 at `use_original_sm=True`, with sin and cos in separate tile blocks.

    The output is 7 sin/cos pairs interleaved as 14 channels upstream, and 2 is a quarter of a tile,
    so the pairs cannot be split on device without a sub-tile slice. The host weight is reordered
    instead: sin into channels 0-6 of one 32-wide block, cos into channels 0-6 of the next. Then
    `sin^2 + cos^2` is an elementwise add whose channel k already holds angle k's squared norm, so
    the normalisation needs no reduction at all, and the padded channels normalise 0 to 0.
    """

    def __init__(self, weights: dict, cfg: ABB3Config, *, to_device):
        self.eps = cfg.epsilon
        self.w_in = to_device(weights["linear_in.weight"].t())
        self.b_in = to_device(weights["linear_in.bias"])
        self.w_init = to_device(weights["linear_initial.weight"].t())
        self.b_init = to_device(weights["linear_initial.bias"])
        self.blocks = []
        for i in range(cfg.no_resnet_blocks):
            self.blocks.append(([to_device(weights[f"layers.{i}.linear_{j}.weight"].t())
                                 for j in (2, 3)],
                                [to_device(weights[f"layers.{i}.linear_{j}.bias"]) for j in (2, 3)]))
        out_w, out_b = weights["linear_out.weight"], weights["linear_out.bias"]
        n = cfg.no_angles
        padded_w = out_w.new_zeros(2 * TILE, out_w.shape[1])
        padded_b = out_b.new_zeros(2 * TILE)
        for part in (0, 1):                       # 0 = sin, 1 = cos, upstream interleaves them
            padded_w[part * TILE:part * TILE + n] = out_w[part::2]
            padded_b[part * TILE:part * TILE + n] = out_b[part::2]
        self.w_out = to_device(padded_w.t())
        self.b_out = to_device(padded_b)

    def __call__(self, s, s_initial):
        h = ops.add(ops.linear(ops.relu(s), self.w_in, self.b_in),
                    ops.linear(ops.relu(s_initial), self.w_init, self.b_init))
        for w, b in self.blocks:
            inner = ops.relu(ops.linear(ops.relu(h), w[0], b[0]))
            h = ops.add(h, ops.linear(inner, w[1], b[1]))
        raw = ops.linear(ops.relu(h), self.w_out, self.b_out)
        width = int(raw.shape[-1])
        sin = ops.slice_dim(raw, -1, 0, TILE)
        cos = ops.slice_dim(raw, -1, TILE, width)
        norm_sq = ops.add(ops.mul(sin, sin), ops.mul(cos, cos))
        norm = ops.sqrt_plus(ops.clamp_min(norm_sq, self.eps), 0.0)
        return (sin, cos), (ops.div(sin, norm), ops.div(cos, norm))


class DeviceBackboneUpdate:
    """Alg. 23 line 10, as six one-column projections.

    Six outputs is a fifth of a tile, so one 6-wide projection would have to be sliced per channel
    below tile granularity. Six projections of width 1 cost six launches on a 128x1 matmul and hand
    back exactly the `[B, N]` scalars the frame is built from.
    """

    def __init__(self, weights: dict, *, to_device):
        w, b = weights["linear.weight"], weights["linear.bias"]
        self.w = [to_device(w[i:i + 1].t()) for i in range(6)]
        self.b = [to_device(b[i:i + 1]) for i in range(6)]

    def __call__(self, s):
        batch, n_tok = [int(d) for d in s.shape][:2]
        return [ops.reshape(ops.linear(s, w, b), [batch, n_tok])
                for w, b in zip(self.w, self.b)]


class DevicePlddtHead:
    """`PerResidueLDDTCaPredictor`: LN, two ReLU linears, 50 bins. Absent from `base-loss`."""

    def __init__(self, weights: dict, *, to_device):
        self.ln_w = to_device(weights["layer_norm.weight"])
        self.ln_b = to_device(weights["layer_norm.bias"])
        self.w = [to_device(weights[f"linear_{i}.weight"].t()) for i in (1, 2, 3)]
        self.b = [to_device(weights[f"linear_{i}.bias"]) for i in (1, 2, 3)]

    def __call__(self, s):
        h = ops.layer_norm(s, self.ln_w, self.ln_b)
        h = ops.relu(ops.linear(h, self.w[0], self.b[0]))
        h = ops.relu(ops.linear(h, self.w[1], self.b[1]))
        return ops.linear(h, self.w[2], self.b[2])


def _scope(weights: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


class DeviceABB3:
    """The whole model: `linear_in_*`, then 8 blocks of IPA / transition / update / angles.

    The pair representation is rebuilt every block because its last channel is the pairwise distance
    between the current frames' translations and the frames move. Only that channel moves, so
    `linear_in_edge` runs once and its 127 channels are shared by all 8 blocks -- and its host
    weight is widened to 128 with a zero column, so the distance is ADDED into the last channel
    rather than concatenated onto 127. Concatenating 127 + 1 on the last axis is not tile-legal, and
    the add is one broadcast multiply by a one-hot constant plus one add, whose backward reduces
    straight back to the distance without a sub-tile slice.

    What comes back is per-block device tensors. The geometry tail -- torsion angles to frames, then
    frames and literature positions to atom14 -- stays in torch on the host, where it is already
    scored in float64 against upstream and where its gather-heavy rigid-group table lookups cost
    nothing on 8x256 residues. That boundary is also what makes the four losses comparable against
    upstream's own `loss.py` without a device port of each.
    """

    def __init__(self, state_dict: dict, cfg: ABB3Config | None = None, *,
                 to_device=None, use_plddt: bool | None = None):
        weights = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        weights = weights or dict(state_dict)
        if use_plddt is None:
            use_plddt = any(k.startswith("plddt.") for k in weights)
        self.cfg = cfg or ABB3Config(use_plddt=use_plddt)
        upload = to_device or to_device_fp32
        self.upload = upload
        h = self.cfg.no_heads_ipa

        edge_w, edge_b = weights["linear_in_edge.weight"], weights["linear_in_edge.bias"]
        wide_w = edge_w.new_zeros(self.cfg.embed_dim, edge_w.shape[1])
        wide_b = edge_b.new_zeros(self.cfg.embed_dim)
        wide_w[:edge_w.shape[0]] = edge_w
        wide_b[:edge_b.shape[0]] = edge_b
        self.w_edge, self.b_edge = upload(wide_w.t()), upload(wide_b)
        self.w_node = upload(weights["linear_in_node.weight"].t())
        self.b_node = upload(weights["linear_in_node.bias"])
        #: The one-hot that puts the distance into the pair tensor's last channel.
        self.dist_channel = to_device_fp32(
            torch.eye(self.cfg.embed_dim)[-1].reshape(1, 1, 1, self.cfg.embed_dim))

        self.ipa, self.ln, self.transition, self.bb_update, self.angles = [], [], [], [], []
        for i in range(self.cfg.no_blocks):
            self.ipa.append(DeviceIPA(_scope(weights, f"ipa_layers.{i}."), self.cfg,
                                      to_device=upload))
            self.ln.append((upload(weights[f"layer_norm_ipa_layers.{i}.weight"]),
                            upload(weights[f"layer_norm_ipa_layers.{i}.bias"])))
            self.transition.append(DeviceTransition(_scope(weights, f"transition_layers.{i}."),
                                                    self.cfg, to_device=upload))
            self.bb_update.append(DeviceBackboneUpdate(_scope(weights, f"bb_update_layers.{i}."),
                                                       to_device=upload))
            self.angles.append(DeviceAngleResnet(_scope(weights, f"angle_resnet_layers.{i}."),
                                                 self.cfg, to_device=upload))
        self.plddt = (DevicePlddtHead(_scope(weights, "plddt."), to_device=upload)
                      if self.cfg.use_plddt else None)

    def _pair_distance(self, rigid: Rigid, square_mask, n_tok: int, batch: int):
        """The masked pairwise distance between the current translations, as `[B, N, N, 1]`."""
        acc = None
        for t in rigid.trans:
            col = ops.reshape(t, [batch, 1, n_tok, 1])
            row = ops.reshape(t, [batch, 1, 1, n_tok])
            diff = ops.sub(col, row)
            term = ops.mul(diff, diff)
            acc = term if acc is None else ops.add(acc, term)
        dist = ops.mul(ops.norm_from_sq(acc, 1e-12), square_mask)
        return ops.reshape(ops.reshape(dist, [batch, n_tok, n_tok]), [batch, n_tok, n_tok, 1])

    def __call__(self, single, pair, square_mask, masked_bias, *, dropout=None) -> dict:
        """`square_mask` is a raw `[B, 1, N, N]` of 0/1 and `masked_bias` is `inf * (mask - 1)`.

        Both are constants derived from the sequence mask on the host. They are passed in rather
        than built here because the training loop builds one per bucket and reuses it across steps.
        """
        cfg = self.cfg
        batch, n_tok = [int(d) for d in single.shape][:2]
        z_initial = ops.linear(pair, self.w_edge, self.b_edge)
        s = ops.linear(single, self.w_node, self.b_node)
        s_initial = s
        rigid = Rigid.identity(batch, n_tok, upload=self.upload)

        out = {k: [] for k in ("quat", "trans", "sin", "cos", "unnorm_sin", "unnorm_cos", "states")}
        for i in range(cfg.no_blocks):
            dist = self._pair_distance(rigid, square_mask, n_tok, batch)
            z = ops.add(z_initial, ops.mul(dist, self.dist_channel))
            s = ops.add(s, self.ipa[i](s, z, rigid.frame(), masked_bias))
            if dropout is not None:
                s = dropout(s)
            s = ops.layer_norm(s, *self.ln[i])
            s = self.transition[i](s, dropout=dropout)
            rigid = rigid.compose_update(self.bb_update[i](s))
            (unnorm_sin, unnorm_cos), (sin, cos) = self.angles[i](s, s_initial)
            out["quat"].append(list(rigid.quat))
            out["trans"].append([ops.scale(t, cfg.trans_scale_factor) for t in rigid.trans])
            out["sin"].append(sin)
            out["cos"].append(cos)
            out["unnorm_sin"].append(unnorm_sin)
            out["unnorm_cos"].append(unnorm_cos)
            out["states"].append(s)
        out["single"] = s
        if self.plddt is not None:
            out["plddt"] = self.plddt(s)
        return out
