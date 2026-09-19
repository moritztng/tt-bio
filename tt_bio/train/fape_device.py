"""The sidechain FAPE on the device. Host inputs, host output, N^2 arithmetic on the card.

Measured reason for it to exist: at batch 64 and 256 tokens the sidechain FAPE is **86 % of the
loss stage and ~17 s of a 38.5 s complete step** -- 8.6 s of forward plus most of an 11.6 s host
backward -- because it builds `[B, N*8, N*14, 3]` tensors, 117 MB per coordinate, and host torch is
memory-bound on them while the card is not.

What makes it portable without moving the geometry tail is the shape of the problem: its INPUTS are
a couple of MB (2,048 frames and 3,584 points per sample) and only its INTERMEDIATES are N^2-scale.
So the frames and positions go up, the pairwise arithmetic runs on the card in
`abodybuilder3_ops`, one per-sample vector comes back, and the taped ops supply the backward --
which is then handed to torch as a normal `autograd.Function`, so the host graph either side of it
does not know the difference.

Two things are deliberately NOT taped, because they are constants and taping them would retain
351 MB of intermediates each for a gradient nobody reads: the target frames' local coordinates, and
the `[B, F, P]` clamp pattern. The pattern is built on the card from two small region vectors rather
than uploaded, which is the difference between 117 MB of PCIe per micro-batch and none.

**The frame axis is chunked, and the forward keeps no chunk on the tape.** Both sums are separable
over frames (sum over points inside a frame, then sum over frames), so a chunk computes its own
partial and the partials add, which is the same arithmetic in a different order. The forward runs
each chunk with raw leaves, so `abodybuilder3_ops`' grad hook declines every op in it and nothing
N^2-scale is retained; the backward rebuilds one chunk at a time with taped leaves, replays it
against that chunk's own incoming gradient, and drops it before starting the next. The chunk's
forward therefore runs twice and its intermediates exist once.

That is a memory requirement rather than a tuning knob, and `scripts/abb3_port/fape_memory_census.py`
is where the requirement is measured. Retaining the chunks -- which is what this file did until
`train-h-fape-24mb` -- costs 4 968 MB at micro-batch 4 and 256 tokens: 451 MB of constants, 3 155 MB
of retained tape and a 1 362 MB backward transient. On top of the model's own tape that does not fit
on a 32 640 MB card, and qb2 measured it OOM at 29.36 MB wanted against 5.56 MB free, which is one
`[4, 512, 3584]` fp32 chunk tensor short. Checkpointing takes the same configuration to 786 MB.

The leaves are rebuilt per chunk rather than sliced from shared ones, because a sliced gradient has
to be put back by allocating zeros and concatenating -- the mistake that made a reduction 2.7x worse
earlier in this port. `pos_row` is the exception: it spans every chunk, it is `[B, 1, P]` rather than
N^2, and one shared leaf is what accumulates its gradient across the chunks in the order the frame
sum visits them.

`scripts/abb3_port/loss_gate.py` scores this against upstream's own `sidechain_loss` in float64,
which is the only thing that matters about it: the reproduction has to train their objective.
"""

from __future__ import annotations

import torch
import ttnn

from .. import abodybuilder3_ops as ops
from ..abodybuilder3 import to_device_fp32
from . import abodybuilder3_grad as grad


def _col(t: torch.Tensor):
    """`[B, F] -> [B, F, 1]` on device: frame-indexed, broadcast over points."""
    b, f = t.shape
    return to_device_fp32(t.reshape(b, f, 1))


def _row(t: torch.Tensor):
    """`[B, P] -> [B, 1, P]` on device: point-indexed, broadcast over frames."""
    b, p = t.shape
    return to_device_fp32(t.reshape(b, 1, p))


def _local_coords(rot_col, trans_col, pos_row):
    """`R_f^T (p - t_f)` for every frame/point pair, as three `[B, F, P]` tensors.

    `Rigid.invert().apply` in the only layout that keeps it cheap: the frame axis on the height, the
    point axis on the width, and the rotation entering as nine `[B, F, 1]` broadcasts. The ops are
    the same for the prediction and the target side; what differs is the leaves. Raw leaves make
    `abodybuilder3_ops`' hook decline, so the target side runs the shipped op for a gradient nobody
    reads, and `Tensor` leaves make the same call a tape node.
    """
    centred = [ops.sub(pos_row[d], trans_col[d]) for d in range(3)]
    out = []
    for c in range(3):
        acc = None
        for d in range(3):
            term = ops.mul(centred[d], rot_col[d][c])
            acc = term if acc is None else ops.add(acc, term)
        out.append(acc)
    return out


def _clamp_pattern(frame_region: torch.Tensor, atom_region: torch.Tensor, near: float,
                   far: float):
    """`[B, F, P]` of `near` where the frame and the atom share a region and `far` where they do not.

    ABodyBuilder3's own change to FAPE (`cdr_clamp`): a frame and an atom in different regions are
    clamped at the larger distance. Built on the card from the two region vectors, untaped, because
    it is a constant and uploading it would be 117 MB of PCIe per micro-batch.
    """
    col = _col(frame_region.float())
    row = _row(atom_region.float())
    differs = ttnn.gtz(ttnn.multiply(ttnn.subtract(col, row), ttnn.subtract(col, row)))
    return ttnn.add(ttnn.multiply(differs, far - near), near)


def _chunk_leaves(pred_rot: torch.Tensor, pred_trans: torch.Tensor, start: int, stop: int,
                  *, taped: bool):
    """One chunk's nine rotation and three translation leaves, `[B, stop-start, 1]` each.

    Uploaded from the host slice rather than sliced on the card, and rebuilt in the backward rather
    than retained, which is the whole of what makes the chunk a checkpoint: 12 columns of
    `stop-start` floats per sample, against the `[B, stop-start, P]` intermediates they generate.
    """
    wrap = grad.param if taped else (lambda x: x)
    rot_col = [[wrap(_col(pred_rot[:, start:stop, d, c].contiguous())) for c in range(3)]
               for d in range(3)]
    trans_col = [wrap(_col(pred_trans[:, start:stop, d].contiguous())) for d in range(3)]
    return rot_col, trans_col


def _chunk_partial(rot_col, trans_col, pos_row, chunk_const, const):
    """A chunk's contribution to the frame sum, `[B, 1, 1]`.

    `inv_frames` and `inv_points` are applied once to the total, not per chunk, so this is the inner
    sum and nothing else. Whether it lands on the tape is decided entirely by its leaves.
    """
    local = _local_coords(rot_col, trans_col, pos_row)
    err2 = None
    for c in range(3):
        term = ops.sub_square(local[c], chunk_const["target_local"][c])
        err2 = term if err2 is None else ops.add(err2, term)
    err = ops.minimum(ops.sqrt_plus(err2, const["eps"]), chunk_const["cap"])
    normed = ops.scale(err, 1.0 / const["length_scale"])
    normed = ops.mul(ops.mul(normed, chunk_const["frames_mask"]), const["positions_mask"])
    return ops.sum_dim(ops.sum_last(normed), 1)


class _DeviceSidechainFape(torch.autograd.Function):
    """Host tensors in, a per-sample loss out, and one chunk of device tape at a time in between."""

    @staticmethod
    def forward(ctx, pred_rot, pred_trans, pred_pos, const):
        # The tape has to be installed while the tail below runs, because that is what turns its op
        # calls into tape nodes. It is installed for the duration and restored after, rather than
        # left to the caller: a caller that forgot would get `ttnn` type errors from inside this
        # file, which is the least useful place to read that mistake.
        previous = ops.grad_hook()
        if previous is None:
            grad.install()
        batch = pred_trans.shape[0]
        pos_row = [_row(pred_pos[:, :, d].contiguous()) for d in range(3)]

        # Each chunk runs off raw leaves, so every op in it is offered to the hook and declined, and
        # the whole chunk collapses to the `[B, 1, 1]` partial as soon as Python drops the last
        # reference to it. Only the partials cross into the taped tail.
        partials = []
        for start, stop, chunk_const in const["chunks"]:
            rot_col, trans_col = _chunk_leaves(pred_rot, pred_trans, start, stop, taped=False)
            partials.append(grad.param(_chunk_partial(rot_col, trans_col, pos_row, chunk_const,
                                                      const)))

        total = None
        for partial in partials:
            total = partial if total is None else ops.add(total, partial)
        per_sample = ops.mul(ops.mul(total, const["inv_frames"]), const["inv_points"])
        ctx.saved = (pred_rot, pred_trans, pred_pos, const, partials, per_sample, batch)
        if previous is None:
            ops.set_grad_hook(None)
        return torch.as_tensor(ttnn.to_torch(per_sample.value).reshape(batch),
                               dtype=pred_pos.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        pred_rot, pred_trans, pred_pos, const, partials, per_sample, batch = ctx.saved
        previous = ops.grad_hook()
        if previous is None:
            grad.install()
        seed = to_device_fp32(grad_out.detach().float().reshape(batch, 1, 1))
        grad.backward([per_sample], [seed])

        n_points = pred_pos.shape[1]
        pos_row = [grad.param(_row(pred_pos[:, :, d].contiguous())) for d in range(3)]

        def pull(p, width):
            return ttnn.to_torch(p.grad).reshape(batch, width).float()

        # One chunk of tape exists at a time: rebuild it, replay it against the gradient its partial
        # received, read the twelve columns out, and drop it before the next chunk allocates. Each
        # chunk owns its own leaves, so the frame-axis gradient is a concatenation rather than a
        # scatter -- no zero-padding and no device-side concat. `pos_row` is shared, so its gradient
        # accumulates over the chunks instead.
        rot_parts, trans_parts = [], []
        for (start, stop, chunk_const), partial in zip(const["chunks"], partials):
            width = stop - start
            rot_col, trans_col = _chunk_leaves(pred_rot, pred_trans, start, stop, taped=True)
            replayed = _chunk_partial(rot_col, trans_col, pos_row, chunk_const, const)
            grad.backward([replayed], [partial.grad])
            rot_parts.append(torch.stack(
                [torch.stack([pull(rot_col[d][c], width) for c in range(3)], -1)
                 for d in range(3)], -2))
            trans_parts.append(torch.stack([pull(trans_col[d], width) for d in range(3)], -1))
            del rot_col, trans_col, replayed
        d_rot = torch.cat(rot_parts, dim=1)
        d_trans = torch.cat(trans_parts, dim=1)
        d_pos = torch.stack([pull(pos_row[d], n_points) for d in range(3)], -1)
        if previous is None:
            ops.set_grad_hook(None)
        return d_rot, d_trans, d_pos, None


def prepare_sidechain_constants(target_rot: torch.Tensor, target_trans: torch.Tensor,
                                target_pos: torch.Tensor, frames_mask: torch.Tensor,
                                positions_mask: torch.Tensor, frame_region: torch.Tensor,
                                atom_region: torch.Tensor, *, length_scale: float = 10.0,
                                clamp_distance: float = 10.0, intercdr_distance: float = 30.0,
                                eps: float = 1e-4, frame_chunk: int = 512) -> dict:
    """Everything the device side needs that does not change with the prediction.

    Built once per batch rather than once per optimizer step. The target local coordinates are the
    expensive part -- three `[B, F, P]` tensors -- and they depend only on the ground truth, so a
    training run pays for them once per batch and not once per step. They are also the whole of what
    this dict costs: 451 MB of the term's 786 MB at micro-batch 4 and 256 tokens, measured by
    `scripts/abb3_port/fape_memory_census.py`.
    """
    pos_row = [_row(target_pos[:, :, d].contiguous()) for d in range(3)]
    eps_f, eps_frames = float(eps), float(eps)
    n_frames = target_trans.shape[1]
    chunks = []
    for start in range(0, n_frames, frame_chunk):
        stop = min(start + frame_chunk, n_frames)
        rot_col, trans_col = _chunk_leaves(target_rot, target_trans, start, stop, taped=False)
        chunks.append((start, stop, {
            "target_local": _local_coords(rot_col, trans_col, pos_row),
            "cap": _clamp_pattern(frame_region[:, start:stop], atom_region, clamp_distance,
                                  intercdr_distance),
            "frames_mask": _col(frames_mask[:, start:stop].float()),
        }))
    return {
        "chunks": chunks,
        "positions_mask": _row(positions_mask.float()),
        # Upstream's fp16-friendly averaging order: divide by the frame count, sum, divide by the
        # point count, which duplicates eps relative to a single division. Kept, because the point
        # of this file is to be the same number.
        "inv_frames": to_device_fp32(
            (1.0 / (eps_frames + frames_mask.float().sum(-1))).reshape(-1, 1, 1)),
        "inv_points": to_device_fp32(
            (1.0 / (eps_f + positions_mask.float().sum(-1))).reshape(-1, 1, 1)),
        "eps": eps,
        "length_scale": length_scale,
    }


def sidechain_fape_device(pred_frames_4x4: torch.Tensor, pred_positions: torch.Tensor,
                          const: dict) -> torch.Tensor:
    """Per-sample sidechain FAPE, computed on the card. `[B]`.

    `pred_frames_4x4` is `[B, F, 4, 4]` and `pred_positions` `[B, P, 3]`, both flattened over the
    residue and group axes exactly as `losses_geometry.sidechain_fape` flattens them, so the two are
    interchangeable at the call site and the gate can score one against the other.
    """
    return _DeviceSidechainFape.apply(pred_frames_4x4[..., :3, :3].contiguous(),
                                      pred_frames_4x4[..., :3, 3].contiguous(),
                                      pred_positions, const)
