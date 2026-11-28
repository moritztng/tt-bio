"""What one chunk of edges costs, so the chunk size is a division and not a search.

`tp_scatter(..., edge_chunk=N)` slices each edge group into fixed-size pieces and accumulates
them, which is what keeps a large complex inside a card's memory. *N itself* is a device
measurement -- the largest chunk that fits beside everything else in the layer -- but a row
without a card can still remove the search from it, because the only unknown that needs a
card is the **budget**, not the cost.

The cost per edge is a property of the layer's irreps, not of the data: every tensor the
chunk allocates has a leading edge axis and a width fixed by the signature. :func:`per_edge`
measures that width by running one chunk under the op recorder of :mod:`tt_bio.diffdock.trace`
and summing what each op returned, and :func:`chunk_for_budget` divides a budget by it. The
measurement is host-torch, so for a device it is a floor rather than a prediction -- ttnn
tilizes to 32x32, keeps both operands of a binary op live, and adds its own L1 scratch. Hence
:func:`chunk_for_budget` takes an explicit *slack*, and the honest use of this module is "the
chunk cannot be larger than this", with the card deciding how much smaller.

Why the per-edge cost is dominated by one tensor: the per-edge tensor-product weight block is
`weight_numel` wide -- 4,164 values at the released ns=48/nv=10 cross signature, against 78
for the node features and 9 for the spherical harmonics. The FC block emits it, the mask
multiplies it (a second copy), and the tensor product reads it. Everything else in the chunk
is noise beside it, which is also why chunking the *edge* axis is the lever that works.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .trace import record_ops


@dataclass(frozen=True)
class EdgeCost:
    """Bytes per edge for one chunked call, measured rather than derived."""

    total_elems: float      # every tensor the chunk allocates, summed, per edge
    peak_elems: float       # the single widest per-edge tensor -- the weight block
    dtype_bytes: int

    @property
    def total_bytes(self) -> float:
        return self.total_elems * self.dtype_bytes

    @property
    def peak_bytes(self) -> float:
        return self.peak_elems * self.dtype_bytes

    def __str__(self) -> str:
        return (f"{self.total_bytes:.0f} B/edge allocated, widest single tensor "
                f"{self.peak_bytes:.0f} B/edge")


def _elems_per_edge(ops, n_edges: int, dtype) -> tuple[float, float]:
    total = 0
    peak = 0
    for op in ops:
        for shape, _ in op.out:
            if not shape or shape[0] != n_edges:
                continue            # not an edge-axis tensor: a node tensor or a constant
            n = 1
            for d in shape:
                n *= d
            total += n
            peak = max(peak, n)
    return total / n_edges, peak / n_edges


def per_edge(run_chunk, n_edges: int, dtype=torch.float32) -> EdgeCost:
    """Run one chunk of *n_edges* edges under the recorder and divide.

    *run_chunk* is a zero-argument callable issuing exactly the work of one chunk -- in
    practice ``lambda: tp_scatter(..., edge_chunk=n_edges)`` on an edge set of that size.
    Only tensors whose leading axis is the edge count are counted: the node accumulators and
    the Wigner constants do not scale with the chunk and belong to the budget, not the slope.
    """
    if n_edges < 1:
        raise ValueError(f"n_edges must be positive, got {n_edges}")
    _, ops = record_ops(run_chunk)
    total, peak = _elems_per_edge(ops, n_edges, dtype)
    return EdgeCost(total, peak, torch.empty((), dtype=dtype).element_size())


def chunk_for_budget(cost: EdgeCost, budget_bytes: float, *, slack: float = 0.5,
                     multiple: int = 32) -> int:
    """The largest chunk that fits in *budget_bytes*, rounded **down** to *multiple*.

    *slack* is what a device costs beyond the host's tensor arithmetic -- tiling to 32x32,
    both operands of a binary op live at once, the framework's own scratch. It is a knob
    rather than a constant because only the card can set it: pass 1.0 to ask for the pure
    host number. Rounding down to 32 keeps the chunk a legal tile height, which is also what
    makes the chunked loop traceable: every slice has the same shape.
    """
    if not 0 < slack <= 1:
        raise ValueError(f"slack must be in (0, 1], got {slack}")
    usable = budget_bytes * slack
    n = int(usable // cost.total_bytes)
    n -= n % multiple
    return max(n, multiple)


def report(cost: EdgeCost, budgets_gib=(1.0, 4.0, 16.0), slack: float = 0.5) -> str:
    """A table for `dfd-device`: the chunk this cost implies at a few plausible budgets."""
    lines = [str(cost), f"slack {slack} (device overhead beyond host tensor arithmetic)", ""]
    for g in budgets_gib:
        n = chunk_for_budget(cost, g * 2 ** 30, slack=slack)
        lines.append(f"  budget {g:>5.1f} GiB -> edge_chunk {n:>8d}")
    return "\n".join(lines)


# --------------------------------------------------------------- the pose axis, not the edge axis

def chunked_score_fn(score_fn, width: int, *, n_torsions: int, reset=None, narrowest: int = 1):
    """``score_fn`` evaluated *width* poses per device call, narrowing when the card refuses one.

    The two chunking axes answer different questions. ``edge_chunk`` splits one convolution so a
    large complex fits; this splits the **pose batch**, which is what ``--batch_size`` names
    upstream and what grows the whole step's memory linearly.

    It is exact, not an approximation, and for the same reason the batch exists at all: the ten
    poses are a disjoint union of ten copies of the same graph (:meth:`StaticGraphs.build`), so no
    message ever crosses between them and a pose's scores do not depend on which other poses rode
    along. Splitting the batch is therefore bit-identical, unlike splitting the *sampler*, whose
    per-step noise draw is one tensor over the whole batch -- chunk there and every pose gets
    different noise, which is why this wraps the score function and leaves
    :func:`tt_bio.diffdock.sampling.sample` alone.

    The refusal ladder is :func:`tt_bio.sample_chunks.denoise_in_chunks`, the one every tt-bio
    sampler shares, so a DiffDock refusal narrows by the same rule and prints the same line as a
    Boltz-2 one. To hand it a single tensor the three scores ride as one row per pose,
    ``[tr | rot | tor]``, ``6 + n_torsions`` wide. A narrowed width persists to the next step:
    the rollout pays for one refused allocation, not nineteen. *reset* drops device state keyed
    on the old width (a captured trace, above all).
    """
    from dataclasses import replace

    from tt_bio.sample_chunks import denoise_in_chunks, resolve_sample_chunk_width

    state = {"width": max(1, int(width))}

    def run_chunk(step):
        def run(chunk, _w):
            tr, rot, tor = score_fn(replace(step, lig_pos=chunk))[:3]
            return torch.cat([tr.reshape(chunk.shape[0], 3), rot.reshape(chunk.shape[0], 3),
                              tor.reshape(chunk.shape[0], n_torsions)], dim=1)

        n_poses = step.lig_pos.shape[0]
        rows, state["width"] = denoise_in_chunks(
            step.lig_pos, resolve_sample_chunk_width(n_poses, state["width"]), run,
            reset=reset, narrowest=narrowest, tag="diffdock")
        return rows[:, :3], rows[:, 3:6], rows[:, 6:].reshape(n_poses * n_torsions)

    run_chunk.width = lambda: state["width"]
    return run_chunk
