"""Rigid-body contact guidance, dense path of OpenDDE v1.2.0 ``opendde/tfg/rigid_contact.py`` (Apache-2.0).

Every update is a proper rigid transformation of the moving chain group; no contact atom
moves on its own. A backtracking gate refuses a step that adds a severe interchain overlap
or deepens an existing one, and a sample whose contacts all lie inside their windows is not
touched. Constants, step sizes, candidate poses, acceptance rules and op order are upstream's.

Two clash cores, chosen by ``core`` (upstream's OPENDDE_RIGID_CORE):
  "off"    upstream's dense path (every moving-fixed pair), op for op; what upstream runs on CPU.
  "auto"   the exact pair-list core of tt_bio.tfg.neighbors (the role of upstream's Triton cell lists): the same
           pairs, so the same energies up to fp32 summation order and the same severe sets. Refinement scores
           the nine shorter backtracking proposals in one call; the coarse searches score every candidate in
           batches and keep the first lowest feasible one below the entry energy, which is the pose the
           sequential strict-improvement loop keeps. Samples that already meet the request are not scored.
  "check"  both; logs the largest coordinate difference and returns the dense result.

The ``OPENDDE_RIGID_*`` environment knobs are replaced by :class:`RigidSchedule`.

Feature keys read (``feats``):
  contact_groups, refine_rigid_contact, search_rigid_contact:
    user_distance_restraint_index [2, C] long, user_distance_restraint_lower_bound [C],
    user_distance_restraint_upper_bound [C], user_rigid_movable_atom [N_atom] bool (optional;
    without it: asym_id [N_token], atom_to_token_idx [N_atom]), ref_element [N_atom, E] one-hot
    (index 0 is hydrogen; refine rejects it).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch

from tt_bio.tfg.neighbors import SparseClash, square_length

# Identical to upstream opendde.data.constants.rdkit_vdws (118 entries, H..Og, index 0 = H).
from tt_bio.data.const import vdw_radii as rdkit_vdws

logger = logging.getLogger(__name__)

SOFT_OVERLAP = 0.85
# Pair-element budget above which the clash test is evaluated in chunks.
EXACT_CLASH_BUDGET = 200_000_000
DEFAULT_LATE_START = 190

_MODES = ("auto", "off", "control", "on")
CORES = ("auto", "off", "check")
# Core used when a caller passes core=None (tests pin "off" for op-for-op parity with upstream).
DEFAULT_CORE = "auto"


@dataclass(frozen=True)
class RigidSchedule:
    """When and whether rigid guidance runs; replaces upstream's OPENDDE_RIGID_* environment knobs.

    mode      OPENDDE_RIGID_CONTACT: "auto", "on", "off" or "control" (see epitope.rigid_mode).
    x0_start  OPENDDE_RIGID_X0_START: first step of the early pass on the denoiser's x0; None turns it off.
    x0_every  OPENDDE_RIGID_X0_EVERY.
    x0_last   OPENDDE_RIGID_X0_LAST: last step of the early pass (the late pass owns the last steps).
    late      OPENDDE_RIGID_LATE: run the late pass on the sampled state.
    start     OPENDDE_RIGID_START: first late-pass step. None is upstream's default, step 190, or the
              last step when the run has fewer than 191 steps. An explicit value must lie inside the run.
    every     OPENDDE_RIGID_EVERY: late-pass refinement stride.
    core      OPENDDE_RIGID_CORE: "auto", "off" or "check" (module docstring); None is DEFAULT_CORE.
    """

    mode: str = "auto"
    x0_start: Optional[int] = 100
    x0_every: int = 3
    x0_last: int = 189
    late: bool = True
    start: Optional[int] = None
    every: int = 2
    core: Optional[str] = None

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError("Invalid rigid contact mode %r" % (self.mode,))
        if self.core is not None:
            resolve_core(self.core)
        if self.x0_start is not None and (
            self.x0_start < 0 or self.x0_every < 1 or self.x0_last < self.x0_start
        ):
            raise ValueError("Invalid x0_start / x0_every / x0_last")


def resolve_core(core):
    core = DEFAULT_CORE if core is None else core
    if core not in CORES:
        raise ValueError("Invalid rigid core %r" % (core,))
    return core


def checked(fn, coords, *args, **kwargs):
    """core="check": run both cores, log how far apart their results are, return the dense one."""
    dense = fn(coords, *args, core="off", **kwargs)
    fast = fn(coords, *args, core="auto", **kwargs)
    logger.info(
        "RIGID_CORE check %s: max |dense - auto| = %.3e A per sample %s",
        fn.__name__,
        (dense - fast).abs().max().item(),
        (dense - fast).abs().flatten(1).amax(1).tolist(),
    )
    return dense


def x0_schedule(schedule: RigidSchedule):
    """(start, every, last) of the early pass, or None when it is off."""
    if schedule.x0_start is None:
        return None
    return schedule.x0_start, schedule.x0_every, schedule.x0_last


def x0_step_active(step_i: int, schedule: RigidSchedule) -> bool:
    """Whether the early pass runs at sampler step ``step_i``."""
    sch = x0_schedule(schedule)
    if sch is None:
        return False
    start, every, last = sch
    return start <= step_i <= last and (step_i - start) % every == 0


def late_pass_enabled(schedule: RigidSchedule) -> bool:
    return bool(schedule.late)


def intervention_schedule(num_steps: int, schedule: RigidSchedule):
    """Late-pass steps: (coarse search steps, refinement steps)."""
    start = (
        min(DEFAULT_LATE_START, num_steps - 1)
        if schedule.start is None
        else int(schedule.start)
    )
    every = int(schedule.every)
    if not 0 <= start < num_steps or every < 1:
        raise ValueError("Invalid rigid start / every for %d steps" % num_steps)
    last = num_steps - 1
    coarse = sorted({start, last})
    refine = sorted({i for i in range(start, num_steps) if i % every == 0} | {last})
    return coarse, refine


def contact_groups(coords, feats):
    """(fixed atom ids, moving atom ids, fixed atom of each contact, moving atom of each contact).

    ``user_rigid_movable_atom`` names the moving chains (a Fab moves as one body). Without it the
    complex must have exactly two chains and the chain of each contact's second atom moves.
    """
    idx = feats["user_distance_restraint_index"]
    mask = feats.get("user_rigid_movable_atom")
    if mask is not None and mask.numel() != 0:
        # Only an absent or empty mask may fall back; a malformed one would move the wrong chain.
        if mask.ndim != 1 or mask.numel() != coords.shape[-2]:
            raise ValueError(
                "user_rigid_movable_atom must be a 1-D mask over %d atoms, got shape %s"
                % (coords.shape[-2], tuple(mask.shape))
            )
        mask = mask.to(torch.bool)
        if not mask.any() or mask.all():
            raise ValueError("movable_chains must name some but not all atoms.")
        moving_ids = torch.where(mask)[0]
        fixed_ids = torch.where(~mask)[0]
        left_moves = mask[idx[0]]
        right_moves = mask[idx[1]]
        if bool((left_moves == right_moves).any()):
            raise ValueError(
                "Every contact must join one atom of the moving group to one fixed atom."
            )
        mov_atoms = torch.where(left_moves, idx[0], idx[1])
        fix_atoms = torch.where(left_moves, idx[1], idx[0])
        return fixed_ids, moving_ids, fix_atoms, mov_atoms
    atom_chain = feats["asym_id"][feats["atom_to_token_idx"]]
    left = torch.unique(atom_chain[idx[0]])
    right = torch.unique(atom_chain[idx[1]])
    if left.numel() != 1 or right.numel() != 1 or left.item() == right.item():
        raise ValueError("Rigid contact requires contacts between two distinct chains.")
    if torch.unique(atom_chain).numel() != 2:
        raise ValueError("Rigid contact without movable_chains supports exactly two chains.")
    fixed_ids = torch.where(atom_chain == left.item())[0]
    moving_ids = torch.where(atom_chain == right.item())[0]
    return fixed_ids, moving_ids, idx[0], idx[1]


def clash_terms(x, fixed, rsum, fraction, want_gradient=False):
    """Repulsion energy sum(overlap**2), severe-overlap summary, depth and gradient against the fixed group.

    Exact branch: the summary is the full [S, M, F] boolean matrix. Chunked branch (above
    EXACT_CLASH_BUDGET pair elements): one severe count per chunk, for reporting; acceptance then
    goes through severe_transition. The gradient is that of 0.5 * 20 * sum(overlap**2).
    """
    samples, moving_n, _ = x.shape
    fixed_n = fixed.shape[-2]
    budget = EXACT_CLASH_BUDGET
    if samples * moving_n * fixed_n <= budget:
        dist = torch.cdist(x, fixed, compute_mode="donot_use_mm_for_euclid_dist").clamp_min(1e-6)
        overlap = torch.relu(fraction * rsum - dist)
        severe = dist < 0.75 * rsum
        depth = torch.relu(0.75 * rsum - dist).amax(dim=(-2, -1))
        energy = overlap.square().sum((-2, -1))
        if not want_gradient:
            return energy, severe, depth, None
        coeff = torch.where(dist > 1e-6, -20.0 * overlap / dist, torch.zeros_like(dist))
        grad = coeff.sum(-1, keepdim=True) * x - torch.bmm(coeff, fixed)
        return energy, severe, depth, grad
    step = max(1, budget // max(1, samples * moving_n))
    energy = torch.zeros(samples, device=x.device, dtype=x.dtype)
    counts = []
    depth = torch.zeros(samples, device=x.device, dtype=x.dtype)
    grad = torch.zeros_like(x) if want_gradient else None
    for start in range(0, fixed_n, step):
        stop = min(start + step, fixed_n)
        block = fixed[:, start:stop]
        radii = rsum[:, start:stop]
        dist = torch.cdist(x, block, compute_mode="donot_use_mm_for_euclid_dist").clamp_min(1e-6)
        overlap = torch.relu(fraction * radii - dist)
        energy = energy + overlap.square().sum((-2, -1))
        severe = dist < 0.75 * radii
        counts.append(severe.sum((-2, -1)))
        depth = torch.maximum(depth, torch.relu(0.75 * radii - dist).amax(dim=(-2, -1)))
        if want_gradient:
            coeff = torch.where(dist > 1e-6, -20.0 * overlap / dist, torch.zeros_like(dist))
            grad = grad + coeff.sum(-1, keepdim=True) * x - torch.bmm(coeff, block)
    return energy, torch.stack(counts, dim=-1), depth, grad


def severe_transition(previous_x, current_x, fixed, rsum):
    """Per sample: True when no severe overlap appeared that was absent before (chunked, pair-exact)."""
    samples, moving_n, _ = previous_x.shape
    fixed_n = fixed.shape[-2]
    step = max(1, EXACT_CLASH_BUDGET // max(1, samples * moving_n))
    ok = torch.ones(samples, device=previous_x.device, dtype=torch.bool)
    for start in range(0, fixed_n, step):
        stop = min(start + step, fixed_n)
        block = fixed[:, start:stop]
        threshold = 0.75 * rsum[:, start:stop]
        was = torch.cdist(previous_x, block, compute_mode="donot_use_mm_for_euclid_dist") < threshold
        now = torch.cdist(current_x, block, compute_mode="donot_use_mm_for_euclid_dist") < threshold
        ok &= ~(now & ~was).any(dim=(-2, -1))
    return ok


def severe_count(summary):
    """Severe overlaps per sample, from either summary form."""
    return summary.sum((-2, -1)) if summary.dtype == torch.bool else summary.sum(-1)


def no_new_severe(previous, current):
    """True per sample when no severe overlap appeared; per-chunk non-increasing counts in the chunked form."""
    if previous.dtype == torch.bool:
        return ~(current & ~previous).any(dim=(-2, -1))
    return (current <= previous).all(-1)


class DenseClash:
    """Upstream's clash test over every moving-fixed pair (clash_terms, no_new_severe / severe_transition)."""

    batched = False

    def __init__(self, fixed, ra, rb):
        self.fixed = fixed
        self.rsum = ra[:, None] + rb[None, :]

    def terms(self, x, want_gradient=False):
        return clash_terms(x, self.fixed, self.rsum, SOFT_OVERLAP, want_gradient=want_gradient)

    def no_new(self, previous_x, previous, x, current):
        if previous.dtype == torch.bool:
            return no_new_severe(previous, current)
        return severe_transition(previous_x, x, self.fixed, self.rsum)

    @staticmethod
    def count(summary):
        return severe_count(summary)


def clash_core(core, fixed, ra, rb):
    """The clash backend for core "off" (DenseClash) or "auto" (neighbors.SparseClash)."""
    return DenseClash(fixed, ra, rb) if core == "off" else SparseClash(fixed, ra, rb)


# Normals per batch of the pair-list coarse search (12 turns x 5 radii candidates each).
SEARCH_CHUNK_NORMALS = 5


def _take_first_best(energy, bad, rows, best_energy, feasible, improving, accepted):
    """Batched form of the sequential search rule for candidates [n, B] of samples ``rows``: a candidate is
    taken when feasible and strictly below the best so far, so the last one taken is the first lowest feasible
    candidate below the entry best. Updates best_energy and the counters in place; returns (index [n],
    taken [n]). ``improving`` counts feasible candidates only (a culled severe candidate has no full energy)."""
    masked = torch.where(bad, torch.full_like(energy, float("inf")), energy)
    running = torch.cat([best_energy[rows, None], masked], 1).cummin(1).values[:, :-1]
    better = masked < running
    feasible.index_add_(0, rows, (~bad).sum(-1))
    improving.index_add_(0, rows, better.sum(-1))
    accepted.index_add_(0, rows, better.sum(-1))
    low, index = masked.min(-1)
    taken = low < best_energy[rows]
    best_energy[rows] = torch.where(taken, low, best_energy[rows])
    return index, taken


def skew_matrix(w, like):
    """Cross-product matrices [S, 3, 3] of w [S, 3], built in the dtype of ``like``."""
    skew = torch.zeros_like(like)
    skew[:, 0, 1], skew[:, 0, 2] = -w[:, 2], w[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = w[:, 2], -w[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -w[:, 1], w[:, 0]
    return skew


def rigid_descent(moving, evaluate, satisfied_fn, n_terms, iterations, clash, needed=None):
    """Shared force/torque descent with backtracking, used by the contact and epitope refinements.

    ``evaluate(x, gradient, terms=None)`` returns (energy, severe, depth, aux[, grad]); ``terms`` hands it
    clash terms already computed for x. ``satisfied_fn(aux)`` marks samples that are left alone. ``n_terms``
    is the number of contact terms (C pairs or K residues); ``clash`` is the backend (clash_core).
    ``needed`` (moving-atom indices) are the atoms evaluate reads besides the clash pairs: given it, the batched
    backtracking proposals are placed for those atoms and the pair list's atoms only, whenever a bound on the
    rigid move proves the pair list valid for all of them (same values: a bmm row does not depend on the others).
    Returns (moving, accepted step count per sample).
    """
    samples = moving.shape[0]
    n_moving = moving.shape[1]
    device = moving.device
    accepted_count = torch.zeros(samples, device=device, dtype=torch.long)
    identity = torch.eye(3, device=device).expand(samples, 3, 3)
    for _ in range(iterations):
        energy, severe, depth, aux, grad = evaluate(moving, True)
        satisfied = satisfied_fn(aux)
        center = moving.mean(1, keepdim=True)
        centered = moving - center
        translation = -0.2 * grad.sum(1) / n_terms
        translation *= 0.5 / torch.linalg.vector_norm(translation, dim=-1, keepdim=True).clamp_min(0.5)
        torque = torch.linalg.cross(centered, grad, dim=-1).sum(1)
        inertia = centered.square().sum((1, 2))[:, None, None] * identity - torch.bmm(
            centered.transpose(1, 2), centered
        )
        rotation = (
            -0.15
            * n_moving
            / n_terms
            * torch.linalg.solve(inertia + 1e-3 * identity, torque[..., None]).squeeze(-1)
        )
        rotation *= 0.03 / torch.linalg.vector_norm(rotation, dim=-1, keepdim=True).clamp_min(0.03)

        def propose(backtrack):
            scale = 0.5**backtrack
            matrix = torch.linalg.matrix_exp(skew_matrix(rotation * scale, identity))
            return torch.bmm(centered, matrix.transpose(1, 2)) + center + translation[:, None] * scale

        def propose_many(backtracks):
            """propose(b) for several b at once [S, B, M, 3]: one batched matrix_exp and bmm."""
            scales = [0.5**b for b in backtracks]
            w = torch.cat([rotation * scale for scale in scales])
            matrix = torch.linalg.matrix_exp(skew_matrix(w, identity.repeat(len(scales), 1, 1)))
            turned = torch.bmm(centered.repeat(len(scales), 1, 1), matrix.transpose(1, 2))
            shift = torch.cat([translation[:, None] * scale for scale in scales])
            return (turned + center.repeat(len(scales), 1, 1) + shift).view(len(scales), samples, n_moving, 3).transpose(0, 1)

        def propose_rows(backtracks, rows):
            """propose_many for the atoms `rows` only; every other atom is NaN (never read)."""
            scales = [0.5**b for b in backtracks]
            w = torch.cat([rotation * scale for scale in scales])
            matrix = torch.linalg.matrix_exp(skew_matrix(w, identity.repeat(len(scales), 1, 1)))
            turned = torch.bmm(centered[:, rows].repeat(len(scales), 1, 1), matrix.transpose(1, 2))
            shift = torch.cat([translation[:, None] * scale for scale in scales])
            out = torch.full((samples, len(scales), n_moving, 3), float("nan"), dtype=moving.dtype)
            out[:, :, rows] = (turned + center.repeat(len(scales), 1, 1) + shift).view(
                len(scales), samples, len(rows), 3).transpose(0, 1)
            return out

        def sparse_rows():
            """Atoms the later proposals need, or None. Every later proposal moves an atom by at most
            0.5 (|w| r_max + |t|) from `moving`, which is d0 from the list's build pose: if that stays a margin inside
            the skin, the pair list's own check would pass too, so it is skipped and only listed atoms are read."""
            pairs = getattr(clash, "list", None)
            if needed is None or pairs is None or pairs.Q is None or pairs.Q.shape != moving.shape:
                return None
            d0 = float(square_length(moving - pairs.Q).max()) ** 0.5
            r_max = float(square_length(centered).max()) ** 0.5
            move = 0.5 * (float(torch.linalg.vector_norm(rotation, dim=-1).max()) * r_max
                          + float(torch.linalg.vector_norm(translation, dim=-1).max()))
            if not d0 + 1.01 * move + 1e-3 < pairs.skin - 2e-3:          # NaN falls back too
                return None
            return torch.unique(torch.cat([pairs.i, needed]))

        found = torch.zeros(samples, device=device, dtype=torch.bool)
        next_coords = moving.clone()
        later = None
        rows = None
        for backtrack in range(10):
            if clash.batched and backtrack >= 1:
                if later is None:
                    # The first proposal usually decides; the other nine go through the clash core together.
                    rows = sparse_rows()
                    if rows is None:
                        proposals = propose_many(range(1, 10))
                        later = proposals, clash.terms(proposals)
                    else:
                        proposals = propose_rows(range(1, 10), rows)
                        later = proposals, clash.terms(proposals, trusted=True)
                proposals, (e, sev, dep, _) = later
                j = backtrack - 1
                proposal = proposals[:, j]
                new_energy, new_severe, new_depth, _ = evaluate(
                    proposal, terms=(e[:, j], sev.item(j), dep[:, j], None)
                )
            else:
                proposal = propose(backtrack)
                new_energy, new_severe, new_depth, _ = evaluate(proposal)
            no_new_clash = clash.no_new(moving, severe, proposal, new_severe)
            accept = (
                (~found)
                & (~satisfied)
                & no_new_clash
                & (new_depth <= depth + 1e-6)
                & (new_energy < energy - 1e-7)
            )
            if accept.any():
                if rows is not None and backtrack >= 1:
                    proposal = propose_many([backtrack])[:, 0]          # the full pose, same values as its rows
                next_coords[accept] = proposal[accept]
            found |= accept
            if (found | satisfied).all():          # later proposals can accept nothing more
                break
        if not found.any():
            break
        moving = next_coords
        accepted_count += found.long()
    return moving, accepted_count


@torch.no_grad()
def refine_rigid_contact(coords, feats, iterations=40, core=None):
    """Rigid refinement of the moving group towards the contact windows. coords [S, N_atom, 3]."""
    core = resolve_core(core)
    if core == "check":
        return checked(refine_rigid_contact, coords, feats, iterations=iterations)
    if coords.ndim != 3:
        raise ValueError("Rigid contact requires [sample, atom, xyz].")
    idx = feats["user_distance_restraint_index"]
    if idx.numel() == 0:
        return coords
    fixed_ids, moving_ids, fix_atoms, mov_atoms = contact_groups(coords, feats)
    fixed_lookup = torch.full((coords.shape[-2],), -1, device=coords.device, dtype=torch.long)
    fixed_lookup[fixed_ids] = torch.arange(len(fixed_ids), device=coords.device)
    li = fixed_lookup[fix_atoms]
    moving_lookup = torch.full((coords.shape[-2],), -1, device=coords.device, dtype=torch.long)
    moving_lookup[moving_ids] = torch.arange(len(moving_ids), device=coords.device)
    ri = moving_lookup[mov_atoms]
    original_dtype = coords.dtype
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float().clone()
    radii = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    atomic_number_index = feats["ref_element"].argmax(-1)
    if (atomic_number_index == 0).any():
        raise ValueError("Rigid contact expects heavy-atom protein input; hydrogen found.")
    clash = clash_core(core, fixed, radii[atomic_number_index[moving_ids]], radii[atomic_number_index[fixed_ids]])
    lower = feats["user_distance_restraint_lower_bound"].float()
    upper = feats["user_distance_restraint_upper_bound"].float()
    entry = moving.clone()

    def evaluate(x, gradient=False, terms=None):
        pair_vec = x[:, ri] - fixed[:, li]
        pair_d = torch.linalg.vector_norm(pair_vec, dim=-1).clamp_min(1e-6)
        violation = torch.relu(pair_d - upper) - torch.relu(lower - pair_d)
        clash_energy, severe, severe_depth, clash_grad = terms or clash.terms(x, want_gradient=gradient)
        energy = 0.5 * (violation.square().sum(-1) + 20.0 * clash_energy)
        if not gradient:
            return energy, severe, severe_depth, pair_d
        grad = clash_grad
        # Below the clamp the energy is constant in the distance, so the gradient is zero there.
        contact_grad = violation[..., None] * pair_vec / pair_d[..., None]
        contact_grad = torch.where((pair_d > 1e-6)[..., None], contact_grad, torch.zeros_like(contact_grad))
        grad.index_add_(1, ri, contact_grad)
        return energy, severe, severe_depth, pair_d, grad

    def satisfied(pair_d):
        return ((pair_d >= lower - 1e-6) & (pair_d <= upper + 1e-6)).all(-1)

    first_energy, first_severe, _, first_d = evaluate(moving)
    moving, accepted_count = rigid_descent(moving, evaluate, satisfied, idx.shape[1], iterations, clash, ri.unique())
    final_energy, final_severe, _, final_d = evaluate(moving)
    intact = clash.no_new(entry, first_severe, moving, final_severe)
    if not bool(intact.all()):
        raise RuntimeError("Rigid contact introduced a severe interchain clash.")
    result = coords.clone()
    result[:, moving_ids] = moving.to(original_dtype)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "RIGID_CONTACT energy %s -> %s; mean_distance %s -> %s; severe_pairs %s -> %s; accepted %s",
            first_energy.tolist(),
            final_energy.tolist(),
            first_d.mean(-1).tolist(),
            final_d.mean(-1).tolist(),
            clash.count(first_severe).tolist(),
            clash.count(final_severe).tolist(),
            accepted_count.tolist(),
        )
    return result


@torch.no_grad()
def search_rigid_contact(coords, feats, core=None):
    """Deterministic coarse pose search over 25 axes x 12 turns x 5 radii around the contact site.

    The moving group is first Kabsch-fitted onto the contact pairs, then rotated and shifted out
    along each axis. Candidates with a severe interchain overlap are rejected; the lowest-energy
    remaining pose replaces the input only if its energy is lower. Satisfied samples are unchanged.
    """
    core = resolve_core(core)
    if core == "check":
        return checked(search_rigid_contact, coords, feats)
    fi, mi, fix_atoms, mov_atoms = contact_groups(coords, feats)
    fixed = coords[:, fi].float()
    moving = coords[:, mi].float()
    source = coords[:, mov_atoms].float()
    target = coords[:, fix_atoms].float()
    sc = source.mean(1, keepdim=True)
    tc = target.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(torch.bmm((source - sc).transpose(1, 2), target - tc))
    diag = torch.eye(3, device=coords.device).expand(len(coords), 3, 3).clone()
    diag[:, 2, 2] = torch.det(torch.bmm(u, vh))
    fit = torch.bmm(torch.bmm(u, diag), vh)
    fitted = torch.bmm(moving - sc, fit)
    fitted_contacts = torch.bmm(source - sc, fit)
    outward = (tc - fixed.mean(1, keepdim=True)).squeeze(1)
    outward /= torch.linalg.vector_norm(outward, dim=-1, keepdim=True).clamp_min(1e-6)
    radii = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    r = radii[feats["ref_element"].argmax(-1)]
    rsum = r[mi][:, None] + r[fi][None, :]
    lower = feats["user_distance_restraint_lower_bound"]
    upper = feats["user_distance_restraint_upper_bound"]

    def score_contact(contact):
        tgt = target if contact.dim() == 3 else target[todo, None]
        distance = torch.linalg.vector_norm(contact - tgt, dim=-1)
        return torch.relu(distance - upper) + torch.relu(lower - distance)

    def score(x, contact):
        distance = torch.linalg.vector_norm(contact - target, dim=-1)
        error = torch.relu(distance - upper) + torch.relu(lower - distance)
        clash_energy, summary, _, _ = clash_terms(x, fixed, rsum, SOFT_OVERLAP)
        severe = severe_count(summary) > 0
        energy = 0.5 * error.square().sum(-1) + 10 * clash_energy
        return energy, severe

    best = moving.clone()
    if core == "off":
        best_energy, bad = score(moving, source)
    else:
        clash = SparseClash(fixed, r[mi], r[fi])
        clash_energy, bad = clash.score(moving[:, None], torch.arange(len(coords), device=coords.device))
        best_energy = 0.5 * score_contact(source).square().sum(-1) + 10 * clash_energy[:, 0]
        bad = bad[:, 0]
    best_energy = torch.where(bad, torch.full_like(best_energy, float("inf")), best_energy)
    entry_distance = torch.linalg.vector_norm(coords[:, mov_atoms].float() - target, dim=-1)
    satisfied = ((entry_distance >= lower - 1e-6) & (entry_distance <= upper + 1e-6)).all(-1)
    normals = [outward]
    for i in range(24):
        z = 1 - 2 * (i + 0.5) / 24
        phi = i * math.pi * (3 - math.sqrt(5))
        rad = math.sqrt(1 - z * z)
        normals.append(
            torch.tensor([rad * math.cos(phi), rad * math.sin(phi), z], device=coords.device).expand(
                len(coords), 3
            )
        )
    accepted = torch.zeros(len(coords), device=coords.device, dtype=torch.long)
    feasible = torch.zeros(len(coords), device=coords.device, dtype=torch.long)
    improving = torch.zeros(len(coords), device=coords.device, dtype=torch.long)
    tested = 0
    turns = [i * math.pi / 6 for i in range(12)]
    radii_out = [5.5, 7.5, 10.0, 15.0, 20.0]
    todo = torch.nonzero(~satisfied).squeeze(1)
    if core != "off":
        # poses are placed for the unsatisfied samples only (per-sample ops: same values)
        t_fit, t_fitted, t_contacts, t_tc = fit[todo], fitted[todo], fitted_contacts[todo], tc[todo]

        def place(chunk):
            """Poses of a chunk of normals in upstream's (normal, turn, radius) order: x [T, P, M, 3], contacts [T, P, C, 3].
            The turns of a normal share one batched matrix_exp and bmm (per-matrix values unchanged)."""
            T, R = len(todo), len(radii_out)
            P = len(chunk) * len(turns) * R
            x = torch.empty(T, P, t_fitted.shape[1], 3, dtype=t_fitted.dtype)
            contact = torch.empty(T, P, t_contacts.shape[1], 3, dtype=t_contacts.dtype)
            for a, normal in enumerate(chunk):
                normal = normal[todo]
                w = torch.cat([normal * turn for turn in turns])                          # [turns * T, 3]
                rotation = torch.linalg.matrix_exp(skew_matrix(w, t_fit.repeat(len(turns), 1, 1))).transpose(1, 2)
                rotated = torch.bmm(t_fitted.repeat(len(turns), 1, 1), rotation).view(len(turns), T, -1, 3)
                rot_contacts = torch.bmm(t_contacts.repeat(len(turns), 1, 1), rotation).view(len(turns), T, -1, 3)
                for b in range(len(turns)):
                    for c, radius in enumerate(radii_out):
                        shift = t_tc + radius * normal[:, None]
                        k = (a * len(turns) + b) * R + c
                        torch.add(rotated[b], shift, out=x[:, k])
                        torch.add(rot_contacts[b], shift, out=contact[:, k])
            return x, contact

        for start in range(0, len(normals), SEARCH_CHUNK_NORMALS):
            chunk = normals[start:start + SEARCH_CHUNK_NORMALS]
            tested += len(chunk) * len(turns) * len(radii_out)
            if todo.numel() == 0:
                continue
            x, contact = place(chunk)
            clash_energy, bad = clash.score(x, todo)
            energy = 0.5 * score_contact(contact).square().sum(-1) + 10 * clash_energy
            index, taken = _take_first_best(energy, bad, todo, best_energy, feasible, improving, accepted)
            pick = torch.nonzero(taken).squeeze(1)
            best[todo[pick]] = x[pick, index[pick]]
    for normal in normals if core == "off" else []:
        for turn in turns:
            rotation = torch.linalg.matrix_exp(skew_matrix(normal * turn, fit))
            rotated = torch.bmm(fitted, rotation.transpose(1, 2))
            rot_contacts = torch.bmm(fitted_contacts, rotation.transpose(1, 2))
            for radius in radii_out:
                shift = tc + radius * normal[:, None]
                proposed = rotated + shift
                proposed_contacts = rot_contacts + shift
                energy, bad = score(proposed, proposed_contacts)
                tested += 1
                feasible += (~bad).long()
                improving += (energy < best_energy).long()
                take = (~bad) & (~satisfied) & (energy < best_energy)
                best[take] = proposed[take]
                best_energy = torch.where(take, energy, best_energy)
                accepted += take.long()
    result = coords.clone()
    result[:, mi] = best.to(coords.dtype)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "RIGID_CONTACT coarse search satisfied=%s candidates=%d feasible=%s improving=%s accepted=%s final_energy=%s",
            satisfied.tolist(),
            tested,
            feasible.tolist(),
            improving.tolist(),
            accepted.tolist(),
            best_energy.tolist(),
        )
    return result
