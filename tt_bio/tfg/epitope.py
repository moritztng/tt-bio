"""Epitope-only rigid guidance, dense path of OpenDDE v1.2.0 ``opendde/tfg/epitope_guidance.py`` (Apache-2.0).

Moves the antibody (the movable group) as one body until at least K of the requested antigen
residues are in heavy-atom contact with its binding face. Only the sampler's coordinates are used.

  reached   an epitope residue is reached when one of its heavy atoms is within GATE of a paratope atom
  request   at least K of the n epitope residues are reached
  energy    0.5 * sum over the K residues closest to the paratope of relu(d_e - TARGET)^2, plus the soft
            clash term of rigid.py (weight 20 * 0.5 in the refinement, 10 in the coarse search)
  updates   proper rigid transformations of the moving group, with rigid.py's force/torque reduction,
            step clipping and backtracking; a move that adds a severe clash or does not lower the energy is refused
  skip      a sample that already meets the request is not touched

The Triton core and Fold-CP paths are not ported (upstream ``OPENDDE_RIGID_CORE=off``).

Feature keys read (``feats``):
  groups, refine_epitope, search_epitope:
    user_rigid_movable_atom [N_atom] bool, user_epitope_atom_index [n, m] long (-1 padded, residue x atom),
    user_epitope_paratope_atom [N_atom] bool, user_epitope_k [1] long, ref_element [N_atom, E] one-hot
  search_epitope additionally: asym_id [N_token], atom_to_token_idx [N_atom]
  rigid_mode: user_epitope_atom_index, user_distance_restraint_index, user_rigid_movable_atom, asym_id,
    atom_to_token_idx
  guide_x0: everything above, plus the contact keys of rigid.py for a contact request
"""

from __future__ import annotations

import logging
import math

import torch

from tt_bio.tfg import rigid as rc
from tt_bio.tfg.neighbors import SparseClash, square_length
from tt_bio.tfg.rigid import RigidSchedule, rdkit_vdws

logger = logging.getLogger(__name__)

GATE = 5.0
TARGET = 4.5
CLASH_WEIGHT_REFINE = 20.0
CLASH_WEIGHT_SEARCH = 10.0
OVERLAP_FRACTION = 0.85
NORMAL_RADIUS = 12.0
TILT_DEGREES = 15.0
TILT_AZIMUTHS = 12
FIBONACCI_AXES = 24
SPINS = 24
STANDOFFS = (2.0, 4.0, 6.0, 9.0, 12.0)
# The anchor is the mean of this fraction of paratope atoms reaching furthest along the binding face.
TIP_FRACTION = 0.1
# Pair-list core batching: approach axes per scoring call (24 spins x 5 standoffs each).
SEARCH_CHUNK_AXES = 2
PATCH_FRACTION = 0.10
PATCH_MINIMUM = 100


def active(feats) -> bool:
    """True when the request carries an epitope."""
    return feats.get("user_epitope_atom_index") is not None


def rigid_mode(feats, tfg_enabled: bool = True, schedule: RigidSchedule = RigidSchedule()) -> str:
    """Rigid mode in force for this input: "on", "off" or "control".

    "auto" resolves to "on" when TFG is enabled and the input carries an epitope request, or contact
    pairs rigid guidance can apply (a movable mask, or exactly two chains); otherwise "off".
    """
    value = schedule.mode
    if value != "auto":
        return value
    if not tfg_enabled:
        return "off"
    if active(feats):
        return "on"
    pairs = feats.get("user_distance_restraint_index")
    if pairs is None or pairs.numel() == 0:
        return "off"
    movable = feats.get("user_rigid_movable_atom")
    if movable is not None and movable.numel() != 0:
        return "on"
    chains = feats["asym_id"][feats["atom_to_token_idx"]]
    return "on" if torch.unique(chains).numel() == 2 else "off"


def _guard(coords):
    if coords.dtype != torch.float32:
        raise ValueError(
            "epitope guidance requires float32 coordinates, got %s; a cast can move an atom across the gate."
            % coords.dtype
        )
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("non-finite coordinates reached the epitope guidance.")


def groups(coords, feats):
    """(fixed ids, moving ids, epitope slots in the fixed group [n, m], valid [n, m], paratope slots in the moving group, K)."""
    mask = feats.get("user_rigid_movable_atom")
    if mask is None or mask.numel() != coords.shape[-2] or mask.ndim != 1:
        raise ValueError("epitope guidance needs user_rigid_movable_atom over all %d atoms." % coords.shape[-2])
    mask = mask.to(torch.bool).to(coords.device)
    if not mask.any() or mask.all():
        raise ValueError("movable_chains must name some but not all atoms.")
    moving_ids = torch.where(mask)[0]
    fixed_ids = torch.where(~mask)[0]
    n_atom = coords.shape[-2]
    fixed_lookup = torch.full((n_atom,), -1, device=coords.device, dtype=torch.long)
    fixed_lookup[fixed_ids] = torch.arange(len(fixed_ids), device=coords.device)
    moving_lookup = torch.full((n_atom,), -1, device=coords.device, dtype=torch.long)
    moving_lookup[moving_ids] = torch.arange(len(moving_ids), device=coords.device)
    epi = feats["user_epitope_atom_index"].to(coords.device)
    valid = epi >= 0
    epi_local = torch.where(valid, fixed_lookup[epi.clamp_min(0)], torch.full_like(epi, -1))
    if bool((epi_local[valid] < 0).any()):
        raise ValueError("an epitope atom lies in the movable group.")
    para_ids = torch.where(feats["user_epitope_paratope_atom"].to(coords.device))[0]
    para_local = moving_lookup[para_ids]
    if para_local.numel() == 0 or bool((para_local < 0).any()):
        raise ValueError("the paratope must be a non-empty subset of the movable group.")
    k = int(feats["user_epitope_k"].reshape(-1)[0])
    if not 1 <= k <= epi.shape[0]:
        raise ValueError("K=%d is outside 1..%d." % (k, epi.shape[0]))
    return fixed_ids, moving_ids, epi_local, valid, para_local, k


def residue_distances(moving, fixed, epi_local, valid, para_local):
    """Per sample and epitope residue: smallest heavy-atom distance to the paratope and the pair giving it.

    Returns (d [S, n], fixed slot [S, n] in 0..m-1, paratope slot [S, n], epitope coords [S, n, m, 3]).
    """
    s, n, m = moving.shape[0], epi_local.shape[0], epi_local.shape[1]
    fixed_epi = fixed[:, epi_local.clamp_min(0).reshape(-1)].reshape(s, n, m, 3)
    para = moving[:, para_local]
    dist = torch.cdist(
        fixed_epi.reshape(s, n * m, 3), para, compute_mode="donot_use_mm_for_euclid_dist"
    ).reshape(s, n, m, -1)
    dist = dist.masked_fill(~valid[None, :, :, None], float("inf"))
    dmin, arg = dist.reshape(s, n, -1).min(-1)
    width = para.shape[1]
    return dmin, arg // width, arg % width, fixed_epi


class ResidueDistances:
    """residue_distances(...)[0] for many rigid poses of the paratope, without the all-pairs cdist.

    Paratope atoms are grouped into residues with bounding spheres fixed in the group's own frame (`body` [S, A, 3],
    the unmoved paratope per sample). Each pose's frame comes from three anchor atoms, the epitope residues are
    taken into it, and a residue pair whose spheres are further apart than a measured distance of that epitope
    residue cannot hold its nearest pair. Only the remaining pairs are measured, in world coordinates with the same
    norm as cdist, so the minima are bit-identical to residue_distances.
    """

    MARGIN = 1e-2                        # covers fp32 rounding of the poses and frames by orders of magnitude

    def __init__(self, body, para_token, epi_slots, valid):
        _, inverse, counts = torch.unique(para_token, return_inverse=True, return_counts=True)
        order = torch.argsort(inverse, stable=True)
        rank = torch.arange(order.numel()) - (torch.cumsum(counts, 0) - counts)[inverse[order]]
        self.para = torch.full((counts.numel(), int(counts.max())), -1, dtype=torch.long)
        self.para[inverse[order], rank] = order                # residue -> paratope slots, -1 padded
        self.pmask = self.para >= 0
        self.epi, self.valid = epi_slots, valid
        self.centre, self.radius = self._spheres(body[:, self.para.clamp_min(0)], self.pmask)
        # anchors: the atom furthest from the centroid, the one furthest from it, the one furthest off their line
        rows = torch.arange(body.shape[0])
        a0 = (body - body.mean(1, keepdim=True)).norm(dim=-1).argmax(1)
        a1 = (body - body[rows, a0][:, None]).norm(dim=-1).argmax(1)
        u = body[rows, a1] - body[rows, a0]
        w = body - body[rows, a0][:, None]
        a2 = torch.linalg.cross(w, u[:, None].expand_as(w), dim=-1).norm(dim=-1).argmax(1)
        self.anchors = torch.stack([a0, a1, a2], 1)
        self.body_frame = self._frame(body[rows[:, None], self.anchors])

    @staticmethod
    def _spheres(x, mask):
        """x [..., R, w, 3], mask [R, w] -> centre [..., R, 3], radius [..., R]."""
        m = mask[..., None].to(x.dtype)
        centre = (x * m).sum(-2) / m.sum(-2).clamp_min(1)
        radius = torch.where(mask, (x - centre[..., None, :]).norm(dim=-1), 0.0).amax(-1)
        return centre, radius

    @staticmethod
    def _frame(p):
        """Anchor atoms p [P, 3, 3] -> (origin [P, 3], orthonormal axes [P, 3, 3] as rows)."""
        e1 = torch.nn.functional.normalize(p[:, 1] - p[:, 0], dim=-1)
        e2 = p[:, 2] - p[:, 0]
        e2 = torch.nn.functional.normalize(e2 - (e2 * e1).sum(-1, keepdim=True) * e1, dim=-1)
        return p[:, 0], torch.stack([e1, e2, torch.linalg.cross(e1, e2, dim=-1)], 1)

    def __call__(self, para, ref, rows, slots=False):
        """para [P, A, 3] posed paratope atoms of samples rows [P], ref [P, slots, 3] epitope atoms -> d [P, n], and
        with `slots` the nearest pair's epitope slot and paratope slot [P, n] (ties: the first in residue_distances'
        flattened (slot, paratope) order, as its min returns)."""
        P, n = para.shape[0], self.epi.shape[0]
        er = ref[:, self.epi]                                                 # [P, n, m, 3]
        cr, rr = self._spheres(er, self.valid)
        o_w, f_w = self._frame(para[torch.arange(P)[:, None], self.anchors[rows]])
        o_b, f_b = self.body_frame[0][rows], self.body_frame[1][rows]
        rot = torch.einsum("pji,pjk->pik", f_b, f_w)                       # world -> group frame
        body = torch.einsum("pij,pnmj->pnmi", rot, er - o_w[:, None, None]) + o_b[:, None, None]
        bc = (body * self.valid[..., None]).sum(2) / self.valid.sum(1, keepdim=True).clamp_min(1)
        centre, radius = self.centre[rows], self.radius[rows]
        gap = square_length(bc[:, :, None] - centre[:, None]).sqrt() - rr[..., None] - radius[:, None]   # [P, n, Q]

        # a measured distance per epitope residue: to the paratope residue whose sphere is nearest
        q0 = gap.argmin(-1)                                                   # [P, n]
        pq = para[torch.arange(P)[:, None, None], self.para[q0].clamp_min(0)]  # [P, n, w, 3]
        sq = square_length(er[:, :, :, None] - pq[:, :, None])               # [P, n, m, w]
        ok = self.valid[None, :, :, None] & self.pmask[q0][:, :, None]
        upper = sq.masked_fill(~ok, float("inf")).amin((2, 3)).sqrt() + self.MARGIN
        # residue pairs that may hold the nearest pair, then the epitope atoms of those that may
        p, r, q = torch.nonzero(gap <= upper[..., None], as_tuple=True)
        near = square_length(body[p, r] - centre[p, q][:, None]).sqrt() - radius[p, q][:, None] <= upper[p, r][:, None]
        t, a = torch.nonzero(near & self.valid[r], as_tuple=True)
        p, r, q = p[t], r[t], q[t]
        pq = para[p[:, None], self.para[q].clamp_min(0)]                    # [T, w, 3]
        # batched cdist: each element is computed from its own two rows exactly as residue_distances' cdist is
        dist = torch.cdist(er[p, r, a][:, None], pq, compute_mode="donot_use_mm_for_euclid_dist")[:, 0]
        dist = dist.masked_fill(~self.pmask[q], float("inf"))               # [T, w]
        best = dist.amin(1)
        pr = p * n + r
        d = torch.full((P * n,), float("inf"), dtype=para.dtype).scatter_reduce_(0, pr, best, "amin")
        if not slots:
            return d.view(P, n)
        A = para.shape[1]
        big = torch.iinfo(torch.long).max
        first = torch.where(dist == best[:, None], self.para[q], big).amin(1)
        tie = best == d[pr]
        flat = torch.full((P * n,), big).scatter_reduce_(0, pr[tie], a[tie] * A + first[tie], "amin")
        flat = torch.where(torch.isinf(d), 0, flat)
        return d.view(P, n), (flat // A).view(P, n), (flat % A).view(P, n)


def reached_count(d):
    return (d <= GATE).sum(-1)


def contact_terms(d, k):
    """Energy over the K closest epitope residues, their violations and indices."""
    # Stable sort: ties keep the canonical row order.
    ordered, order = torch.sort(d, dim=-1, stable=True)
    vals, idx = ordered[:, :k], order[:, :k]
    violation = torch.relu(vals - TARGET)
    return 0.5 * violation.square().sum(-1), violation, idx


def contact_energy_and_gradient(x, fixed, epi_local, valid, para_local, k, distances=None):
    """Contact energy [S], its gradient on the moving atoms [S, M, 3] and residue distances [S, n].

    Each counted residue pulls its nearest paratope atom along the pair with force relu(d - TARGET);
    residues sharing a paratope atom add up. `distances` (a ResidueDistances over these samples' paratope) gives
    the same residue distances and nearest pairs without the all-pairs cdist.
    """
    samples, n_moving = x.shape[0], x.shape[1]
    if distances is None:
        d, a_slot, p_slot, fixed_epi = residue_distances(x, fixed, epi_local, valid, para_local)
    else:
        fixed_epi = fixed[:, epi_local.clamp_min(0).reshape(-1)]
        d, a_slot, p_slot = distances(x[:, para_local], fixed_epi, torch.arange(samples), slots=True)
        fixed_epi = fixed_epi.view(samples, *epi_local.shape, 3)
    energy, violation, top = contact_terms(d, k)
    rows = torch.arange(samples, device=x.device)[:, None].expand(-1, k)
    fa = fixed_epi[rows, top, a_slot.gather(1, top)]
    mp = para_local[p_slot.gather(1, top)]
    vec = x[rows, mp] - fa
    dist = torch.linalg.vector_norm(vec, dim=-1).clamp_min(1e-6)
    contrib = violation[..., None] * vec / dist[..., None]
    grad = torch.zeros_like(x)
    grad.view(-1, 3).index_add_(0, (rows * n_moving + mp).reshape(-1), contrib.reshape(-1, 3))
    return energy, grad, d


def status(entry_ok, final_ok, moved):
    """Per-sample status: already_satisfied_skipped, delivered, partial or search_failed."""
    out = []
    for e, f, m in zip(entry_ok.tolist(), final_ok.tolist(), moved.tolist()):
        out.append(
            "already_satisfied_skipped" if e else ("delivered" if f else ("partial" if m else "search_failed"))
        )
    return out


def _atom_radii(coords, feats, moving_ids, fixed_ids):
    """vdW radii of the moving and the fixed atoms."""
    table = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    element = feats["ref_element"].argmax(-1)
    if (element == 0).any():
        raise ValueError("epitope guidance expects heavy-atom input; an atom with element index 0 was found.")
    return table[element[moving_ids]], table[element[fixed_ids]]


def _radii(coords, feats, moving_ids, fixed_ids):
    ra, rb = _atom_radii(coords, feats, moving_ids, fixed_ids)
    return ra[:, None] + rb[None, :]


@torch.no_grad()
def refine_epitope(coords, feats, iterations=40, core=None):
    """Rigid refinement of the movable group towards the epitope request. coords [S, N_atom, 3] float32."""
    core = rc.resolve_core(core)
    if core == "check":
        return rc.checked(refine_epitope, coords, feats, iterations=iterations)
    if coords.ndim != 3:
        raise ValueError("epitope guidance requires [sample, atom, xyz].")
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    original_dtype = coords.dtype
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float().clone()
    if core != "off" and bool((reached_count(residue_distances(moving, fixed, epi_local, valid, para_local)[0]) >= k).all()):
        # the descent leaves satisfied samples alone: with every sample satisfied it moves nothing
        logger.info("EPITOPE_GUIDANCE refine: every sample already meets the request K=%d", k)
        return coords.clone()
    clash = rc.clash_core(core, fixed, *_atom_radii(coords, feats, moving_ids, fixed_ids))
    distances = None if core == "off" else ResidueDistances(
        moving[:, para_local], feats["atom_to_token_idx"][moving_ids[para_local]],
        torch.arange(epi_local.numel()).view(epi_local.shape), valid)

    def evaluate(x, gradient=False, terms=None):
        contact_energy, contact_grad, d = contact_energy_and_gradient(
            x, fixed, epi_local, valid, para_local, k, distances)
        clash_energy, severe, depth, clash_grad = terms or clash.terms(x, want_gradient=gradient)
        energy = contact_energy + 0.5 * CLASH_WEIGHT_REFINE * clash_energy
        if not gradient:
            return energy, severe, depth, d
        return energy, severe, depth, d, clash_grad + contact_grad

    def satisfied(d):
        return reached_count(d) >= k

    first_energy, first_severe, _, first_d = evaluate(moving)
    entry_ok = reached_count(first_d) >= k
    entry = moving.clone()
    moving, accepted = rc.rigid_descent(moving, evaluate, satisfied, k, iterations, clash, para_local)
    final_energy, final_severe, _, final_d = evaluate(moving)
    intact = clash.no_new(entry, first_severe, moving, final_severe)
    if not bool(intact.all()):
        raise RuntimeError("Epitope guidance introduced a severe interchain clash.")
    result = coords.clone()
    result[:, moving_ids] = moving.to(original_dtype)
    if logger.isEnabledFor(logging.INFO):
        # The status is read off the returned coordinates, not the internal state.
        final_d, _, _, _ = residue_distances(
            result[:, moving_ids].float(), result[:, fixed_ids].float(), epi_local, valid, para_local
        )
        final_ok = reached_count(final_d) >= k
        logger.info(
            "EPITOPE_GUIDANCE refine K=%d n=%d reached_before=%s reached_after=%s energy %s -> %s status=%s accepted=%s",
            k,
            epi_local.shape[0],
            reached_count(first_d).tolist(),
            reached_count(final_d).tolist(),
            first_energy.tolist(),
            final_energy.tolist(),
            status(entry_ok, final_ok, accepted > 0),
            accepted.tolist(),
        )
    return result


def _helper_axis(v):
    """Per row of v [S, 3]: the x axis, or y when |v_x| > 0.9."""
    x = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    y = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    return torch.where((v[:, 0].abs() > 0.9)[:, None], y, x)


def rotation_aligning(a, b):
    """Rotations [S, 3, 3] taking unit vectors a to b, built in float64 about a x b by atan2(|a x b|, a . b).

    Parallel and antiparallel pairs use an explicit perpendicular axis.
    """
    a64, b64 = a.double(), b.double()
    a64 = a64 / torch.linalg.vector_norm(a64, dim=-1, keepdim=True).clamp_min(1e-12)
    b64 = b64 / torch.linalg.vector_norm(b64, dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.linalg.cross(a64, b64, dim=-1)
    s_ = torch.linalg.vector_norm(v, dim=-1)
    c = (a64 * b64).sum(-1)
    theta = torch.atan2(s_, c)
    helper = _helper_axis(a64)
    perp = torch.linalg.cross(a64, helper, dim=-1)
    perp = perp / torch.linalg.vector_norm(perp, dim=-1, keepdim=True).clamp_min(1e-12)
    degenerate = s_ < 1e-9
    axis = torch.where(degenerate[:, None], perp, v / s_.clamp_min(1e-300)[:, None])
    theta = torch.where(
        degenerate,
        torch.where(c < 0, torch.full_like(theta, math.pi), torch.zeros_like(theta)),
        theta,
    )
    return rotation_about(axis, theta).to(a.dtype)


def rotation_about(axis, angle):
    """Rodrigues rotations [S, 3, 3] about unit axes [S, 3] by angles [S] or a scalar, computed in float64."""
    dtype_out = axis.dtype
    axis = axis.double()
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    s = axis.shape[0]
    angle = torch.as_tensor(angle, device=axis.device, dtype=torch.float64).expand(s)
    skew = torch.zeros(s, 3, 3, device=axis.device, dtype=torch.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = torch.eye(3, device=axis.device, dtype=torch.float64).expand(s, 3, 3)
    rot = (
        eye
        + torch.sin(angle)[:, None, None] * skew
        + (1 - torch.cos(angle))[:, None, None] * torch.bmm(skew, skew)
    )
    return rot.to(dtype_out)


def approach_axes(fixed, fixed_chain, epi_local, valid, centroid):
    """Unit approach directions (list of [S, 3]) pointing away from the antigen, and a note on what was used.

    Epitope on one chain: the outward local normal (smallest principal axis of that chain's atoms
    within NORMAL_RADIUS of the epitope centroid) plus TILT_AZIMUTHS tilts of TILT_DEGREES; a sample
    with a degenerate normal falls back to chain-centroid-to-epitope (or +z). FIBONACCI_AXES sphere
    axes are always appended. A cross-chain epitope gets only the Fibonacci axes.
    """
    s = fixed.shape[0]
    device, dtype = fixed.device, fixed.dtype
    axes = []
    slot_atoms = epi_local[valid]
    chains = torch.unique(fixed_chain[slot_atoms])
    if chains.numel() == 1:
        pool = fixed[:, fixed_chain == chains[0]]
        near = torch.linalg.vector_norm(pool - centroid[:, None], dim=-1) <= NORMAL_RADIUS
        ok = near.sum(-1) >= 10
        w = near.to(dtype)
        cnt = w.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (pool * w[..., None]).sum(1) / cnt
        diff = (pool - mean[:, None]) * w[..., None]
        cov = torch.bmm(diff.transpose(1, 2), diff) / cnt[..., None]
        evals, evecs = torch.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        outward = centroid - pool.mean(1)
        sign = torch.sign((normal * outward).sum(-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        normal = normal * sign[:, None]
        finite = torch.isfinite(normal).all(-1)
        ok = ok & finite & (evals[:, 1] > 1e-6)
        fallback = outward / torch.linalg.vector_norm(outward, dim=-1, keepdim=True).clamp_min(1e-12)
        bad_fb = ~torch.isfinite(fallback).all(-1) | (torch.linalg.vector_norm(outward, dim=-1) < 1e-6)
        fallback = torch.where(
            bad_fb[:, None],
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(fallback),
            fallback,
        )
        base = torch.where(
            ok[:, None],
            normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-9),
            fallback,
        )
        helper = _helper_axis(base)
        e1 = torch.linalg.cross(base, helper, dim=-1)
        e1 = e1 / torch.linalg.vector_norm(e1, dim=-1, keepdim=True).clamp_min(1e-9)
        e2 = torch.linalg.cross(base, e1, dim=-1)
        axes.append(base)
        theta = math.radians(TILT_DEGREES)
        for i in range(TILT_AZIMUTHS):
            phi = 2.0 * math.pi * i / TILT_AZIMUTHS
            axes.append(math.cos(theta) * base + math.sin(theta) * (math.cos(phi) * e1 + math.sin(phi) * e2))
        note = (
            "normal+tilts+fibonacci"
            if bool(ok.all())
            else "normal+tilts+fibonacci(fallback_normal_for_samples=%s)" % torch.where(~ok)[0].tolist()
        )
    else:
        note = "fibonacci_only(cross_chain_epitope)"
    for i in range(FIBONACCI_AXES):
        z = 1 - 2 * (i + 0.5) / FIBONACCI_AXES
        phi = i * math.pi * (3 - math.sqrt(5))
        r = math.sqrt(1 - z * z)
        axes.append(
            torch.tensor([r * math.cos(phi), r * math.sin(phi), z], device=device, dtype=dtype).expand(s, 3)
        )
    return axes, note


def candidate_energy(x, fixed, rsum, epi_local, valid, para_local, k):
    """Coarse-search objective: (contact energy + CLASH_WEIGHT_SEARCH * clash energy [S], has severe clash [S])."""
    d, _, _, _ = residue_distances(x, fixed, epi_local, valid, para_local)
    contact_energy, _, _ = contact_terms(d, k)
    clash_energy, summary, _, _ = rc.clash_terms(x, fixed, rsum, OVERLAP_FRACTION)
    return contact_energy + CLASH_WEIGHT_SEARCH * clash_energy, rc.severe_count(summary) > 0


def interface_patch(moving, fixed, fraction=PATCH_FRACTION, minimum=PATCH_MINIMUM):
    """Per sample, the n moving atoms closest to any fixed atom, n = min(M, max(minimum, ceil(fraction * M))). [S, n, 3]."""
    s_, m_ = moving.shape[0], moving.shape[1]
    n = min(m_, max(minimum, int(math.ceil(fraction * m_))))
    out = []
    for i in range(s_):
        d = torch.cdist(moving[i], fixed[i], compute_mode="donot_use_mm_for_euclid_dist").min(-1).values
        idx = torch.topk(-d, n, dim=-1).indices
        out.append(moving[i, idx])
    return torch.stack(out)


@torch.no_grad()
def search_epitope(coords, feats, core=None):
    """Coarse pose search from the epitope geometry alone.

    Each candidate turns the binding face (group centre to paratope centre) towards the epitope
    along an approach axis, spins it SPINS times about the axis and places the paratope tip at each
    of STANDOFFS from the epitope centroid. Candidates with a severe clash are dropped; the best
    remaining one replaces the pose only if its energy is lower. Satisfied samples are unchanged.
    With the pair-list core (rigid module docstring) the candidates are scored in batches of axes.
    """
    core = rc.resolve_core(core)
    if core == "check":
        return rc.checked(search_epitope, coords, feats)
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float()
    samples = coords.shape[0]
    ra, rb = _atom_radii(coords, feats, moving_ids, fixed_ids)
    asym = feats["asym_id"][feats["atom_to_token_idx"]].to(coords.device)
    fixed_chain = asym[fixed_ids]
    d0, _, _, fixed_epi = residue_distances(moving, fixed, epi_local, valid, para_local)
    entry_ok = reached_count(d0) >= k
    if bool(entry_ok.all()):
        logger.info(
            "EPITOPE_GUIDANCE coarse skipped: every sample already meets the request K=%d n=%d", k, epi_local.shape[0]
        )
        return coords
    resid_centres = (fixed_epi * valid[None, :, :, None]).sum(2) / valid.sum(1)[None, :, None].clamp_min(1)
    centroid = resid_centres.mean(1)
    # Paratope = every movable atom: the binding face comes from the pose (interface patch).
    auto_face = int(para_local.numel()) == int(moving.shape[1])
    para_pos = interface_patch(moving, fixed) if auto_face else moving[:, para_local]
    para_centre = para_pos.mean(1)
    group_centre = moving.mean(1)
    face = para_centre - group_centre
    face = face / torch.linalg.vector_norm(face, dim=-1, keepdim=True).clamp_min(1e-6)
    reach = (para_pos * face[:, None]).sum(-1)
    n_tip = max(1, int(math.ceil(TIP_FRACTION * para_pos.shape[1])))
    tip_idx = torch.topk(reach, n_tip, dim=-1).indices
    tip = torch.gather(para_pos, 1, tip_idx[..., None].expand(-1, -1, 3)).mean(1)
    axes, note = approach_axes(fixed, fixed_chain, epi_local, valid, centroid)

    best = moving.clone()
    tested = 0
    feasible = torch.zeros(samples, device=coords.device, dtype=torch.long)
    improving = torch.zeros(samples, device=coords.device, dtype=torch.long)
    accepted = torch.zeros(samples, device=coords.device, dtype=torch.long)
    if core != "off":
        clash = SparseClash(fixed, ra, rb)
        # The contact term only reads the epitope atoms: hand it those, so a batch of poses can share them.
        n_epi, m_epi = epi_local.shape
        epi_atoms = fixed[:, epi_local.clamp_min(0).reshape(-1)]
        epi_slots = torch.arange(n_epi * m_epi, device=coords.device).reshape(n_epi, m_epi)
        distances = ResidueDistances(moving[:, para_local], feats["atom_to_token_idx"][moving_ids[para_local]],
                                     epi_slots, valid)

        def score_many(x, rows):
            """x [n, B, M, 3] poses of samples rows [n] -> (energy, severe) [n, B]."""
            n, b = x.shape[:2]
            flat = x.reshape(n * b, *x.shape[2:])
            ref = epi_atoms[rows].repeat_interleave(b, 0)
            contact = contact_terms(distances(flat[:, para_local], ref, rows.repeat_interleave(b)), k)[0].view(n, b)
            clash_energy, bad = clash.score(x, rows)
            return contact + CLASH_WEIGHT_SEARCH * clash_energy, bad

        cur_energy, cur_bad = score_many(moving[:, None], torch.arange(samples, device=coords.device))
        cur_energy, cur_bad = cur_energy[:, 0], cur_bad[:, 0]
        best_energy = torch.where(cur_bad, torch.full_like(cur_energy, float("inf")), cur_energy)
        todo = torch.nonzero(~entry_ok).squeeze(1)

        # poses are placed for the unsatisfied samples only (per-sample ops: same values)
        t_face, t_moving, t_tip, t_centroid = face[todo], moving[todo], tip[todo], centroid[todo]

        spin_angle = torch.tensor([2.0 * math.pi * i / SPINS for i in range(SPINS)], dtype=torch.float64)
        standoff = torch.tensor(STANDOFFS)

        def placements(chunk):
            """Every placement of a chunk of axes -> [n, axis * spin * standoff, M, 3], in the loop order (axis,
            spin, standoff). Rows are independent in every op, so batching them gives the per-axis values."""
            n, A = todo.numel(), len(chunk)
            u = torch.stack([v[todo] for v in chunk])                                   # [A, n, 3]
            align = rotation_aligning(t_face.repeat(A, 1), -u.reshape(-1, 3))
            aligned = torch.bmm((t_moving - t_tip[:, None]).repeat(A, 1, 1), align.transpose(1, 2))
            uu = u[:, None].expand(A, SPINS, n, 3).reshape(-1, 3)
            spin = rotation_about(uu, spin_angle[None, :, None].expand(A, SPINS, n).reshape(-1))
            rotated = torch.bmm(aligned.view(A, 1, n, -1, 3).expand(A, SPINS, n, -1, 3).reshape(A * SPINS * n, -1, 3),
                                spin.transpose(1, 2)).view(A, SPINS, n, -1, 3)
            shift = t_centroid + standoff[:, None, None] * u[:, None]                    # [A, R, n, 3]
            x = rotated[:, :, None] + shift[:, None, :, :, None]                         # [A, spin, R, n, M, 3]
            return x.permute(3, 0, 1, 2, 4, 5).reshape(n, A * SPINS * len(STANDOFFS), -1, 3)

        for start in range(0, len(axes), SEARCH_CHUNK_AXES):
            chunk = axes[start:start + SEARCH_CHUNK_AXES]
            tested += len(chunk) * SPINS * len(STANDOFFS)
            if todo.numel() == 0:
                continue
            x = placements(chunk)
            energy, bad = score_many(x, todo)
            index, taken = rc._take_first_best(energy, bad, todo, best_energy, feasible, improving, accepted)
            pick = torch.nonzero(taken).squeeze(1)
            best[todo[pick]] = x[pick, index[pick]]
    else:
        rsum = ra[:, None] + rb[None, :]
        cur_energy, cur_bad = candidate_energy(moving, fixed, rsum, epi_local, valid, para_local, k)
        best_energy = torch.where(cur_bad, torch.full_like(cur_energy, float("inf")), cur_energy)

    def score(x):
        return candidate_energy(x, fixed, rsum, epi_local, valid, para_local, k)

    for u in axes if core == "off" else []:
        align = rotation_aligning(face, -u)
        aligned = torch.bmm(moving - tip[:, None], align.transpose(1, 2))
        for i in range(SPINS):
            spin = rotation_about(u, 2.0 * math.pi * i / SPINS)
            rotated = torch.bmm(aligned, spin.transpose(1, 2))
            for radius in STANDOFFS:
                proposed = rotated + (centroid + radius * u)[:, None]
                energy, bad = score(proposed)
                tested += 1
                feasible += (~bad).long()
                improving += (energy < best_energy).long()
                take = (~bad) & (~entry_ok) & (energy < best_energy)
                best = torch.where(take[:, None, None], proposed, best)
                best_energy = torch.where(take, energy, best_energy)
                accepted += take.long()
    result = coords.clone()
    result[:, moving_ids] = best.to(coords.dtype)
    if logger.isEnabledFor(logging.INFO):
        d1, _, _, _ = residue_distances(
            result[:, moving_ids].float(), result[:, fixed_ids].float(), epi_local, valid, para_local
        )
        logger.info(
            "EPITOPE_GUIDANCE coarse axes=%s candidates=%d feasible=%s improving=%s accepted=%s K=%d n=%d "
            "reached_before=%s reached_after=%s energy_before=%s energy_after=%s skipped_already_satisfied=%s",
            note,
            tested,
            feasible.tolist(),
            improving.tolist(),
            accepted.tolist(),
            k,
            epi_local.shape[0],
            reached_count(d0).tolist(),
            reached_count(d1).tolist(),
            cur_energy.tolist(),
            best_energy.tolist(),
            entry_ok.tolist(),
        )
    return result


@torch.no_grad()
def guide_x0_contact(x0, feats, step_i, core=None):
    """Contact request on x0 [S, N_atom, 3]: refine, coarse search for samples still off, refine again."""
    orig_dtype = x0.dtype
    y = x0.float()
    if y.ndim != 3:
        raise ValueError("contact x0 guidance expects [sample, atom, xyz]")
    y = rc.refine_rigid_contact(y, feats, iterations=40, core=core)
    y = rc.search_rigid_contact(y, feats, core=core)
    y = rc.refine_rigid_contact(y, feats, iterations=40, core=core)
    logger.info("RIGID_CONTACT x0 hook step=%d", step_i)
    return y.to(orig_dtype)


@torch.no_grad()
def guide_x0(x0, feats, step_i, schedule: RigidSchedule = RigidSchedule(), tfg_enabled: bool = True):
    """Early pass: rigidly move the movable group of the denoiser's x0 [S, N_atom, 3] towards the request.

    Returns x0 unchanged when the pass is not scheduled at ``step_i`` or rigid mode is not "on".
    """
    mode = rigid_mode(feats, tfg_enabled, schedule)
    if (
        mode == "on"
        and not active(feats)
        and rc.x0_step_active(step_i, schedule)
        and feats.get("user_distance_restraint_index") is not None
        and feats["user_distance_restraint_index"].numel() > 0
    ):
        return guide_x0_contact(x0, feats, step_i, schedule.core)
    if mode != "on" or not (active(feats) and rc.x0_step_active(step_i, schedule)):
        return x0
    orig_dtype = x0.dtype
    y = x0.float()
    if len(y.shape[:-3]) != 0:
        raise ValueError("guide_x0 expects coordinates without extra batch dimensions")
    # Smallest move first; only samples still missing the request get the coarse re-placement.
    y = refine_epitope(y, feats, iterations=40, core=schedule.core)
    y = search_epitope(y, feats, core=schedule.core)
    y = refine_epitope(y, feats, iterations=40, core=schedule.core)
    logger.info("EPITOPE_GUIDANCE x0 hook step=%d", step_i)
    return y.to(orig_dtype)
