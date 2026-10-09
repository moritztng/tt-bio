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


def reached_count(d):
    return (d <= GATE).sum(-1)


def contact_terms(d, k):
    """Energy over the K closest epitope residues, their violations and indices."""
    # Stable sort: ties keep the canonical row order.
    ordered, order = torch.sort(d, dim=-1, stable=True)
    vals, idx = ordered[:, :k], order[:, :k]
    violation = torch.relu(vals - TARGET)
    return 0.5 * violation.square().sum(-1), violation, idx


def contact_energy_and_gradient(x, fixed, epi_local, valid, para_local, k):
    """Contact energy [S], its gradient on the moving atoms [S, M, 3] and residue distances [S, n].

    Each counted residue pulls its nearest paratope atom along the pair with force relu(d - TARGET);
    residues sharing a paratope atom add up.
    """
    samples, n_moving = x.shape[0], x.shape[1]
    d, a_slot, p_slot, fixed_epi = residue_distances(x, fixed, epi_local, valid, para_local)
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


def _radii(coords, feats, moving_ids, fixed_ids):
    table = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    element = feats["ref_element"].argmax(-1)
    if (element == 0).any():
        raise ValueError("epitope guidance expects heavy-atom input; an atom with element index 0 was found.")
    return table[element[moving_ids]][:, None] + table[element[fixed_ids]][None, :]


@torch.no_grad()
def refine_epitope(coords, feats, iterations=40):
    """Rigid refinement of the movable group towards the epitope request. coords [S, N_atom, 3] float32."""
    if coords.ndim != 3:
        raise ValueError("epitope guidance requires [sample, atom, xyz].")
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    original_dtype = coords.dtype
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float().clone()
    rsum = _radii(coords, feats, moving_ids, fixed_ids)

    def evaluate(x, gradient=False):
        contact_energy, contact_grad, d = contact_energy_and_gradient(x, fixed, epi_local, valid, para_local, k)
        clash_energy, severe, depth, clash_grad = rc.clash_terms(
            x, fixed, rsum, OVERLAP_FRACTION, want_gradient=gradient
        )
        energy = contact_energy + 0.5 * CLASH_WEIGHT_REFINE * clash_energy
        if not gradient:
            return energy, severe, depth, d
        return energy, severe, depth, d, clash_grad + contact_grad

    def satisfied(d):
        return reached_count(d) >= k

    first_energy, first_severe, _, first_d = evaluate(moving)
    entry_ok = reached_count(first_d) >= k
    entry = moving.clone()
    moving, accepted = rc.rigid_descent(moving, evaluate, satisfied, k, iterations, fixed, rsum)
    final_energy, final_severe, _, final_d = evaluate(moving)
    intact = (
        rc.no_new_severe(first_severe, final_severe)
        if first_severe.dtype == torch.bool
        else rc.severe_transition(entry, moving, fixed, rsum)
    )
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
def search_epitope(coords, feats):
    """Coarse pose search from the epitope geometry alone.

    Each candidate turns the binding face (group centre to paratope centre) towards the epitope
    along an approach axis, spins it SPINS times about the axis and places the paratope tip at each
    of STANDOFFS from the epitope centroid. Candidates with a severe clash are dropped; the best
    remaining one replaces the pose only if its energy is lower. Satisfied samples are unchanged.
    """
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float()
    samples = coords.shape[0]
    rsum = _radii(coords, feats, moving_ids, fixed_ids)
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

    def score(x):
        return candidate_energy(x, fixed, rsum, epi_local, valid, para_local, k)

    best = moving.clone()
    cur_energy, cur_bad = score(moving)
    best_energy = torch.where(cur_bad, torch.full_like(cur_energy, float("inf")), cur_energy)
    tested = 0
    feasible = torch.zeros(samples, device=coords.device, dtype=torch.long)
    improving = torch.zeros(samples, device=coords.device, dtype=torch.long)
    accepted = torch.zeros(samples, device=coords.device, dtype=torch.long)
    for u in axes:
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
def guide_x0_contact(x0, feats, step_i):
    """Contact request on x0 [S, N_atom, 3]: refine, coarse search for samples still off, refine again."""
    orig_dtype = x0.dtype
    y = x0.float()
    if y.ndim != 3:
        raise ValueError("contact x0 guidance expects [sample, atom, xyz]")
    y = rc.refine_rigid_contact(y, feats, iterations=40)
    y = rc.search_rigid_contact(y, feats)
    y = rc.refine_rigid_contact(y, feats, iterations=40)
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
        return guide_x0_contact(x0, feats, step_i)
    if mode != "on" or not (active(feats) and rc.x0_step_active(step_i, schedule)):
        return x0
    orig_dtype = x0.dtype
    y = x0.float()
    if len(y.shape[:-3]) != 0:
        raise ValueError("guide_x0 expects coordinates without extra batch dimensions")
    # Smallest move first; only samples still missing the request get the coarse re-placement.
    y = refine_epitope(y, feats, iterations=40)
    y = search_epitope(y, feats)
    y = refine_epitope(y, feats, iterations=40)
    logger.info("EPITOPE_GUIDANCE x0 hook step=%d", step_i)
    return y.to(orig_dtype)
