"""ABodyBuilder3's geometry losses in torch: FAPE, supervised chi and pLDDT, on the host.

These run on the host on purpose. They consume the device model's per-block frames and angles
through the geometry tail (`torsion_angles_to_frames`, then atom14), which is a few thousand
gather-heavy operations on `[B, N, 8]` frames and `[B, N, 14, 3]` atoms -- no FLOPs worth a kernel,
and every rigid-group table lookup is an index into a 21-row table. What the boundary buys is that
these functions are comparable against upstream's own `openfold/utils/loss.py` op for op in
float64, which `scripts/abb3_port/loss_gate.py` does, and that the gradient reaching the device is
a vector-Jacobian seed per output rather than a scalar (`abodybuilder3_grad.backward`).

Transcribed from `Exscientia/ABodyBuilder3` `src/abodybuilder3/openfold/utils/loss.py`, taking the
branches `ABB3Loss` actually takes: `compute_fape` (:85), `backbone_loss` (:171),
`sidechain_loss` (:222), `fape_loss` (:313), `supervised_chi_loss` (:337), `lddt` (:429),
`lddt_loss` (:501) and `final_output_backbone_loss` (:1673). Weights from `params.yaml`: FAPE 1.0
with backbone 0.5 and sidechain 1.0, supervised chi 0.5 with `angle_norm_weight` 0.02, pLDDT 0.01.

**The three violation terms are not here, and that is upstream's own gating rather than a gap.**
`ABB3Loss.forward` adds `violation_loss_bondlength`, `violation_loss_bondangle` and
`violation_loss_clash` only when `finetune` is set, and the released `base-loss` recipe is 1,000
epochs without them followed by 500 with. Stage 1 is 193,512 of the steps.

**One upstream defect this file reproduces on purpose.** `supervised_chi_loss` builds its
`chi_pi_periodic` flags with `torch.einsum("...ij,jk->ik", one_hot, table)`, and that output spec
has no ellipsis, so the leading dims are SUMMED rather than kept. At batch 1 it is a 0/1 flag per
residue, as intended. At batch B it is a count in `[0, B]`, so `shifted_mask = 1 - 2 * count` runs
over `{1, -1, -3, ... }` instead of `{+1, -1}`: for a count of 2 or more the shifted error is
enormous and `minimum` never selects it, which disables the pi-periodicity handling for that
position. Upstream computes this loss per micro-batch of 8 (`stages/train.py:64-73`), so the
released checkpoints were trained with it mostly disabled. `batch_collapsed_periodicity=True` is
the default because the reproduction target is their number, not their intent; passing False gives
the per-sample flag the docstring describes. It is a decision for the campaign, not for this file,
and the gate scores both.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..af2_reference import CHI_PI_PERIODIC, quat_to_rot

#: Upstream's `residue_constants.restype_num + 1`: 20 amino acids plus unknown.
RESTYPE_BINS = 21
#: `residue_constants.atom_order["CA"]` in the atom14 layout, which is the same as atom37 for the
#: first three: N, CA, C.
CA_INDEX = 1


def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim, eps: float = 1e-4
                ) -> torch.Tensor:
    """`tensor_utils.masked_mean`, including its epsilon in the denominator and not the numerator."""
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)


def rigid_from_tensor_7(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`Rigid.from_tensor_7` at `normalize_quats=False`, then to a rotation matrix.

    Upstream's `backbone_loss` converts the trajectory's quaternions to matrices and keeps them
    (`:183-186`), with a comment saying why it does not round-trip back to a quaternion. The device
    port emits quaternions that are already unit, so not normalising here matches upstream exactly
    and also does not hide a drifting quaternion if one ever appears.
    """
    return quat_to_rot(t[..., :4]), t[..., 4:]


def rigid_from_tensor_4x4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`Rigid.from_tensor_4x4`: the rotation and translation blocks of a homogeneous transform."""
    return t[..., :3, :3], t[..., :3, 3]


def invert_apply(rot: torch.Tensor, trans: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """`Rigid.invert().apply(points)`: `R^T (p - t)`, with the frame axis broadcast over points."""
    centred = points - trans.unsqueeze(-2)
    return torch.einsum("...ji,...nj->...ni", rot, centred)


def compute_fape(pred_rot: torch.Tensor, pred_trans: torch.Tensor, target_rot: torch.Tensor,
                 target_trans: torch.Tensor, frames_mask: torch.Tensor,
                 pred_positions: torch.Tensor, target_positions: torch.Tensor,
                 positions_mask: torch.Tensor, length_scale: float,
                 l1_clamp_distance: Optional[float] = None,
                 l1_clamp_distance_large: Optional[float] = None,
                 cdr_mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """Alg. 28, in upstream's exact averaging order.

    The two-stage normalisation at the end is upstream's fp16-friendly form and not the textbook
    one: it divides by the frame count, sums, then divides by the position count, which duplicates
    `eps` relative to a single division. Kept, because the difference is a real if tiny one and the
    point of this file is to be comparable.
    """
    # Upstream writes this as `pred_frames.invert()[..., None].apply(positions[..., None, :, :])`:
    # a frame axis of 1 appended to the frames and a frame axis inserted before the points, so the
    # two broadcast into [*, N_frames, N_pts, 3]. Here the frame axis is already the last batch
    # dim of the rotation, so only the POSITIONS need the insert.
    local_pred = invert_apply(pred_rot, pred_trans, pred_positions.unsqueeze(-3))
    local_target = invert_apply(target_rot, target_trans, target_positions.unsqueeze(-3))
    error_dist = torch.sqrt(torch.sum((local_pred - local_target) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        if cdr_mask is not None and l1_clamp_distance_large is not None:
            # ABodyBuilder3's own change: a frame and an atom in DIFFERENT regions are clamped at
            # the larger distance, same-region pairs at the smaller one.
            error_dist = torch.where(cdr_mask,
                                     error_dist.clamp(min=0, max=l1_clamp_distance_large),
                                     error_dist.clamp(min=0, max=l1_clamp_distance))
        else:
            error_dist = error_dist.clamp(min=0, max=l1_clamp_distance)

    normed = error_dist / length_scale
    normed = normed * frames_mask[..., None]
    normed = normed * positions_mask[..., None, :]
    normed = torch.sum(normed, dim=-1)
    normed = normed / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed = torch.sum(normed, dim=-1)
    return normed / (eps + torch.sum(positions_mask, dim=-1))


def backbone_fape(traj: torch.Tensor, backbone_rigid_tensor: torch.Tensor,
                  backbone_rigid_mask: torch.Tensor, use_clamped_fape=None,
                  clamp_distance: float = 10.0, loss_unit_distance: float = 10.0,
                  eps: float = 1e-4) -> torch.Tensor:
    """FAPE on the backbone frames of EVERY block, averaged. `traj` is `[blocks, B, N, 7]`.

    The ground truth is unsqueezed to a leading block axis rather than the loss being looped over
    blocks, which is what makes the mean an unweighted average over all 8 -- Alg. 20's auxiliary
    loss on intermediate blocks, not just the final one.
    """
    pred_rot, pred_trans = rigid_from_tensor_7(traj)
    gt_rot, gt_trans = rigid_from_tensor_4x4(backbone_rigid_tensor)
    clamp = clamp_distance if use_clamped_fape else None
    loss = compute_fape(pred_rot, pred_trans, gt_rot.unsqueeze(0), gt_trans.unsqueeze(0),
                        backbone_rigid_mask.unsqueeze(0), pred_trans, gt_trans.unsqueeze(0),
                        backbone_rigid_mask.unsqueeze(0), length_scale=loss_unit_distance,
                        l1_clamp_distance=clamp, eps=eps)
    return torch.mean(loss)


def sidechain_fape(sidechain_frames: torch.Tensor, sidechain_atom_pos: torch.Tensor,
                   rigidgroups_gt_frames: torch.Tensor, rigidgroups_alt_gt_frames: torch.Tensor,
                   rigidgroups_gt_exists: torch.Tensor,
                   renamed_atom14_gt_positions: torch.Tensor,
                   renamed_atom14_gt_exists: torch.Tensor, alt_naming_is_better: torch.Tensor,
                   cdr_mask: torch.Tensor, clamp_distance: float = 10.0,
                   intercdr_distance: float = 30.0, length_scale: float = 10.0,
                   eps: float = 1e-4) -> torch.Tensor:
    """FAPE on the 8 rigid-group frames of the LAST block only, against the renamed ground truth.

    Last block only, which is upstream's `[-1]` and not an oversight: the sidechain frames of an
    intermediate block are built from that block's angles, and Alg. 20 supervises intermediates on
    the backbone alone.
    """
    flat = sidechain_inputs(sidechain_frames, sidechain_atom_pos, rigidgroups_gt_frames,
                            rigidgroups_alt_gt_frames, rigidgroups_gt_exists,
                            renamed_atom14_gt_positions, renamed_atom14_gt_exists,
                            alt_naming_is_better, cdr_mask)
    pred_rot, pred_trans = rigid_from_tensor_4x4(flat["pred_frames"])
    gt_rot, gt_trans = rigid_from_tensor_4x4(flat["gt_frames"])
    pair_mask = flat["frame_region"].unsqueeze(-1) != flat["atom_region"].unsqueeze(-2)
    return compute_fape(pred_rot, pred_trans, gt_rot, gt_trans, flat["frames_mask"],
                        flat["pred_positions"], flat["gt_positions"], flat["positions_mask"],
                        length_scale=length_scale, l1_clamp_distance=clamp_distance,
                        l1_clamp_distance_large=intercdr_distance, cdr_mask=pair_mask, eps=eps)


def sidechain_inputs(sidechain_frames, sidechain_atom_pos, rigidgroups_gt_frames,
                     rigidgroups_alt_gt_frames, rigidgroups_gt_exists,
                     renamed_atom14_gt_positions, renamed_atom14_gt_exists,
                     alt_naming_is_better, cdr_mask) -> dict:
    """The sidechain FAPE's inputs, flattened over the residue and group axes.

    Split out so the host and device implementations share one flattening rather than two: the
    renaming blend, the `[B, N, 8]` frames becoming `[B, N*8]` and the `[B, N, 14]` atoms becoming
    `[B, N*14]`, and the per-residue region labels expanding to match. A second copy of this is
    exactly how a device port ends up scoring a different quantity than the host it was verified
    against.
    """
    renamed_gt = ((1.0 - alt_naming_is_better[..., None, None, None]) * rigidgroups_gt_frames
                  + alt_naming_is_better[..., None, None, None] * rigidgroups_alt_gt_frames)
    lead = sidechain_frames.shape[1:-4]
    return {
        "pred_frames": sidechain_frames[-1].reshape(*lead, -1, 4, 4),
        "gt_frames": renamed_gt.reshape(*lead, -1, 4, 4),
        "frames_mask": rigidgroups_gt_exists.reshape(*lead, -1),
        "pred_positions": sidechain_atom_pos[-1].reshape(*lead, -1, 3),
        "gt_positions": renamed_atom14_gt_positions.reshape(*lead, -1, 3),
        "positions_mask": renamed_atom14_gt_exists.reshape(*lead, -1),
        "frame_region": cdr_mask.unsqueeze(-1).expand(*cdr_mask.shape, 8).reshape(*lead, -1),
        "atom_region": cdr_mask.unsqueeze(-1).expand(*cdr_mask.shape, 14).reshape(*lead, -1),
    }


def compute_renamed_ground_truth_dense(batch: dict, atom14_pred_positions: torch.Tensor,
                                       eps: float = 1e-10) -> dict:
    """Alg. 26, renameSymmetricGroundTruthAtoms: pick the better of two ground-truth namings.

    A handful of residue types have a symmetric pair of side-chain atoms whose labels are
    arbitrary (ASP's two carboxyl oxygens, and so on). Scoring against the wrong labelling pulls
    the prediction toward a mirror of itself, so every loss uses whichever of the two the
    prediction is already closer to. `atom14_alt_gt_positions` and `atom14_atom_is_ambiguous` are
    in their dataset, so this is a selection and not a search.

    **This is upstream's dense form and it is expensive: O(N^2 x 14^2).** It builds
    `[B, N, N, 14, 14]` distance tensors -- three of them, plus two lddt tensors and a mask -- which
    at micro-batch 8 and 256 tokens is 102M elements each, 410 MB per tensor in fp32. On a GPU that
    is nothing; in host torch it is the single most expensive part of a step. The mask is zero
    except where `atom14_atom_is_ambiguous` is 1, which is two atoms on a few residue types, so the
    computation restricts by about two orders of magnitude -- but the restriction has to be verified
    against this form, so this form exists first and the step measurement prices it.
    """
    def pair_dists(x):
        return torch.sqrt(eps + torch.sum(
            (x[..., None, :, None, :] - x[..., None, :, None, :, :]) ** 2, dim=-1))

    # `no_grad` around the whole thing, and it is exact rather than an approximation: every output
    # of this function is either a ground-truth tensor or derived from one through
    # `alt_per_res < per_res`, a comparison, so nothing here has a gradient path to the prediction.
    # Upstream builds the graph anyway and pays nothing for it on a GPU. In host torch it was
    # measured at 25.12 s of a 69 s step, plus its share of the 13.4 s host backward, for
    # intermediates whose gradient is discarded.
    with torch.no_grad():
        pred = pair_dists(atom14_pred_positions)
        gt = pair_dists(batch["atom14_gt_positions"])
        alt = pair_dists(batch["atom14_alt_gt_positions"])
        lddt = torch.sqrt(eps + (pred - gt) ** 2)
        alt_lddt = torch.sqrt(eps + (pred - alt) ** 2)

        exists = batch["atom14_gt_exists"]
        ambiguous = batch["atom14_atom_is_ambiguous"]
        mask = (exists[..., None, :, None] * ambiguous[..., None, :, None]
                * exists[..., None, :, None, :] * (1.0 - ambiguous[..., None, :, None, :]))
        per_res = torch.sum(mask * lddt, dim=(-1, -2, -3))
        alt_per_res = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))
        better = (alt_per_res < per_res).to(atom14_pred_positions.dtype)
    return {
        "alt_naming_is_better": better,
        "renamed_atom14_gt_positions": ((1.0 - better[..., None, None])
                                        * batch["atom14_gt_positions"]
                                        + better[..., None, None]
                                        * batch["atom14_alt_gt_positions"]),
        "renamed_atom14_gt_exists": ((1.0 - better[..., None]) * exists
                                     + better[..., None] * batch["atom14_alt_gt_exists"]),
    }


def compute_renamed_ground_truth(batch: dict, atom14_pred_positions: torch.Tensor,
                                 eps: float = 1e-10) -> dict:
    """Alg. 26, restricted to the atom pairs that can contribute. Exactly the dense result.

    The dense form is 71 % of the loss stage and 36 % of a complete step -- 25.26 s of 67.1 s at
    batch 64 and 256 tokens -- because it builds `[B, N, N, 14, 14]` tensors, about 1.2 GB of
    intermediates per micro-batch, and host torch is memory-bound on them. `no_grad` around it
    changes nothing (measured: 25.12 -> 25.26 s), because the cost is the traffic and not the graph.

    What makes it restrictable is the mask, not an approximation. A term survives only where the
    FIRST atom is ambiguous and exists (`exists * ambiguous`) and the second exists and is NOT
    ambiguous, and ambiguity is a property of a handful of residue types -- two carboxyl oxygens,
    two ring carbons. So the first index runs over the ambiguous atoms that are actually present,
    typically a few hundred of the `N * 14` slots, rather than all of them. Everything else is the
    same arithmetic in the same order per term, including the epsilon inside each distance.

    `compute_renamed_ground_truth_dense` is kept and `scripts/abb3_port/loss_gate.py` scores this
    against it, because a restriction that quietly drops a contributing pair is exactly the kind of
    optimisation that looks like a speedup and is a silent loss change.
    """
    exists = batch["atom14_gt_exists"]
    ambiguous = batch["atom14_atom_is_ambiguous"]
    gt_pos = batch["atom14_gt_positions"]
    alt_pos = batch["atom14_alt_gt_positions"]
    dtype = atom14_pred_positions.dtype

    with torch.no_grad():
        first = (exists * ambiguous) > 0                      # [B, N, 14] the ambiguous side
        second = exists * (1.0 - ambiguous)                   # [B, N, 14] the reference side
        idx = first.nonzero(as_tuple=False)                   # [K, 3] -> (b, i, a)
        per_res = torch.zeros(exists.shape[:-1], dtype=dtype, device=exists.device)
        alt_per_res = torch.zeros_like(per_res)
        if len(idx):
            b, i, a = idx[:, 0], idx[:, 1], idx[:, 2]

            def dists(x_sel, x_all):
                return torch.sqrt(eps + torch.sum(
                    (x_sel[:, None, None, :] - x_all) ** 2, dim=-1))

            d_pred = dists(atom14_pred_positions[b, i, a], atom14_pred_positions[b])
            d_gt = dists(gt_pos[b, i, a], gt_pos[b])
            d_alt = dists(alt_pos[b, i, a], alt_pos[b])
            weight = second[b]                                # [K, N, 14]
            flat = per_res.reshape(-1)
            offset = b * per_res.shape[1] + i
            flat.index_add_(0, offset,
                            (weight * torch.sqrt(eps + (d_pred - d_gt) ** 2)).sum(dim=(-1, -2)))
            alt_per_res.reshape(-1).index_add_(
                0, offset,
                (weight * torch.sqrt(eps + (d_pred - d_alt) ** 2)).sum(dim=(-1, -2)))
        better = (alt_per_res < per_res).to(dtype)

    return {
        "alt_naming_is_better": better,
        "renamed_atom14_gt_positions": ((1.0 - better[..., None, None]) * gt_pos
                                        + better[..., None, None] * alt_pos),
        "renamed_atom14_gt_exists": ((1.0 - better[..., None]) * exists
                                     + better[..., None] * batch["atom14_alt_gt_exists"]),
    }


def fape_loss(out: dict, batch: dict, *, backbone_weight: float = 0.5,
              sidechain_weight: float = 1.0) -> torch.Tensor:
    """`params.yaml` `loss.fape`: backbone 0.5 and sidechain 1.0, the pair weighted 1.0 overall."""
    bb = backbone_fape(out["frames"], batch["backbone_rigid_tensor"],
                       batch["backbone_rigid_mask"], batch.get("use_clamped_fape"))
    sc = sidechain_fape(out["sidechain_frames"], out["positions"],
                        batch["rigidgroups_gt_frames"], batch["rigidgroups_alt_gt_frames"],
                        batch["rigidgroups_gt_exists"], batch["renamed_atom14_gt_positions"],
                        batch["renamed_atom14_gt_exists"], batch["alt_naming_is_better"],
                        batch["cdr_mask"])
    return torch.mean(backbone_weight * bb + sidechain_weight * sc)


def supervised_chi_loss(angles_sin_cos: torch.Tensor, unnormalized_angles_sin_cos: torch.Tensor,
                        aatype: torch.Tensor, seq_mask: torch.Tensor, chi_mask: torch.Tensor,
                        chi_angles_sin_cos: torch.Tensor, chi_weight: float = 1.0,
                        angle_norm_weight: float = 0.02, eps: float = 1e-6,
                        batch_collapsed_periodicity: bool = True) -> torch.Tensor:
    """Alg. 27. See this module's docstring for `batch_collapsed_periodicity`."""
    pred = angles_sin_cos[..., 3:, :]
    one_hot = F.one_hot(aatype.long(), RESTYPE_BINS).to(angles_sin_cos.dtype)
    table = torch.as_tensor(CHI_PI_PERIODIC, dtype=angles_sin_cos.dtype,
                            device=angles_sin_cos.device)
    if batch_collapsed_periodicity:
        periodic = torch.einsum("...ij,jk->ik", one_hot, table)
    else:
        periodic = one_hot @ table

    true_chi = chi_angles_sin_cos.unsqueeze(0)
    shifted = (1 - 2 * periodic).unsqueeze(-1) * true_chi
    sq_err = torch.sum((true_chi - pred) ** 2, dim=-1)
    sq_err = torch.minimum(sq_err, torch.sum((shifted - pred) ** 2, dim=-1))
    sq_err = sq_err.permute(*range(len(sq_err.shape))[1:-2], 0, -2, -1)
    loss = chi_weight * masked_mean(chi_mask[..., None, :, :], sq_err, dim=(-1, -2, -3))

    angle_norm = torch.sqrt(torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(*range(len(norm_error.shape))[1:-2], 0, -2, -1)
    loss = loss + angle_norm_weight * masked_mean(seq_mask[..., None, :, None], norm_error,
                                                  dim=(-1, -2, -3))
    return torch.mean(loss)


def lddt(pred_pos: torch.Tensor, true_pos: torch.Tensor, mask: torch.Tensor,
         cutoff: float = 15.0, eps: float = 1e-10, per_residue: bool = True) -> torch.Tensor:
    """Per-residue lDDT on one atom per residue. `mask` keeps its trailing 1 axis, as upstream's does."""
    n = mask.shape[-2]
    d_true = torch.sqrt(eps + torch.sum((true_pos[..., None, :] - true_pos[..., None, :, :]) ** 2,
                                        dim=-1))
    d_pred = torch.sqrt(eps + torch.sum((pred_pos[..., None, :] - pred_pos[..., None, :, :]) ** 2,
                                        dim=-1))
    scored = ((d_true < cutoff) * mask * mask.transpose(-1, -2)
              * (1.0 - torch.eye(n, device=mask.device, dtype=mask.dtype)))
    l1 = torch.abs(d_true - d_pred)
    score = 0.25 * ((l1 < 0.5).to(l1.dtype) + (l1 < 1.0).to(l1.dtype)
                    + (l1 < 2.0).to(l1.dtype) + (l1 < 4.0).to(l1.dtype))
    dims = (-1,) if per_residue else (-2, -1)
    return (1.0 / (eps + torch.sum(scored, dim=dims))) * (eps + torch.sum(scored * score,
                                                                          dim=dims))


def lddt_loss(logits: torch.Tensor, pred_pos: torch.Tensor, true_pos: torch.Tensor,
              atom_mask: torch.Tensor, resolution: torch.Tensor, cutoff: float = 15.0,
              no_bins: int = 50, min_resolution: float = 0.1, max_resolution: float = 3.0,
              eps: float = 1e-10) -> torch.Tensor:
    """pLDDT: cross-entropy against the CA lDDT of the last block, binned into 50.

    The target is detached, which is the whole design of the head: it predicts its own accuracy and
    must not be able to improve the loss by moving the structure.
    """
    ca_pred = pred_pos[..., CA_INDEX, :]
    ca_true = true_pos[..., CA_INDEX, :]
    ca_mask = atom_mask[..., CA_INDEX:CA_INDEX + 1]
    score = lddt(ca_pred, ca_true, ca_mask, cutoff=cutoff, eps=eps).detach()
    bins = torch.clamp(torch.floor(score * no_bins).long(), max=no_bins - 1)
    errors = softmax_cross_entropy(logits, F.one_hot(bins, num_classes=no_bins).to(logits.dtype))
    flat_mask = ca_mask.squeeze(-1)
    loss = torch.sum(errors * flat_mask, dim=-1) / (eps + torch.sum(flat_mask, dim=-1))
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    return torch.mean(loss)


def final_output_backbone_loss(out: dict, batch: dict) -> torch.Tensor:
    """`params.yaml` `loss.final_output_backbone_loss` at weight 0.5: the last block, twice.

    Backbone FAPE and the chi loss again on the final block alone, the second with
    `angle_norm_weight` 0 so only the angle error counts. ABodyBuilder3's own addition on top of
    AlphaFold's loss set.
    """
    bb = backbone_fape(out["frames"][-1].unsqueeze(0), batch["backbone_rigid_tensor"],
                       batch["backbone_rigid_mask"], batch.get("use_clamped_fape"))
    angle = supervised_chi_loss(out["angles"][-1].unsqueeze(0),
                                out["unnormalized_angles"][-1].unsqueeze(0), batch["aatype"],
                                batch["seq_mask"], batch["chi_mask"],
                                batch["chi_angles_sin_cos"], chi_weight=1.0,
                                angle_norm_weight=0.0)
    return bb + angle
