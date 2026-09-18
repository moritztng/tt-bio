#!/usr/bin/env python3
"""`tt_bio/train/losses_geometry.py` against upstream's own `openfold/utils/loss.py`, float64.

Host-only, no device. The losses are pure tensor functions of a batch dict, so the batch here is
synthetic rather than loaded: random frames, random ground truth, random masks with real structure
(a padded sample, a CDR region, some absent atoms). That is the right input for this check --
what is being verified is that our arithmetic is upstream's, and a real structure would exercise
strictly fewer of the branches than masks chosen to hit them.

**Upstream's float32 rigid floor is lifted here too.** OpenFold's `Rotation.__init__` casts to
float32 unconditionally (`rigid_utils.py:315-318`), and `Rigid.invert()` builds a new `Rotation`, so
upstream's FAPE rotates float64 coordinates with a float32 matrix and a float64-vs-float64
comparison against it stalls at 4.3e-10 -- upstream's error, not ours. Same shim as
`reference_gate.py`, imported rather than copied.

Every loss is scored twice, once at batch 1 and once at batch 3, because the defect this file
exists to pin down only appears at batch > 1: upstream's `supervised_chi_loss` builds its
pi-periodic flags with an einsum whose output spec drops the ellipsis, so the leading dims are
summed. At batch 1 that is the intended 0/1 flag; at batch B it is a count.

Run: PYTHONPATH=$PWD python3 scripts/abb3_port/loss_gate.py --upstream ~/abb3_src/ABodyBuilder3/src
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tt_bio.train import losses_geometry as ours

DT = torch.float64


def synthetic_batch(batch: int, n_tok: int, blocks: int, seed: int) -> tuple[dict, dict]:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, dtype=DT) * scale

    def rand_quat(*shape):
        q = rand(*shape, 4)
        return q / q.norm(dim=-1, keepdim=True)

    def rand_rigid_4x4(*shape):
        from tt_bio.af2_reference import quat_to_rot
        out = torch.zeros(*shape, 4, 4, dtype=DT)
        out[..., :3, :3] = quat_to_rot(rand_quat(*shape))
        out[..., :3, 3] = rand(*shape, 3, scale=8.0)
        out[..., 3, 3] = 1.0
        return out

    seq_mask = torch.ones(batch, n_tok, dtype=DT)
    seq_mask[-1, -3:] = 0.0                       # a padded sample
    atom_exists = (torch.rand(batch, n_tok, 14, generator=g, dtype=DT) > 0.2).to(DT)
    atom_exists = atom_exists * seq_mask.unsqueeze(-1)
    chi_mask = (torch.rand(batch, n_tok, 4, generator=g, dtype=DT) > 0.3).to(DT)
    true_chi = rand(batch, n_tok, 4, 2)
    true_chi = true_chi / true_chi.norm(dim=-1, keepdim=True)
    angles = rand(blocks, batch, n_tok, 7, 2)
    unnorm = angles * (1.0 + 0.3 * rand(blocks, batch, n_tok, 7, 1).abs())
    angles = angles / angles.norm(dim=-1, keepdim=True)

    out = {
        "frames": torch.cat([rand_quat(blocks, batch, n_tok),
                             rand(blocks, batch, n_tok, 3, scale=8.0)], dim=-1),
        "sidechain_frames": rand_rigid_4x4(blocks, batch, n_tok, 8),
        "positions": rand(blocks, batch, n_tok, 14, 3, scale=8.0),
        "angles": angles,
        "unnormalized_angles": unnorm,
        "plddt": rand(batch, n_tok, 50),
    }
    b = {
        "aatype": torch.randint(0, 21, (batch, n_tok), generator=g),
        "seq_mask": seq_mask,
        "chi_mask": chi_mask,
        "chi_angles_sin_cos": true_chi,
        "backbone_rigid_tensor": rand_rigid_4x4(batch, n_tok),
        "backbone_rigid_mask": seq_mask.clone(),
        "rigidgroups_gt_frames": rand_rigid_4x4(batch, n_tok, 8),
        "rigidgroups_alt_gt_frames": rand_rigid_4x4(batch, n_tok, 8),
        "rigidgroups_gt_exists": (torch.rand(batch, n_tok, 8, generator=g, dtype=DT) > 0.25).to(DT),
        "renamed_atom14_gt_positions": rand(batch, n_tok, 14, 3, scale=8.0),
        "renamed_atom14_gt_exists": atom_exists,
        "alt_naming_is_better": (torch.rand(batch, n_tok, generator=g, dtype=DT) > 0.5).to(DT),
        # region 2 is CDR-H3 in B1's numbering; a contiguous middle stretch is the realistic shape
        "cdr_mask": (torch.arange(n_tok) >= n_tok // 3) & (torch.arange(n_tok) < 2 * n_tok // 3),
        "use_clamped_fape": torch.ones(1, dtype=DT),
        "atom14_gt_positions": rand(batch, n_tok, 14, 3, scale=8.0),
        "atom14_atom_exists": atom_exists,
        # Alg. 26's inputs. `atom14_atom_is_ambiguous` marks the symmetric side-chain pairs, so it
        # is sparse in reality; a random quarter of the slots is used here because what the check
        # needs is for `alt_naming_is_better` to come back MIXED -- an all-zero or all-one verdict
        # would agree with upstream for the wrong reason.
        "atom14_alt_gt_positions": rand(batch, n_tok, 14, 3, scale=8.0),
        "atom14_gt_exists": atom_exists,
        "atom14_alt_gt_exists": (torch.rand(batch, n_tok, 14, generator=g, dtype=DT) > 0.2).to(DT),
        "atom14_atom_is_ambiguous": (torch.rand(batch, n_tok, 14, generator=g,
                                                dtype=DT) > 0.75).to(DT),
        "resolution": torch.full((batch,), 2.0, dtype=DT),
    }
    b["cdr_mask"] = b["cdr_mask"].expand(batch, n_tok).clone()
    return out, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="path to ABodyBuilder3/src")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, args.upstream)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from abodybuilder3.openfold.utils import loss as up
    from abodybuilder3.openfold.utils import rigid_utils
    from reference_gate import _Fp32FloorLifted

    torch.set_grad_enabled(False)
    rows = []
    for batch in (1, 3):
        out, b = synthetic_batch(batch, args.tokens, args.blocks, args.seed)
        cfg_bb = {"clamp_distance": 10.0, "loss_unit_distance": 10.0, "eps": 1e-4}
        cfg_sc = {"clamp_distance": 10.0, "intercdr_distance": 30.0, "length_scale": 10.0,
                  "eps": 1e-4}

        floor = _Fp32FloorLifted(rigid_utils)
        got_bb = ours.backbone_fape(out["frames"], b["backbone_rigid_tensor"],
                                    b["backbone_rigid_mask"], b["use_clamped_fape"])
        with floor:
            want_bb = up.backbone_loss(traj=out["frames"], **{**b, **cfg_bb})
        rows.append((f"backbone FAPE   [batch {batch}]", got_bb, want_bb))

        got_sc = ours.sidechain_fape(out["sidechain_frames"], out["positions"],
                                     b["rigidgroups_gt_frames"], b["rigidgroups_alt_gt_frames"],
                                     b["rigidgroups_gt_exists"],
                                     b["renamed_atom14_gt_positions"],
                                     b["renamed_atom14_gt_exists"], b["alt_naming_is_better"],
                                     b["cdr_mask"])
        with floor:
            want_sc = up.sidechain_loss(out["sidechain_frames"], out["positions"],
                                    **{**b, **cfg_sc})
        rows.append((f"sidechain FAPE  [batch {batch}]", got_sc.mean(), want_sc.mean()))

        got_chi = ours.supervised_chi_loss(out["angles"], out["unnormalized_angles"],
                                           b["aatype"], b["seq_mask"], b["chi_mask"],
                                           b["chi_angles_sin_cos"], chi_weight=1.0,
                                           angle_norm_weight=0.02)
        want_chi = up.supervised_chi_loss(out["angles"], out["unnormalized_angles"],
                                          **{**b, "chi_weight": 1.0, "angle_norm_weight": 0.02})
        rows.append((f"supervised chi  [batch {batch}]", got_chi, want_chi))

        got_plddt = ours.lddt_loss(out["plddt"], out["positions"][-1],
                                   b["atom14_gt_positions"], b["atom14_atom_exists"],
                                   b["resolution"])
        want_plddt = up.lddt_loss(out["plddt"], out["positions"][-1], b["atom14_gt_positions"],
                                  b["atom14_atom_exists"], b["resolution"])
        rows.append((f"pLDDT           [batch {batch}]", got_plddt, want_plddt))

        # Alg. 26's renaming, which the sidechain FAPE consumes. Scored on all three outputs it
        # returns, because a wrong `alt_naming_is_better` is a silent sign flip on the symmetric
        # residues rather than an error.
        got_ren = ours.compute_renamed_ground_truth(b, out["positions"][-1])
        with floor:
            want_ren = up.compute_renamed_ground_truth(b, out["positions"][-1])
        # The restricted form against our own dense form as well as against upstream, because a
        # restriction that drops a contributing pair agrees with neither and must be caught by both.
        dense = ours.compute_renamed_ground_truth_dense(b, out["positions"][-1])
        assert torch.equal(got_ren["alt_naming_is_better"], dense["alt_naming_is_better"]), (
            "the restricted Alg. 26 chose a different naming than the dense form")
        frac = got_ren["alt_naming_is_better"].mean().item()
        assert 0.05 < frac < 0.95, (
            f"alt_naming_is_better came back {frac:.2f} -- an all-or-nothing verdict agrees with "
            f"upstream without testing the selection")
        for key in ("alt_naming_is_better", "renamed_atom14_gt_positions",
                    "renamed_atom14_gt_exists"):
            rows.append((f"renamed gt {key.split('_')[-1]:<10}[batch {batch}]",
                         got_ren[key].sum(), want_ren[key].sum()))

        got_fin = ours.final_output_backbone_loss(out, b)
        with floor:
            want_fin = up.final_output_backbone_loss(out, {**b, **cfg_bb})
        rows.append((f"final-block extra [batch {batch}]", got_fin, want_fin))

        # And the defect, isolated: the corrected per-sample periodicity against upstream's.
        fixed = ours.supervised_chi_loss(out["angles"], out["unnormalized_angles"], b["aatype"],
                                         b["seq_mask"], b["chi_mask"], b["chi_angles_sin_cos"],
                                         chi_weight=1.0, angle_norm_weight=0.02,
                                         batch_collapsed_periodicity=False)
        rows.append((f"  ^ corrected periodicity, as a DELTA not a match [batch {batch}]",
                     fixed, want_chi))

    bad = 0
    for name, got, want in rows:
        got_v, want_v = float(got), float(want)
        delta = abs(got_v - want_v) / max(abs(want_v), 1e-30)
        informational = name.startswith("  ^")
        ok = informational or delta < 1e-12
        bad += not ok
        tag = "info" if informational else ("ok  " if ok else "FAIL")
        print(f"  {tag} {name:<46} ours {got_v:.10f}  upstream {want_v:.10f}  rel {delta:.2e}")
    print(f"\n{'PASS' if not bad else f'FAIL: {bad} losses'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
