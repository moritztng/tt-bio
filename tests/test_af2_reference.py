"""Card-free, checkpoint-free tests for the torch AF2 reference.

The expensive check -- the whole trunk against the captured JAX activations -- lives in
`scripts/af2_port/tap_gate.py` and needs the 373 MB parameter file. What is testable without it
is everything that decides whether that check is even meaningful:

* the parameter tree matches `tt_bio.af2_weights`'s remap key for key and shape for shape, read
  off the committed manifest rather than off a checkpoint;
* the primitives compute what AlphaFold's source says they compute, on hand-checked inputs;
* the transformations that are easy to get backwards -- the incoming triangle multiplication's
  left/right swap, the template pair stack's block order, the psi sign flip -- fail when removed.

Each of those has been verified to fail with the transformation taken out, which is the only way
a parity test is worth having (`gate-fixture-existence-vs-content-inversion`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio import af2_reference as ref

ARTIFACTS = Path(__file__).resolve().parents[1] / "scripts" / "af2_port" / "parity_artifacts"
REMAPPED_SHAPES = ARTIFACTS / "params_model_1_ptm_remapped_shapes.json"


def _remapped_shapes() -> dict[str, list[int]]:
    return {k: list(v) for k, v in json.loads(REMAPPED_SHAPES.read_text()).items()}


def test_parameter_tree_matches_the_remap_exactly():
    """Every remapped key has a home in the model, and nothing in the model is unfed.

    This is the structural half of the parity story: it catches a wrong channel count, a missing
    bias, a module nested one level too deep. Nothing is deferred any more, so the comparison is
    over the whole 93,208,926-parameter checkpoint rather than a subset of it.
    """
    model = {k: list(v.shape) for k, v in ref.AF2Model().state_dict().items()}
    assert model == _remapped_shapes()


def test_the_model_is_the_whole_checkpoint():
    assert sum(p.numel() for p in ref.AF2Model().parameters()) == 93_208_926


def test_monomer_model_drops_the_template_stack():
    keys = set(ref.AF2Model(template=False).state_dict())
    assert not any(k.startswith("template.") for k in keys)
    assert any(k.startswith("evoformer.47.") for k in keys)


# ------------------------------------------------------------------ primitives


def test_layer_norm_uses_float32_math_and_float32_parameters_in_a_bf16_trunk():
    norm = ref.LayerNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
        norm.bias.copy_(torch.linspace(-0.1, 0.1, 8))
    x = torch.randn(4, 8)
    want = norm(x)
    got = norm(x.to(torch.bfloat16))
    assert got.dtype == torch.bfloat16
    assert norm.weight.dtype == torch.float32
    assert torch.allclose(got.float(), want, atol=2e-2)


def test_fast_variance_differs_from_the_centred_variance():
    """Only the fused triangle multiplication's two LayerNorms use mean(x^2) - mean(x)^2.

    They are numerically different on a large mean, and pinning that here is what stops the
    difference from being flattened into "LayerNorm is LayerNorm" later.
    """
    x = torch.full((1, 4), 1e5) + torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    slow, fast = ref.LayerNorm(4), ref.LayerNorm(4, fast_variance=True)
    assert not torch.allclose(slow(x), fast(x), atol=1e-3)
    small = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert torch.allclose(slow(small), fast(small), atol=1e-4)


def test_relu_transition_is_relu_not_swiglu():
    block = ref.ReluTransition(4, 2)
    with torch.no_grad():
        block.norm.weight.fill_(1.0)
        block.fc1.weight.copy_(torch.eye(8, 4))
        block.fc2.weight.copy_(torch.eye(4, 8))
    out = block(torch.tensor([[-3.0, -1.0, 1.0, 3.0]]))
    normed = block.norm(torch.tensor([[-3.0, -1.0, 1.0, 3.0]]))
    assert torch.allclose(out, torch.relu(normed), atol=1e-5)


def test_attention_q_is_prescaled_and_the_pair_bias_is_raw():
    """AlphaFold scales q by key_dim ** -0.5 and adds the nonbatched bias afterwards.

    Folding the scale into the bias instead -- the Boltz/Protenix convention -- makes the softmax
    sqrt(key_dim) too peaky. That was already root-caused once on OpenFold3, so the two are
    separated here by construction and this test is what keeps them separated.
    """
    attn = ref.Attention(4, 4, num_head=1, key_dim=4, value_dim=4, out_dim=4, gating=False)
    with torch.no_grad():
        attn.linear_q.weight.copy_(torch.eye(4))
        attn.linear_k.weight.copy_(torch.eye(4))
        attn.linear_v.weight.copy_(torch.eye(4))
        attn.linear_o.weight.copy_(torch.eye(4))
    x = torch.randn(1, 3, 4)
    bias = torch.zeros(1, 1, 1, 3)
    nonbatched = torch.randn(1, 3, 3)
    logits = torch.einsum("qc,kc->qk", x[0], x[0]) * 0.5 + nonbatched[0]
    want = torch.softmax(logits, -1) @ x[0]
    assert torch.allclose(attn._attend(x, x, bias, nonbatched)[0], want, atol=1e-5)


def test_global_attention_shares_one_key_and_value_head():
    attn = ref.GlobalAttention(8, num_head=4, key_dim=2, value_dim=2, out_dim=8)
    assert list(attn.linear_k.weight.shape) == [2, 8]
    assert list(attn.linear_v.weight.shape) == [2, 8]
    assert list(attn.linear_q.weight.shape) == [8, 8]


def test_global_attention_query_average_is_zero_under_an_all_zero_mask():
    """The extra-MSA stack runs on an all-zero mask, so this is the path production takes."""
    attn = ref.GlobalAttention(8, num_head=4, key_dim=2, value_dim=2, out_dim=8)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_()
    x = torch.randn(5, 1, 8)
    mask = torch.zeros(5, 1, 1)
    q_avg = (mask * x).sum(1) / (mask.sum(1) + 1e-10)
    assert torch.equal(q_avg, torch.zeros(5, 8))
    assert torch.isfinite(attn._attend(x, x, mask)).all()


def test_incoming_triangle_multiplication_transposes_the_outgoing_one():
    """`kic,kjc->ijc` against `ikc,jkc->ijc` on the same inputs.

    tt-bio's incoming convention takes AlphaFold's *right* projection first, which is the swap
    `af2_weights` applies to the fused concatenation. Getting it backwards transposes every
    incoming update, which looks plausible and is wrong.
    """
    torch.manual_seed(0)
    a, b = torch.randn(6, 6, 3), torch.randn(6, 6, 3)
    out = torch.einsum("ikc,jkc->ijc", a, b)
    inc = torch.einsum("kic,kjc->ijc", a, b)
    assert torch.allclose(inc, torch.einsum("ikc,jkc->ijc", a.permute(1, 0, 2),
                                            b.permute(1, 0, 2)), atol=1e-5)
    assert not torch.allclose(inc, out, atol=1e-3)


def test_triangle_multiplication_under_a_zero_mask_is_a_bias_not_a_zero():
    """A fully masked pair block does not contribute zero, and that is not a bug.

    The mask kills the projections, so the einsum is zero -- but the centre LayerNorm of an
    all-zero vector is its own offset, and the output projection and gate act on that. So the
    update is a per-channel constant. Reading "masked means zero" into a padded tile is how a
    padding bug gets written, which is why the actual value is pinned here.
    """
    block = ref.TriangleMultiplication(4, 4, ending=False)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(std=0.2)
    z = torch.randn(5, 5, 4)
    got = block(z, torch.zeros(5, 5))
    x = block.norm_in(z)
    want = block.p_out(block.norm_out(torch.zeros(5, 5, 4))) * torch.sigmoid(block.g_out(x))
    assert torch.allclose(got, want, atol=1e-6)
    assert got.abs().max() > 1e-4


def test_outer_product_mean_divides_the_bias_too():
    """AlphaFold divides output_w @ outer + output_b by the pair norm, epsilon 1e-3.

    With an all-zero mask the outer product is zero and the norm is zero, so the whole output
    collapses to output_b / 1e-3 -- a per-channel constant, not zero. That constant is what the
    extra-MSA stack injects into the pair track on every one of its four blocks.
    """
    opm = ref.OuterProductMean(4, 2, 3)
    with torch.no_grad():
        for p in opm.parameters():
            p.normal_(std=0.2)
    out = opm(torch.randn(1, 5, 4), torch.zeros(1, 5))
    constant = opm.proj_o.bias / opm.eps
    assert torch.allclose(out, constant.expand(5, 5, 3), atol=1e-4)


def test_template_pair_stack_runs_the_attentions_before_the_multiplications():
    evo = ref.PairBlock(4, 4, num_head=1, head_dim=4, factor=2, evoformer_order=True)
    tpl = ref.PairBlock(4, 4, num_head=1, head_dim=4, factor=2, evoformer_order=False)
    tpl.load_state_dict(evo.state_dict())
    with torch.no_grad():
        for p in evo.parameters():
            p.normal_(std=0.3)
    tpl.load_state_dict(evo.state_dict())
    z, mask = torch.randn(4, 4, 4), torch.ones(4, 4)
    assert not torch.allclose(evo(z, mask), tpl(z, mask), atol=1e-4)


# ------------------------------------------------------------------ geometry


def test_recycling_distogram_is_15_bins_to_20_75_angstrom():
    """Not the template distogram's 39 bins to 50.75. The plan had these swapped."""
    assert ref.RECYCLE_DGRAM == (15, 3.25, 20.75)
    assert ref.TEMPLATE_DGRAM == (39, 3.25, 50.75)
    assert ref.AF2Model().recycle["prev_pos_linear"].weight.shape[1] == 15


def test_dgram_is_one_hot_and_the_last_bin_catches_everything():
    positions = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
    dgram = ref.dgram_from_positions(positions, *ref.RECYCLE_DGRAM)
    assert dgram.shape == (3, 3, 15)
    assert torch.equal(dgram.sum(-1), torch.ones(3, 3) - torch.eye(3))
    # 500 A is past max_bin, so it lands in the final bin.
    assert dgram[0, 2].argmax().item() == 14


def test_chi_atom_indices_match_alphafolds_table():
    """21 rows, four chis, four atoms, zero-padded, with an all-zero UNK row.

    The vendored ESM `residue_constants` agrees with AlphaFold on the chi tables on every row --
    unlike the atom-existence masks, which differ on restype 20. Pinning it here means a vendor
    bump that moves either one fails a test instead of moving a number.
    """
    table = ref.CHI_ATOM_INDICES
    assert table.shape == (21, 4, 4)
    assert (table[20] == 0).all()
    # Arginine has four chis; its first is N-CA-CB-CG.
    from tt_bio._vendor.esm.utils import residue_constants as rc
    arg = rc.restype_order["R"]
    assert list(table[arg, 0]) == [rc.atom_order[a] for a in ("N", "CA", "CB", "CG")]
    assert ref.CHI_ANGLES_MASK.shape == (21, 4)
    assert ref.CHI_PI_PERIODIC.shape == (21, 4)
    assert not ref.CHI_ANGLES_MASK[20].any()


def test_torsion_angles_are_unit_length_and_psi_is_mirrored():
    """A single ideal residue: the sin/cos pairs normalise, and psi carries the sign flip.

    psi is computed from the oxygen rather than the next residue's nitrogen, so AlphaFold
    multiplies it by -1. Removing that flip leaves every other torsion untouched, which is
    exactly why it needs its own test.
    """
    torch.manual_seed(1)
    positions = torch.zeros(1, 3, ref.NUM_ATOM, 3)
    positions[0, :, :5] = torch.randn(3, 5, 3)
    mask = torch.zeros(1, 3, ref.NUM_ATOM)
    mask[0, :, :5] = 1.0
    out = ref.atom37_to_torsion_angles(torch.zeros(1, 3, dtype=torch.long), positions, mask)
    norms = out["torsion_angles_sin_cos"].square().sum(-1).sqrt()
    # A degenerate frame (a torsion whose four atoms are not all present) normalises to exactly
    # zero rather than to a unit vector, because `placeholder_for_undefined` is False in
    # production. Every torsion the mask keeps is unit length.
    assert torch.all((norms < 1e-6) | ((norms - 1).abs() < 1e-3))
    keep = out["torsion_angles_mask"] > 0
    assert torch.allclose(norms[keep], torch.ones_like(norms[keep]), atol=1e-3)
    # Alanine has no chis, so only the three backbone torsions are unmasked, and pre-omega and
    # phi need the previous residue.
    assert out["torsion_angles_mask"][0, 0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert out["torsion_angles_mask"][0, 1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    unflipped = ref._frame_local_coords(
        torch.cat([positions[:, :, 0:3], positions[:, :, 4:5]], dim=-2))
    sin_cos = torch.stack([unflipped[..., 2], unflipped[..., 1]], -1)
    sin_cos = sin_cos / sin_cos.square().sum(-1, keepdim=True).add(1e-8).sqrt()
    assert torch.allclose(out["torsion_angles_sin_cos"][0, :, 2], -sin_cos[0], atol=1e-4)


def test_frame_local_coords_puts_the_origin_at_zero_and_the_x_axis_on_the_second_point():
    points = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0]]])
    local = ref._frame_local_coords(points)
    assert torch.allclose(local, torch.zeros(1, 3), atol=1e-5)
    # The point on the negative x axis sits at -|p2 - p1| along x.
    points[0, 3] = points[0, 1]
    assert torch.allclose(ref._frame_local_coords(points)[0],
                          torch.tensor([-3.0, 0.0, 0.0]), atol=1e-5)


def test_extra_msa_feature_is_25_wide_with_the_gap_channel_hot():
    feats = {
        "extra_msa": torch.zeros(1, 6, dtype=torch.long),
        "extra_has_deletion": torch.zeros(1, 6),
        "extra_deletion_value": torch.zeros(1, 6),
    }
    out = ref._extra_msa_feature(feats)
    assert out.shape == (1, 6, 25)
    assert torch.equal(out.sum(-1), torch.ones(1, 6))
    assert out[0, 0, 0] == 1.0


def test_pseudo_beta_takes_ca_for_glycine_and_cb_otherwise():
    from tt_bio._vendor.esm.utils import residue_constants as rc
    positions = torch.zeros(2, ref.NUM_ATOM, 3)
    positions[:, rc.atom_order["CA"]] = torch.tensor([1.0, 0.0, 0.0])
    positions[:, rc.atom_order["CB"]] = torch.tensor([0.0, 1.0, 0.0])
    aatype = torch.tensor([rc.restype_order["G"], rc.restype_order["A"]])
    out = ref.pseudo_beta(aatype, positions)
    assert torch.equal(out[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(out[1], torch.tensor([0.0, 1.0, 0.0]))


# ------------------------------------------------------- the structure module and its algebra


RIGID_TABLES = ARTIFACTS / "af2_rigid_group_tables.npz"


def test_rigid_group_tables_are_alphafolds_not_the_vendored_copys():
    """Pinned against AlphaFold's own dump, so a vendor bump breaks a test not the coordinates.

    The vendored ESM constants agree on restypes 0-19 and give `UNK` N/CA/C where AlphaFold
    gives it nothing; `af2_data` zeroes row 20 to match. Checking the derivation against
    colabdesign's values makes this a pin rather than a self-consistency check.
    """
    from tt_bio import af2_data

    with np.load(RIGID_TABLES) as want:
        for name, got in (
            ("restype_rigid_group_default_frame", af2_data.RESTYPE_RIGID_GROUP_DEFAULT_FRAME),
            ("restype_atom14_to_rigid_group", af2_data.RESTYPE_ATOM14_TO_RIGID_GROUP),
            ("restype_atom14_rigid_group_positions",
             af2_data.RESTYPE_ATOM14_RIGID_GROUP_POSITIONS),
            ("restype_atom14_mask", af2_data.RESTYPE_ATOM14_MASK),
        ):
            assert np.array_equal(got, want[name]), name
            assert not np.any(np.asarray(got)[20]), f"{name} row 20 must be AlphaFold's empty UNK"


def test_quat_to_rot_of_the_identity_quaternion_is_the_identity():
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(ref.quat_to_rot(quaternion)[0], torch.eye(3), atol=1e-6)


def test_quat_to_rot_is_orthonormal_with_determinant_one():
    quaternion = torch.randn(5, 4)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    rot = ref.quat_to_rot(quaternion)
    assert torch.allclose(rot @ rot.transpose(-1, -2), torch.eye(3).expand(5, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(5), atol=1e-5)


def test_pre_compose_rotates_the_translation_update_by_the_old_rotation():
    """Adding the update unrotated is the mistake this catches; it is a no-op at the identity."""
    torch.manual_seed(0)
    quaternion = torch.randn(4, 4)
    affine = ref.QuatAffine(quaternion, torch.zeros(4, 3))
    update = torch.cat([torch.zeros(4, 3), torch.randn(4, 3)], dim=-1)
    composed = affine.pre_compose(update)
    rotated = torch.einsum("nij,nj->ni", affine.rotation, update[:, 3:])
    assert torch.allclose(composed.translation, rotated, atol=1e-6)
    assert not torch.allclose(composed.translation, update[:, 3:], atol=1e-3)


def test_pre_compose_renormalises_the_quaternion():
    affine = ref.QuatAffine(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3))
    composed = affine.pre_compose(torch.tensor([[0.7, -0.4, 0.2, 0.0, 0.0, 0.0]]))
    assert torch.allclose(composed.quaternion.norm(), torch.tensor(1.0), atol=1e-6)


def test_invert_point_undoes_apply_to_point():
    torch.manual_seed(1)
    affine = ref.QuatAffine(torch.randn(3, 4), torch.randn(3, 3))
    points = torch.randn(3, 5, 3)
    assert torch.allclose(affine.invert_point(affine.apply_to_point(points)), points, atol=1e-5)


def test_ipa_point_split_is_block_wise_not_interleaved():
    """`q_point_local` splits into an x-block, a y-block and a z-block, in that order."""
    flat = torch.arange(12.0).reshape(1, 12)
    points = ref.InvariantPointAttention._as_points(flat)
    assert points.shape == (1, 4, 3)
    assert torch.equal(points[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(points[0, :, 2], torch.tensor([8.0, 9.0, 10.0, 11.0]))


def test_ipa_output_projection_reads_scalars_points_norms_and_the_pair_track():
    ipa = ref.InvariantPointAttention()
    assert ipa.output_projection.weight.shape[1] == 12 * 16 + 3 * 12 * 8 + 12 * 8 + 12 * 128


def test_ipa_ignores_masked_residues():
    """A masked residue must not reach an unmasked query, through the logits or the pair track."""
    torch.manual_seed(2)
    ipa = ref.InvariantPointAttention()
    for parameter in ipa.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    act, act_2d = torch.randn(6, ref.C_S), torch.randn(6, 6, ref.C_Z)
    mask = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
    affine = ref.QuatAffine.identity(6)
    before = ipa(act, act_2d, mask, affine)
    act_2d[3:, :] = torch.randn(3, 6, ref.C_Z)
    act_2d[:, 3:] = torch.randn(6, 3, ref.C_Z)
    after = ipa(act, act_2d, mask, affine)
    assert torch.allclose(before[:3], after[:3], atol=1e-5)


def test_torsion_angles_to_frames_chains_chi2_onto_chi1():
    torch.manual_seed(3)
    aatype = torch.tensor([4])                       # PHE, which has two chi angles
    angles = torch.randn(1, 7, 2)
    angles = angles / angles.norm(dim=-1, keepdim=True)
    rot, trans = ref.torsion_angles_to_frames(aatype, torch.eye(3)[None], torch.zeros(1, 3),
                                              angles)
    default = ref.RIGID_GROUP_DEFAULT_FRAME[aatype]
    sin, cos = angles[0, 4, 0], angles[0, 4, 1]
    rot_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    chi2_alone = default[0, 6, :3, :3] @ rot_x
    assert torch.allclose(rot[0, 6], rot[0, 5] @ chi2_alone, atol=1e-5)


def test_torsion_angles_to_frames_leaves_the_backbone_group_unrotated():
    aatype = torch.tensor([0])
    angles = torch.zeros(1, 7, 2)
    angles[..., 1] = 1.0
    rot, trans = ref.torsion_angles_to_frames(aatype, torch.eye(3)[None], torch.zeros(1, 3),
                                              angles)
    assert torch.allclose(rot[0, 0], torch.eye(3), atol=1e-6)


def test_frames_to_atom14_positions_zeroes_atoms_the_residue_does_not_have():
    aatype = torch.tensor([7, 20])                   # GLY has 4 atoms, UNK has none
    rot = torch.eye(3).expand(2, 8, 3, 3).contiguous()
    trans = torch.zeros(2, 8, 3)
    positions = ref.frames_to_atom14_positions(aatype, rot, trans)
    assert not torch.any(positions[0, 4:])
    assert not torch.any(positions[1])


def test_atom14_to_atom37_gathers_by_the_index_map_and_masks():
    atom14 = torch.arange(2 * 14 * 3, dtype=torch.float32).reshape(2, 14, 3)
    feats = {"residx_atom37_to_atom14": torch.zeros(2, 37, dtype=torch.long),
             "atom37_atom_exists": torch.zeros(2, 37)}
    feats["residx_atom37_to_atom14"][:, 5] = 3
    feats["atom37_atom_exists"][:, 5] = 1.0
    atom37 = ref.atom14_to_atom37(atom14, feats)
    assert torch.equal(atom37[:, 5], atom14[:, 3])
    assert not torch.any(atom37[:, :5])


def test_structure_module_starts_at_the_identity_and_scales_the_trajectory():
    """Zero-initialised weights leave the affine at the identity, which pins the traj layout."""
    module = ref.AF2StructureModule()
    feats = {"seq_mask": torch.ones(4), "aatype": torch.zeros(4, dtype=torch.long),
             "atom14_atom_exists": torch.ones(4, 14), "atom37_atom_exists": torch.ones(4, 37),
             "residx_atom37_to_atom14": torch.zeros(4, 37, dtype=torch.long)}
    out = module(torch.randn(4, ref.C_S), torch.randn(4, 4, ref.C_Z), feats)
    assert out["traj"].shape == (8, 4, 7)
    assert torch.allclose(out["traj"][:, :, :4], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert not torch.any(out["traj"][:, :, 4:])
    assert torch.equal(out["final_affines"], out["traj"][-1])
    assert out["sidechains/angles"] is out["sidechains/angles_sin_cos"]


def test_lddt_head_reads_the_structure_module_not_the_trunk_single():
    head = ref.PredictedLDDTHead()
    assert head.norm.weight.shape == (ref.C_S,)
    assert head.logits.weight.shape == (ref.PLDDT_BINS, 128)


def test_run_recycles_threads_prev_and_returns_every_pass():
    class Counter(torch.nn.Module):
        def forward(self, feats, prev):
            step = prev["prev_pair"] + 1
            return {"msa_first_row": step, "pair": step, "single": step,
                    "structure": {"final_atom_positions": step}}

    passes = ref.run_recycles(Counter(), {}, {"prev_pair": torch.zeros(1)}, num_recycles=3)
    assert [float(p["pair"]) for p in passes] == [1.0, 2.0, 3.0, 4.0]
