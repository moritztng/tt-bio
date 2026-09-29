"""`tt_bio/abodybuilder3_reference.py` against `af2_reference`, and pinned against itself.

The strongest check on the ABodyBuilder3 reference needs upstream's package: score every output of
our module against Exscientia's own `StructureModule` in float64. That is
`scripts/abb3_port/reference_gate.py --upstream`, it passes at 8.7e-14, and CI cannot run it because
CI does not have the package. Which is exactly the gap `tt_bio/train/losses.py` sat in -- correct,
verified once against ByteDance's own module, and unprotected against regression for months. So the
part that needs nothing external lives here.

**What each test in this file proves, stated so the pin is not mistaken for a correctness proof:**

* `test_ipa_matches_af2_reference` and `test_batched_frame_functions_match_af2_reference` are real
  checks. Our IPA is `af2_reference.InvariantPointAttention`'s algorithm at ABodyBuilder3's dims,
  and our batched torsion/atom14 functions are that file's two with a sample axis in front, so
  float64 agreement is a statement about arithmetic and not about a recorded number.
* `test_reference_digests_are_pinned` proves **nothing moved**. It does not prove the reference was
  ever right -- the two tests above and the upstream gate are what do that. It catches a refactor
  that changes a number, which is the failure mode a reference of record actually has.
* `test_the_pin_actually_breaks` is the control, and it edits a COPY OF THE SOURCE ON DISK. A
  control that rebinds a module attribute instead can silently pass: a constant already bound as a
  default argument at def-time does not change when the module attribute does, so the perturbation
  never reaches the code under test and the control "passes" by measuring nothing.

Host-only: torch and numpy, no ttnn, no device.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest
import torch

from tt_bio import af2_reference as afr
from tt_bio.abodybuilder3_reference import (ABB3Config, ABB3StructureModule,
                                            InvariantPointAttention, frames_to_atom14_positions,
                                            single_and_pair_features, torsion_angles_to_frames)

DT = torch.float64
SEED = 0
TOKENS = 24
BATCH = 2

# Digests of every output of a fixed-seed model on a fixed input, in float64. Regenerate ONLY with
# a reason: a failing run prints the replacement block: a diff here is either a
# bug or a deliberate change to the reference, and both need saying out loud in the commit message.
GOLDEN = {
    "frames": (950.532097027419, 22564.27373835959),
    "sidechain_frames": (12184.95365503489, 201786.2628651583),
    "unnormalized_angles": (62616.800243459846, 3266295243.3574376),
    "angles": (240.05750284646575, 2688.0),
    "positions": (10755.192456239023, 228191.3040497437),
    "states": (1597.3154961938878, 8441.783813429623),
    "single": (-179.23232988711348, 1227.571620333318),
    "plddt": (-3610.216544833568, 848629.6795962822),
}
#: Relative, with an absolute floor for the digests that sit near zero. These are float64 sums and
#: reproduce bit for bit on one build, but they span nine orders of magnitude -- 2688 for the
#: unit-norm angles against 3.3e9 for the un-normalised ones -- so a single absolute bar is either
#: below float64 spacing at the top end or blind to a sign flip at the bottom.
DIGEST_RTOL = 1e-12
DIGEST_ATOL = 1e-9


def _fixed_inputs(dtype=DT):
    g = torch.Generator().manual_seed(SEED + 1)
    aatype = torch.randint(0, 21, (BATCH, TOKENS), generator=g)
    is_heavy = (torch.arange(TOKENS) < TOKENS // 2).expand(BATCH, TOKENS).clone()
    residue_index = torch.arange(TOKENS).expand(BATCH, TOKENS).clone()
    single, pair = single_and_pair_features(aatype, is_heavy, residue_index, dtype=dtype)
    mask = torch.ones(BATCH, TOKENS, dtype=dtype)
    mask[-1, -3:] = 0.0  # a padded sample, so the mask path is exercised and not assumed
    return single, pair, aatype, mask


def _fixed_model(module=None, *, use_plddt=True):
    """A model whose every parameter is random but reproducible.

    Deliberately not the released initialisation: that leaves `linear_out` and the backbone update
    at zero, which would hide a sign error in either.
    """
    mod = module or importlib.import_module("tt_bio.abodybuilder3_reference")
    torch.manual_seed(SEED)
    model = mod.ABB3StructureModule(mod.ABB3Config(use_plddt=use_plddt)).to(DT)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    return model.eval()


def _digests(out: dict) -> dict:
    """Sum and sum-of-squares per output. Two moments catch a sign flip that one would not."""
    return {k: (v.sum().item(), v.square().sum().item()) for k, v in out.items()}


@torch.no_grad()
def test_ipa_matches_af2_reference():
    """Our IPA is `af2_reference`'s algorithm at ABodyBuilder3's dims, in float64.

    The two constants that genuinely differ are set to `af2_reference`'s here -- `eps` 1e-8 rather
    than upstream's 1e-7 point-norm epsilon, and `inf` 1e5 rather than 1e7 -- because the point of
    this test is the arithmetic between them, and those two are configuration that
    `scripts/abb3_port/reference_gate.py` checks against upstream's own values.
    """
    cfg = ABB3Config(epsilon=1e-8, inf=1e5)
    torch.manual_seed(SEED)
    ours = InvariantPointAttention(cfg).to(DT)
    for p in ours.parameters():
        p.normal_(0.0, 0.3)
    theirs = afr.InvariantPointAttention(
        c_s=cfg.embed_dim, c_z=cfg.embed_dim, num_head=cfg.no_heads_ipa, num_scalar_qk=cfg.c_ipa,
        num_scalar_v=cfg.c_ipa, num_point_qk=cfg.no_qk_points, num_point_v=cfg.no_v_points).to(DT)
    for dst, src in (("q_scalar", "linear_q"), ("kv_scalar", "linear_kv"),
                     ("q_point_local", "linear_q_points"),
                     ("kv_point_local", "linear_kv_points"), ("attention_2d", "linear_b"),
                     ("output_projection", "linear_out")):
        getattr(theirs, dst).weight.copy_(getattr(ours, src).weight)
        getattr(theirs, dst).bias.copy_(getattr(ours, src).bias)
    theirs.point_weights.copy_(ours.head_weights)

    g = torch.Generator().manual_seed(SEED + 2)
    n = TOKENS
    s = torch.randn(n, cfg.embed_dim, generator=g, dtype=DT)
    z = torch.randn(n, n, cfg.embed_dim, generator=g, dtype=DT)
    quat = torch.randn(n, 4, generator=g, dtype=DT)
    trans = torch.randn(n, 3, generator=g, dtype=DT) * 8.0
    affine = afr.QuatAffine(quat, trans)
    mask = torch.ones(n, dtype=DT)

    mine = ours(s.unsqueeze(0), z.unsqueeze(0), affine, mask.unsqueeze(0)).squeeze(0)
    ref = theirs(s, z, mask.unsqueeze(-1), affine)
    err = (mine - ref).abs().max().item()
    assert err < 1e-10, f"IPA differs from af2_reference by {err:.3e} in float64"


@torch.no_grad()
def test_batched_frame_functions_match_af2_reference():
    """The batched torsion and atom14 functions against the unbatched originals: exactly equal.

    Not approximately. The batched versions do the same operations in the same order with a sample
    axis in front, so anything above zero means an axis moved.
    """
    g = torch.Generator().manual_seed(SEED + 3)
    n = TOKENS
    aatype = torch.randint(0, 21, (n,), generator=g)
    quat = torch.randn(n, 4, generator=g, dtype=DT)
    trans = torch.randn(n, 3, generator=g, dtype=DT) * 8.0
    affine = afr.QuatAffine(quat, trans)
    angles = torch.randn(n, 7, 2, generator=g, dtype=DT)
    angles = angles / angles.square().sum(-1, keepdim=True).sqrt()

    rot, tr = torsion_angles_to_frames(aatype.unsqueeze(0), affine.rotation.unsqueeze(0),
                                       trans.unsqueeze(0), angles.unsqueeze(0))
    rot_r, tr_r = afr.torsion_angles_to_frames(aatype, affine.rotation, trans, angles)
    assert torch.equal(rot[0], rot_r)
    assert torch.equal(tr[0], tr_r)
    pos = frames_to_atom14_positions(aatype.unsqueeze(0), rot, tr)
    assert torch.equal(pos[0], afr.frames_to_atom14_positions(aatype, rot_r, tr_r))


def _as_golden_block(got: dict) -> str:
    """The replacement `GOLDEN` block, printed only on failure.

    On failure because the values are what the reader needs at exactly that moment, and only then:
    a pin that prints its own expected values on every green run trains whoever reads CI to paste
    them back without asking why they moved.
    """
    body = "".join(f'    "{k}": ({a!r}, {b!r}),\n' for k, (a, b) in got.items())
    return "GOLDEN = {\n" + body + "}"


@torch.no_grad()
def test_reference_digests_are_pinned():
    """Every output of a fixed-seed model on a fixed input, pinned. Proves nothing moved."""
    model = _fixed_model()
    out = model(*_fixed_inputs())
    got = _digests(out)
    block = _as_golden_block(got)
    assert set(got) == set(GOLDEN), f"output keys changed: {sorted(set(got) ^ set(GOLDEN))}"
    for key, (s, sq) in GOLDEN.items():
        assert got[key][0] == pytest.approx(s, rel=DIGEST_RTOL, abs=DIGEST_ATOL), \
            f"{key} sum moved: {got[key][0]!r} against {s!r}.\n{block}"
        assert got[key][1] == pytest.approx(sq, rel=DIGEST_RTOL, abs=DIGEST_ATOL), \
            f"{key} sum of squares moved: {got[key][1]!r} against {sq!r}.\n{block}"


@torch.no_grad()
def test_the_pin_actually_breaks():
    """The control: perturb a COPY OF THE SOURCE ON DISK and check the digests move.

    The perturbation is the pair-bias scale in the IPA, `sqrt(1/3)`, which is one of the three
    equal-variance factors Alg. 22 balances. It is scaled by 0.9 -- small enough that the model
    still runs and produces plausible structures, which is the kind of change a pin has to catch.

    On disk and not by monkeypatching, because the copy has to be *compiled* with the change: a
    constant bound as a default argument at def-time, or folded into a module-level table, does not
    move when a module attribute is rebound, and a control built that way passes while measuring
    nothing. The copy lands inside the package so its relative imports still resolve, and is
    removed in `finally`.
    """
    src_path = Path(afr.__file__).with_name("abodybuilder3_reference.py")
    src = src_path.read_text()
    needle = "logits = logits + math.sqrt(1.0 / 3) * self.linear_b(z).movedim(-1, -3)"
    assert needle in src, ("the control's perturbation site has moved; fix the control rather than "
                           "deleting it, or it silently stops testing anything")
    perturbed = src.replace(needle, needle.replace("math.sqrt(1.0 / 3)", "0.9 * math.sqrt(1.0 / 3)"))
    assert perturbed != src, "the perturbation did not apply, so this control proves nothing"

    copy_path = src_path.with_name("_abb3_reference_negative_control.py")
    try:
        copy_path.write_text(perturbed)
        module = importlib.import_module("tt_bio._abb3_reference_negative_control")
        importlib.reload(module)
        assert "0.9 * math.sqrt(1.0 / 3)" in Path(module.__file__).read_text(), \
            "the module under test is not the perturbed copy"
        out = _fixed_model(module)(*_fixed_inputs())
        got = _digests(out)
    finally:
        copy_path.unlink(missing_ok=True)
        cache = copy_path.parent / "__pycache__"
        for stale in cache.glob("_abb3_reference_negative_control.*"):
            stale.unlink(missing_ok=True)

    moved = [k for k in GOLDEN
             if got[k][0] != pytest.approx(GOLDEN[k][0], rel=DIGEST_RTOL, abs=DIGEST_ATOL)
             or got[k][1] != pytest.approx(GOLDEN[k][1], rel=DIGEST_RTOL, abs=DIGEST_ATOL)]
    assert set(moved) == set(GOLDEN), (
        f"a 10 % change to the IPA pair-bias scale left {sorted(set(GOLDEN) - set(moved))} "
        f"unchanged, so the pin does not cover those outputs")
