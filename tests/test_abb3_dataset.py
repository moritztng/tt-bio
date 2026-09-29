"""The real-split micro-batch assembler, gated without a card and without the dataset staged.

The interesting failures here are the ones a mask cannot save you from. A padded residue is
masked out of every stage-1 loss, so it is tempting to pad everything with zeros -- but
``0 * NaN`` is ``NaN``, so any degenerate value computed BEFORE the mask is applied survives it.
Two fields have that property and both are checked below: the 4x4 rigid frames, which FAPE
inverts, and the chi sin/cos targets, which get normalised.

The fixtures are built here rather than read from disk so the gate runs anywhere, and the same
checks are re-run against the real staged structures when they are present. A gate that skips
wherever the data is absent is a gate that never runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tt_bio.train.abb3_dataset import (SabdabFvs, UNKNOWN_AATYPE, bucket_tokens,
                                       resolve_split)

#: A stand-in card. Passing one keeps these tests off the hardware; the laziness of the REAL
#: resolution is checked separately below, without opening anything.
_STUB = object()

REAL_STRUCTURES = Path("/home/ttuser/abb3_data/data/structures/structures")
REAL_SPLIT = Path("/home/ttuser/abb3_src/ABodyBuilder3/data/split.csv")
REAL_TRUE = Path("/home/ttuser/abb3/base-loss/true")


def _fake_structure(n: int, seed: int = 0) -> dict:
    """Upstream's own field set at their shapes, with plausible values.

    Copied from what a real ``structures/*.pt`` holds rather than from what the losses happen to
    read today, so a loss that starts reading another field fails here instead of at step 1 of a
    multi-day run.
    """
    g = torch.Generator().manual_seed(seed)

    def frames(*shape):
        out = torch.zeros(*shape, 4, 4)
        out[..., 0, 0] = out[..., 1, 1] = out[..., 2, 2] = out[..., 3, 3] = 1.0
        out[..., :3, 3] = torch.randn(*shape, 3, generator=g)
        return out

    chi = torch.randn(n, 4, 2, generator=g)
    return {
        "aatype": torch.randint(0, 20, (n,), generator=g),
        "seq_mask": torch.ones(n),
        "is_heavy": (torch.arange(n) < n // 2).float(),
        "residue_index": torch.arange(n),
        "chi_mask": (torch.rand(n, 4, generator=g) > 0.3).float(),
        "chi_angles_sin_cos": chi / chi.norm(dim=-1, keepdim=True),
        "backbone_rigid_tensor": frames(n),
        "backbone_rigid_mask": torch.ones(n),
        "rigidgroups_gt_frames": frames(n, 8),
        "rigidgroups_alt_gt_frames": frames(n, 8),
        "rigidgroups_gt_exists": (torch.rand(n, 8, generator=g) > 0.25).float(),
        "atom14_gt_positions": torch.randn(n, 14, 3, generator=g),
        "atom14_alt_gt_positions": torch.randn(n, 14, 3, generator=g),
        "atom14_gt_exists": torch.ones(n, 14),
        "atom14_alt_gt_exists": torch.ones(n, 14),
        "atom14_atom_is_ambiguous": (torch.rand(n, 14, generator=g) > 0.9).float(),
        "atom14_atom_exists": torch.ones(n, 14),
        "cdr_mask": (torch.arange(n) >= n // 3).long(),
        "resolution": 1.9,
    }


@pytest.fixture
def staged(tmp_path):
    """Three structures of different lengths, so padding is actually exercised."""
    ids = []
    for i, n in enumerate((227, 241, 209)):
        name = f"fake{i}_H0-L0"
        torch.save(_fake_structure(n, seed=i), tmp_path / f"{name}.pt")
        ids.append(name)
    return ids, tmp_path


def _cfg():
    from tt_bio.abodybuilder3_reference import ABB3Config
    return ABB3Config(use_plddt=False, no_blocks=8)


def _no_device(monkeypatch):
    """Keep the uploads on the host, so the assembler is gated without a card.

    The two input one-hots are expanded on the card by `abb3_features_device`; here they fall
    back to the untouched reference, which is also what that module is checked against in
    `test_abb3_features_device.py`. So this stand-in is the thing the device path has to equal,
    not a second approximation of it.
    """
    import tt_bio.abodybuilder3 as A
    import tt_bio.train.abb3_features_device as F
    from tt_bio.abodybuilder3_reference import single_and_pair_features
    monkeypatch.setattr(A, "to_device_fp32", lambda t: t)
    monkeypatch.setattr(F, "input_features_device",
                        lambda aatype, is_heavy, residue_index, *, device, **kw:
                        single_and_pair_features(aatype, is_heavy, residue_index))


def test_the_assembler_produces_exactly_the_contract_the_step_reads(staged, monkeypatch):
    """The field set is compared against `step_gate.synthetic_micro_batch`, which is the
    contract B2's step actually reads. A missing target is a KeyError at step 1 of a run."""
    _no_device(monkeypatch)
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / "scripts" / "abb3_port")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from step_gate import synthetic_micro_batch

    ids, root = staged
    cfg = _cfg()
    mb = SabdabFvs(ids, root, cfg, device=_STUB).batch([0, 1, 2])
    syn = synthetic_micro_batch(cfg, 3, mb["tokens"], 0, None)

    assert set(mb["targets"]) == set(syn["targets"]), (
        f"missing {sorted(set(syn['targets']) - set(mb['targets']))}, "
        f"extra {sorted(set(mb['targets']) - set(syn['targets']))}")
    mismatched = {k: (tuple(mb["targets"][k].shape), tuple(v.shape))
                  for k, v in syn["targets"].items()
                  if tuple(mb["targets"][k].shape) != tuple(v.shape)}
    assert not mismatched, mismatched
    for key in ("single_d", "pair_d", "square_d", "bias_d", "aatype"):
        assert key in mb


def test_the_token_axis_is_a_whole_number_of_tiles(staged, monkeypatch):
    _no_device(monkeypatch)
    ids, root = staged
    mb = SabdabFvs(ids, root, _cfg(), device=_STUB).batch([0, 1, 2])
    assert mb["tokens"] % 32 == 0
    assert mb["tokens"] == 256, "241 is the longest of the three, so the bucket is 256"
    assert bucket_tokens(1) == 32 and bucket_tokens(256) == 256 and bucket_tokens(257) == 288


def test_padding_is_masked_and_the_two_degenerate_pads_are_not_zero(staged, monkeypatch):
    """`0 * NaN` is `NaN`, so a mask cannot remove a degenerate value already computed.

    Frames are inverted by FAPE before the mask lands, and the chi targets are normalised. A
    zero 3x3 block is not a rotation and a zero 2-vector normalises to NaN, so those two pad to
    the identity and to a unit vector. This is the check that would catch a well-meaning
    simplification to `torch.zeros`.
    """
    _no_device(monkeypatch)
    ids, root = staged
    ds = SabdabFvs(ids, root, _cfg(), device=_STUB)
    mb = ds.batch([0, 1, 2])
    t, n_tok = mb["targets"], mb["tokens"]
    for j, n in enumerate((227, 241, 209)):
        assert t["seq_mask"][j, :n].all() and not t["seq_mask"][j, n:].any()
        assert t["aatype"][j, n:].eq(UNKNOWN_AATYPE).all()
        for key in ("atom14_gt_exists", "chi_mask", "rigidgroups_gt_exists",
                    "backbone_rigid_mask"):
            assert not t[key][j, n:].any(), f"{key} is not zero on padding"
        eye = torch.eye(4)
        for key in ("backbone_rigid_tensor", "rigidgroups_gt_frames",
                    "rigidgroups_alt_gt_frames"):
            pad = t[key][j, n:]
            assert torch.allclose(pad, eye.expand_as(pad)), f"{key} pads to something that is "\
                f"not a rotation; FAPE inverts it before the mask is applied"
        chi = t["chi_angles_sin_cos"][j, n:]
        assert torch.allclose(chi.norm(dim=-1), torch.ones(chi.shape[:-1])), \
            "chi targets pad to a non-unit vector, which normalises to NaN"
    assert not any(torch.isnan(v).any() for v in t.values() if torch.is_floating_point(v))
    assert n_tok == 256


def test_an_fv_longer_than_a_fixed_axis_is_refused_rather_than_cropped(staged, monkeypatch):
    """Cropping would be a different experiment than the one being reproduced."""
    _no_device(monkeypatch)
    ids, root = staged
    ds = SabdabFvs(ids, root, _cfg(), device=None, tokens=128)
    with pytest.raises(ValueError, match="different experiment"):
        ds.batch([0])


def test_a_missing_structure_is_caught_at_construction_not_at_step_one(staged, tmp_path):
    ids, root = staged
    with pytest.raises(FileNotFoundError, match="no .pt"):
        SabdabFvs(ids + ["nope_H0-L0"], root, _cfg(), device=_STUB)


def test_device_resolves_lazily_so_a_dp_driver_can_build_a_dataset_holding_no_card(staged):
    """Constructing a dataset must open nothing. A contract term, not tidiness.

    Data parallelism here is one process per chip, so the driver that spawns the ranks has to
    reach the launcher holding no card. A dataset that resolved a device in its constructor
    would take a chip in the driver and the launcher would refuse to start.

    Checked without a card and without ``hasattr``, which CALLS a property to find out whether
    it exists -- the exact move ``catalogue.load`` avoids for this reason.
    """
    from tt_bio.train import catalogue
    from tt_bio.train.abb3_dataset import SabdabFvs, SyntheticFvs, _LazyDevice

    ids, root = staged
    for ds in (SabdabFvs(ids, root, _cfg()), SyntheticFvs(8, _cfg(), 4, 256)):
        assert isinstance(type(ds).__dict__["device"], _LazyDevice)
        assert vars(ds).get("_device") is None, "the constructor resolved a device"
        # Member for member, what catalogue.load itself does.
        missing = [m for m in catalogue.REQUIRED_DATASET_MEMBERS
                   if not (hasattr(type(ds), m) or m in vars(ds))]
        assert not missing, f"{type(ds).__name__} is missing {missing}"
        assert vars(ds).get("_device") is None, "the contract check itself opened a device"
        ds.device = _STUB
        assert ds.device is _STUB


def test_abodybuilder3_is_registered_so_tier_0_can_reach_it():
    """`tt-bio finetune --model abodybuilder3` resolves without importing the run loop."""
    from tt_bio.train import abb3_dataset, catalogue
    abb3_dataset.register()
    assert "abodybuilder3" in catalogue.names()
    abb3_dataset.register()          # idempotent: registering twice is not an error
    assert catalogue.names().count("abodybuilder3") == 1


def test_the_adapter_refuses_a_root_with_no_structures_and_names_where_to_get_them(tmp_path):
    from tt_bio.train.abb3_dataset import abodybuilder3_adapter
    with pytest.raises(NotImplementedError, match="structures"):
        abodybuilder3_adapter(tmp_path)


@pytest.mark.skipif(not (REAL_STRUCTURES.is_dir() and REAL_SPLIT.is_file()),
                    reason="the SAbDab structures are not staged on this host")
def test_the_same_checks_hold_on_the_real_staged_split(monkeypatch):
    """The fixtures above prove the code; this proves the code against upstream's own files."""
    _no_device(monkeypatch)
    ids = resolve_split(REAL_SPLIT, REAL_TRUE if REAL_TRUE.is_dir() else None)["train"]
    assert len(ids) == 8395
    mb = SabdabFvs(ids, REAL_STRUCTURES, _cfg(), device=_STUB).batch([0, 1, 2, 3])
    assert mb["tokens"] % 32 == 0
    t = mb["targets"]
    assert not any(torch.isnan(v).any() for v in t.values() if torch.is_floating_point(v))
    n0 = int(t["seq_mask"][0].sum())
    assert 150 < n0 <= mb["tokens"]
    assert not t["seq_mask"][0, n0:].any()
