"""`tt-bio train --model rfd4 --dry-run` and `dryrun.plan(..., model='rfd4')`.

Verifies that RFD4 dry-run predictions come from real measured Blackhole hardware
entries (never UNMEASURED at crop 256, and never fabricated for unmeasured sizes or
architectures) and that impossible DRAM budgets or broken model names refuse loudly.
"""

from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner


BH_28_GIB_IN_GB = 28 * (1024 ** 3) / 1e9  # 30.064771072 GB


def test_rfd4_dry_run_cli_fits_without_importing_device_stack():
    """`tt-bio train --model rfd4 --dry-run` prints a real measured fit, no ttnn/torch."""
    from tt_bio.train.cli import train
    from tt_bio.train.dryrun import UNMEASURED

    before = set(sys.modules)
    res = CliRunner().invoke(train, ["--model", "rfd4", "--dry-run"])
    assert res.exit_code == 0, res.output
    new_mods = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not (new_mods & {"ttnn", "torch"}), f"dry-run imported {new_mods & {'ttnn', 'torch'}}"

    assert "plan: fits at 256 tokens on 1 chip(s)" in res.output, res.output
    assert UNMEASURED not in res.output, res.output
    assert "15.83 GB" in res.output and "2.00 s/step" in res.output, res.output
    assert "4e31c35c74" in res.output, res.output


def test_rfd4_plan_crop_256_fits_blackhole_28_gib_and_default_budget():
    """Crop 256 fits both BH 28 GiB and default allocator DRAM with measured timing."""
    from tt_bio.train.dryrun import UNMEASURED, needs_exact, plan

    for budget_gb in (28.0, BH_28_GIB_IN_GB):
        p = plan(tokens=256, model="rfd4", chips=1, frozen_trunk=False, card_gb=budget_gb)
        assert p.verdict == "fits" and p.verdict != UNMEASURED
        assert p.measured is True and p.fits is True
        assert abs(p.replica_gb - 15.833) < 1e-6
        assert abs(p.occupancy - 15.833 / budget_gb) < 1e-6
        assert p.occupancy < 1.0
        assert p.seconds_per_step == pytest.approx(2.00)
        assert p.sources

    accum = plan(tokens=256, model="rfd4", chips=1, global_batch=2, frozen_trunk=False)
    assert accum.verdict == "fits" and accum.seconds_per_step == pytest.approx(4.00)
    assert not needs_exact("rfd4", 256)
    assert not needs_exact("rfd4", 384)


@pytest.mark.parametrize("chips,want_step,want_peak_gb", [
    (4, 2.04, 19.409),
    (8, 1.91, 19.079),
    (16, 2.06, 18.322),
    (32, 2.58, 17.850),
])
def test_rfd4_plan_galaxy_spmd_widths_return_measured_numbers(chips, want_step, want_peak_gb):
    """Galaxy SPMD widths 4/8/16/32 return measured MeshTrace step time and per-chip DRAM."""
    from tt_bio.train.dryrun import plan

    p = plan(tokens=256, model="rfd4", chips=chips, frozen_trunk=False,
             card_gb=BH_28_GIB_IN_GB)
    assert p.verdict == "fits" and p.fits is True
    assert p.seconds_per_step == pytest.approx(want_step)
    assert p.replica_gb == pytest.approx(want_peak_gb)
    assert p.occupancy < 1.0
    assert any("24ad49b68a" in s for s in p.sources)


def test_rfd4_unmeasured_crop_arch_and_chip_width_are_honest():
    """Crops > 256, Wormhole, and unmeasured chip counts never fabricate numbers."""
    from tt_bio.train.dryrun import UNMEASURED, plan

    for big_crop in (384, 512):
        p = plan(tokens=big_crop, model="rfd4", chips=1, frozen_trunk=False)
        assert p.verdict == UNMEASURED
        assert p.measured is False and p.fits is None and p.seconds_per_step is None

    for wh in ("wh", "wormhole", "wormhole_b0"):
        p = plan(tokens=256, model="rfd4", chips=1, frozen_trunk=False, arch=wh)
        assert p.verdict == UNMEASURED
        assert p.measured is False and p.fits is None and p.seconds_per_step is None
        assert "wormhole" in p.why.lower()

    for unmeasured_chips in (2, 3, 64):
        p = plan(tokens=256, model="rfd4", chips=unmeasured_chips, frozen_trunk=False)
        assert p.verdict == "fits" and p.fits is True
        assert p.seconds_per_step is None
        assert UNMEASURED in p.why


def test_rfd4_every_table_entry_passes_provenance_audit():
    """Every RFD4 entry across all dry-run tables carries a valid openable citation."""
    from tt_bio.train.dryrun import (
        DEVICE_PATH_LARGEST_FIT,
        FORWARD_FITS_BY_MODEL,
        FORWARD_OOM_BY_MODEL,
        LARGEST_MEASURED_TO_FIT,
        TRAINED_TRUNK_BOUND_BY_MODEL,
        WEIGHTS_SPMD_STEP_S,
        WEIGHTS_STEP_S,
        provenance_gap,
    )

    assert "rfd4" in FORWARD_FITS_BY_MODEL
    assert "rfd4" in TRAINED_TRUNK_BOUND_BY_MODEL
    assert "rfd4" in FORWARD_OOM_BY_MODEL
    assert "rfd4" in DEVICE_PATH_LARGEST_FIT
    assert "rfd4" in LARGEST_MEASURED_TO_FIT
    assert "rfd4" in WEIGHTS_STEP_S
    assert "rfd4" in WEIGHTS_SPMD_STEP_S

    for entry in (
        *FORWARD_FITS_BY_MODEL["rfd4"].values(),
        *TRAINED_TRUNK_BOUND_BY_MODEL["rfd4"].values(),
        *WEIGHTS_STEP_S["rfd4"].values(),
        *WEIGHTS_SPMD_STEP_S["rfd4"][256].values(),
        DEVICE_PATH_LARGEST_FIT["rfd4"],
        LARGEST_MEASURED_TO_FIT["rfd4"],
    ):
        assert provenance_gap(entry[-1]) is None, entry[-1]

    # Negative control: missing or unopenable citation must be caught by provenance_gap.
    assert provenance_gap("") is not None
    assert provenance_gap("measured on a card once") is not None


def test_rfd4_negative_impossible_dram_budget_refuses_loudly(monkeypatch):
    """Impossible DRAM budget refuses explicitly in both Python API and CLI --dry-run."""
    from tt_bio.train import dryrun
    from tt_bio.train.cli import train

    # Single-chip RFD4 needs 15.833 GB; 8 GB and 12 GB must refuse.
    for tiny_gb in (8.0, 12.0):
        p = dryrun.plan(tokens=256, model="rfd4", chips=1, frozen_trunk=False, card_gb=tiny_gb)
        assert p.verdict == "refused"
        assert p.fits is False
        assert p.seconds_per_step is None
        assert p.replica_gb == pytest.approx(15.833)
        assert p.occupancy > 1.0
        assert "exceeds" in p.why

    # 32-chip SPMD needs 17.850 GB; 16 GB must refuse.
    p32 = dryrun.plan(tokens=256, model="rfd4", chips=32, frozen_trunk=False, card_gb=16.0)
    assert p32.verdict == "refused" and p32.fits is False

    # Non-positive DRAM budget or invalid arch raises ValueError.
    with pytest.raises(ValueError, match="card_gb must be positive"):
        dryrun.plan(tokens=256, model="rfd4", frozen_trunk=False, card_gb=0.0)
    with pytest.raises(ValueError, match="unknown arch"):
        dryrun.plan(tokens=256, model="rfd4", frozen_trunk=False, arch="hopper")

    # CLI exits non-zero when dry-run refuses on DRAM budget.
    real_plan = dryrun.plan
    monkeypatch.setattr(dryrun, "plan", lambda **kw: real_plan(**kw, card_gb=8.0))
    res = CliRunner().invoke(train, ["--model", "rfd4", "--dry-run"])
    assert res.exit_code != 0, res.output
    assert "refused" in res.output and "refusing to start" in res.output, res.output


def test_rfd4_negative_broken_model_name_refuses_in_cli_and_unmeasured_in_api():
    """A broken model name fails loudly on the CLI and never fabricates a fit in plan()."""
    from tt_bio.train.cli import train
    from tt_bio.train.dryrun import UNMEASURED, plan

    res = CliRunner().invoke(train, ["--model", "rfd4-nonexistent", "--dry-run"])
    assert res.exit_code == 2, res.output
    assert "Invalid value for '--model'" in res.output, res.output

    p = plan(tokens=256, model="rfd4-nonexistent", frozen_trunk=False)
    assert p.verdict == UNMEASURED and p.fits is None and p.seconds_per_step is None
