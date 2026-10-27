"""`design --devices N` opens only card N for the in-process design models (issue #26).

`tt-bio design --model rfd4 --devices 1` printed "rfd4 does not use --devices; ignoring" and
then opened all four /dev/tenstorrent nodes of a 4-card Blackhole host: get_device() opens and
leases every VISIBLE chip, and nothing narrowed the visible set. rfd3's single-card path chose
the compute chip with TT_BIO_LOGICAL_DEVICE_ID, which left the same four chips visible.
PXDesign dropped --devices the same way. All three now pin TT_VISIBLE_DEVICES to the requested
card before anything opens, so an external scheduler can run one process per card.

Host-only: the cards are a faked /dev listing and the model run is replaced by a recorder.
"""
from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from tt_bio import main as tt_main
from tt_bio import runtime

FOUR_CARDS = [f"/dev/tenstorrent/{i}" for i in range(4)]


@pytest.fixture
def four_card_host(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob", lambda pat: list(FOUR_CARDS))
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: {})
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)


@pytest.fixture
def rfd4_run(monkeypatch, tmp_path):
    seen = {}

    def record(*a, **k):
        seen["visible"] = os.environ.get("TT_VISIBLE_DEVICES")
    monkeypatch.setattr(tt_main, "_run_rfd4_cli", record)
    spec = tmp_path / "in.cif"
    spec.write_text("data_x\n")

    def run(*extra):
        res = CliRunner().invoke(tt_main.cli, ["design", str(spec), "--model", "rfd4",
                                               "--out_dir", str(tmp_path / "o"), *extra])
        return res, seen
    return run


def test_rfd4_devices_pins_the_one_card(four_card_host, rfd4_run):
    res, seen = rfd4_run("--devices", "1")
    assert res.exit_code == 0, res.output
    assert seen["visible"] == "1"
    assert "does not use --devices" not in res.output


def test_rfd4_without_devices_is_unchanged(four_card_host, rfd4_run):
    res, seen = rfd4_run()
    assert res.exit_code == 0, res.output
    assert seen["visible"] is None


def test_rfd4_refuses_a_card_list_before_opening_anything(four_card_host, rfd4_run):
    res, seen = rfd4_run("--devices", "0,1")
    assert res.exit_code == 2 and "one card per process" in res.output
    assert seen == {}


def test_rfd4_refuses_an_absent_card(four_card_host, rfd4_run):
    res, seen = rfd4_run("--devices", "7")
    assert res.exit_code == 2 and "not available" in res.output
    assert seen == {}


def test_rfd4_devices_narrows_an_ambient_visible_set(four_card_host, rfd4_run, monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "1,2")
    res, seen = rfd4_run("--devices", "2")
    assert res.exit_code == 0, res.output
    assert seen["visible"] == "2"


def test_pxdesign_devices_pins_the_one_card(four_card_host, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(tt_main, "_run_pxdesign_cli",
                        lambda *a, **k: seen.setdefault("visible", os.environ.get("TT_VISIBLE_DEVICES")))
    target = tmp_path / "t.yaml"
    target.write_text("x: 1\n")
    res = CliRunner().invoke(tt_main.cli, ["design", str(target), "--model", "pxdesign",
                                           "--devices", "3"])
    assert res.exit_code == 0, res.output
    assert seen["visible"] == "3"


def test_rfd3_single_card_pins_visibility_not_the_logical_id(four_card_host, monkeypatch):
    from tt_bio.rfd3 import design as rfd3

    seen = {}

    def record(*a, **k):
        seen["visible"] = os.environ.get("TT_VISIBLE_DEVICES")
        seen["logical"] = os.environ.get("TT_BIO_LOGICAL_DEVICE_ID")
        return []
    monkeypatch.setattr(rfd3, "_run_design_jobs", record)
    monkeypatch.setattr(rfd3.InputSpecification, "from_dict", classmethod(lambda cls, raw: _Spec()))
    rfd3.run_design({"a": {}}, "/tmp/unused", checkpoint_dir="x", devices=[2])
    assert seen == {"visible": "2", "logical": None}


class _Spec:
    def validate(self):
        pass
