"""TT_VISIBLE_DEVICES accepts PCI BDFs, and the P300 1x1 mesh descriptor is lone-chip only.

Issue #11, the two smaller defects:

(a) ttnn's device open accepts a PCI BDF (``0000:01:00.0``) in TT_VISIBLE_DEVICES, but the
    predict path did ``int()`` on each token and died with
    ``ValueError: invalid literal for int()``. ``runtime.visible_device_indices`` now resolves
    a BDF to its UMD index through the tenstorrent sysfs class, and everything that reads the
    variable (device detection, the device lease's card naming, the P300 mesh rule) goes
    through it.

(b) ``ensure_p300_mesh_descriptor()`` forced the 1x1 P300 mesh-graph descriptor whenever a
    P300 chip existed on the host, so a process opening a whole P300 board pair failed with
    "Physical chip id 0 not found in control plane chip mapping". The descriptor is a
    lone-chip crutch (a single P300 is a custom topology ttnn cannot open bare), so it now
    applies only when exactly one chip is visible. Per-worker pinning is unaffected: each
    worker sees exactly one chip and still gets it.

Host-only — no device, no network. sysfs and the descriptor lookup are monkeypatched.
"""
from __future__ import annotations

import pytest

from tt_bio import device_lease, runtime
from tt_bio import main as tt_main

BDFS = {"0000:01:00.0": 0, "0000:02:00.0": 1, "0000:03:00.0": 2}
MGD = "/fake/p150_mesh_graph_descriptor.textproto"


@pytest.fixture
def bdf_map(monkeypatch):
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: dict(BDFS))


class TestVisibleDeviceIndices:
    def test_plain_indices(self, bdf_map):
        assert runtime.visible_device_indices("0,2") == [0, 2]

    def test_full_bdf(self, bdf_map):
        assert runtime.visible_device_indices("0000:02:00.0") == [1]

    def test_bdf_without_domain(self, bdf_map):
        assert runtime.visible_device_indices("01:00.0") == [0]

    def test_mixed_tokens(self, bdf_map):
        assert runtime.visible_device_indices("2,0000:01:00.0") == [2, 0]

    def test_unknown_bdf_names_the_index_form(self, bdf_map):
        with pytest.raises(ValueError, match="UMD index"):
            runtime.visible_device_indices("0000:09:00.0")

    def test_detect_devices_filters_on_a_bdf(self, monkeypatch, bdf_map):
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "0000:02:00.0")
        monkeypatch.setattr(runtime.glob, "glob",
                            lambda pat: ["/dev/tenstorrent/0", "/dev/tenstorrent/1",
                                         "/dev/tenstorrent/2"])
        assert runtime.detect_tenstorrent_devices(None, 0, 8) == [1]


class TestLeaseCardNaming:
    def test_bdf_visible_leases_the_resolved_index(self, monkeypatch, bdf_map):
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "0000:02:00.0")
        assert device_lease.physical_card() == "1"

    def test_logical_id_indexes_the_resolved_list(self, monkeypatch, bdf_map):
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "0000:02:00.0,0000:01:00.0")
        monkeypatch.setenv("TT_BIO_LOGICAL_DEVICE_ID", "1")
        assert device_lease.physical_card() == "0"


@pytest.fixture
def p300_host(monkeypatch):
    """A two-chip P300 host with a findable descriptor and no explicit override."""
    monkeypatch.setattr(tt_main, "_detect_p300_devices", lambda: [0, 1])
    monkeypatch.setattr(tt_main, "_find_ttnn_mesh_graph_descriptor", lambda name: MGD)
    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)


class TestP300MeshDescriptor:
    def test_lone_p300_gets_the_1x1(self, monkeypatch, p300_host):
        monkeypatch.setattr(tt_main, "_visible_tt_devices", lambda: [0])
        assert tt_main._p300_mesh_descriptor() == MGD

    def test_board_pair_gets_none(self, monkeypatch, p300_host):
        monkeypatch.setattr(tt_main, "_visible_tt_devices", lambda: [0, 1])
        assert tt_main._p300_mesh_descriptor() is None

    def test_lone_non_p300_chip_gets_none(self, monkeypatch, p300_host):
        monkeypatch.setattr(tt_main, "_visible_tt_devices", lambda: [7])
        assert tt_main._p300_mesh_descriptor() is None

    def test_explicit_device_asks_about_that_chip(self, p300_host):
        assert tt_main._p300_mesh_descriptor(device=1) == MGD

    def test_lone_p300_via_bdf_env(self, monkeypatch, p300_host, bdf_map):
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "0000:01:00.0")
        assert tt_main._p300_mesh_descriptor() == MGD

    def test_pair_via_env_gets_none(self, monkeypatch, p300_host, bdf_map):
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,0000:02:00.0")
        assert tt_main._p300_mesh_descriptor() is None

    def test_explicit_descriptor_env_wins(self, p300_host):
        env = {"TT_MESH_GRAPH_DESC_PATH": "/mine.textproto"}
        assert tt_main.ensure_p300_mesh_descriptor(env) is None
        assert env == {"TT_MESH_GRAPH_DESC_PATH": "/mine.textproto"}

    def test_workers_get_the_1x1_when_the_parent_sees_the_pair(self, monkeypatch, p300_host):
        """Per-worker pinning is the one-chip case by construction, MGD included."""
        monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
        assignments = tt_main._build_worker_device_assignments([0, 1])
        assert assignments[0]["mesh_graph_descriptor"] == MGD
        assert assignments[1]["mesh_graph_descriptor"] == MGD
