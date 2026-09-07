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


class TestDeviceNumbering:
    """UMD device index vs ``/dev/tenstorrent/N`` node number, on a box where they differ.

    The Galaxy ``UF-EV-A13-GWH02`` has 32 chips on four root complexes and the kernel
    driver probes them out of PCI order: nodes 0-7 are buses c1-c8, 8-15 are 81-88,
    16-23 are 01-08, 24-31 are 41-48. Measured 2026-09-02 by cross-checking all 32
    device-lease files against ``lsof`` on every node. The old ``tt_bdf_to_index`` read
    the node number as the UMD index, which is wrong for every one of the 32 cards, so
    an occupancy check on a node acted on a chip it never tested.
    """

    GALAXY_NODES = {
        **{n: f"0000:c{n + 1:x}:00.0" for n in range(8)},          # nodes 0-7   -> bus c1-c8
        **{n: f"0000:8{n - 7:x}:00.0" for n in range(8, 16)},      # nodes 8-15  -> bus 81-88
        **{n: f"0000:0{n - 15:x}:00.0" for n in range(16, 24)},    # nodes 16-23 -> bus 01-08
        **{n: f"0000:4{n - 23:x}:00.0" for n in range(24, 32)},    # nodes 24-31 -> bus 41-48
    }

    @pytest.fixture
    def galaxy(self, monkeypatch):
        monkeypatch.setattr(runtime, "tt_dev_node_bdfs", lambda: dict(self.GALAXY_NODES))

    def test_umd_index_is_bdf_rank_not_node_number(self, galaxy):
        by_bdf = runtime.tt_bdf_to_index()
        assert by_bdf["0000:01:00.0"] == 0     # node 16
        assert by_bdf["0000:08:00.0"] == 7     # node 23
        assert by_bdf["0000:41:00.0"] == 8     # node 24
        assert by_bdf["0000:81:00.0"] == 16    # node 8
        assert by_bdf["0000:c1:00.0"] == 24    # node 0

    def test_bdf_sort_is_numeric_not_lexicographic_on_the_bus(self, galaxy):
        """``c1`` must rank above ``81``: a naive int() on a hex bus would invert these."""
        by_bdf = runtime.tt_bdf_to_index()
        assert by_bdf["0000:c8:00.0"] == 31
        assert by_bdf["0000:88:00.0"] == 23

    def test_node_map_is_a_permutation_with_no_fixed_point(self, galaxy):
        to_node = runtime.umd_index_to_dev_node()
        assert sorted(to_node) == list(range(32))
        assert sorted(to_node.values()) == list(range(32))
        # The whole point: not one card's UMD index equals its node number here, so
        # reading one as the other is never even accidentally right.
        assert not [k for k, n in to_node.items() if k == n]
        assert to_node[0] == 16 and to_node[7] == 23 and to_node[16] == 8 and to_node[24] == 0

    def test_node_map_round_trips(self, galaxy):
        to_node = runtime.umd_index_to_dev_node()
        to_umd = runtime.dev_node_to_umd_index()
        assert all(to_umd[node] == idx for idx, node in to_node.items())

    def test_identity_box_still_maps_identically(self, monkeypatch):
        """A box probed in PCI order is unchanged, so single-root hosts keep their numbering."""
        monkeypatch.setattr(runtime, "tt_dev_node_bdfs",
                            lambda: {n: f"0000:0{n + 1:x}:00.0" for n in range(4)})
        assert runtime.umd_index_to_dev_node() == {0: 0, 1: 1, 2: 2, 3: 3}
