"""A chip that enumerates with no device node behind it.

The shape a wedged or half-attached card takes: the tenstorrent sysfs class still lists it, so
every index check passes, and the user gets a UMD throw that blames their TT_VISIBLE_DEVICES.
Reproduced on `.108` over a tmpfs `/dev/tenstorrent` in a private mount namespace, 2026-09-30.
"""
import pytest

from tt_bio import runtime


@pytest.fixture
def one_chip(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "umd_index_to_dev_node", lambda: {0: 16, 1: 17})
    return tmp_path


def test_a_chip_whose_node_is_gone_is_named_with_its_node(monkeypatch, one_chip):
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: p != "/dev/tenstorrent/16")
    assert runtime.missing_device_nodes([0]) == [(0, 16)]
    assert runtime.missing_device_nodes([1]) == []


def test_the_refusal_says_the_chip_is_missing_and_not_the_setting(monkeypatch, one_chip):
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: p != "/dev/tenstorrent/16")
    with pytest.raises(RuntimeError) as caught:
        runtime.refuse_missing_device_nodes([0])
    said = str(caught.value)
    assert "chip 0 (/dev/tenstorrent/16)" in said
    assert "not your TT_VISIBLE_DEVICES" in said
    assert "ls /dev/tenstorrent" in said and "tt-smi" in said and "dmesg" in said


def test_a_present_node_is_not_refused(monkeypatch, one_chip):
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: True)
    assert runtime.missing_device_nodes([0, 1]) == []
    assert runtime.refuse_missing_device_nodes([0, 1]) is None


def test_every_requested_chip_is_named_not_just_the_first(monkeypatch, one_chip):
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: False)
    with pytest.raises(RuntimeError) as caught:
        runtime.refuse_missing_device_nodes([0, 1])
    assert "chip 0 (/dev/tenstorrent/16)" in str(caught.value)
    assert "chip 1 (/dev/tenstorrent/17)" in str(caught.value)


def test_a_host_with_no_tenstorrent_sysfs_checks_nothing(monkeypatch):
    # Every CPU run: TT_VISIBLE_DEVICES may be set and there is nothing to refuse.
    monkeypatch.setattr(runtime, "umd_index_to_dev_node", lambda: {})
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: False)
    assert runtime.missing_device_nodes([0, 7]) == []
    assert runtime.refuse_missing_device_nodes([0, 7]) is None


def test_an_index_sysfs_does_not_map_is_left_to_the_index_check(monkeypatch, one_chip):
    # visible_device_indices() already refuses an out-of-range chip with its own sentence;
    # this must not shadow that with a node message about a chip that was never listed.
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: False)
    assert runtime.missing_device_nodes([99]) == []


def test_a_node_held_by_another_tenant_is_not_a_fault(monkeypatch, one_chip):
    # Existence only: this never opens the node, so a chip another row is folding on cannot
    # be refused here as broken.
    opened = []
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: True)
    monkeypatch.setattr(runtime.os, "open", lambda *a, **k: opened.append(a) or 3)
    assert runtime.missing_device_nodes([0, 1]) == []
    assert opened == []
