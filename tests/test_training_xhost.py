"""The cross-host transport, tested where it can fail: the frames and the protocol above them.

The real two-host run is measured in `perf/train_xhost/`; what belongs in a test is everything
that does not need a second box. `SshPeers(local=True)` runs the SAME receiver program over a
real pipe with ssh left out, so a malformed frame, a missing atomic rename or a retirement that
does not cross fails here instead of 40 minutes into a run.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tt_bio.train.hostreduce import HostReduce, master_hash
from tt_bio.train.xhost import SshPeers, parse_rendezvous


def test_plain_path_parses_to_itself_and_no_peers():
    d, peers = parse_rendezvous("/dev/shm/abb3-dp")
    assert d == Path("/dev/shm/abb3-dp") and peers == []


def test_peers_come_off_the_rendezvous_spec():
    d, peers = parse_rendezvous("/dev/shm/abb3-dp+ttuser@tt-quietbox2,ttuser@pc")
    assert d == Path("/dev/shm/abb3-dp")
    assert peers == ["ttuser@tt-quietbox2", "ttuser@pc"]


def test_single_host_reduce_opens_no_transport(tmp_path):
    comm = HostReduce(tmp_path / "rv", 0, 2)
    assert comm.transport is None and comm.peers == []


def _wait_for(path: Path, timeout: float = 30.0):
    t0 = time.monotonic()
    while not path.exists():
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.01)


def test_put_rm_and_rmtree_cross_the_channel(tmp_path):
    far = tmp_path / "far"
    peers = SshPeers(["stand-in"], far, local=True)
    try:
        blob = np.arange(1000, dtype=np.float32).tobytes()
        peers.put("grad-000000001/0.npy", blob)
        landed = far / "grad-000000001" / "0.npy"
        _wait_for(landed)
        assert landed.read_bytes() == blob
        # No temporary left behind: the reader polls for existence, so a `.xhost` file that
        # survived would be a half-written array waiting to be loaded as a whole one.
        assert not list(landed.parent.glob(".*"))
        peers.remove("grad-000000001/0.npy")
        t0 = time.monotonic()
        while landed.exists() and time.monotonic() - t0 < 30:
            time.sleep(0.01)
        assert not landed.exists()
        peers.put("hash-000000002/0.bin", b"digest")
        _wait_for(far / "hash-000000002" / "0.bin")
        peers.remove_tree()
        t0 = time.monotonic()
        while far.exists() and time.monotonic() - t0 < 30:
            time.sleep(0.01)
        assert not far.exists()
        assert peers.bytes_sent == len(blob) + len(b"digest")
    finally:
        peers.close()


def test_a_name_that_escapes_the_rendezvous_is_refused(tmp_path):
    far = tmp_path / "far"
    peers = SshPeers(["stand-in"], far, local=True)
    try:
        peers.put("../escaped.npy", b"x")
        t0 = time.monotonic()
        while peers.peers[0].proc.poll() is None and time.monotonic() - t0 < 30:
            time.sleep(0.01)
        assert peers.peers[0].proc.poll() == 1
        assert not (tmp_path / "escaped.npy").exists()
        with pytest.raises(RuntimeError, match="exited 1"):
            peers.check()
    finally:
        peers.close()


def test_two_disjoint_rendezvous_reduce_to_the_same_bits(tmp_path):
    """Two ranks, two separate directories, one sum: the cross-host arrangement without ssh.

    Each rank writes only into its OWN directory and sees the other's file only because the
    transport mirrors it, which is what two boxes do. A rank that quietly read the peer's
    directory would pass a shared-directory test and stall on a real second box.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    comms = {
        0: HostReduce(a, 0, 2, timeout=60.0, transport=SshPeers(["to-b"], b, local=True)),
        1: HostReduce(b, 1, 2, timeout=60.0, transport=SshPeers(["to-a"], a, local=True)),
    }
    results: dict = {}

    def one(rank):
        comm = comms[rank]
        for step in (1, 2, 3):
            vec = np.arange(4096, dtype=np.float32) * (rank + 1) + step
            total = comm.allreduce(vec, step=step)
            digest = master_hash([total])
            comm.check_equal(digest, step=step)
            results.setdefault(rank, []).append(digest.hex())

    threads = [threading.Thread(target=one, args=(r,)) for r in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
        assert not th.is_alive(), "a rank is still waiting: the mirror never delivered"
    assert results[0] == results[1], "the two ranks summed to different bits"
    base = np.arange(4096, dtype=np.float32)
    assert results[0][0] == master_hash([(base + 1) + (base * 2 + 1)]).hex()
    for step in (1, 2):
        # Retirement crosses the wire too, or every peer accumulates a gradient per step.
        for d in (a / f"grad-{step:09d}", b / f"grad-{step:09d}"):
            assert not d.exists() or not list(d.iterdir()), f"{d} was not retired"
    for comm in comms.values():
        comm.transport.close()
