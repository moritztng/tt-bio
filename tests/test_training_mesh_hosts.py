"""The host axis of ``Mesh``, tested where the original defect was: at the entry point.

`train-j-multihost` proved two boxes train one model and left `tt_bio/train/mesh.py` raising
``NotImplementedError`` on ``hosts != 1``, so the capability existed and nothing a user could
type reached it. A test that only checked the raise was gone would close the same tier boundary
one step further along and be blind to the next one, so the tests here go through to the
transport: a mesh composes the rendezvous spec, ``HostReduce`` parses its peers off that spec,
and two ranks in two separate directories reduce to the same bits over it.

The transport is `SshPeers(local=True)`, the same receiver program over a pipe with ssh left
out, which is how `tests/test_training_xhost.py` tests the frames. Host-only, no ttnn, no card.
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import numpy as np
import pytest

from tt_bio.train.hostreduce import HostReduce, master_hash
from tt_bio.train.mesh import MAX_CHIPS_ONE_BOX, Mesh
from tt_bio.train.xhost import SshPeers, parse_rendezvous

ROOT = Path(__file__).resolve().parent.parent
#: The spec `train-j-multihost` measured its cross-host run on, verbatim from its state doc and
#: from `perf/train_xhost/`. The point of quoting it literally is that the mesh has to compose
#: THAT string and not a plausible variant of it.
J_SPEC = "/dev/shm/abb3-dp+ttuser@tt-quietbox2"


# --------------------------------------------------------------- the axis exists

def test_a_mesh_across_hosts_constructs_and_names_its_peers():
    mesh = Mesh({"dp": [0, 1]}, hosts=["ttuser@tt-quietbox2"])
    assert mesh.hosts == 2
    assert mesh.peers == ("ttuser@tt-quietbox2",)
    assert mesh.axis("dp").width == 2
    assert "ttuser@tt-quietbox2" in str(mesh)


def test_a_bare_host_count_constructs_and_says_what_is_missing_at_the_rendezvous():
    """``hosts=2`` with nobody named is honest, not an error. Naming is owed at the spec.

    Refusing to construct is the defect this row exists to remove, and moving the refusal from
    ``hosts != 1`` to ``peers == ()`` would be the same refusal with a longer reach.
    """
    mesh = Mesh({"dp": [0]}, hosts=2)
    assert mesh.hosts == 2 and mesh.peers == ()
    with pytest.raises(ValueError, match="none of them is named"):
        mesh.rendezvous("/dev/shm/abb3-dp")


def test_one_host_gives_back_the_bare_path():
    assert Mesh({"dp": [0, 1]}).rendezvous("/dev/shm/abb3-dp") == "/dev/shm/abb3-dp"
    assert Mesh({"dp": [0]}).hosts == 1


def test_the_chip_cap_is_per_box_and_the_host_count_multiplies_it():
    assert Mesh({"dp": list(range(8))}, hosts=["a@b"]).chips == 2 * MAX_CHIPS_ONE_BOX
    with pytest.raises(ValueError, match="beyond 8"):
        Mesh({"dp": list(range(9))}, hosts=["a@b"])
    with pytest.raises(ValueError, match="beyond 4"):
        Mesh({"dp": list(range(5))})


@pytest.mark.parametrize("peer", ["", "  ", "ttuser@a,ttuser@b", "ttuser@a+b", "tt user@a"])
def test_a_destination_that_would_corrupt_the_spec_is_refused_where_it_is_given(peer):
    """The two separators are the format. A peer carrying one parses back as a different list.

    Refused at construction rather than at ``rendezvous``, because the far side is where it
    would otherwise show up: as a rank waiting forever for a file nobody is sending.
    """
    with pytest.raises(ValueError, match="not an ssh destination"):
        Mesh({"dp": [0, 1]}, hosts=[peer])


def test_a_repeated_peer_is_refused():
    with pytest.raises(ValueError, match="repeats a peer"):
        Mesh({"dp": [0, 1, 2]}, hosts=["a@b", "a@b"])


# --------------------------------------------------------------- one transport, one protocol

def test_the_spec_a_mesh_composes_is_the_one_the_transport_parses():
    mesh = Mesh({"dp": [0, 1]}, hosts=["ttuser@tt-quietbox2"])
    assert mesh.rendezvous("/dev/shm/abb3-dp") == J_SPEC
    assert parse_rendezvous(mesh.rendezvous("/dev/shm/abb3-dp")) == (
        Path("/dev/shm/abb3-dp"), list(mesh.peers))

    three = Mesh({"dp": [0, 1, 2]}, hosts=["ttuser@tt-quietbox2", "ttuser@pc"])
    assert three.hosts == 3
    assert parse_rendezvous(three.rendezvous(Path("/dev/shm/abb3-dp"))) == (
        Path("/dev/shm/abb3-dp"), list(three.peers))


def test_the_mesh_composes_the_same_spec_as_the_gate_that_measured_the_run():
    """Two independent composers, one string. The evidence that there is no second mechanism.

    `scripts/train_xhost/xhost_gate.py` built the spec for every rank of the measured two-host
    run. If the mesh's version drifts from it, one of them is a second protocol and the run's
    numbers stop describing what a user gets.
    """
    spec = importlib.util.spec_from_file_location(
        "_xhost_gate_for_test", ROOT / "scripts" / "train_xhost" / "xhost_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    hosts = ["tt-quietbox", "tt-quietbox2"]
    for me, peers in (("tt-quietbox", ["ttuser@tt-quietbox2"]),
                      ("tt-quietbox2", ["ttuser@tt-quietbox"])):
        theirs = gate.rendezvous_for(me, hosts, "/dev/shm/abb3-dp", "ttuser")
        ours = Mesh({"dp": [0, 1]}, hosts=peers).rendezvous("/dev/shm/abb3-dp")
        assert ours == theirs, f"on {me}: mesh says {ours!r}, the gate says {theirs!r}"


def test_the_mesh_adds_no_transport_of_its_own():
    """`mesh.py` may reach the wire only through `xhost.py`, and must not restate its format.

    A second copy of the separator is how one protocol becomes two: the copy stays behind when
    the original changes, and the failure lands on the far host in the middle of a run.
    """
    src = (ROOT / "tt_bio" / "train" / "mesh.py").read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from ")) and "typing" not in ln]
    assert any("from .xhost import" in ln for ln in imports), (
        "mesh.py no longer takes the rendezvous format from xhost.py, so the two can disagree")
    for forbidden in ("subprocess", "socket", "paramiko", "ttnn.distributed"):
        assert not any(forbidden in ln for ln in imports), (
            f"mesh.py imports {forbidden}: the host axis is a rendezvous string, and anything "
            f"that opens a connection here is a second transport beside xhost.py")
    # The literal the format is made of appears once, where it is defined and explained.
    assert src.count('_PEER_LIST_SEP = ","') == 1
    assert src.count('"+"') == 0, "mesh.py restates PEER_SEP instead of importing it"


# --------------------------------------------------------------- reaching it end to end

def test_two_ranks_reduce_across_a_mesh_composed_rendezvous(tmp_path):
    """The capability, reached the way a user reaches it: build a mesh, hand over its spec.

    Each rank gets its OWN directory and sees the other's file only because the transport
    mirrors it, which is what two boxes do. Nothing here knows the peer list except the string
    the mesh composed, so the assertion at the end is that the axis reaches J's transport.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    # qb1's mesh names qb2 and qb2's names qb1, exactly as each host's own spec does.
    meshes = {0: Mesh({"dp": [0, 1]}, hosts=["stand-in-b"]),
              1: Mesh({"dp": [0, 1]}, hosts=["stand-in-a"])}
    specs = {0: meshes[0].rendezvous(a), 1: meshes[1].rendezvous(b)}
    comms = {
        0: HostReduce(specs[0], 0, 2, timeout=60.0,
                      transport=SshPeers(["stand-in-b"], b, local=True)),
        1: HostReduce(specs[1], 1, 2, timeout=60.0,
                      transport=SshPeers(["stand-in-a"], a, local=True)),
    }
    # The peers HostReduce will use came off the mesh's string, not out of the test.
    for rank, comm in comms.items():
        assert comm.peers == list(meshes[rank].peers)
    assert comms[0].dir == a and comms[1].dir == b

    digests: dict = {}

    def one(rank):
        comm = comms[rank]
        for step in (1, 2, 3):
            vec = np.arange(2048, dtype=np.float32) * (rank + 1) + step
            total = comm.allreduce(vec, step=step)
            digest = master_hash([total])
            comm.check_equal(digest, step=step)
            digests.setdefault(rank, []).append(digest.hex())

    threads = [threading.Thread(target=one, args=(r,)) for r in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
        assert not th.is_alive(), "a rank is still waiting: the mesh's peers never reached the "\
                                  "transport, so nothing mirrored"
    try:
        assert digests[0] == digests[1], "the two ranks summed to different bits"
        base = np.arange(2048, dtype=np.float32)
        assert digests[0][0] == master_hash([(base + 1) + (base * 2 + 1)]).hex()
    finally:
        for comm in comms.values():
            comm.transport.close()


def test_the_control_a_mesh_with_no_peers_mirrors_nothing(tmp_path):
    """Negative control: without the peers the same arrangement must FAIL to reduce.

    Without this the test above could pass because the two ranks found each other some other
    way -- a shared directory, a stale file -- rather than because the mesh's peers reached the
    transport. One host, one spec, no mirror, so rank 0 waits and times out.
    """
    a = tmp_path / "a"
    a.mkdir()
    spec = Mesh({"dp": [0, 1]}).rendezvous(a)
    assert spec == str(a)
    comm = HostReduce(spec, 0, 2, timeout=1.0, poll=0.01)
    assert comm.peers == [] and comm.transport is None
    with pytest.raises(TimeoutError):
        comm.allreduce(np.arange(16, dtype=np.float32), step=1)


# --------------------------------------------------------------- the numbers around it

#: The cross-host record `train-j-multihost` wrote. The docs quote it, so it is the thing to
#: check them against: the 4-step arm in the same file was the settling link and was superseded
#: by the 12-step steady state below it, and the shipped doc quoted the settling arm for a day.
RECORD = ROOT / "perf" / "train_xhost" / "scaling_qb1_qb2.txt"
SETTLING = ("14.096", "1.493x", "74.6 %")
STEADY = ("13.545", "1.554x", "77.7 %")
DOCS = ("README.md", "docs/training.md")


@pytest.mark.skipif(not RECORD.is_file(), reason="the cross-host record is not on this tree")
def test_the_record_still_holds_both_arms_so_this_gate_knows_which_is_which():
    """Negative control: the check below is only meaningful while both numbers exist.

    If the record ever stops carrying the settling arm, "the docs do not quote 14.096" becomes
    true for free and the gate goes quiet without anyone deciding it should.
    """
    text = RECORD.read_text()
    for n in SETTLING + STEADY:
        assert n.split()[0] in text, f"{n} is not in {RECORD.name} any more; re-cut this gate"
    assert "median of steps 2..12: 13.545 s" in text


@pytest.mark.skipif(not RECORD.is_file(), reason="the cross-host record is not on this tree")
@pytest.mark.parametrize("doc", DOCS)
def test_the_docs_quote_the_steady_state_and_not_the_settling_arm(doc):
    text = (ROOT / doc).read_text()
    if "QuietBoxes" not in text and "Multi-host" not in text:
        pytest.skip(f"{doc} makes no cross-host claim")
    quoted = [n for n in STEADY if n in text]
    assert quoted, (
        f"{doc} makes a cross-host claim and quotes none of the measured steady-state figures "
        f"{STEADY} from {RECORD.name}")
    stale = [n for n in SETTLING if n in text]
    assert not stale, (
        f"{doc} quotes {stale}, which is the 4-step settling arm. `train-j-multihost` "
        f"superseded it in the same record: the steady state is the median of steps 2-12, "
        f"{STEADY[0]} s, {STEADY[1]} at {STEADY[2]}")


@pytest.mark.parametrize("doc", DOCS)
def test_the_docs_say_the_wifi_efficiency_is_a_floor(doc):
    """77.7 % read as the fleet's ceiling is the misreading this caveat exists to prevent.

    The link is the whole gap and qb2 has no cable, so a reader with wired hosts should expect
    much closer to the 94.3 % the same two chips give in one box.
    """
    text = (ROOT / doc).read_text()
    if "WiFi" not in text:
        pytest.skip(f"{doc} makes no cross-host claim")
    assert "floor" in text or "no cable" in text, (
        f"{doc} quotes the cross-host efficiency without saying it is close to a floor. "
        f"Measured over WiFi at 27.8-32.5 MB/s; the same 28.4 MB at 1000 Mb/s is ~0.24 s each "
        f"way, near 2 % of the step instead of 13-18 %")


def test_the_doc_snippet_is_the_api_that_shipped():
    """The exact call `docs/training.md` prints, executed. A snippet nobody runs rots."""
    text = (ROOT / "docs" / "training.md").read_text()
    snippet = 'train.Mesh({"dp": [0, 1]}, hosts=["ttuser@tt-quietbox2"])'
    assert snippet in text, "docs/training.md no longer shows the multi-host mesh call"
    assert 'mesh.rendezvous("/dev/shm/abb3-dp")' in text
    assert Mesh({"dp": [0, 1]}, hosts=["ttuser@tt-quietbox2"]).rendezvous(
        "/dev/shm/abb3-dp") == J_SPEC
