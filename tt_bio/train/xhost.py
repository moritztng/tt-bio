"""The cross-host leg of the shared-directory all-reduce: same protocol, longer wire.

``tt_bio/train/hostreduce.py`` is a rendezvous over a directory. Every rank writes its file,
waits for the others to appear, and sums them in rank order. On one box that directory is
``/dev/shm`` and the exchange is a memcpy. Across two boxes there is no shared directory, so
this module gives each host its OWN copy of the rendezvous and pushes every file that lands in
it to the other hosts. The protocol above the transport does not change: a rank still writes,
still waits for ``world`` files, still sums 0..world-1. That is deliberate, and it is why the
per-rank master hashes stay bit-identical across a host boundary -- the reduction order is a
property of the protocol, not of the wire.

**One ssh process per peer host, for the life of the run.** A fresh ssh per transfer pays a
handshake per step. The orchestrator's probe measured 1.87-2.02 s for a 28,446,060 B exchange
with a handshake each time; the persistent channel here removes that term. The receiver is a
stdlib program fed through stdin, so a peer needs python3 and nothing else -- no wheel to
install on the far host, no launcher, no daemon to leave running.

**Frames are ``<op> <nbytes> <name>``.** ``put`` is written to a temporary name and renamed into
place, because the reader on the far side polls for existence and would otherwise load a
half-written array as a valid one. ``rm`` retires a file, which matters more here than it looks:
a rank retires its own previous-step file locally, and without the same retirement crossing the
wire every peer accumulates 28.4 MB per rank per step until /dev/shm is gone.

**A peer that dies fails the run, it does not degrade it.** If the ssh channel drops, the next
publish raises. The alternative -- carrying on locally -- is the diverging-replica failure the
hash check exists to catch, arrived at deliberately.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

__all__ = ["parse_rendezvous", "SshPeers", "PEER_SEP"]

#: Separates the rendezvous directory from the peer hosts that mirror it:
#: ``/dev/shm/abb3-dp+ttuser@tt-quietbox2``. In the argument rather than the environment, and
#: rather than a file inside the rendezvous, because the supervisor deletes that directory
#: before every restart -- a peer list living there would vanish on the first watchdog reset and
#: the resumed world would wait for files nobody was sending.
PEER_SEP = "+"

_RECEIVER = r"""
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
inp = sys.stdin.buffer
while True:
    header = inp.readline()
    if not header:
        break
    op, size, name = header.decode().rstrip("\n").split(" ", 2)
    size = int(size)
    if ".." in name.split("/"):
        raise SystemExit("refusing a name that escapes the rendezvous: " + name)
    dest = root / name if name else root
    if op == "put":
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / ("." + dest.name + ".xhost")
        left = size
        with open(tmp, "wb") as fh:
            while left:
                chunk = inp.read(min(1 << 22, left))
                if not chunk:
                    raise SystemExit("truncated frame for " + name)
                fh.write(chunk)
                left -= len(chunk)
        os.replace(tmp, dest)
    elif op == "rm":
        try:
            dest.unlink()
        except FileNotFoundError:
            pass
    elif op == "rmtree":
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
    else:
        raise SystemExit("unknown op " + op)
"""


def parse_rendezvous(spec) -> tuple:
    """``dir[+peer[,peer...]]`` -> ``(Path(dir), [peer, ...])``.

    A plain path parses to itself and no peers, so the single-host case reaches exactly the
    code it reached before this module existed.
    """
    head, sep, tail = str(spec).partition(PEER_SEP)
    peers = [p for p in tail.split(",") if p] if sep else []
    return Path(head), peers


def _ssh_argv(dest: str, root: Path) -> list:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", dest,
            "python3", "-u", "-c", _RECEIVER, str(root)]


def _local_argv(dest: str, root: Path) -> list:
    """The same receiver with no wire in front of it, which is how the frames get tested.

    A test that mocks the transport tests the mock. This runs the real receiver program over a
    real pipe and only leaves out ssh, so a malformed frame or a missing rename fails here
    rather than on the far host in the middle of a run.
    """
    return [sys.executable, "-u", "-c", _RECEIVER, str(root)]


class _Peer:
    """One persistent ssh channel, with a writer thread so the sends overlap the barrier.

    The queue is bounded: a link slower than the step would otherwise let the sender run ahead
    and buffer a gradient per step in memory. Bounded means the rank blocks instead, which is
    the honest behaviour -- the run is then link-bound and the step time says so.
    """

    def __init__(self, dest: str, root: Path, *, depth: int = 2, argv=_ssh_argv):
        self.dest = dest
        self.proc = subprocess.Popen(argv(dest, root), stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=sys.stderr)
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.error = None
        self.sent = 0
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            op, name, blob = item
            try:
                size = len(blob) if blob is not None else 0
                self.proc.stdin.write(f"{op} {size} {name}\n".encode())
                if blob is not None:
                    self.proc.stdin.write(blob)
                    self.sent += size
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                self.error = RuntimeError(
                    f"the channel to {self.dest} is gone ({e}); ssh exited "
                    f"{self.proc.poll()}. A rank on that host is unreachable, so this world "
                    f"cannot complete a step -- the supervisor restarts all ranks together")
                break
        # Drain whatever is left so a blocked producer is released rather than deadlocked on a
        # dead channel: it raises on its next publish, which is where the error belongs.
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def publish(self, op: str, name: str, blob) -> None:
        if self.error is not None:
            raise self.error
        self.q.put((op, name, blob))
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        try:
            self.q.put(None, timeout=5.0)
            self.thread.join(timeout=30.0)
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=30.0)
        except Exception:
            self.proc.kill()


class SshPeers:
    """The transport :class:`~tt_bio.train.hostreduce.HostReduce` publishes through.

    ``root`` is the same absolute path on every host, so a file's name inside the rendezvous
    identifies it everywhere and the far side needs no translation table.
    """

    def __init__(self, peers, root, *, local: bool = False):
        self.root = Path(root)
        argv = _local_argv if local else _ssh_argv
        self.peers = [_Peer(p, self.root, argv=argv) for p in peers]

    @property
    def dests(self) -> list:
        return [p.dest for p in self.peers]

    @property
    def bytes_sent(self) -> int:
        return sum(p.sent for p in self.peers)

    def put(self, name: str, blob: bytes) -> None:
        for p in self.peers:
            p.publish("put", name, blob)

    def remove(self, name: str) -> None:
        for p in self.peers:
            p.publish("rm", name, None)

    def remove_tree(self, name: str = "") -> None:
        for p in self.peers:
            p.publish("rmtree", name, None)

    def check(self) -> None:
        """Raise if any channel has died. Called from the barrier's wait loop.

        Without it a dead peer is indistinguishable from a slow one until the rendezvous
        timeout, which is 15 minutes by default and would hide the real error behind it.
        """
        for p in self.peers:
            if p.error is not None:
                raise p.error
            if p.proc.poll() is not None:
                raise RuntimeError(
                    f"ssh to {p.dest} exited {p.proc.returncode} while the rendezvous was "
                    f"still open")

    def close(self) -> None:
        for p in self.peers:
            p.close()

    def __str__(self) -> str:
        return (f"SshPeers({', '.join(self.dests)} -> {self.root}, "
                f"{self.bytes_sent / 1e6:.0f} MB sent)")


def local_clean(spec) -> None:
    """Remove the local rendezvous named by ``spec``, ignoring the peer part.

    A caller that passes the whole spec to ``shutil.rmtree`` removes nothing and says nothing,
    because ``/dev/shm/abb3-dp+ttuser@host`` is not a path that exists.
    """
    root, _ = parse_rendezvous(spec)
    shutil.rmtree(root, ignore_errors=True)
