"""Host-local, physical-card-keyed device lease enforced at the device-open choke point.

Every tt-bio device acquisition routes through ``get_device()`` -> ``ttnn.open_device``
(``tenstorrent.py``). Wrapping that single choke point with an exclusive, auto-releasing
file lock makes it IMPOSSIBLE for two processes on the same host to open the same physical
TT card at once, no matter how the job was launched -- fleet worker, detached campaign,
cross-host fanout, or a manual one-off. This replaces the previous convention-only "remember
to write a lease" rule (a human-followed PLAYBOOKS note) with enforcement in code.

Mechanism -- one exclusive ``flock()`` held on a per-card lock file for the entire lifetime
the device is open:

* Serialization: a second opener of the same card blocks up to a bounded timeout, then fails
  with an explicit ``DeviceInUseError`` naming the holder -- never a silent fd-level collision.
* Stale reclaim / crash safety: ``flock`` is released by the kernel on ANY process death --
  clean exit, SIGTERM, or SIGKILL -- so a crashed, killed, or orphaned holder (e.g. a leaked
  mp.spawn worker) never leaves a phantom claim. There is no pid-liveness scan to get wrong:
  a dead holder's lock is simply gone, so the next acquire succeeds immediately.
* No self-deadlock: the fleet dispatcher writes its lease JSON without holding an ``flock``,
  so a worker opening its own already-dispatch-leased card acquires instantly and just
  overwrites the metadata with its live pid.

Shared view with the fleet dispatcher: the lock/metadata files use the same directory and the
same ``<host>-card<N>.json`` naming the dispatcher already uses, so the dispatcher and the
device-open observe ONE consistent set of leases. Collision-freedom does NOT depend on that
sharing, though -- because both colliding jobs pass through this same choke point, they
serialize here even if the dispatcher never sees the lease.
"""

import errno
import fcntl
import json
import os
import signal
import socket
import threading
import time

# Bounded wait so a genuinely-contended card fails loudly instead of hanging a job forever.
DEFAULT_TIMEOUT_S = float(os.environ.get("TT_BIO_LEASE_TIMEOUT", "120"))
_POLL_S = 0.25


class DeviceInUseError(RuntimeError):
    """Raised when the physical card cannot be leased within the timeout."""


def lease_dir():
    """Directory holding the per-card lease files.

    Defaults to the fleet's lease directory when a ``~/.coworker`` tree exists (so the
    dispatcher's ``pick_card`` and this device-open share one view), else a neutral
    host-local directory for a standalone tt-bio install. ``TT_BIO_LEASE_DIR`` overrides.
    """
    d = os.environ.get("TT_BIO_LEASE_DIR")
    if d:
        return d
    coworker = os.path.expanduser("~/.coworker")
    if os.path.isdir(coworker):
        return os.path.join(coworker, "state", "leases")
    return "/tmp/tt-bio-device-leases"


def lease_host():
    """Host label used in the lease filename. ``TT_BIO_LEASE_HOST`` overrides the hostname
    so it can be aligned with whatever name the dispatcher uses for this host."""
    return os.environ.get("TT_BIO_LEASE_HOST") or socket.gethostname()


def physical_card():
    """The physical card ``get_device()`` will open, named exactly as the fleet names it.

    A worker (or manual run) exports ``TT_VISIBLE_DEVICES`` to the physical chip(s) it may
    use; ``get_device`` then opens logical id ``TT_BIO_LOGICAL_DEVICE_ID`` (default 0), which
    is the ``logical``-th visible chip. Keying the lease on that same value makes the lease
    filename identical to the fleet dispatcher's ``<host>-card<N>.json`` for card N.
    """
    logical = int(os.environ.get("TT_BIO_LOGICAL_DEVICE_ID", "0"))
    visible = os.environ.get("TT_VISIBLE_DEVICES", "")
    if visible.strip():
        # Tokens may be PCI BDFs, which ttnn also accepts; resolve to UMD indices so
        # the lease file keeps the fleet's <host>-card<N>.json naming (issue #11).
        from tt_bio.runtime import visible_device_indices
        cards = visible_device_indices(visible)
        return str(cards[logical] if logical < len(cards) else cards[0])
    return str(logical)


def _holder_label():
    """Identity written into the lease. The fleet exports ``TT_BIO_LEASE_HOLDER=worker:<name>``
    at dispatch; a detached/manual job with no such env falls back to its own pid."""
    return os.environ.get("TT_BIO_LEASE_HOLDER") or f"pid:{os.getpid()}"


class DeviceLease:
    """An exclusive lease on one physical TT card, held for as long as the device is open."""

    def __init__(self, card=None, timeout=DEFAULT_TIMEOUT_S):
        self.card = str(card if card is not None else physical_card())
        self.host = lease_host()
        self.timeout = timeout
        self.dir = lease_dir()
        self.path = os.path.join(self.dir, f"{self.host}-card{self.card}.json")
        self._fd = None
        self._lock = threading.Lock()

    def _read_holder(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return None

    def _write_metadata(self):
        meta = {
            "host": self.host,
            "card": self.card,
            "holder": _holder_label(),
            "pid": os.getpid(),
            "acquired": time.time(),
            "released": None,
        }
        # We hold the exclusive flock, so this rewrite is safe against other leasers.
        # Single truncate + single write keeps a lock-free reader (pick_card) from
        # observing a torn record for these small payloads.
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, (json.dumps(meta) + "\n").encode())
        os.fsync(self._fd)

    def acquire(self):
        """Atomically claim the card. Blocks up to ``timeout`` if a LIVE process holds it,
        then raises :class:`DeviceInUseError`. Returns ``self``."""
        os.makedirs(self.dir, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    os.close(fd)
                    raise
                if time.time() >= deadline:
                    holder = self._read_holder()
                    os.close(fd)
                    who = "?"
                    if holder:
                        who = f"{holder.get('holder')} (pid {holder.get('pid')})"
                    raise DeviceInUseError(
                        f"physical card {self.card} on {self.host} is in use by {who}; "
                        f"waited {self.timeout:.0f}s. Refusing to open it concurrently "
                        f"(would collide at the fd level)."
                    )
                time.sleep(_POLL_S)
        self._fd = fd
        try:
            self._write_metadata()
        except Exception:
            pass  # metadata is observability only; the flock is the real lease
        return self

    def release(self):
        """Release the lease: mark the metadata free (best effort) and drop the flock.

        Idempotent and safe to call from atexit. The kernel also drops the flock on process
        death, so a missed release (e.g. SIGKILL) never leaves the card falsely held."""
        with self._lock:
            if self._fd is None:
                return
            fd, self._fd = self._fd, None
            try:
                meta = self._read_holder() or {}
                meta["released"] = time.time()
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, (json.dumps(meta) + "\n").encode())
            except Exception:
                pass
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# Process-lifetime guard: who may still be holding a card
# ---------------------------------------------------------------------------
_ARMED = False


def install_parent_death_guard(parent_pid: int, *, exit_code: int = 70) -> None:
    """Die with the process that owns this one, from inside any call.

    PR_SET_PDEATHSIG is delivered by the kernel the moment the parent dies, so it
    covers the case a between-jobs ``getppid()`` check cannot: a process orphaned in
    the middle of device work, still holding its chip and its card lease. One was
    found holding /dev/tenstorrent/0 and card 3's flock for 6 h at 100% CPU; another,
    a gate scorer, held card 1 for 1 h 43 m and killed the next gate's first leg with
    DeviceInUseError.

    The kernel sends the signal when the parent THREAD that created us exits, not when
    the parent process does, so a caller spawning from a thread pool must not let that
    thread return while the child is still alive.
    """
    if not parent_pid:
        return
    try:
        import ctypes

        ctypes.CDLL(None).prctl(1, int(signal.SIGTERM))  # 1 = PR_SET_PDEATHSIG
    except Exception:
        pass
    # The parent can die between our spawn and the prctl above. man 2 prctl: no signal
    # is sent if it is already gone, so re-check by hand.
    if os.getppid() != parent_pid:
        os._exit(exit_code)


def arm_orphan_guard() -> None:
    """A process about to take a card must not outlive whoever wanted the result.

    ``TT_BIO_PARENT_PID`` names the process that spawned us: the predict CLI for its
    workers, a gate driver for its scorers and folds. Arm only when the value really is
    our parent. A stale value inherited from a dead grandparent means we are a detached
    fleet job, and those must keep running, so the test is exact rather than "the
    variable is set".

    SIGTERM keeps its default disposition here on purpose. A Python handler only runs
    when the interpreter does, and a process wedged inside a ttnn call would then hold
    its card forever, which is the failure this guard exists to end. Workers can afford
    a handler because ``run_worker_loop``'s heartbeat backstops it.
    """
    global _ARMED
    if _ARMED:
        return
    parent = int(os.environ.get("TT_BIO_PARENT_PID") or 0)
    if not parent or parent != os.getppid():
        return
    _ARMED = True
    install_parent_death_guard(parent)
