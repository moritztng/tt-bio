"""Run several independent trajectories on one card at the same time.

A gradient round is a host column and a device column laid end to end, and they never
overlap: the round's own dataflow puts every host second on the current round's device
result, so there is nothing inside one trajectory to pipeline (measured,
`state/perf10/bcx-OVERLAP.md`: 2.630 s of host-busy-device-idle with 0.034 s in which
nothing at all runs). The seconds are only recoverable across INDEPENDENT work, and a
design campaign has a supply of it -- BindCraft 2 draws one trajectory after another and
they share nothing but the weights.

So: one thread per trajectory, one lock over the card. The lock is taken at the device
SEAM -- the whole of one on-card forward or backward -- and not around a whole fold and not
around a single ttnn call. The card is one serial resource, so serialising the seams costs
nothing that was not already serial, and the host gaps between a trajectory's seams are
exactly the seconds the other one runs in.

Everything that was one-trajectory-at-a-time state is keyed by SLOT rather than made
thread-local at the point of use, because the caller that needs it is sometimes the
trajectory's own thread (the route selection) and sometimes a JAX callback (the seam).
`jax.pure_callback` runs on the thread that entered the jitted function, which is what
makes one thread-local enough for both.

Off unless a caller asks for it. `interleave()` is the whole switch; nothing here is read
from the environment and nothing changes behaviour for a process that never calls it.
"""
from __future__ import annotations

import contextlib
import threading
import time

#: The trajectory this thread is running. "" is the only slot a single-trajectory process
#: ever has, and every slot-keyed structure defaults to it, so an un-interleaved run keeps
#: exactly the state layout it had before this module existed.
_LOCAL = threading.local()

#: Bytes one in-flight trajectory can hold on the card at once, measured on the composed
#: BindCraft 2 round at n=288 (`state/perf10/bcx-p10-duotraj.md` leg 1): 3.746 GB inside a
#: seam, 1.668 GB banked between seams, over a 0.559 GB floor of weights and masks the
#: trajectories share. The refusal below prices a new trajectory at the in-seam peak minus
#: the shared floor, which is the most one of them can add.
TRAJECTORY_BYTES = int(3.2 * 2**30)


def slot() -> str:
    """Which trajectory this thread is running, or "" when it is the only one."""
    return getattr(_LOCAL, "slot", "")


@contextlib.contextmanager
def trajectory(name: str):
    """Run this block as trajectory `name`. Restores the previous slot on the way out."""
    prev = slot()
    _LOCAL.slot = str(name)
    try:
        yield
    finally:
        _LOCAL.slot = prev


class DeviceGate:
    """One lock over every on-card seam, with the wait and the hold counted per slot.

    Reentrant: a seam that calls another seam (the template stack inside a fold) must not
    deadlock against itself, and the inner acquire is free.

    The two counters are the row's whole measurement. `held` is device work and is serial by
    construction; `waited` is a trajectory standing at the card door, and the difference
    between `waited` and zero is the interference the amortised round pays for.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._depth = threading.local()
        self.waited: dict[str, float] = {}
        self.held: dict[str, float] = {}
        self.entries: dict[str, int] = {}
        #: `(slot, thread name)` seen at each seam, counted. The slot is only trustworthy if
        #: the seam runs on the trajectory's own thread, and whether it does is a property of
        #: the traced program rather than of jax, so it is recorded and not assumed.
        self.threads: dict[str, int] = {}

    @contextlib.contextmanager
    def held_for(self, tag: str = "", slot: str = ""):
        depth = getattr(self._depth, "n", 0)
        if depth:                       # already ours; count it once, at the outer seam
            self._depth.n = depth + 1
            try:
                yield
            finally:
                self._depth.n = depth
            return
        s = slot
        t0 = time.perf_counter()
        self._lock.acquire()
        t1 = time.perf_counter()
        self._depth.n = 1
        self.waited[s] = self.waited.get(s, 0.0) + (t1 - t0)
        self.entries[s] = self.entries.get(s, 0) + 1
        key = f"{s or '-'}@{threading.current_thread().name}"
        self.threads[key] = self.threads.get(key, 0) + 1
        try:
            yield
        finally:
            self._depth.n = depth
            self.held[s] = self.held.get(s, 0.0) + (time.perf_counter() - t1)
            self._lock.release()

    def report(self) -> dict:
        return {"waited_s": {k: round(v, 3) for k, v in self.waited.items()},
                "held_s": {k: round(v, 3) for k, v in self.held.items()},
                "seams": dict(self.entries), "slot_at_seam": dict(self.threads)}


def serialize(gate: DeviceGate, *classes, methods=("_primal", "_taped", "_backward")):
    """Put `gate` around every named method of every class. Returns an undo callable.

    The classes rather than a list of names, so a stack that grows a fourth seam is covered
    by naming its class here and not by remembering to add a string. A method the class does
    not define is skipped rather than raising: the three BindCraft 2 splices share the three
    names, and a model that does not is not a reason to refuse.
    """
    undo = []
    for cls in classes:
        for name in methods:
            orig = cls.__dict__.get(name)
            if orig is None:
                continue

            def make(orig, tag):
                def seam(self, traj, *a, **kw):
                    # Every device seam takes its trajectory as its first argument, baked into
                    # the traced program: the seam runs on one of XLA:CPU's pool threads, so
                    # there is nothing on this thread that says whose work it is.
                    with gate.held_for(tag, traj):
                        return orig(self, traj, *a, **kw)
                seam.__name__ = getattr(orig, "__name__", tag)
                seam.__doc__ = getattr(orig, "__doc__", None)
                seam.__wrapped__ = orig
                return seam
            setattr(cls, name, make(orig, f"{cls.__name__}.{name}"))
            undo.append((cls, name, orig))

    def restore():
        for cls, name, orig in undo:
            setattr(cls, name, orig)
    return restore


def free_device_bytes() -> int:
    """Free DRAM on the open card, or 0 when the read fails or no card is open."""
    try:
        import ttnn
        from tt_bio import tenstorrent
        if tenstorrent._device is None:
            return 0
        mv = ttnn.get_memory_view(tenstorrent._device, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_free_per_bank) * int(mv.num_banks)
    except Exception:
        return 0


def refuse_if_it_will_not_fit(extra: int, *, per_trajectory: int = TRAJECTORY_BYTES) -> None:
    """Raise unless the card has room for `extra` more trajectories in flight.

    Reads the allocator rather than inferring from a model: a footprint is the one thing a
    numerical fixture cannot see, and crashing a size a user gets today is worse than being
    slower than it could have been. The refusal quotes what it asked for and what was free,
    because the two answers "it does not fit" and "it is fragmented" need different fixes.
    """
    if extra <= 0:
        return
    free = free_device_bytes()
    if free == 0:
        return                      # no card open yet; the seam check below is the backstop
    want = extra * per_trajectory
    if free < want:
        raise MemoryError(
            f"interleaving {extra + 1} trajectories needs {want / 2**30:.2f} GB beyond what "
            f"is already resident and the card has {free / 2**30:.2f} GB free. Run them one "
            f"at a time, or lower per_trajectory if this model holds less than "
            f"{per_trajectory / 2**30:.2f} GB in flight.")


@contextlib.contextmanager
def interleave(*classes, trajectories: int = 2, per_trajectory: int = TRAJECTORY_BYTES):
    """Serialise the device seams of `classes` so `trajectories` threads can share the card.

    The switch, and the whole of it. Outside this block nothing in tt-bio behaves
    differently, which is what keeps a lever that changes the resident footprint off by
    default.

        with duotraj.interleave(EvoformerOnDevice, ExtraMsaOnDevice) as gate:
            duotraj.run(gate, [lambda: design(0), lambda: design(1)])
    """
    refuse_if_it_will_not_fit(trajectories - 1, per_trajectory=per_trajectory)
    gate = DeviceGate()
    restore = serialize(gate, *classes)
    try:
        yield gate
    finally:
        restore()


def run(runners, *, names=None, ready=None, ready_timeout=900.0) -> list:
    """Run each callable in `runners` on its own thread, each in its own slot.

    `ready` staggers the start: every runner after the first waits for that event before it
    begins. Two trajectories entering their first JAX trace at the same moment would be
    measuring two concurrent compiles, which is not the thing being measured and is also the
    one part of the design loop with process-wide trace state. The caller sets the event when
    the first trajectory is past its compile round. Without it they all start together.

    Returns their results in order. A runner that raises re-raises here once every thread has
    finished, so one trajectory failing does not leave the other one orphaned on the card.
    """
    names = list(names or [f"t{i}" for i in range(len(runners))])
    out: list = [None] * len(runners)
    err: list = [None] * len(runners)

    def go(i, fn, name):
        if i and ready is not None and not ready.wait(ready_timeout):
            err[i] = TimeoutError(f"{name} waited {ready_timeout:.0f}s for the first "
                                  f"trajectory to clear its compile round and it never did")
            return
        with trajectory(name):
            try:
                out[i] = fn()
            except BaseException as exc:        # StopAfterRounds is a BaseException
                err[i] = exc

    threads = [threading.Thread(target=go, args=(i, fn, n), name=f"duotraj:{n}")
               for i, (fn, n) in enumerate(zip(runners, names))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in err:
        if e is not None:
            raise e
    return out
