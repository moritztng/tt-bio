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
import sys
import threading
import time

#: The trajectory this thread is running. "" is the only slot a single-trajectory process
#: ever has, and every slot-keyed structure defaults to it, so an un-interleaved run keeps
#: exactly the state layout it had before this module existed.
_LOCAL = threading.local()

#: Bytes one in-flight trajectory can hold on the CARD at once, measured on the composed
#: BindCraft 2 round at n=288 (`state/perf10/bcx-p10-duotraj.md` leg 1): 3.746 GB inside a
#: seam, 1.668 GB banked between seams, over a 0.559 GB floor of weights and masks the
#: trajectories share. The refusal below prices a new trajectory at the in-seam peak minus
#: the shared floor, which is the most one of them can add.
TRAJECTORY_BYTES = int(3.2 * 2**30)

#: Bytes one more in-flight trajectory takes on the HOST, and this is the one that binds. The
#: first nine-round interleaved arm was OOM-KILLED by the host at 13.2 GB anon-rss on a 31 GB
#: box while device DRAM never went past 3.75 GB of 31.9. Measured on the serial arm, where
#: the two trajectories run in one process one after the other: trajectory 1 reaches 8.22 GB
#: high-water over nine rounds and trajectory 2 adds 2.75 GB on top of that. 3.5 GB is that
#: marginal figure with a margin, and it is per ADDITIONAL trajectory, not per trajectory.
TRAJECTORY_HOST_BYTES = int(3.5 * 2**30)


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


#: The gate in force, or None when nothing is interleaved. A module global rather than an
#: argument threaded through nine call sites, because `card()` has to be reachable from inside
#: a model's own forward without that model taking a scheduler parameter.
GATE: "DeviceGate | None" = None


@contextlib.contextmanager
def card(slot: str = "", tag: str = ""):
    """Hold the card for the duration of this block. A no-op when nothing is interleaved.

    Wrap the DEVICE region of a seam, not the whole seam. The difference is the whole lever:
    measured on the composed BindCraft 2 round, wrapping whole seam methods held the gate for
    8.6 s of a 9.1 s round -- 95 % -- because a seam also pads its inputs, builds its cotangent
    buffers and slices its results, none of which touches the card. A gate held across that
    serialises exactly the host seconds the interleave exists to overlap, and caps the lever at
    1.05x before it starts.
    """
    gate = GATE
    if gate is None:
        yield
        return
    with gate.held_for(tag, slot):
        yield


def free_device_bytes() -> int:
    """Free DRAM on the open card, or 0 when the read fails or no card is open.

    A process that has not imported ttnn has no card open, so this answers without
    importing it: the auto default asks before anything on the design path has, and
    pulling the whole device stack in to be told there is no device costs seconds and a
    pile of teardown noise for an answer already known.
    """
    if "ttnn" not in sys.modules:
        return 0
    try:
        import ttnn
        from tt_bio import tenstorrent
        if tenstorrent._device is None:
            return 0
        mv = ttnn.get_memory_view(tenstorrent._device, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_free_per_bank) * int(mv.num_banks)
    except Exception:
        return 0


def free_host_bytes() -> int:
    """MemAvailable, the kernel's own estimate of what a new allocation can have."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def refuse_if_it_will_not_fit(extra: int, *, per_trajectory: int = TRAJECTORY_BYTES,
                              per_trajectory_host: int = TRAJECTORY_HOST_BYTES) -> None:
    """Raise unless BOTH the card and the box have room for `extra` more trajectories.

    The host check is not decoration and it is not second: the card has 28 GB free on the
    measured round and the box had 14, and it is the box that killed the first interleaved
    arm. Reading both rather than inferring from a model, because a footprint is the one thing
    a numerical fixture cannot see and crashing a size a user gets today is worse than being
    slower than it could have been. Each refusal quotes what it asked for and what was free,
    since "it does not fit" and "it is fragmented" need different fixes.
    """
    if extra <= 0:
        return
    want_host = extra * per_trajectory_host
    free_host = free_host_bytes()
    if free_host and free_host < want_host:
        raise MemoryError(
            f"interleaving {extra + 1} trajectories needs {want_host / 2**30:.2f} GB of HOST "
            f"memory beyond what this process already holds and the box has "
            f"{free_host / 2**30:.2f} GB available. This is the limit that binds: the card "
            f"has an order of magnitude more headroom than the box. Run them one at a time, "
            f"or free the box.")
    free = free_device_bytes()
    if free == 0:
        return                      # no card open yet; the host check above still applied
    want = extra * per_trajectory
    if free < want:
        raise MemoryError(
            f"interleaving {extra + 1} trajectories needs {want / 2**30:.2f} GB beyond what "
            f"is already resident and the card has {free / 2**30:.2f} GB free. Run them one "
            f"at a time, or lower per_trajectory if this model holds less than "
            f"{per_trajectory / 2**30:.2f} GB in flight.")


#: The most trajectories `auto` will ever choose. A fourth read 6.976 s a round against three
#: at 6.992 on the same card and sitting (`state/perf10/bcx-p10-tritraj.md`), which is nothing:
#: at three, 0.26 s of gate idle a round is all that is left to fill. Past this the host memory
#: is spent for no seconds back.
AUTO_CAP = 3

#: What a WHOLE campaign's host high-water looks like, which is not what the guard above prices.
#: `refuse_if_it_will_not_fit` charges each additional trajectory and the first one nothing,
#: which is the right question for a number the caller chose and the wrong one for choosing the
#: number: when `auto` runs, the first trajectory has not grown either. Measured on real PD-L1
#: campaigns (`state/perf10/bcx-p10-campaign.md`, `bcx-p10-tritraj.md`): 8.2 GB high-water with
#: one trajectory, 14.2-15.3 with two, 19.3-19.5 with three. That is ~8.5 GB for the first and
#: ~5.6 GB for each one after it, rounded up in both places because the cost of being wrong here
#: is a campaign the kernel kills at round 200.
AUTO_BASE_HOST_BYTES = int(8.5 * 2**30)
AUTO_EXTRA_HOST_BYTES = int(6.0 * 2**30)

#: Left to the rest of the box. A campaign that takes MemAvailable to zero is one the kernel
#: starts reclaiming against, and this is the margin `auto` keeps so it never sits exactly on
#: the boundary `refuse_if_it_will_not_fit` refuses at.
AUTO_HOST_RESERVE_BYTES = int(2.0 * 2**30)


def host_rss_bytes() -> int:
    """What this process already holds, or 0 when it cannot be read."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def auto_trajectories(cap: int = AUTO_CAP) -> "tuple[int, str]":
    """How many trajectories this box can hold, and one sentence saying why.

    Reads the box (and the card, when one is open) and returns the largest count up to `cap`
    that fits, floored at 1. The caller prints the reason, because a default that silently
    picks a different number on two boxes is a default nobody can reason about.

    **An unreadable box chooses 1, never `cap`.** `free_host_bytes()` returns 0 when
    /proc/meminfo is not there or not readable, and the two ways of being wrong do not cost
    the same: one is a campaign slower than it could have been, the other is one the kernel
    kills at round 200 with the designs it had.
    """
    cap = max(1, int(cap))
    free = free_host_bytes()
    gb = 1 / 2**30
    if not free:
        return 1, "free host memory could not be read, and an unreadable box is not a roomy one"
    held = host_rss_bytes()

    def peak(n: int) -> int:
        return AUTO_BASE_HOST_BYTES + AUTO_EXTRA_HOST_BYTES * (n - 1)

    def needs(n: int) -> int:
        return max(0, peak(n) - held) + AUTO_HOST_RESERVE_BYTES

    on_card = free_device_bytes()
    room_on_card = cap if not on_card else max(1, 1 + on_card // TRAJECTORY_BYTES)
    for n in range(cap, 1, -1):
        if n > room_on_card:
            continue
        if needs(n) <= free:
            return n, (f"{free * gb:.1f} GB of host memory is free and {n} of them peak near "
                       f"{peak(n) * gb:.0f} GB")
    if on_card and room_on_card < 2:
        return 1, (f"the card has {on_card * gb:.1f} GB free, under the "
                   f"{TRAJECTORY_BYTES * gb:.1f} GB a second trajectory holds in flight")
    return 1, (f"{free * gb:.1f} GB of host memory is free and a second trajectory needs "
               f"{needs(2) * gb:.1f} GB")


#: Gradient rounds entered per slot, and the event that fires once a slot has its FIRST round
#: behind it. Two trajectories entering their first JAX trace together would compile at the same
#: time, which is the one part of a design loop with process-wide trace state, so the caller
#: starts trajectory i only after i-1 has cleared its compile round. `interleave` resets both.
_ROUNDS: dict[str, int] = {}
_CLEARED: dict[str, threading.Event] = {}


def compile_round_cleared(name: str) -> threading.Event:
    """The event set once trajectory `name` has entered its second gradient round."""
    return _CLEARED.setdefault(str(name), threading.Event())


def round_entered() -> None:
    """Count a gradient round for this thread's trajectory. Inert when nothing is interleaved.

    Called at the top of the design model's `sequence_gradients`, which is the design loop
    itself and runs on the trajectory's own thread, so the slot is trustworthy here. A compile
    of the next length bucket runs on a daemon thread of its own and lands in slot `""`, which
    is nobody's trajectory and sets no event.
    """
    if GATE is None:
        return
    s = slot()
    n = _ROUNDS[s] = _ROUNDS.get(s, 0) + 1
    if n == 2:
        compile_round_cleared(s).set()


@contextlib.contextmanager
def interleave(trajectories: int = 2, per_trajectory: int = TRAJECTORY_BYTES):
    """Install the gate so `trajectories` threads can share one card.

    The switch, and the whole of it. Outside this block `card()` is a no-op and nothing in
    tt-bio behaves differently, which is what keeps a lever that changes the resident
    footprint off by default.

        with duotraj.interleave(trajectories=2) as gate:
            duotraj.run([lambda: design(0), lambda: design(1)])
    """
    refuse_if_it_will_not_fit(trajectories - 1, per_trajectory=per_trajectory)
    global GATE
    _ROUNDS.clear()
    _CLEARED.clear()
    was, GATE = GATE, DeviceGate()
    try:
        yield GATE
    finally:
        GATE = was


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
