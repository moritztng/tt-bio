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

import atexit
import contextlib
import fcntl
import os
import pathlib
import sys
import threading
import time

from tt_bio.envflags import env_int

#: The trajectory this thread is running. "" is the only slot a single-trajectory process
#: ever has, and every slot-keyed structure defaults to it, so an un-interleaved run keeps
#: exactly the state layout it had before this module existed.
_LOCAL = threading.local()

#: The token axis every constant below was measured at: the PD-L1 design, a 146 aa binder.
REFERENCE_TOKENS = 288

#: Bytes one in-flight trajectory adds on the CARD at `REFERENCE_TOKENS`. Measured twice, and the
#: charge covers the larger: 3.746 GB inside a seam over a 0.559 GB floor on a Blackhole chip
#: (`state/perf10/bcx-p10-duotraj.md` leg 1), and **4.119 GB on one chip of a Wormhole Galaxy**
#: (`state/b2p-wh.md`), where two trajectories ran a campaign to its stop condition and three do
#: not fit the 11.40 GB the pricing leaves for them. `trajectory_bytes` scales it by the square.
#:
#: 3.2 GiB was under both figures and the composed-path surcharge was quietly making up the
#: difference, so dropping that surcharge where it cannot run has to be paid for here: at 3.2 GiB
#: and no surcharge, a 12 GiB chip approves the three trajectories it was measured to refuse.
TRAJECTORY_BYTES = int(3.9 * 2**30)

#: Bytes one more in-flight trajectory takes on the HOST at `REFERENCE_TOKENS`. The first
#: nine-round interleaved arm was OOM-KILLED by the host at 13.2 GB anon-rss on a 31 GB box while
#: device DRAM never went past 3.75 GB of 31.9. Measured serially, trajectory 2 adds 2.75 GB on
#: top of trajectory 1's 8.22 GB high-water; 3.5 GB is that with a margin.
TRAJECTORY_HOST_BYTES = int(3.5 * 2**30)

#: How the host figures grow with the token axis, per token squared, above `REFERENCE_TOKENS`.
#: Measured on one shipped trajectory (`state/bgx-traj.md`): host high-water 7.79, 8.73, 10.95 and
#: 15.03 GB at 288, 384, 512 and 704 tokens fits 1.77e-5 GB/token^2, and the second and third
#: trajectory at 512 add 7.94 GB each against 4.47 at 288, which is 1.94e-5. Rounded up.
HOST_BYTES_PER_TOKEN2 = 2.0e-5 * 2**30

#: DRAM one chip has, as the allocator reports it, by the PCI device id the kernel driver puts
#: in `/sys/class/tenstorrent/tenstorrent!N/device/device` (the ids are tt-kmd's own, from
#: `enumerate.h`). 31.875 GiB on Blackhole, the p150a and one p300 chip alike; 12 GiB on a
#: Wormhole chip, which is what `tenstorrent.atom_pair_budget_bytes` already prices that part at.
CARD_BYTES_BY_PCI_ID = {0x401E: int(12 * 2**30), 0xB140: int(31.875 * 2**30)}

#: What a card is worth when the host will not say which part it is. The SMALLEST part tt-bio
#: runs on, not the largest, the same way `tenstorrent.l1_resident_budget_bytes()` falls back to
#: Wormhole's L1: over-pricing the card starts trajectories that do not fit, under-pricing it
#: starts one that does.
CARD_BYTES = int(12 * 2**30)

#: Where the kernel driver publishes one node per chip.
TT_SYSFS_CLASS = "/sys/class/tenstorrent"

#: DRAM `auto` leaves unspent, for the fragmentation a nearly full card refuses on: at 704 tokens
#: three trajectories died with 197 MB free in total and no contiguous 11.5 MB piece of it.
CARD_RESERVE_BYTES = int(1.0 * 2**30)


def card_total_bytes() -> int:
    """DRAM one chip of this host has, answered without opening a device.

    `auto` runs before ttnn is imported, so the size of the card it is pricing cannot come from
    the allocator. The kernel driver publishes the part anyway: every chip has a
    `/sys/class/tenstorrent` node whose PCI device id says Wormhole or Blackhole, and the two
    differ by **2.65x**. Priced as a Blackhole chip, a 12 GiB Wormhole chip approves 3
    trajectories of a 288-token design at 5.3 GB each, which is 16 GB on a card that holds 11.

    Chips only disagree on a host with two parts in it, and then the smallest one wins: the
    count is chosen once, before anything knows which chip the run lands on.
    """
    try:
        nodes = sorted(pathlib.Path(TT_SYSFS_CLASS).iterdir())
    except OSError:
        return CARD_BYTES
    sizes = []
    for node in nodes:
        try:
            pci = int((node / "device" / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        sizes.append(CARD_BYTES_BY_PCI_ID.get(pci, CARD_BYTES))
    return min(sizes) if sizes else CARD_BYTES


#: What the composed triangle attention adds per token CUBED when the fused arm declines. The
#: fused arm's circular buffers do not fit L1 at every axis, the fit is not monotone in the axis
#: and nothing on the host predicts it: the device refuses it in round 1. Where it declines, the
#: [N,4,N,N] scores are resident and the footprint jumps: one trajectory at 544 tokens peaks
#: 20.02 GB over its floor on qb2 card 0 where the fused fit says 10.2, and holds 25.75 GB on
#: qb1's p150a (`state/bgx-traj.md`, `state/bgx-size.md`). That is 65 and 86 bytes per token^3
#: over the fused figure; 96 covers both.
COMPOSED_BYTES_PER_TOKEN3 = 96

#: Whether the fused arm's pad-up is on: `tenstorrent._TRIATT_HIFI_PAD_UP_TILES`, read from the
#: same variable with the same default because importing `tenstorrent` imports ttnn and `auto`
#: runs before anything has. With it on, the fused HiFi forward served every call at 544, 608,
#: 736, 832 and 864 on a p150a (1296/0 each) and at 544 and 608 on a Wormhole Galaxy chip, which
#: were the axes it used to decline at, so the composed surcharge prices a path that no longer
#: runs (`state/b2p-ceiling.md`).
_PAD_UP_ON = env_int("TT_BIO_TRIATT_HIFI_PAD_UP", 2) > 0


def _scale(tokens: int) -> float:
    """(tokens / 288)^2, and never under 1: the footprint is quadratic in the axis, and a design
    smaller than the one measured is priced as the one measured rather than extrapolated down."""
    return max(1.0, (int(tokens) / REFERENCE_TOKENS) ** 2)


def trajectory_bytes(tokens: int) -> int:
    """What one in-flight trajectory adds on the card at this token axis, over the shared floor.

    The fused path, measured on qb2 card 0, one shipped trajectory, peak minus round-boundary
    floor: 2.88, 5.08, 6.90, 8.98, 11.51, 14.18 and 17.14 GB at 288, 384, 448, 512, 576, 640 and
    704 tokens, 3.46e-5 GB x tokens^2 to within 0.07 GB, which `TRAJECTORY_BYTES` scaled by the
    square covers. Under the pad-up the same charge covers the padded route's resident peaks on a
    p150a, floor included: 13.00, 16.01, 23.07, 29.18 and 31.39 GB at 544, 608, 736, 832 and 864.

    Only with the pad-up off can the fused arm decline and the composed path hold the scores,
    and then `COMPOSED_BYTES_PER_TOKEN3` is added: 20.02 GB measured at 544, 25.8 charged.
    """
    n = max(int(tokens), REFERENCE_TOKENS)
    composed = 0 if _PAD_UP_ON else COMPOSED_BYTES_PER_TOKEN3 * n ** 3
    return int(TRAJECTORY_BYTES * _scale(n) + composed)


def trajectory_floor_bytes(tokens: int) -> int:
    """What the card holds between rounds, shared by every trajectory: weights, masks, the
    per-size constants. Measured 0.80 GB at 288 to 2.18 at 704, 3.35e-6 GB x tokens^2 + 0.52,
    priced here at 3.5e-6 and 0.6."""
    return int((3.5e-6 * int(tokens) ** 2 + 0.6) * 2**30)


def trajectory_host_bytes(tokens: int) -> int:
    """What one more trajectory takes on the host at this token axis."""
    grow = max(0, int(tokens) ** 2 - REFERENCE_TOKENS ** 2)
    return int(TRAJECTORY_HOST_BYTES + HOST_BYTES_PER_TOKEN2 * grow)


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


def refuse_if_it_will_not_fit(extra: int, *, tokens: int) -> None:
    """Raise unless BOTH the card and the box have room for `extra` more trajectories.

    `tokens` is the design's padded token axis. Both footprints grow with its square, and a
    guard priced at one size approves a count at another that cannot fit: at 704 tokens three
    trajectories need about 54 GB of a 31.9 GB card.

    The host check is not decoration: on the 288-token round the card had 28 GB free and the box
    14, and it is the box that killed the first interleaved arm. Crashing a size a user gets today
    is worse than being slower than it could have been, so each refusal quotes what it asked for
    and what was free.
    """
    if extra <= 0:
        return
    gb = 1 / 2**30
    want_host = extra * trajectory_host_bytes(tokens)
    free_host = free_host_bytes()
    if free_host and free_host < want_host:
        raise MemoryError(
            f"interleaving {extra + 1} trajectories of {tokens} tokens needs "
            f"{want_host * gb:.2f} GB of HOST memory beyond what this process already holds and "
            f"the box has {free_host * gb:.2f} GB available. Run them one at a time, or free "
            f"the box.")
    # None has started, so the card holds at most the shared floor: all of them are priced.
    want = (extra + 1) * trajectory_bytes(tokens)
    free = free_device_bytes() or card_total_bytes() - trajectory_floor_bytes(tokens)
    if free < want:
        raise MemoryError(
            f"interleaving {extra + 1} trajectories of {tokens} tokens needs {want * gb:.2f} GB "
            f"on the card and it has {free * gb:.2f} GB for them. Run them one at a time "
            f"(trajectories_per_card=1), or use fewer.")


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


def host_rss_bytes(pid: "int | str" = "self") -> int:
    """What a process already holds, this one by default, or 0 when it cannot be read."""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


#: Where each `auto` leaves the host peak it planned for. One campaign per chip is how a box with
#: several chips is used, and a Galaxy owner starts 32 of them from one shell loop: every one reads
#: MemAvailable before any has grown and every one sees the whole box. On a Wormhole Galaxy with
#: 278 GB free that is 32 x 2 trajectories planned at 14.5 GB each, 464 GB. A campaign that started
#: first and has not yet grown into its plan is memory this one cannot have, so `auto` subtracts
#: what live siblings planned and do not hold yet. A file per process, named by pid and start time
#: so a recycled pid is not read as the campaign that left it.
HOST_CLAIMS_DIR = "/tmp/tt-bio-host-claims"


def _start_time(pid: int) -> str:
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return ""


@contextlib.contextmanager
def _host_claims():
    """Yield what live sibling campaigns planned and do not hold yet, and a function that records
    this process's own plan.

    Both happen under one lock, so two campaigns deciding in the same second still see each
    other. Anything unreadable counts as nothing, which is what `auto` did before claims existed.
    """
    me = os.getpid()
    d = pathlib.Path(HOST_CLAIMS_DIR)
    try:
        if not d.is_dir():
            d.mkdir()
            d.chmod(0o1777)                  # every user's campaigns share one view of the box
        lock = open(d / ".lock", "a")
        if (d / ".lock").stat().st_uid == os.getuid():
            (d / ".lock").chmod(0o666)
    except OSError:
        yield 0, lambda peak: None
        return
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = 0
        for f in d.glob("*-*"):
            pid, _, start = f.name.partition("-")
            if not pid.isdigit() or int(pid) == me:
                continue
            try:
                if _start_time(int(pid)) != start:
                    f.unlink(missing_ok=True)
                    continue
                pending += max(0, int(f.read_text()) - host_rss_bytes(int(pid)))
            except (OSError, ValueError):
                pass

        def record(peak: int) -> None:
            ours = d / f"{me}-{_start_time(me)}"
            try:
                ours.write_text(str(int(peak)))
            except OSError:
                return
            atexit.register(ours.unlink, missing_ok=True)

        yield pending, record


def auto_trajectories(tokens: "int | None", cap: int = AUTO_CAP) -> "tuple[int, str]":
    """How many trajectories of `tokens` tokens this box and card can hold, and why.

    Returns the largest count up to `cap` that fits on both, floored at 1. The caller prints the
    reason, because a default that silently picks a different number on two boxes is a default
    nobody can reason about.

    **When in doubt it is 1.** An unreadable box or an unknown token axis chooses 1, never
    `cap`: one way of being wrong is a slower campaign, the other is one that dies at round 1 or
    is killed at round 200 with the designs it had. At sizes where only one fits, 1 is also what
    it returns, and 1 is BindCraft 2's own loop, unchanged.
    """
    cap = max(1, int(cap))
    gb = 1 / 2**30
    if not tokens:
        return 1, "the design's token axis could not be read, so its footprint is unknown"
    free = free_host_bytes()
    if not free:
        return 1, "free host memory could not be read, and an unreadable box is not a roomy one"
    held = host_rss_bytes()
    grow = HOST_BYTES_PER_TOKEN2 * max(0, int(tokens) ** 2 - REFERENCE_TOKENS ** 2)

    def peak(n: int) -> int:
        return int(AUTO_BASE_HOST_BYTES + grow + (AUTO_EXTRA_HOST_BYTES + grow) * (n - 1))

    def needs(n: int) -> int:
        return max(0, peak(n) - held) + AUTO_HOST_RESERVE_BYTES

    with _host_claims() as (siblings, record):
        n, why = _choose(tokens, cap, free - siblings, peak, needs)
        record(peak(n))
    if siblings:
        why += (f", after the {siblings * gb:.1f} GB other campaigns on this box planned for and "
                f"do not hold yet")
    return n, why


def _choose(tokens: int, cap: int, free: int, peak, needs) -> "tuple[int, str]":
    gb = 1 / 2**30
    per = trajectory_bytes(tokens)
    open_card = free_device_bytes()
    room = (open_card or card_total_bytes() - trajectory_floor_bytes(tokens)) - CARD_RESERVE_BYTES
    on_card = max(1, min(cap, room // per))
    for n in range(on_card, 1, -1):
        if needs(n) <= free:
            return n, (f"{n} of them at {tokens} tokens hold about {n * per * gb:.0f} GB of the "
                       f"card and peak near {peak(n) * gb:.0f} GB of the "
                       f"{free * gb:.1f} GB of host memory free")
    if on_card < 2:
        # `per` is an upper bound, so said as "one trajectory holds N GB of the card" it reads as
        # a refusal at the largest axis that works, which is the opposite of what it means.
        return 1, (f"at {tokens} tokens one trajectory is priced at up to {per * gb:.1f} GB "
                   f"against the {room * gb:.1f} GB the card has for them, so a second one does "
                   f"not fit and it runs one")
    return 1, (f"{free * gb:.1f} GB of host memory is free and a second trajectory at {tokens} "
               f"tokens needs {needs(2) * gb:.1f} GB")


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
def interleave(trajectories: int = 2, *, tokens: int):
    """Install the gate so `trajectories` threads can share one card.

    The switch, and the whole of it. Outside this block `card()` is a no-op and nothing in
    tt-bio behaves differently, which is what keeps a lever that changes the resident
    footprint off by default.

        with duotraj.interleave(trajectories=2, tokens=288) as gate:
            duotraj.run([lambda: design(0), lambda: design(1)])
    """
    refuse_if_it_will_not_fit(trajectories - 1, tokens=tokens)
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
