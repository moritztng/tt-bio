"""Hold the Blackhole ARC clock at burst for the length of a run.

A fold blocks on hundreds of host syncs, and the clock governor drops the chip back to its 800 MHz
floor in the gaps, so the run pays a clock it never needed to pay. Measured on qb2 (p300c, Boltz-2
at 512 aa, eight folds per arm alternating inside one process): 18.99 s median on the governor
against 14.987 s with the clock held at 1350 MHz, so the fold is 1.267x faster. That ratio is a
floor rather than an estimate, because each forced fold leaves the governor boosted into the
control fold that follows it; the first control fold of the session, from a cold governor, was
21.130 s.

Two things this is not. It is not doing less of the model's own work: the same steps, the same
recycles, the same kernels, one clock. And it is not a precision change. Across two sessions and 33
folds the held arm never wrote a structure the governor arm did not also write, and in the second
session every fold in both arms was bit-identical. The first session's governor arm did produce
three different structures to the held arm's one, but that is the run-to-run nondeterminism
Blackhole already shows at 512 aa, not something the clock fixes: one session is not enough to
claim a steady clock buys determinism, and this code does not claim it.

Off by default, for two reasons worth stating plainly. It costs about 30 W of extra card power
while a fold is running (56-57 W against 98-103 W), and it drives the chip through an ARC message
tt-kmd does not document. So it is opt-in per run:

    TT_BIO_AICLK=1350 tt-bio predict target.yaml

The documented alternative, `TENSTORRENT_IOCTL_SET_POWER_STATE` with `TT_POWER_FLAG_MAX_AI_CLK`,
is accepted by firmware 19.11 and does not move the clock. That was already suspected from a
saturating matmul, which proves nothing either way because such a load is pinned at the governor's
loaded ceiling anyway; it has now been measured against a real fold, which is the case that has
the idle gaps, and it is still flat.
"""
from __future__ import annotations

import atexit
import fcntl
import os
import signal
import struct
import threading
from pathlib import Path

# _IO(TENSTORRENT_IOCTL_MAGIC, 17) -- tt-kmd owns the ARC message queue and multiplexes it over
# every open fd, so this rides alongside whatever UMD is doing on the chip instead of racing it.
_IOCTL_SMC_MSG = (0xFA << 8) | 17
_POST, _POLL = 1 << 0, 1 << 1
_FORCE_AICLK = 0x33            # argument is the target in MHz; 0 hands the clock back
_LAYOUT = "=IIII8I"            # struct tenstorrent_smc_msg: message[0] type, [1..7] arguments

# How far under target the chip has to read before the clock counts as lost. The governor's rungs
# are hundreds of MHz apart, so this only has to clear telemetry jitter.
_SAG_MHZ = 50
_POLL_S = 0.25

_hold: "_Hold | None" = None


def _aiclk(node: int) -> int:
    return int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read_text())


def _open_nodes() -> list:
    """Which /dev/tenstorrent/N this process has open.

    TT_VISIBLE_DEVICES is a UMD logical id and is not the device node, so the force has to be
    aimed off our own fd table rather than off the environment.
    """
    nodes = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent/"):
            tail = target.rsplit("/", 1)[1]
            if tail.isdigit():
                nodes.add(int(tail))
    return sorted(nodes)


class _Hold:
    """A forced AICLK on every chip this process has open, and the thread that keeps it there.

    The force is chip state, not fd state: it outlives the fd it was sent on and it outlives the
    process. That is why release is wired to atexit and to the fatal signals as well as to the
    normal path -- a run that died without releasing would leave the card at burst and ~40 W above
    idle until somebody noticed.

    It is also not quite sticky. Any legacy open of the same chip makes tt-kmd re-send an
    aggregated power state whose AICLK bit is clear (`chardev.c:987`, the default for an open
    without O_APPEND, which is every UMD open), and that puts the clock back under the governor.
    Measured rate on a Boltz-2 512 aa run: one clear in thirteen folds. Cheap to repair, so the
    watchdog repairs it rather than the caller discovering a silently slow fold.
    """

    def __init__(self, nodes: list, mhz: int):
        self.mhz = mhz
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.reasserts = 0
        # O_APPEND so that opening this fd does not itself emit the legacy "min AI clock"
        # aggregate that we are about to fight (`no_power_contrib`, chardev.c:1074).
        self.fds = {n: os.open(f"/dev/tenstorrent/{n}", os.O_RDWR | os.O_APPEND) for n in nodes}
        self._apply(mhz)
        self.thread = threading.Thread(target=self._watch, name="tt-bio-aiclk", daemon=True)
        self.thread.start()

    def _smc(self, fd: int, msg_type: int, *args: int) -> tuple:
        msg = [msg_type] + list(args) + [0] * (7 - len(args))
        fcntl.ioctl(fd, _IOCTL_SMC_MSG, struct.pack(_LAYOUT, 48, _POST, 0, 0, *msg))
        buf = bytearray(struct.pack(_LAYOUT, 48, _POLL, 0, 0, *([0] * 8)))
        for _ in range(400):                       # 2 s at the 5 ms retry below
            try:
                fcntl.ioctl(fd, _IOCTL_SMC_MSG, buf, True)
            except OSError as e:
                if e.errno == 11:                  # EAGAIN: the response is not back yet
                    self.stop.wait(0.005)
                    continue
                raise
            status = struct.unpack(_LAYOUT, bytes(buf))[4] & 0xFF
            return status
        raise TimeoutError(f"no ARC response to message 0x{msg_type:02X}")

    def _apply(self, mhz: int) -> None:
        with self.lock:
            for node, fd in self.fds.items():
                status = self._smc(fd, _FORCE_AICLK, mhz)
                # 0xFF is the firmware saying it does not know the message.
                if status != 0:
                    raise OSError(
                        f"FORCE_AICLK({mhz}) refused on /dev/tenstorrent/{node}, "
                        f"status 0x{status:02X}")

    def _watch(self) -> None:
        while not self.stop.wait(_POLL_S):
            for node, fd in self.fds.items():
                try:
                    if _aiclk(node) >= self.mhz - _SAG_MHZ:
                        continue
                except (OSError, ValueError):
                    continue
                with self.lock:
                    if self.stop.is_set():
                        return
                    try:
                        self._smc(fd, _FORCE_AICLK, self.mhz)
                    except (OSError, TimeoutError):
                        continue
                    self.reasserts += 1

    def release(self) -> None:
        self.stop.set()
        for node, fd in list(self.fds.items()):
            try:
                self._smc(fd, _FORCE_AICLK, 0)
            except (OSError, TimeoutError):
                pass
            os.close(fd)
        self.fds.clear()


def _install_signal_release() -> None:
    """Release on the signals that would otherwise skip atexit.

    A library taking over signal handlers is rude, so this only happens once the caller has asked
    for a hold, and it chains to whatever handler was already installed instead of replacing it.
    """
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        previous = signal.getsignal(sig)

        def handler(signum, frame, previous=previous):
            release()
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            signal.signal(sig, handler)
        except ValueError:            # not the main thread; atexit still covers the normal path
            return


def engage(arch: str) -> int:
    """Hold the clock if $TT_BIO_AICLK asks for it. Returns the target in MHz, or 0 for off.

    Call once the device is open, so the chips to aim at can be read off our own fd table.
    """
    global _hold
    want = os.environ.get("TT_BIO_AICLK", "").strip().lower()
    if _hold is not None or want in ("", "0", "off", "false", "no"):
        return 0
    if not want.isdigit():
        raise ValueError(
            f"TT_BIO_AICLK={want!r} is not a clock: give a target in MHz, e.g. TT_BIO_AICLK=1350 "
            "(the burst clock measured on Blackhole p300c), or leave it unset for the governor.")
    if arch != "blackhole":
        # 0x33 is a Blackhole ARC message. On another architecture that opcode is not known to
        # mean this, and guessing at a power-management message is not worth a clock.
        raise ValueError(
            f"TT_BIO_AICLK is Blackhole-only; this host reports {arch!r}. Unset it to run on the "
            "governor.")
    nodes = _open_nodes()
    if not nodes:
        raise RuntimeError("TT_BIO_AICLK was asked for before any Tenstorrent chip was open")
    _hold = _Hold(nodes, int(want))
    atexit.register(release)
    _install_signal_release()
    return _hold.mhz


def release() -> None:
    """Hand the clock back to the governor. Safe to call twice, and on a process that never held."""
    global _hold
    hold, _hold = _hold, None
    if hold is not None:
        hold.release()


def status() -> dict:
    """What the hold is doing, for the perf harnesses. Empty when the clock is not held."""
    return {} if _hold is None else {"mhz": _hold.mhz, "nodes": sorted(_hold.fds),
                                     "reasserts": _hold.reasserts}
