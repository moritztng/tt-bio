"""One place decides whether an AICLK reading is a measurement.

A Blackhole whose ARC firmware has died still enumerates, still opens and still computes.
What it stops doing is answering telemetry: ``tt_aiclk`` returns 4294967295 (0xFFFFFFFF) and
nothing raises. So a perf run on that chip does not fail, it succeeds and writes an artifact
stamped with a 4.29-billion-MHz clock. That happened on 2026-09-26: a five-minute anchor on
qb1 card 3 wrote a ``round_events.json`` in which every AICLK sample was the sentinel. The
file named its card correctly and looked structurally fine.

Two kinds of caller, two behaviours, one predicate:

* a **sampler** describes the host. It reports ``None`` and keeps going, exactly as it
  already does for a node it cannot read. :func:`read` is its door.
* a **harness about to stamp a number** refuses. A fold timed against a clock nobody can read
  is not a measurement, and writing it down is worse than not measuring, because the artifact
  looks fine. :func:`require` is its door.

Shell readers reach the same predicate through ``perf/lib/aiclk.sh``, which shells out to
``python3 -m tt_bio.aiclk`` when it can and falls back to the same range test inline.

``tt_heartbeat`` and ``tt_serial`` are read off the same dead ARC and answer with the same
sentinel. What transfers to them is :func:`is_dead_arc` and not :func:`sane`: a heartbeat of
4211 is ordinary and a clock of 4211 MHz is not, so the plausibility bound is AICLK's alone.
Their readers are still blind and are a follow-up, not this change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

__all__ = ["ARC_DEAD", "MAX_PLAUSIBLE_MHZ", "SYSFS", "DeadARC", "is_dead_arc", "sane",
           "parse", "node_path", "read", "require"]

#: What every telemetry attribute on a dead ARC answers with.
ARC_DEAD = 0xFFFFFFFF

#: Nothing Tenstorrent ships clocks near this. Blackhole idles at 800 MHz and bursts to 1350.
#: The bound is the predicate rather than an equality against :data:`ARC_DEAD` alone, so a
#: torn read or a second sentinel is rejected by the same door instead of needing its own.
MAX_PLAUSIBLE_MHZ = 3000

SYSFS = Path("/sys/class/tenstorrent")


class DeadARC(RuntimeError):
    """Raised where a clock is about to be written down and there is no clock to write."""


def is_dead_arc(text) -> bool:
    """True when a telemetry read is the dead-ARC sentinel, whatever attribute it came from.

    The only part of this module that is not about clocks. ``tt_heartbeat`` and ``tt_serial``
    answer with the same value off the same dead ARC, and their readers are as blind to it as
    the clock readers were.
    """
    try:
        return int(str(text).split()[0]) == ARC_DEAD
    except (IndexError, ValueError):
        return False


def sane(mhz) -> bool:
    """True when *mhz* could plausibly be a clock this chip ran at."""
    return isinstance(mhz, int) and not isinstance(mhz, bool) and 0 < mhz <= MAX_PLAUSIBLE_MHZ


def parse(text: str) -> Optional[int]:
    """The MHz in a sysfs read, or ``None`` when it is not a clock.

    Takes the raw text so a caller that already has the bytes -- a replayed log line, a
    captured artifact -- runs the same predicate as a live read.
    """
    try:
        mhz = int(text.split()[0])
    except (AttributeError, IndexError, ValueError):
        return None
    return mhz if sane(mhz) else None


def node_path(node: Union[int, str, Path], sysfs: Optional[Path] = None) -> Path:
    """The ``tt_aiclk`` file for a device node, given a node number or a path to either."""
    if isinstance(node, int):
        return Path(sysfs or SYSFS) / f"tenstorrent!{node}" / "tt_aiclk"
    p = Path(node)
    return p if p.name == "tt_aiclk" else p / "tt_aiclk"


def read(node: Union[int, str, Path] = 0, sysfs: Optional[Path] = None) -> Optional[int]:
    """This node's AICLK in MHz, or ``None`` for a node that would not read *or would lie*.

    The sampler's contract: a dead ARC and an absent node are both "no clock here", because
    to a sampler they are the same fact and neither is worth crashing over.
    """
    try:
        return parse(node_path(node, sysfs).read_text())
    except OSError:
        return None


def require(node: Union[int, str, Path] = 0, sysfs: Optional[Path] = None) -> int:
    """This node's AICLK, or :class:`DeadARC`. For a caller that is about to stamp it."""
    path = node_path(node, sysfs)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise DeadARC(f"{path}: unreadable ({exc})") from exc
    mhz = parse(raw)
    if mhz is None:
        stripped = raw.strip()
        why = ("the ARC is dead (0xFFFFFFFF)" if stripped == str(ARC_DEAD)
               else f"not a plausible clock (0 < MHz <= {MAX_PLAUSIBLE_MHZ})")
        raise DeadARC(f"{path} reads {stripped!r}: {why}")
    return mhz


def main(argv=None) -> int:
    """``python3 -m tt_bio.aiclk [node]`` -- MHz on stdout, or ``DEAD``/``NA`` and a non-zero
    exit. This is the shell readers' door; see ``perf/lib/aiclk.sh``."""
    argv = sys.argv[1:] if argv is None else argv
    node: Union[int, str] = argv[0] if argv else 0
    if isinstance(node, str) and node.isdigit():
        node = int(node)
    try:
        print(require(node))
    except DeadARC as exc:
        print("NA" if "unreadable" in str(exc) else "DEAD")
        print(exc, file=sys.stderr)
        return 2 if "unreadable" in str(exc) else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
