"""Provenance on every run by default: the clock, the seed, the sha, the config, the bar.

**The clock is sampled DURING the run, not before it.** That is the whole reason this is a
module and not a dict literal. On Blackhole the AICLK sets the fold time -- the 512 aa cell
reads 21.90 s at 800 MHz, 17.34 s at ~1063 and 14.69 s in the 1350 burst -- so a number
recorded against a clock read before the work started is not a measurement of the work. Three
investigations on this fleet blamed a bad board, a firmware regression and a UMD flag before a
tt-flash and a 4-chip reset cleared a max arbiter latched at 800 MHz. A sampler that runs
while the step runs is what makes that visible in the record instead of in the next campaign.

Sampling is sysfs reads and ``/proc/<pid>/fd`` readlinks. Nothing here opens a device, so it
cannot perturb what it measures and it does not need a card lease. It samples **every** node
and separately records which nodes the process has open, because ``tt-smi``'s index and the
``/dev/tenstorrent/N`` node number are not the same thing on this host -- a sampler keyed to
``TT_VISIBLE_DEVICES`` can watch a different chip than the one computing, which is how A1's
first pass read 800 MHz flat off an idle card.

The accuracy field is a deviation against a bar with the seed floor beside it, never a bare
"passed". Parity here means accurate enough, not bit-exact: at 512 aa the structural kill bar
is 0.60 A while re-running with a different seed moves the structure 1.84 A, so a reading
without the floor next to it cannot be judged.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["Provenance", "during", "record", "clocks", "git_sha", "KILL_BAR_A",
           "SEED_FLOOR_A", "ARTIFACT_CLOCK_MHZ"]


_SYSFS = Path("/sys/class/tenstorrent")

# The 512 aa structural bar and the seed scatter beside it, both measured. A lever that moves
# the structure less than the seed floor is smaller than variation already accepted.
KILL_BAR_A = 0.60
SEED_FLOOR_A = 1.84

# Below this the chip is not running at speed and a timing is an artifact of the governor
# rather than of the code. A result recorded under it gets flagged, not reported as a
# regression: that misdiagnosis has cost this fleet three investigations.
ARTIFACT_CLOCK_MHZ = 1200


def _nodes() -> List[int]:
    try:
        return sorted(int(p.name.split("!")[1]) for p in _SYSFS.glob("tenstorrent!*"))
    except OSError:
        return []


def clocks() -> Dict[int, Optional[int]]:
    """Every card's AICLK in MHz, by device node. ``None`` for a node that would not read."""
    out = {}
    for n in _nodes():
        try:
            out[n] = int((_SYSFS / f"tenstorrent!{n}" / "tt_aiclk").read_text().strip())
        except (OSError, ValueError):
            out[n] = None
    return out


def open_nodes(pid: Optional[int] = None) -> List[int]:
    """The device nodes this process has open. Which chip ran it, read off the process."""
    pid = os.getpid() if pid is None else pid
    found = set()
    try:
        for fd in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if "/dev/tenstorrent/" in target:
                try:
                    found.add(int(target.rsplit("/", 1)[1]))
                except ValueError:
                    pass
    except (OSError, PermissionError):
        pass
    return sorted(found)


def git_sha(repo: Optional[Path] = None) -> str:
    """The commit the code ran at, with ``-dirty`` when the tree is not clean.

    ``-dirty`` is not cosmetic. A sha that does not describe the code that ran is worse than
    no sha, because it looks reproducible.
    """
    root = Path(repo) if repo else Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return f"{sha[:12]}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class Provenance:
    """One run's record. ``aiclk`` is populated by the sampler, so it cannot be set by hand.

    ``deviation_a``/``bar_a``/``seed_floor_a`` are the accuracy triple. Set ``deviation_a``
    and ``passed`` answers itself; leave it unset and the record says the accuracy was not
    measured rather than implying it was fine.
    """

    seed: Optional[int] = None
    git_sha: str = "unknown"
    config: dict = field(default_factory=dict)
    aiclk: Dict[str, object] = field(default_factory=dict)
    device_nodes: List[int] = field(default_factory=list)
    seconds: Optional[float] = None
    deviation_a: Optional[float] = None
    bar_a: float = KILL_BAR_A
    seed_floor_a: float = SEED_FLOOR_A

    @property
    def clock_mhz(self) -> Optional[float]:
        """The median AICLK over the samples, on the nodes the process actually held."""
        return self.aiclk.get("median")

    @property
    def clock_is_artifact(self) -> bool:
        """True when the chip was too slow for a timing off this run to mean anything."""
        m = self.clock_mhz
        return m is not None and m < ARTIFACT_CLOCK_MHZ

    @property
    def passed(self) -> Optional[bool]:
        if self.deviation_a is None:
            return None
        return self.deviation_a <= self.bar_a

    def summary(self) -> str:
        clk = "no clock sampled" if self.clock_mhz is None else (
            f"{self.clock_mhz:.0f} MHz median during"
            + (f" -- BELOW {ARTIFACT_CLOCK_MHZ} MHz, any timing here is a governor artifact"
               if self.clock_is_artifact else ""))
        acc = ("accuracy not measured" if self.deviation_a is None else
               f"{self.deviation_a:.3f} A against a {self.bar_a:.2f} A bar, seed floor "
               f"{self.seed_floor_a:.2f} A -- {'PASS' if self.passed else 'FAIL'}")
        secs = "" if self.seconds is None else f", {self.seconds:.2f} s"
        return (f"seed {self.seed}, {self.git_sha}, nodes {self.device_nodes or 'none'}, "
                f"{clk}{secs}\n  {acc}")

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("seed", "git_sha", "config", "aiclk", "device_nodes", "seconds",
              "deviation_a", "bar_a", "seed_floor_a")}
        d["clock_is_artifact"] = self.clock_is_artifact
        d["passed"] = self.passed
        return d

    def write(self, path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, default=str) + "\n")


class _Sampler(threading.Thread):
    """Reads every card's AICLK on an interval while the run runs.

    Daemon, so a run that dies does not leave it holding the interpreter open. It records the
    per-sample maximum across the nodes the process holds, which is the clock the work saw --
    a minimum across all nodes would report whichever idle co-tenant card is parked lowest.
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._done = threading.Event()
        self.samples: List[float] = []
        self.nodes: set = set()

    def run(self) -> None:
        while not self._done.is_set():
            mine = open_nodes()
            self.nodes |= set(mine)
            c = clocks()
            watch = [c[n] for n in (mine or list(c)) if c.get(n) is not None]
            if watch:
                self.samples.append(float(max(watch)))
            self._done.wait(self.interval)

    def stop(self) -> dict:
        self._done.set()
        self.join(timeout=self.interval * 4)
        s = sorted(self.samples)
        if not s:
            return {"samples": 0, "median": None, "min": None, "max": None,
                    "why": "no sysfs AICLK node readable; this host has no Tenstorrent card "
                           "visible, so there is no clock to report"}
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
        return {"samples": len(s), "median": median, "min": s[0], "max": s[-1]}


def record(*, seed: Optional[int] = None, config: Optional[dict] = None,
           repo: Optional[Path] = None) -> Provenance:
    """A record with everything that is knowable before the run. No clock yet, by design."""
    return Provenance(seed=seed, git_sha=git_sha(repo), config=dict(config or {}))


@contextmanager
def during(*, seed: Optional[int] = None, config: Optional[dict] = None,
           repo: Optional[Path] = None, interval: float = 0.5):
    """Sample the clock for the duration of the block and fill in the record.

    ::

        with train.provenance.during(seed=0, config=cfg) as prov:
            run()
        print(prov.summary())

    The sampler starts before the block and stops after it, so every sample is from inside the
    work. Wall time is recorded here too, since a duration and the clock it ran at are only
    meaningful together.
    """
    prov = record(seed=seed, config=config, repo=repo)
    sampler = _Sampler(interval=interval)
    sampler.start()
    t0 = time.monotonic()
    try:
        yield prov
    finally:
        prov.seconds = time.monotonic() - t0
        prov.aiclk = sampler.stop()
        prov.device_nodes = sorted(sampler.nodes)
