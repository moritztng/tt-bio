"""Who else held a Tenstorrent node WHILE the run ran, sampled rather than snapshotted.

A before-and-after check cannot see a cotenant that arrives and leaves inside the measurement,
and on this box that is the common case rather than the corner one: `bfp8-l1-chunking` takes and
releases a card in **4-second** leases, so a one-shot check either side of a five-minute
measurement is blind to dozens of them. That is not hypothetical -- this row's first 2-chip
reading was verified free immediately before and immediately after and is therefore *unprovable*
rather than wrong, which is why it had to be taken again.

**And the node is not the card number.** `TT_VISIBLE_DEVICES=3` opened `/dev/tenstorrent/0` on
qb1. So a check that compares cotenants against a logical card number reads a conflict where
there is none and misses the one there is. This asks the kernel which node THIS process opened,
after the device is up, and splits what it finds three ways, because the three mean different
things:

* **same node** -- another process computing on our chip. Contention for the chip itself;
* **own rank** -- one of this run's other data-parallel ranks. Expected, not contention;
* **elsewhere** -- a different chip on this host. Contention for the host cores and the power
  rail, which for a step that is 59 % host torch matters as much as the chip does.

``step_time.device_holders`` is called rather than reimplemented: it already counts the pids
whose ``fd`` directory it could not read instead of dropping them, and a quiet check that
silently discards unknowns is blind exactly where it matters.

**Own ranks are recognised by an environment token, not by their command line, and the first
cut of this file got that wrong in a way worth recording.** ``device_holders`` truncates each
cmdline to 90 characters, and a rank's is
``/home/ttuser/tt-bio-dev/env/bin/python3 /home/ttuser/.coworker/wt/train-b3-train/scripts/a`` --
cut off before ``repro.py``, so a substring test for the script name never matched and every
2-chip measurement reported itself as contended by its own sibling. An instrument that cries
wolf on the healthy case is worse than no instrument, because the next contended reading gets
explained away too. The supervisor stamps ``ABB3_RUN_ID`` into each rank's environment and this
reads it out of ``/proc/<pid>/environ``, which is exact and cannot be truncated away.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

__all__ = ["CotenantSampler"]

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "abb3_port"


def _holders():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from step_time import device_holders, our_nodes
    return device_holders, our_nodes


class CotenantSampler(threading.Thread):
    """Poll the holders of every ``/dev/tenstorrent/*`` node for the life of the run.

    Daemon, so a run killed by a watchdog reset does not leave it holding the interpreter open.
    ``run_id`` identifies this run's own ranks; it defaults to this process's own
    ``ABB3_RUN_ID``, so a sibling rank is reported as a sibling instead of inflating the
    cotenant count. An empty ``run_id`` means no rank can be recognised, and then every
    sibling counts as foreign -- honest, and it is why the supervisor always sets it.
    """

    #: The environment variable the supervisor stamps into every rank of one run.
    RUN_ID_ENV = "ABB3_RUN_ID"

    def __init__(self, *, interval: float = 2.0, run_id: str | None = None):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.run_id = run_id if run_id is not None else os.environ.get(self.RUN_ID_ENV, "")
        self._done = threading.Event()
        self.samples = 0
        self.same_node_samples = 0
        self.elsewhere_samples = 0
        self.same_node: dict = {}
        self.elsewhere: dict = {}
        self.own_ranks: dict = {}
        self.blind_pids: set = set()
        self.nodes: set = set()

    def run(self) -> None:
        device_holders, our_nodes = _holders()
        me = os.getpid()
        while not self._done.is_set():
            try:
                mine = our_nodes()
                self.nodes |= set(mine)
                holders, blind = device_holders()
                self.blind_pids |= set(blind)
                # tt-smi opens a node to read telemetry, so the clock sampler would otherwise
                # appear as a cotenant of its own measurement.
                holders = [h for h in holders if "tt-smi" not in h[2] and h[0] != me]
                own_pids = {h[0] for h in holders if self._is_own_rank(h[0])}
                same = [h for h in holders if h[1] in mine and h[0] not in own_pids]
                own = [h for h in holders if h[0] in own_pids]
                other = [h for h in holders
                         if h[1] not in mine and h[0] not in own_pids]
                for pid, node, cmd in same:
                    self.same_node[pid] = f"{node} {cmd}"
                for pid, node, cmd in other:
                    self.elsewhere[pid] = f"{node} {cmd}"
                for pid, node, cmd in own:
                    self.own_ranks[pid] = f"{node} {cmd}"
                self.samples += 1
                self.same_node_samples += bool(same)
                self.elsewhere_samples += bool(other)
            except Exception:
                # A sampler that can raise takes the run with it. A failed read is a lost
                # sample and the count below is what says how many were actually taken.
                pass
            self._done.wait(self.interval)

    def _is_own_rank(self, pid: int) -> bool:
        """Is ``pid`` another rank of THIS run? Read from its environment, not its cmdline.

        ``/proc/<pid>/environ`` is readable for our own user's processes and is not truncated,
        which is exactly what the cmdline test could not manage. A pid we cannot read is not
        claimed as our own: an unknown counted as a sibling would hide a real cotenant.
        """
        if not self.run_id:
            return False
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                env = fh.read()
        except OSError:
            return False
        # Exact entry match on the NUL-separated list, not a substring of the whole blob: a
        # substring test would match a different variable that merely contains our id.
        return f"{self.RUN_ID_ENV}={self.run_id}".encode() in env.split(b"\0")

    def stop(self) -> dict:
        self._done.set()
        self.join(timeout=self.interval * 4)
        return self.as_dict()

    @property
    def clean(self) -> bool:
        """True only if nothing foreign ever appeared, anywhere on the host, during the run.

        Deliberately strict about ``elsewhere``: 59 % of this step is host torch on an 8-core
        CPU, so a foreign process on another chip is contending for the cores this step needs.
        A measurement taken beside one is an upper bound on time, which is a different claim
        from a clean reading and has to be labelled as one.
        """
        return self.samples > 0 and not self.same_node and not self.elsewhere

    def as_dict(self) -> dict:
        return {"samples": self.samples, "nodes": sorted(self.nodes),
                "same_node_samples": self.same_node_samples,
                "elsewhere_samples": self.elsewhere_samples,
                "same_node": self.same_node, "elsewhere": self.elsewhere,
                "own_ranks": self.own_ranks,
                "blind_pids": sorted(self.blind_pids), "clean": self.clean}

    def summary(self) -> str:
        if not self.samples:
            return "COTENANCY: NO SAMPLES -- this measurement carries no cleanliness evidence"
        if self.clean:
            return (f"COTENANCY: clean over {self.samples} samples DURING -- nodes "
                    f"{sorted(self.nodes)}, no foreign holder of any node on this host, "
                    f"{len(self.own_ranks)} own rank(s), {len(self.blind_pids)} unreadable pid(s)")
        return (f"COTENANCY: CONTENDED -- {self.same_node_samples}/{self.samples} samples had a "
                f"foreign holder of OUR node(s) {sorted(self.nodes)} "
                f"{list(self.same_node.values())[:3]}, and {self.elsewhere_samples}/"
                f"{self.samples} had one elsewhere on the host "
                f"{list(self.elsewhere.values())[:3]}. This timing is an UPPER bound, not a "
                f"clean reading")
