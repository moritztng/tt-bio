"""What a fair TT measurement records about the box it ran on, shared by the campaign harnesses
(perf/rfd4/fair, perf/mol1/fair): foreign CPU and device holders sampled during a batch, the board type,
and FAIR.md history carried across verdicts."""
import json
import os
import statistics
import subprocess
import threading
import time



class Contention:
    """Foreign CPU cores and foreign device holders, sampled every `period` s while open."""

    def __init__(self, period=2.0):
        self.cores, self.holders = [], set()
        self._stop = threading.Event()
        self._period = period
        self._hz = os.sysconf("SC_CLK_TCK")

    @staticmethod
    def _box_busy():
        f = open("/proc/stat").readline().split()[1:]
        v = [int(x) for x in f]
        return sum(v) - v[3] - v[4]  # all minus idle and iowait

    @staticmethod
    def _own_busy():
        f = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        return sum(int(x) for x in f[11:15])  # utime stime cutime cstime

    @staticmethod
    def device_holders():
        """Pids, other than this process and its children, with a /dev/tenstorrent node open."""
        me = os.getpid()
        out = set()
        for p in os.listdir("/proc"):
            if not p.isdigit() or int(p) == me:
                continue
            try:
                if int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1]) == me:
                    continue  # our own tt-smi sampler
                for fd in os.listdir(f"/proc/{p}/fd"):
                    if os.readlink(f"/proc/{p}/fd/{fd}").startswith("/dev/tenstorrent"):
                        out.add(f"{p}:{open(f'/proc/{p}/comm').read().strip()}")
                        break
            except (OSError, ValueError, IndexError):
                continue
        return out

    @staticmethod
    def _proc_cpu():
        """{pid: (comm, utime+stime ticks)} for every process except this one and its children."""
        me, out = os.getpid(), {}
        for p in os.listdir("/proc"):
            if not p.isdigit() or int(p) == me:
                continue
            try:
                head, tail = open(f"/proc/{p}/stat").read().rsplit(")", 1)
                f = tail.split()
                if int(f[1]) != me:
                    out[p] = (head.split("(", 1)[1], int(f[11]) + int(f[12]))
            except (OSError, ValueError, IndexError):
                continue
        return out

    def _sample(self):
        box, own, t = self._box_busy(), self._own_busy(), time.monotonic()
        if t > self._last[2]:
            self.cores.append(max(0.0, (box - self._last[0]) - (own - self._last[1])) / self._hz / (t - self._last[2]))
        self._last = (box, own, t)
        self.holders |= self.device_holders()
        for p, (comm, _) in self._proc_cpu().items():  # short-lived processes exit before __exit__ can rank them
            if p not in self._seen:
                self._seen.add(p)
                self.spawned[comm] = self.spawned.get(comm, 0) + 1

    @classmethod
    def wait_quiet(cls, cap, window=10.0, timeout=1800.0):
        """Block until a `window`-s sample of foreign cores has a median at or under `cap`; seconds waited.

        A batch on a loud box is discarded, so it costs a full batch for no admitted line. Waiting here is
        cheaper. Gives up after `timeout` and lets the batch run (and be judged) anyway.
        """
        t0 = time.monotonic()
        while True:
            with cls(period=1.0) as c:
                time.sleep(window)
            med = statistics.median(c.cores) if c.cores else 0.0
            if med <= cap or time.monotonic() - t0 > timeout:
                return round(time.monotonic() - t0 - window, 1), round(med, 2), c.summary()
            time.sleep(20)

    def _run(self):
        while not self._stop.wait(self._period):
            self._sample()

    def __enter__(self):
        self._last = (self._box_busy(), self._own_busy(), time.monotonic())
        self._t0, self._cpu0 = self._last[2], self._proc_cpu()
        self._seen, self.spawned = set(self._cpu0), {}
        self.holders |= self.device_holders()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=10)
        self._sample()  # the tail since the last tick, so a short batch is never unsampled
        dt = max(time.monotonic() - self._t0, 1e-9)
        use = [(n - self._cpu0.get(p, (c, 0))[1], f"{p}:{c}") for p, (c, n) in self._proc_cpu().items()]
        self.top = [{"proc": k, "cores": round(d / self._hz / dt, 2)} for d, k in sorted(use, reverse=True)[:5]
                    if d / self._hz / dt >= 0.05]
        self.top_cores = round(sum(d for d, _ in use) / self._hz / dt, 2)
        return False

    def summary(self):
        c = sorted(self.cores)
        return {"foreign_cores_median": round(statistics.median(c), 2) if c else None,
                "foreign_cores_max": round(c[-1], 2) if c else None,
                "foreign_device_holders": sorted(self.holders),
                "foreign_top": getattr(self, "top", []),
                # foreign cores the surviving processes do not account for: work by processes that exited
                "foreign_unattributed_cores": round(max(0.0, statistics.mean(c) - getattr(self, "top_cores", 0)), 2)
                if c else None,
                "foreign_spawned": dict(sorted(self.spawned.items(), key=lambda kv: -kv[1])[:8])}


def board_type():
    """The granted chip's board type as tt-smi reports it (index 0 under TT_VISIBLE_DEVICES)."""
    import clocksample  # here, not at import: the CPU-only verdict scripts import this module without tt_bio
    try:
        d = json.loads(subprocess.run([clocksample.TT_SMI, "-s"], capture_output=True, text=True, timeout=60).stdout)
        return d["device_info"][0]["board_info"].get("board_type")
    except Exception as exc:  # noqa: BLE001 -- recorded, and --part is then unverified
        return f"unknown ({type(exc).__name__})"


HISTORY = "## Earlier verdicts"


def carry_history(path, fair, written):
    """The new FAIR.md, with every earlier verdict kept under one history heading.

    A fair number is evidence: rerunning a verdict script must not delete the verdict it supersedes
    (it did, 2026-10-23, and the history had to be pasted back from a tool log).
    """
    if not path.exists():
        return fair
    old = path.read_text()
    head, _, tail = old.partition(HISTORY)
    older = tail.strip()
    block = [HISTORY, "", f"### Superseded at {written}", "", head.strip()]
    if older:
        block += ["", older]
    return fair.rstrip() + "\n\n" + "\n".join(block) + "\n"
