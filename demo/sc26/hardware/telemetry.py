"""Per-chip telemetry for the SC26 booth: clock, power, temperature, liveness and what each chip folds.

Every reading comes from the files the tenstorrent kernel driver already publishes under
``/sys/class/tenstorrent/tenstorrent!N/``: ``tt_aiclk`` and ``tt_heartbeat`` on the class node, and
power, temperature, current and core voltage in its hwmon directory. Reading them is a handful of
small file reads per chip. Nothing here opens a device, takes a chip lock or spawns ``tt-smi``, so
sampling cannot disturb a fold, and a fold cannot make sampling hang.

``tt-smi`` is only a fallback for a driver too old to publish ``tt_aiclk``. It then runs at most once
at a time, in its own process group, under a timeout, and is reaped on every path.

Three things a chip can say that are not readings, and how each one renders:

* A dead ARC answers ``tt_aiclk`` and ``tt_heartbeat`` with 0xFFFFFFFF without raising (tt-smi masks
  the same value to 65535). That is ``state: "resetting"``, never a 4.3 GHz clock.
* A heartbeat that stops advancing is a chip whose firmware has stopped, whatever its clock says.
* A node that disappears is a board being reset (``tt-smi -r`` takes both chips of a p300).

Fold activity comes from the engine, which appends one JSON line per fold start and end to an
events file (``record_fold``). The AICLK quoted next to a fold is the median of the samples taken
*during* that fold, so the number on screen carries the clock it was measured at.
"""
from __future__ import annotations

import json
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

SYSFS = Path("/sys/class/tenstorrent")
EVENTS = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "sc26" / "folds.jsonl"

#: Values a dead or resetting ARC returns in place of a reading: the sysfs u32 and tt-smi's 16-bit mask.
SENTINELS = {0xFFFFFFFF, 0xFFFF}
#: A healthy Blackhole heartbeat ticks about 10 times a second; no tick in this long is a stopped chip.
HEARTBEAT_STALL_S = 2.0
#: A Blackhole AICLK idles at its 800 MHz floor by design and ramps up under work.
IDLE_MHZ = 800
#: Samples kept per chip: 15 minutes at 4 Hz, enough to cover the longest fold the demo runs.
HISTORY = 3600


def reading(raw) -> int | None:
    """An integer telemetry value as read, or None when it is not a reading.

    Accepts what both sources produce: sysfs's ``"800\\n"`` and tt-smi's right-aligned ``" 800"``
    or ``"1350.0"``. Empty, ``N/A``, negative and sentinel values all come back as None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    try:
        v = int(float(s)) if re.fullmatch(r"\d+(?:\.\d*)?", s) else None
    except ValueError:
        v = None
    return None if v is None or v in SENTINELS else v


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:  # missing attribute, node vanishing mid-reset, or hwmon's ENODATA for fan1_input
        return None


@dataclass
class Raw:
    """One chip's files as read at time ``t``. None is a value the chip did not give."""
    node: int
    bdf: str
    board: str | None
    t: float
    aiclk: int | None
    heartbeat: int | None
    power_w: float | None
    temp_c: float | None
    vcore_v: float | None
    current_a: float | None
    power_max_w: float | None = None


def _scaled(raw: str | None, scale: float) -> float | None:
    v = reading(raw)
    return None if v is None else v / scale


def _bdf_key(bdf: str) -> tuple[int, ...]:
    return tuple(int(p, 16) for p in re.split(r"[:.]", bdf) if p)


class Chips:
    """The tenstorrent sysfs class under ``root``, read without touching a device."""

    def __init__(self, root: Path = SYSFS):
        self.root = Path(root)

    def nodes(self) -> dict[int, str]:
        """``/dev/tenstorrent/N`` node -> PCI BDF, for every node present now."""
        out = {}
        for d in self.root.glob("tenstorrent!*"):
            out[int(d.name.rsplit("!", 1)[1])] = Path(os.path.realpath(d / "device")).name.lower()
        return out

    def cards(self) -> dict[int, int]:
        """Card index (what ``TT_VISIBLE_DEVICES`` names) -> node.

        UMD numbers chips in PCI order, not in the driver's probe order, so the card index is the
        node's BDF rank. Same rule as ``tt_bio.runtime.tt_bdf_to_index``.
        """
        nodes = self.nodes()
        return {i: n for i, n in enumerate(sorted(nodes, key=lambda n: _bdf_key(nodes[n])))}

    def read(self, node: int, bdf: str) -> Raw:
        d = self.root / f"tenstorrent!{node}"
        hw = next(iter(sorted((d / "device" / "hwmon").glob("hwmon*"))), None)
        hwr = (lambda f: _read(hw / f)) if hw else (lambda f: None)
        board = (_read(d / "tt_card_type") or "").strip() or None
        return Raw(node=node, bdf=bdf, board=None if board == "unknown" else board, t=time.time(),
                   aiclk=reading(_read(d / "tt_aiclk")), heartbeat=reading(_read(d / "tt_heartbeat")),
                   power_w=_scaled(hwr("power1_input"), 1e6), temp_c=_scaled(hwr("temp1_input"), 1e3),
                   vcore_v=_scaled(hwr("in0_input"), 1e3), current_a=_scaled(hwr("curr1_input"), 1e3),
                   power_max_w=_scaled(hwr("power1_max"), 1e6))

    def has_sysfs_clock(self) -> bool:
        return any((d / "tt_aiclk").exists() for d in self.root.glob("tenstorrent!*"))


_TT_SMI_LOCK = threading.Lock()


def tt_smi_aiclk(cmd=("tt-smi", "-s"), timeout: float = 5.0) -> list[int | None] | None:
    """AICLK per chip from one ``tt-smi -s`` snapshot, for a driver with no ``tt_aiclk`` in sysfs.

    tt-smi talks to the chip, so it can hang on a wedged one. It therefore runs in its own session
    under a timeout, and on timeout its whole process group is sent SIGTERM and then reaped, so no
    tt-smi outlives this call. Only one runs at a time; a caller that finds one in flight gets None.
    """
    if not _TT_SMI_LOCK.acquire(blocking=False):
        return None
    try:
        p = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _reap(p)
            return None
        try:
            info = json.loads(out)["device_info"]
        except (ValueError, KeyError, TypeError):
            return None
        return [reading((dev.get("telemetry") or {}).get("aiclk")) for dev in info]
    finally:
        _TT_SMI_LOCK.release()


def _reap(p: subprocess.Popen) -> None:
    """Stop a stuck reader and everything it started, and wait for it, so nothing is orphaned."""
    for sig, grace in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            break
        try:
            p.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            continue
    p.wait()


def record_fold(event: str, card: int, path: Path = EVENTS, **fields) -> None:
    """Append one fold event for the telemetry to pick up. Called by the engine.

    ``event`` is ``start``, ``done`` or ``fail``; ``done`` carries ``seconds``, the fold's own wall
    time. A single short line written with O_APPEND is atomic, so several workers can share a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"t": round(time.time(), 3), "event": event, "card": card, **fields},
                      separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


class Folds:
    """Per-card fold ledger, built by tailing the events file ``record_fold`` writes."""

    def __init__(self, path: Path = EVENTS):
        self.path, self.pos, self.ino = Path(path), 0, None
        self.current: dict[int, dict] = {}
        self.done: dict[int, deque] = {}

    def poll(self) -> None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return
        if st.st_ino != self.ino or st.st_size < self.pos:  # replaced or truncated: read it again
            self.ino, self.pos, self.current, self.done = st.st_ino, 0, {}, {}
        with self.path.open("rb") as f:
            f.seek(self.pos)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1  # a line still being written waits for the next poll
        self.pos += end
        for line in chunk[:end].splitlines():
            try:
                self._apply(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue

    def _apply(self, e: dict) -> None:
        card = int(e["card"])
        if e["event"] == "start":
            self.current[card] = e
        elif e["event"] in ("done", "fail"):
            start = self.current.pop(card, None)
            if e["event"] == "done" and e.get("seconds"):
                done = self.done.setdefault(card, deque())
                done.append({**(start or {}), **e, "t0": (start or {}).get("t")})
                # Only today's folds are read. Keeping every fold since the file began grew the
                # engine's memory, and the time to build a snapshot, by the day.
                while done[0]["t"] < e["t"] - 86400:
                    done.popleft()


def _midnight(now: float) -> float:
    lt = time.localtime(now)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


class Monitor:
    """Samples every chip at ``rate_hz`` in a daemon thread; ``snapshot()`` is the stream message."""

    def __init__(self, root: Path = SYSFS, events: Path = EVENTS, rate_hz: float = 4.0):
        self.chips, self.folds, self.period = Chips(root), Folds(events), 1.0 / rate_hz
        self.hist: dict[int, deque] = {}
        # card -> (heartbeat, monotonic time it last changed). Monotonic, so an NTP step at the booth
        # cannot make a beating chip look stalled.
        self.last_beat: dict[int, tuple[int, float]] = {}
        self.latest: dict[int, Raw] = {}
        self.card_of: dict[int, int] = {}
        self.cost_s = deque(maxlen=240)  # wall time of each sample, for the overhead figure
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.expected: set[int] = set()

    # ---- sampling -------------------------------------------------------------------------------
    def sample(self) -> None:
        t0 = time.perf_counter()
        cards, nodes = self.chips.cards(), self.chips.nodes()
        raws = {c: self.chips.read(n, nodes[n]) for c, n in cards.items()}
        if not self.chips.has_sysfs_clock() and raws:
            clocks = tt_smi_aiclk() or []
            for c, mhz in zip(sorted(raws), clocks):
                raws[c].aiclk = mhz
        self.folds.poll()
        with self.lock:
            self.expected |= set(raws)
            self.card_of = cards
            for c, r in raws.items():
                self.latest[c] = r
                self.hist.setdefault(c, deque(maxlen=HISTORY)).append((r.t, r.aiclk, r.power_w, r.temp_c))
                if r.heartbeat is None or r.heartbeat != self.last_beat.get(c, (None,))[0]:
                    self.last_beat[c] = (r.heartbeat, time.monotonic())
            for c in self.expected - set(raws):  # a node that vanished: its board is being reset
                self.latest.pop(c, None)
        self.cost_s.append(time.perf_counter() - t0)

    def start(self) -> "Monitor":
        def loop():
            while True:
                t = time.monotonic()
                try:
                    self.sample()
                except Exception:  # a telemetry glitch must never take the demo's stream down
                    pass
                time.sleep(max(0.0, self.period - (time.monotonic() - t)))
        self.sample()
        self.thread = threading.Thread(target=loop, name="sc26-telemetry", daemon=True)
        self.thread.start()
        return self

    # ---- reading --------------------------------------------------------------------------------
    def state(self, card: int, now: float) -> str:
        r = self.latest.get(card)
        if r is None:
            return "resetting"
        since = self.last_beat.get(card, (None, time.monotonic()))[1]
        if r.aiclk is None or r.heartbeat is None or r.board is None or time.monotonic() - since > HEARTBEAT_STALL_S:
            return "resetting"
        if card in self.folds.current:
            return "folding"
        # Above the idle floor with no demo fold on it: the chip is working on somebody else's job.
        return "busy" if r.aiclk > IDLE_MHZ else "idle"

    def clock_during(self, card: int, t0: float, t1: float) -> dict | None:
        """Median and minimum AICLK over the samples taken between ``t0`` and ``t1``."""
        mhz = [a for t, a, _, _ in self.hist.get(card, ()) if t0 < t <= t1 and a is not None]
        if not mhz:
            return None
        return {"median": int(statistics.median(mhz)), "min": min(mhz), "n": len(mhz)}

    def snapshot(self) -> dict:
        now = time.time()
        day = _midnight(now)
        with self.lock:
            chips = [self._chip(c, now, day) for c in sorted(self.expected)]
            cost = sorted(self.cost_s)
        return {"type": "chips", "t": round(now, 3), "source": "live", "rate_hz": round(1 / self.period, 2),
                "sample_ms": round(1e3 * cost[len(cost) // 2], 3) if cost else None, "chips": chips}

    def _chip(self, card: int, now: float, day: float) -> dict:
        r, state = self.latest.get(card), self.state(card, now)
        live = state != "resetting"
        hist = [h for h in self.hist.get(card, ()) if h[0] >= now - 60]
        cur = self.folds.current.get(card)
        today = [f for f in self.folds.done.get(card, []) if f["t"] >= day]
        last = today[-1] if today else None
        res = sum(f.get("residues") or 0 for f in today)
        sec = sum(f["seconds"] for f in today if f.get("residues"))
        out = {
            "card": card, "node": self.card_of.get(card), "bdf": r.bdf if r else None,
            "board": r.board if r else None, "state": state,
            "aiclk_mhz": r.aiclk if r and live else None,
            "power_w": r.power_w if r and live else None,
            "temp_c": r.temp_c if r and live else None,
            "vcore_v": r.vcore_v if r and live else None,
            "power_max_w": r.power_max_w if r else None,
            "power_60s": [None if p is None else round(p, 1) for _, _, p, _ in hist[::4]],
            "folding": None if cur is None else {
                k: cur.get(k) for k in ("model", "name", "residues")} | {"elapsed_s": round(now - cur["t"], 1)},
            "folds_today": len(today),
            "residues_per_s_today": round(res / sec, 1) if sec else None,
            "last_fold": None,
        }
        if last:
            clk = self.clock_during(card, last["t0"], last["t"]) if last.get("t0") else None
            out["last_fold"] = {k: last.get(k) for k in ("model", "name", "residues", "seconds")} | {
                "residues_per_s": round(last["residues"] / last["seconds"], 1) if last.get("residues") else None,
                "aiclk_during": clk}
        return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Print one telemetry snapshot of every chip as JSON.")
    ap.add_argument("--events", type=Path, default=EVENTS)
    ap.add_argument("--seconds", type=float, default=2.5, help="sample this long first, so liveness is known")
    a = ap.parse_args()
    m = Monitor(events=a.events).start()
    time.sleep(a.seconds)
    print(json.dumps(m.snapshot(), indent=1))


if __name__ == "__main__":
    main()
