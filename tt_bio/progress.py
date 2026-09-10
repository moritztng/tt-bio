"""Real-time terminal progress display for tt-bio predict.

Uses Rich Live to show per-device status, stage progress bars,
and a rolling log of completed structures. Communicates with
worker processes via a multiprocessing Queue.
"""

import os
import sys as _sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty

# Live display refresh rate (Hz). Override with TT_BIO_REFRESH_HZ; higher makes
# fast stages (e.g. short diffusion) animate more smoothly at a small CPU cost.
REFRESH_HZ = max(1.0, float(os.environ.get("TT_BIO_REFRESH_HZ", "10")))

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

# ── Stage model ───────────────────────────────────────────────────────────
#
#   Start → MSA → Prep → Trunk(recycling) → Diffusion(steps) → Confidence → Save
#
# A stage owns a slice of the bar proportional to how long it takes. These
# starting weights are the median of four models measured at 20 aa,
# single-sequence, warm, on a p150a (`predict` stage timestamps, 2026-09-09):
#
#   model       total   prep   trunk   diffusion
#   esmfold2     21 s    43%     14%       38%
#   rf3         148 s    33%     41%       26%
#   opendde     161 s    17%     57%       25%
#   openfold3   234 s    16%     38%       46%
#
# No table fits all four, which is the point: they are only used until the first
# structure finishes, after which the display divides the bar by the seconds
# each stage actually took on this model, on this machine, in this run. The
# table they replace gave prep 6%, so the bar sat near 15% for the first third
# to half of every fold.
STAGE_ORDER = ("start", "msa", "prep", "trunk", "diffusion", "confidence", "saving")
DEFAULT_WEIGHTS = {"start": 0.01, "msa": 0.08, "prep": 0.25, "trunk": 0.35,
                   "diffusion": 0.28, "confidence": 0.02, "saving": 0.01}

# A stage with no measurement yet still gets this sliver of the bar, so it does
# not become a zero-width band the bar sits frozen in.
UNMEASURED_WEIGHT = 0.005

MIN_BAR, MAX_BAR = 8, 20
W_WORKER, W_NAME, W_STAGE, W_CNT = 18, 18, 18, 6
RECENT_MAX = 8


def _bands(weights: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Cumulative [start, end) of the bar owned by each stage, in order."""
    total = sum(max(0.0, weights.get(s, 0.0)) for s in STAGE_ORDER) or 1.0
    out, acc = {}, 0.0
    for s in STAGE_ORDER:
        w = max(0.0, weights.get(s, 0.0)) / total
        out[s] = (acc, acc + w)
        acc += w
    return out


def _columns(width: int) -> tuple[int, int, int, int]:
    """Column widths that fit `width` columns.

    The table used to be five fixed columns totalling 88 cells. In an 80-column
    terminal Rich shrank the bar and capped it with an ellipsis, so a finished
    bar and a 90% bar rendered identically. Give the bar whatever is left after
    the text columns, and shrink the text columns when even that is not enough.
    """
    pad = 8  # one space either side of five columns, minus the suppressed edges
    widths = {"worker": W_WORKER, "name": W_NAME, "stage": W_STAGE}
    floors = {"worker": 10, "name": 10, "stage": 12}
    short = (sum(widths.values()) + W_CNT + pad + MIN_BAR) - width
    for col in ("worker", "name", "stage"):
        if short <= 0:
            break
        take = min(short, widths[col] - floors[col])
        widths[col] -= take
        short -= take
    bar = width - (sum(widths.values()) + W_CNT + pad)
    return widths["worker"], widths["name"], max(MIN_BAR, min(bar, MAX_BAR)), widths["stage"]


@dataclass
class DeviceState:
    worker_id: str
    device_id: int | str
    host: str = ""
    accelerator: str = ""
    label: str = ""
    name: str = ""
    stage: str = "idle"
    step: int = 0
    total_steps: int = 0
    done: int = 0
    assigned: int = 0
    # Per-target stage timing, folded into the run's measured weights on "done".
    stage_since: float = 0.0
    stage_secs: dict[str, float] = field(default_factory=dict)
    # The bar for one target never moves backwards, even when a completed
    # target rewrites the stage weights mid-flight.
    frac_max: float = 0.0


class ProgressDisplay:
    """Drives a Rich Live display from a multiprocessing.Queue of events.

    Each event carries the worker's identity (host/accelerator/label) plus a
    kind-specific payload:
        {"worker": str, "dev": int, "event": "loading"}
        {"worker": str, "dev": int, "event": "start",   "name": str}
        {"worker": str, "dev": int, "event": "stage",   "stage": str, "step": int, "total": int}
        {"worker": str, "dev": int, "event": "done",    "name": str, "time": float, "status": str}
    """

    def __init__(self, queue, total: int, n_workers: int, model: str | None = None):
        self.queue = queue
        self.total = total
        self.n_workers = n_workers
        self.model = model

        self.devices: dict[str, DeviceState] = {}
        self.completed = 0
        self.failed = 0
        self.recent: list[dict] = []
        self.start_time = time.time()
        # When the first target actually started, i.e. with model load excluded.
        # An ETA that prices the one-off load into every remaining target is the
        # 55x over-estimate this replaces.
        self.first_start: float | None = None
        # Seconds per stage summed over finished targets; the bar's weights.
        self.stage_secs: dict[str, float] = {}

        self._lock = threading.Lock()
        self._console = Console(stderr=True)
        self._live = None
        self._thread = None
        self._stop = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self._live = Live(
            self._render(), console=self._console,
            refresh_per_second=REFRESH_HZ, transient=False,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._live:
            self._drain()
            self._live.update(self._render())
            self._live.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── background thread ─────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            self._drain()
            if self._live:
                self._live.update(self._render())
            self._stop.wait(1.0 / REFRESH_HZ)

    def _drain(self):
        while True:
            try:
                ev = self.queue.get_nowait()
            except (Empty, EOFError):
                break
            with self._lock:
                self._handle(ev)

    def _enter_stage(self, d: DeviceState, stage: str, now: float):
        """Charge the time since the last transition to the stage being left."""
        if d.stage_since and d.stage in STAGE_ORDER:
            d.stage_secs[d.stage] = d.stage_secs.get(d.stage, 0.0) + (now - d.stage_since)
        d.stage = stage
        d.stage_since = now

    def _handle(self, ev: dict):
        now = time.time()
        dev = ev.get("dev", 0)
        worker_id = str(ev.get("worker", dev))
        kind = ev["event"]

        # Every event carries the worker's host/accelerator/label in its meta,
        # so make sure the DeviceState reflects the latest values.
        d = self.devices.get(worker_id)
        if d is None:
            d = DeviceState(worker_id=worker_id, device_id=dev)
            self.devices[worker_id] = d
        if "host" in ev:
            d.host = ev["host"]
        if "accelerator" in ev:
            d.accelerator = ev["accelerator"]
        if "label" in ev:
            d.label = ev["label"]

        if kind == "loading":
            d.stage = "loading"
            d.stage_since = 0.0
            d.name = ""
        elif kind == "start":
            d.name = ev["name"]
            # "Starting" until the worker names a real stage. Claiming an MSA
            # stage here was wrong for every single-sequence fold: the live view
            # said MSA while results.json recorded "msa": false for the same job.
            d.step = 0
            d.total_steps = 0
            d.assigned += 1
            d.stage_secs = {}
            d.frac_max = 0.0
            d.stage_since = 0.0
            self._enter_stage(d, "start", now)
            if self.first_start is None:
                self.first_start = now
        elif kind == "stage":
            stage = ev.get("stage", d.stage)
            if stage != d.stage:
                self._enter_stage(d, stage, now)
            d.step = ev.get("step", 0)
            d.total_steps = ev.get("total", 0)
        elif kind == "done":
            self._enter_stage(d, "done", now)
            self.completed += 1
            ok = ev.get("status") == "ok"
            if not ok:
                self.failed += 1
            else:
                # Only a finished fold describes how the stages divide the work;
                # a job that failed in featurization would make `prep` look free.
                for s, secs in d.stage_secs.items():
                    self.stage_secs[s] = self.stage_secs.get(s, 0.0) + secs
            d.done += 1
            # Terminal state for this slot — "done" (green) on success, "failed"
            # (red) on error. We fill the bar either way: idle would render 0%
            # and look stuck. The next "start" resets it.
            d.stage = "done" if ok else "failed"
            self.recent.append(ev)
            if len(self.recent) > RECENT_MAX:
                self.recent.pop(0)

    # ── progress model ────────────────────────────────────────────────────

    def _weights(self) -> dict[str, float]:
        if not self.stage_secs:
            return DEFAULT_WEIGHTS
        floor = UNMEASURED_WEIGHT * sum(self.stage_secs.values())
        return {s: self.stage_secs.get(s, floor) for s in STAGE_ORDER}

    def _frac(self, d: DeviceState, bands: dict[str, tuple[float, float]]) -> float:
        s = d.stage
        if s in ("idle", "loading"):
            return 0.0
        if s in ("done", "failed"):
            return 1.0
        base, end = bands.get(s, (d.frac_max, d.frac_max))
        if d.total_steps > 0:
            # Steps are emitted 0-based, so step N means N steps are behind us
            # and the band is (N/total) consumed.
            f = base + (end - base) * min(d.step / d.total_steps, 1.0)
        else:
            # Stepless stage (MSA search / prep / confidence / saving): park
            # mid-band so the bar is never empty and still advances forward.
            f = (base + end) / 2
        d.frac_max = max(d.frac_max, f)
        return d.frac_max

    def _work_done(self, bands) -> float:
        """Progress in whole-target units: finished targets plus in-flight fractions."""
        inflight = sum(self._frac(d, bands) for d in self.devices.values()
                       if d.stage not in ("idle", "loading", "done", "failed"))
        return min(float(self.total), self.completed + inflight)

    # ── rendering ─────────────────────────────────────────────────────────

    @staticmethod
    def _bar(frac: float, width: int, failed: bool = False) -> Text:
        filled = int(frac * width)
        txt = Text()
        txt.append("█" * filled, style="red" if failed else "green")
        txt.append("░" * (width - filled), style="bright_black")
        return txt

    @staticmethod
    def _hms(seconds: float) -> str:
        s = max(0, int(seconds))
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    @staticmethod
    def _stage_label(d: DeviceState) -> str:
        if d.stage == "idle":
            return "·"
        if d.stage == "loading":
            return "Loading model…"
        if d.stage == "trunk":
            return f"Trunk {d.step}/{d.total_steps}" if d.total_steps else "Trunk"
        if d.stage == "diffusion":
            return f"Diffusion {d.step}/{d.total_steps}" if d.total_steps else "Diffusion"
        return {"start": "Starting", "msa": "MSA", "prep": "Featurize",
                "confidence": "Confidence", "saving": "Saving",
                "done": "Done", "failed": "Failed"}.get(d.stage, d.stage)

    def _render(self) -> Group:
        with self._lock:
            return self._build()

    def _build(self) -> Group:
        elapsed = time.time() - self.start_time
        bands = _bands(self._weights())
        w_worker, w_name, w_bar, w_stage = _columns(max(40, self._console.width))
        work = self._work_done(bands)

        # ── header ────────────────────────────────────────────────────────
        # The percentage counts in-flight work, so a single fold no longer reads
        # "0/1 (0%)" for its entire run while its own bar sits at 80%.
        pct = f"{int(100 * work / self.total)}%" if self.total else "–"
        hdr = Text("  ")
        hdr.append("tt-bio", style="bold cyan")
        if self.model:
            color = {"boltz2": "bold green",
                     "esmfold2": "bold magenta",
                     "esmfold2-fast": "bold magenta"}.get(self.model, "bold yellow")
            hdr.append(f"  {self.model}", style=color)
        visible_workers = self.n_workers or len(self.devices)
        hdr.append(f"  {visible_workers} worker{'s' if visible_workers != 1 else ''}", style="dim")
        hdr.append(f"  {self.completed}/{self.total}", style="bold")
        hdr.append(f" ({pct})", style="dim")
        if self.failed:
            hdr.append(f"  {self.failed} failed", style="red")
        hdr.append(f"  {self._hms(elapsed)}", style="dim")
        # Rate measured over folding time only, against work actually done —
        # not elapsed/completed, which charged every remaining target for the
        # one-off model load and read 55 s left with 1 s of work to go.
        # Below 5% of a single target the sample is too thin to divide by.
        if self.first_start is not None and 0.05 <= work < self.total:
            folding = time.time() - self.first_start
            eta = folding / work * (self.total - work)
            hdr.append(f"  ~{self._hms(eta)} left", style="dim italic")

        table_width = w_worker + w_name + w_bar + w_stage + W_CNT + 8
        sep = Text("  " + "─" * table_width, style="bright_black")

        # ── device rows ───────────────────────────────────────────────────
        tbl = Table(show_header=False, box=None, padding=(0, 1),
                    pad_edge=False, expand=False)
        tbl.add_column("worker", style="dim", width=w_worker, justify="right", no_wrap=True)
        tbl.add_column("name", width=w_name, no_wrap=True)
        tbl.add_column("bar", width=w_bar, no_wrap=True)
        tbl.add_column("stage", width=w_stage, no_wrap=True)
        tbl.add_column("cnt", style="dim", width=W_CNT, justify="right", no_wrap=True)

        for worker_id in sorted(self.devices):
            d = self.devices[worker_id]
            failed = d.stage == "failed"
            active = d.stage not in ("idle", "loading")
            stage_style = "bold red" if failed else "bold cyan" if active else "dim"
            tbl.add_row(
                (d.label or f"device {d.device_id}")[:w_worker],
                Text(d.name[:w_name] if d.name else "·",
                     style="bold" if active else "dim"),
                self._bar(self._frac(d, bands), w_bar, failed),
                Text(self._stage_label(d)[:w_stage], style=stage_style),
                f"{d.done}/{d.assigned}" if d.assigned else "",
            )

        # ── recent log ────────────────────────────────────────────────────
        log_lines = []
        for r in self.recent[-RECENT_MAX:]:
            ln = Text("  ")
            if r.get("status") == "ok":
                ln.append("✓ ", style="green")
                ln.append(r.get("name", "?"))
                ln.append(f"  {r.get('time', 0):.1f}s", style="dim")
            else:
                # Just the symbol + name here; the full reason is printed once
                # in the post-run failure summary, so showing a lossy one-line
                # clip too would be redundant.
                ln.append("✗ ", style="red")
                ln.append(r.get("name", "?"), style="red")
            log_lines.append(ln)

        parts = [Text(""), hdr, sep, Text(""), tbl]
        if log_lines:
            parts.append(Text(""))
            parts.append(sep)
            parts.extend(log_lines)

        return Group(*parts)


class NullDisplay:
    """No-op display — used when neither Rich nor debug logging is wanted."""
    def __init__(self, queue, **_kw): self.queue = queue
    def start(self): pass
    def stop(self):
        # drain silently so the queue doesn't block
        while True:
            try: self.queue.get_nowait()
            except Exception: break

    def __enter__(self): return self
    def __exit__(self, *_): self.stop()


class DebugDisplay:
    """Drop-in replacement for ProgressDisplay that prints simple text lines.

    Same start()/stop() interface so callers don't need to branch.
    Runs a background thread that drains the queue, just like ProgressDisplay.
    """

    def __init__(self, queue, **_kw):
        self.queue = queue
        self._stop = threading.Event()
        self._thread = None
        self._last = {}  # last line printed per worker, to collapse duplicate stage emits

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._drain()

    def _loop(self):
        while not self._stop.is_set():
            self._drain()
            self._stop.wait(0.25)

    def _drain(self):
        while True:
            try:
                ev = self.queue.get_nowait()
            except (Empty, EOFError):
                break
            dev = ev.get("dev", 0)
            who = ev.get("label") or ev.get("worker") or f"dev {dev}"
            kind = ev["event"]
            if kind == "loading":
                line = f"[{who}] loading model…"
            elif kind == "start":
                line = f"[{who}] {ev.get('name', '?')}"
            elif kind == "stage":
                s, step, total = ev.get("stage", ""), ev.get("step", 0), ev.get("total", 0)
                line = f"[{who}]   {s} {step}/{total}" if total else f"[{who}]   {s}"
                # Some models emit the same stage twice in a row (e.g. msa); show it once.
                if line == self._last.get(who):
                    continue
            elif kind == "done":
                sym = "✓" if ev.get("status") == "ok" else "✗"
                line = f"[{who}] {sym} {ev.get('name', '?')} — {ev.get('time', 0):.1f}s"
            else:
                continue
            self._last[who] = line
            # Timestamp every line — a technical log should show when each stage ran.
            # stderr, the same stream the Rich view uses: progress is diagnostics,
            # and stdout carries the result summary a caller parses.
            print(f"{time.strftime('%H:%M:%S')}  {line}", flush=True, file=_sys.stderr)


def make_progress_fn(queue, device_id: int | str, worker_id: str | None = None, metadata: dict | None = None):
    """Return a lightweight callback for Boltz2.progress_fn.

    Workers call: model.progress_fn = make_progress_fn(queue, dev_id, worker_id, metadata)
    The model then calls progress_fn("stage", step=N, total=M) at key points.
    """
    metadata = metadata or {}

    def _fn(stage: str, step: int = 0, total: int = 0):
        try:
            queue.put_nowait({
                "dev": device_id, "worker": str(worker_id or device_id), "event": "stage",
                "stage": stage, "step": step, "total": total,
                **metadata,
            })
        except Exception:
            pass  # never block the model
    return _fn
