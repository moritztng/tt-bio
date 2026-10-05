"""The SC26 fold service: warm workers on the chips, one stream to the browser.

    python3 demo/sc26/engine/server.py --chips 0,1,2,3          # live, four chips
    python3 demo/sc26/engine/server.py --replay-only            # no chip needed

Serves the app's static files and a WebSocket at /stream on http://127.0.0.1:8626/, the one
local origin the kiosk talks to. The messages are specified in demo/sc26/PROTOCOL.md.

Standard library only, so the booth box needs nothing installed beyond tt-bio itself. (A
recording from before protocol 2 is converted at startup with engine/trajectory.py, which needs
numpy; tt-bio's environment has it.)
"""
import argparse
import asyncio
import base64
import gzip
import hashlib
import itertools
import json
import os
import signal
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO / "hardware"))
import telemetry  # noqa: E402  per-chip sysfs telemetry and the fold ledger, also stdlib only
from stages import Stages  # noqa: E402
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
AMINO = set("ACDEFGHIKLMNPQRSTVWY")
MIME = {".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
        ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
        ".woff2": "font/woff2", ".wasm": "application/wasm", ".jsonl": "application/x-ndjson",
        ".glb": "model/gltf-binary", ".mp4": "video/mp4", ".webm": "video/webm"}


def aiclk(node):
    try:
        v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read_text().split()[0])
        return None if v == 0xFFFFFFFF else v
    except (OSError, ValueError, IndexError):
        return None


def sd_notify(msg):
    """Tell systemd something (sd_notify(3)); a no-op outside a unit that listens."""
    addr = os.environ.get("NOTIFY_SOCKET", "")
    if not addr:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(msg.encode(), "\0" + addr[1:] if addr[0] == "@" else addr)
    except OSError:
        pass


class Hub:
    """Every connected browser; a message goes to all of them."""

    def __init__(self):
        self.clients = set()

    def send(self, msg):
        data = msg if isinstance(msg, str) else json.dumps(msg, separators=(",", ":"))
        for c in list(self.clients):
            c.send_text(data)


def key(ev):
    """Which protein a fold is, as the app's Director counts them: its name, or its sequence."""
    return ev.get("name") or ev.get("sequence")


class Folds:
    """Finished folds' coordinates, sent only to a page that asks for them (GET /fold/<id>). The
    stream itself carries no coordinates: a page shows one fold at a time, so it pulls the ones it
    will keep, at the pace its link allows. Holds the newest live fold and the newest recording of
    each protein, and at most `cap` folds in all."""

    def __init__(self, cap=64):
        self.cap, self.entries = cap, {}   # (protein, source) -> entry, oldest first

    def add(self, start, done):
        fr = done["frames"]
        q16 = fr["q16"] if isinstance(fr["q16"], bytes) else base64.b64decode(fr["q16"])
        xyz = done["xyz"] if isinstance(done["xyz"], bytes) else base64.b64decode(done["xyz"])
        plddt = done.get("plddt") or []
        summary = {k: v for k, v in done.items() if k not in ("xyz", "plddt", "frames")} | {
            k: start.get(k) for k in ("name", "sequence", "n_atoms", "chains") if start.get(k) is not None} | {
            "n_frames": len(fr["step"]), "plddt_mean": round(sum(plddt) / len(plddt), 4) if plddt else None}
        meta = {k: v for k, v in start.items() if k not in ("type", "t", "t_wall")} | {
            k: v for k, v in done.items() if k not in ("type", "xyz", "frames")}
        e = {"summary": summary, "meta": meta, "frames": fr, "q16": q16, "xyz": xyz, "n_atoms": len(xyz) // 12}
        slot = (key(summary), summary.get("source"))
        self.entries.pop(slot, None)
        self.entries[slot] = e
        while len(self.entries) > self.cap:
            self.entries.pop(next(iter(self.entries)))
        return summary

    def get(self, fid):
        return next((e for e in self.entries.values() if e["summary"]["id"] == fid), None)

    def summaries(self):
        return [e["summary"] for e in self.entries.values()]

    @staticmethod
    def body(e, every):
        """One fold for the page: a little-endian u32 length, the JSON metadata gzipped (it is mostly
        atom names, and shrinks about 15x) and zero-padded to a multiple of 4 bytes, the final
        structure as float32, then every `every`-th packed state as int16 (PROTOCOL.md "GET /fold").
        The final structure is always included."""
        fr, n = e["frames"], e["n_atoms"]
        pick = list(range(0, len(fr["scale"]), max(1, every)))
        meta = e["meta"] | {"every": every, "step": [fr["step"][i] for i in pick] + [fr["step"][-1]],
                            "t": [fr["t"][i] for i in pick] + [fr["t"][-1]],
                            "origin": [fr["origin"][i] for i in pick], "scale": [fr["scale"][i] for i in pick]}
        mb = gzip.compress(json.dumps(meta, separators=(",", ":")).encode(), 6, mtime=0)
        size = 6 * n
        return b"".join([struct.pack("<I", len(mb)), mb, b"\0" * (-(len(mb) + 4) % 4), e["xyz"]] +
                        [e["q16"][i * size:(i + 1) * size] for i in pick])


class Pacer:
    """At most one stage and one frame event per fold every `gap` seconds on the wire: a chip lane
    redraws a few times a second, while a sampler reports up to 80 steps a second. A new stage and
    a stage's last step always go out."""

    def __init__(self, gap=0.25):
        self.gap, self.last = gap, {}

    def due(self, ev):
        k, now = (ev.get("id"), ev["type"]), time.monotonic()
        stage, prev = ev.get("stage"), self.last.get(k)
        step, end = ev.get("step"), ev.get("of", ev.get("total"))
        last = end is not None and step is not None and step >= end - (ev["type"] == "frame")
        if prev and prev[0] == stage and now - prev[1] < self.gap and not last:
            return False
        self.last[k] = (stage, now)
        return True

    def forget(self, fid):
        for k in [k for k in self.last if k[0] == fid]:
            del self.last[k]


class Chip:
    """One chipworker.py process. Restarted when it exits; stopped with SIGINT. SIGKILL only right
    before a board reset (Service.reset_board)."""

    def __init__(self, svc, chip, args):
        self.svc, self.chip, self.args = svc, chip, args
        self.node = chip
        self.model = args.models[chip % len(args.models)]   # several models: the chips take turns
        self.state, self.job, self.proc = "starting", None, None
        self.folds, self.last_event, self.ready_at, self.restarts = 0, time.monotonic(), None, 0
        self.last_fold = None
        self.warm = None   # the warm-up fold the chip last reported: {name, n_res, stage}
        self.failures, self.stalled, self.last_reset = 0, False, None  # consecutive unclean exits

    def status(self):
        j = self.job
        # what the chip is on, so a page that connects mid-fold names it and counts from the chip's start
        doing = j and {k: j.get(k) for k in ("id", "kind", "name", "n_res", "t_wall", "plan")}
        return {"chip": self.chip, "state": self.state, "job": j and j["id"], "doing": doing,
                "aiclk_mhz": aiclk(self.node), "folds": self.folds, "restarts": self.restarts,
                "last_fold": self.last_fold, **({"warming": self.warm} if self.state == "warming" else {})}

    async def run(self):
        while not self.svc.stopping:
            await self.svc.board_free(self)
            if self.svc.stopping:
                break
            env = dict(os.environ, TT_VISIBLE_DEVICES=str(self.chip), TT_BIO_LEASE_CARDS=str(self.chip),
                       TT_BIO_LEASE_HOLDER=os.environ.get("TT_BIO_LEASE_HOLDER", f"sc26-demo:chip{self.chip}"))
            self.set_state("warming")
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", self.args.worker, "--chip", str(self.chip),
                "--workers", str(len(self.svc.chips)), "--model", self.model,
                *(["--warm", self.args.attract] if self.model != "esmfold2" and self.args.attract else []),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=open(self.svc.logdir / f"chip{self.chip}.stderr", "a"), env=env,
                limit=64 * 1024 * 1024)
            self.last_event = time.monotonic()
            async for line in self.proc.stdout:
                self.last_event = time.monotonic()
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                self.on_event(ev, line.decode().rstrip("\n"))
            rc = await self.proc.wait()
            if self.job is not None:  # it died holding a fold: say so, and give a visitor's fold to another chip
                job, self.job = self.job, None
                self.svc.publish({"type": "fold_error", "id": job["id"], "chip": self.chip,
                                   "reason": "chip_lost", "t_wall": time.time()})
                self.svc.ledger("fail", self.chip)
                self.svc.requeue(job)
            if self.svc.stopping:
                self.set_state("recovering", rc=rc)
                break
            stalled, self.stalled = self.stalled, False
            unclean = rc != 0 or stalled
            self.failures = self.failures + 1 if unclean else 0
            self.set_state("recovering", rc=rc)
            self.restarts += 1
            # A stall is a device call that never returned: the chip stays wedged, a plain restart
            # only sits in warm-up until --warm-s, so the board is reset at once.
            if unclean and (stalled or self.failures >= self.args.reset_after) and self.svc.reset_ok(self):
                await self.svc.reset_board(self)
            await asyncio.sleep(min(60, 5 * max(1, self.failures)))

    def on_event(self, ev, raw):
        t = ev.get("type")
        if t == "chip":
            st = ev["state"]
            if st == "ready":
                self.job = None if self.state == "busy" else self.job
                self.set_state("ready")
                self.svc.dispatch()
            elif st == "busy" and self.job and ev.get("job") == self.job["id"]:
                # the chip has taken the job: its clock for this fold starts here, and so does the bar
                self.set_state("busy", job=self.job["id"], doing=self.status()["doing"])
            elif st in ("warming", "stopped"):
                self.warm = {k: ev.get(k) for k in ("name", "n_res", "stage")} if st == "warming" else None
                self.set_state(st, **({k: v for k, v in self.warm.items() if v is not None} if self.warm else {}))
            return
        job = self.job
        if job and ev.get("id") == job["id"]:
            ev["kind"] = job["kind"]
            if job.get("name"):
                ev.setdefault("name", job["name"])  # the chipworker never sees an attract fold's name
            if t == "fold_start":  # an attract fold is a gallery pick: it carries the pick's words, as its recording does
                for k in ("story", "pdb", "chains"):
                    if job.get(k) is not None:
                        ev.setdefault(k, job[k])
            raw = json.dumps(ev, separators=(",", ":"))
            self.svc.record(job, raw)
            if t in ("fold_start", "stage", "fold_done"):   # when each stage began, for the next plan
                job.setdefault("marks", []).append(["start" if t == "fold_start" else ev.get("stage", "done"),
                                                    ev.get("seconds" if t == "fold_done" else "t")])
            if t == "fold_done":
                self.svc.stages.add(self.model, job["n_res"], job["marks"])
                self.folds += 1
                self.failures = 0
                self.last_fold = {k: ev.get(k) for k in ("name", "n_res", "seconds", "aiclk_mhz", "t_wall")}
                self.svc.ledger("done", self.chip, seconds=ev.get("seconds"))
            elif t == "fold_error":
                self.svc.ledger("fail", self.chip)
            if t in ("fold_done", "fold_error"):
                self.svc.finish(job, ev)
                self.job = None
                if t == "fold_error" and ev.get("reason") in ("stopped", "out_of_memory"):
                    self.svc.requeue(job)  # the chip was stopped or is recycling: a visitor's fold moves on
        self.svc.publish(ev)

    def set_state(self, st, **kw):
        if self.svc.resetting(self) and st != "resetting":
            kw["worker"] = st  # the lane keeps saying "resetting" until the board is back
            st = "resetting"
        self.state = st
        self.svc.hub.send({"type": "chip", "chip": self.chip, "state": st, "aiclk_mhz": aiclk(self.node),
                           "t_wall": time.time(), **kw})

    def start(self, job):
        self.job, self.state = job, "busy"
        self.job["chip"], self.job["started"], self.job["t_wall"] = self.chip, time.monotonic(), time.time()
        self.job["n_res"] = len(job["sequence"].replace(":", ""))
        self.job["plan"] = self.svc.stages.plan(self.model, self.job["n_res"])
        self.proc.stdin.write((json.dumps({k: job[k] for k in ("id", "sequence", "seed", "yaml") if job.get(k) is not None})
                               + "\n").encode())
        self.svc.ledger("start", self.chip, model=self.model, name=job.get("name"), residues=job["n_res"])

    def preempt(self):
        if self.proc and self.proc.returncode is None:
            self.proc.send_signal(signal.SIGUSR1)

    def stop(self):
        """SIGINT: the worker drops its fold and closes the chip cleanly."""
        if self.proc and self.proc.returncode is None:
            self.proc.send_signal(signal.SIGINT)


class Replay:
    """Plays recorded folds through the same protocol, labelled source=replay."""

    def __init__(self, svc, dirs):
        self.svc = svc
        self.files = {}   # path -> the recording's events, its coordinates decoded once
        for f in (f for d in dirs for f in Path(d).glob("*.jsonl")):
            evs = self.load(f)
            if evs and evs[0].get("model") in svc.args.models and evs[-1]["type"] == "fold_done":
                self.files[f] = evs
        # largest first, so the first replays after a start are the big complexes
        self.order = sorted(self.files, key=lambda f: (-self.files[f][0].get("n_res", 0), f.name))
        self.task = None
        # every recording can be pulled from the start, so a page that loads fills its stage at once
        for f in reversed(self.order):
            start, done = self.files[f][0], self.files[f][-1]
            tag = dict(id=f"g-{f.stem}", chip=None, source="replay", kind="replay", recorded=f.stem, t_wall=time.time())
            svc.folds.add(start | tag, done | tag)

    @staticmethod
    def load(path):
        lines = path.read_text().splitlines()
        if '"xyz":' in lines[1 if len(lines) > 1 else 0]:   # protocol 1: coordinates in every frame
            try:
                import trajectory   # numpy; a recording written since protocol 2 needs nothing
            except ImportError:
                print(f"skipping {path.name}: protocol 1, convert it with engine/trajectory.py", flush=True)
                return None
            lines = trajectory.convert(lines)
        evs = [ev for ev in map(json.loads, lines) if ev.get("type") in ("fold_start", "stage", "frame", "fold_done")]
        done = evs[-1] if evs else {}
        if "frames" not in done:
            return None
        done["xyz"] = base64.b64decode(done["xyz"])
        done["frames"]["q16"] = base64.b64decode(done["frames"]["q16"])
        return evs

    async def play(self, path, jid, kind="replay"):
        prev = None
        for ev in self.files[path]:
            if prev is not None and "t" in ev:
                await asyncio.sleep(max(0.0, min(2.0, ev["t"] - prev)))
            prev = ev.get("t", prev)
            self.svc.publish(ev | dict(id=jid, chip=None, source="replay", kind=kind, t_wall=time.time(),
                                       recorded=path.stem))

    async def loop(self):
        """Recordings between the live folds, so every gallery pick reaches the screen even before
        the chips have folded it live, and with no chip at all the screen is never blank. With a
        chip live they come at a slower pace; the app picks what takes the stage, and a live fold of
        a pick replaces its recording there."""
        for path in itertools.cycle(self.order or [None]):
            if path is None:
                await asyncio.sleep(5)
                continue
            await self.play(path, f"r{next(self.svc.ids)}")
            live = any(c.state in ("ready", "busy") for c in self.svc.chips)
            await asyncio.sleep(self.svc.args.replay_gap_live if live else self.svc.args.replay_gap)


class Service:
    def __init__(self, args):
        self.args, self.hub, self.stopping = args, Hub(), False
        self.ids = itertools.count(1)
        self.logdir = Path(args.logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.recdir = Path(args.record) if args.record else None
        if self.recdir:
            self.recdir.mkdir(parents=True, exist_ok=True)
        self.chips = [Chip(self, c, args) for c in args.chips]
        self.visitors = deque()
        # The attract rotation is the gallery's picks (gallery/build.py writes attract.json), so the chips
        # fold the proteins the stage shows. A long fold holds its chip for 35-94 s and only reaches a
        # point where a visitor can take the chip once per trunk recycle (up to 14 s apart at 833
        # residues), so at most all chips but one are on a long fold: one chip always turns over
        # every 5-13 s, and a visitor's fold preempts that one first.
        picks = json.loads(Path(args.attract).read_text()) if args.attract else []
        is_long = lambda a: len(a["sequence"].replace(":", "")) > args.long_res
        self.attract_long = itertools.cycle([a for a in picks if is_long(a)] or [None])
        self.attract_short = itertools.cycle([a for a in picks if not is_long(a)] or [None])
        self.attract = bool(picks)
        self.folds, self.pacer, self.starting = Folds(), Pacer(), {}   # starting: id -> fold_start, until done
        self.replay = Replay(self, args.replay)
        self.stages = Stages(DEMO / "gallery" / "store")   # what each stage takes on this box, by length
        self.open_files = {}
        self._preempt_armed = False
        # Chips that share a board are reset together (qb2: a p300 board carries chips 0,1 and 2,3).
        self.boards = [set(map(int, b.split(","))) for b in args.boards.split()] if args.boards else []
        self.board_idle = {}  # board index -> asyncio.Event, cleared while that board is reset
        self.monitor = None if args.no_telemetry else telemetry.Monitor(events=Path(args.fold_events))

    def ledger(self, event, chip, **fields):
        """Tell the telemetry about a fold, so the lanes show it with the clock it ran at."""
        if self.monitor:
            telemetry.record_fold(event, chip, path=self.monitor.folds.path, **fields)

    def board_of(self, chip):
        for i, b in enumerate(self.boards):
            if chip.chip in b:
                return i
        return f"chip{chip.chip}"

    def resetting(self, chip):
        ev = self.board_idle.get(self.board_of(chip))
        return ev is not None and not ev.is_set()

    async def board_free(self, chip):
        ev = self.board_idle.get(self.board_of(chip))
        if ev is not None:
            await ev.wait()

    def reset_ok(self, chip):
        gap = self.args.reset_min_gap
        return bool(self.args.reset_cmd) and not self.resetting(chip) and \
            (chip.last_reset is None or time.monotonic() - chip.last_reset > gap)

    async def reset_board(self, chip):
        """Stop every worker on the chip's board (SIGINT, then SIGTERM), reset the board with
        --reset-cmd, bounded by --reset-timeout, then let the workers start again. The other boards
        keep folding, and the replay loop fills the screen if no chip is left.

        A worker stuck inside a device call never runs its signal handlers. It gets SIGKILL, but
        only here, right before the reset: a killed worker leaves its chip unopenable, and the
        reset is what makes it openable again. Killing first means nothing holds the chip while
        it is reset, and a worker that never exits cannot keep its lane dark."""
        b = self.board_of(chip)
        mates = [c for c in self.chips if self.board_of(c) == b]
        ev = self.board_idle[b] = asyncio.Event()
        for c in mates:
            c.set_state("resetting")
            c.last_reset = time.monotonic()
        try:
            for c in mates:
                c.stop()
            deadline = time.monotonic() + self.args.term_s
            for c in mates:
                await self._wait_exit(c.proc, deadline - time.monotonic())
                if c.proc and c.proc.returncode is None:
                    c.proc.terminate()
            deadline = time.monotonic() + self.args.term_s
            for c in mates:
                await self._wait_exit(c.proc, deadline - time.monotonic())
                if c.proc and c.proc.returncode is None:
                    c.proc.kill()
                    await self._wait_exit(c.proc, 10)
            ids = ",".join(str(c.chip) for c in mates)
            t0 = time.monotonic()
            try:
                p = await asyncio.create_subprocess_exec(
                    *self.args.reset_cmd.split(), ids, stdin=asyncio.subprocess.DEVNULL,
                    stdout=open(self.logdir / "reset.log", "a"), stderr=asyncio.subprocess.STDOUT)
                rc = await asyncio.wait_for(p.wait(), self.args.reset_timeout)
            except asyncio.TimeoutError:
                p.terminate()
                rc = "timeout"
            except OSError as e:
                rc = f"{type(e).__name__}"
            self.hub.send({"type": "reset", "chips": [c.chip for c in mates], "rc": rc,
                           "seconds": round(time.monotonic() - t0, 1), "t_wall": time.time()})
        finally:
            ev.set()
            for c in mates:
                c.failures = 0
                c.set_state("recovering")

    @staticmethod
    async def _wait_exit(proc, timeout):
        if proc and proc.returncode is None and timeout > 0:
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout)
            except asyncio.TimeoutError:
                pass

    def job(self, seq, kind, seed=0, extra=None):
        return {"id": f"{kind[0]}{next(self.ids)}", "sequence": seq, "seed": seed, "kind": kind, **(extra or {})}

    def submit(self, seq):
        seq = "".join(seq.split()).upper()
        if not seq or set(seq) - AMINO:
            return {"type": "rejected", "reason": "letters", "allowed": "".join(sorted(AMINO))}
        if not (self.args.min_len <= len(seq) <= self.args.max_len):
            return {"type": "rejected", "reason": "length", "min": self.args.min_len, "max": self.args.max_len}
        job = self.job(seq, "visitor")
        self.visitors.append(job)
        self.dispatch()
        pos = list(self.visitors).index(job) + 1 if job in self.visitors else 0
        return {"type": "queued", "id": job["id"], "position": pos, "n_res": len(seq)}

    def requeue(self, job):
        if job["kind"] == "visitor" and not job.get("retried"):
            job["retried"] = True
            self.visitors.appendleft(job)
        self.dispatch()

    def dispatch(self):
        free = [c for c in self.chips if c.state == "ready" and c.job is None]
        while free and self.visitors:
            free.pop(0).start(self.visitors.popleft())
        if self.visitors and not self._preempt_armed:
            # Every chip is busy. Short attract folds end on their own within about a second, so
            # give them that long; a visitor still waiting then takes an attract fold's chip,
            # which drops it at its next sampler step.
            self._preempt_armed = True
            asyncio.get_running_loop().call_later(self.args.preempt_after, self._preempt)
        if self.attract:
            for c in free:
                on_long = sum(1 for x in self.chips if x.job and x.job.get("long"))
                a = next(self.attract_long) if on_long < len(self.chips) - 1 else None
                a = a or next(self.attract_short) or next(self.attract_long)
                c.start(self.job(a["sequence"], "attract", a.get("seed", 0),
                                 {k: a.get(k) for k in ("name", "story", "pdb", "chains", "yaml")} |
                                 {"long": len(a["sequence"].replace(":", "")) > self.args.long_res}))

    def _preempt(self):
        self._preempt_armed = False
        if not self.visitors:
            return
        busy = sorted((c for c in self.chips if c.state == "busy" and c.job and c.job["kind"] == "attract"
                       and not c.job.get("preempted")), key=lambda c: c.job.get("long", False))  # short folds first
        for c in busy[:len(self.visitors)]:
            c.job["preempted"] = True
            c.preempt()

    def finish(self, job, ev):
        f = self.open_files.pop(job["id"], None)
        if f:
            f.close()
            if ev["type"] != "fold_done":
                f_path = Path(f.name)
                f_path.unlink(missing_ok=True)

    def record(self, job, raw):
        if not self.recdir:
            return
        f = self.open_files.get(job["id"])
        if f is None:
            f = self.open_files[job["id"]] = open(self.recdir / f"{time.strftime('%Y%m%dT%H%M%S')}-{job['id']}.jsonl", "w")
        f.write(raw + "\n")

    def publish(self, ev):
        """A fold event, live or replayed, onto the stream. The stream carries no coordinates: a
        fold_start goes out without its atoms and a fold_done as a summary, and the fold is kept in
        self.folds for the pages that ask for it. Stage and frame ticks are paced (Pacer)."""
        t, fid = ev.get("type"), ev.get("id")
        if t == "fold_start":
            self.starting[fid] = ev
            ev = {k: v for k, v in ev.items() if k != "atoms"}
        elif t in ("stage", "frame"):
            if not self.pacer.due(ev):
                return
            ev = {k: v for k, v in ev.items() if k not in ("xyz", "x0", "R", "T")}
        elif t in ("fold_done", "fold_error"):
            self.pacer.forget(fid)
            start = self.starting.pop(fid, None)
            if t == "fold_done":   # without coordinates (a test worker) it is news, with nothing to pull
                ev = {"type": t} | (self.folds.add(start, ev) if start and "frames" in ev else
                                    {k: v for k, v in ev.items() if k not in ("xyz", "plddt", "frames")})
        self.hub.send(ev)

    # sysfs looks healthy through most of a `tt-smi -r`, so the counters alone showed a board being
    # reset as "ready". The engine knows better: a chip it has stalled, is resetting or is bringing
    # back says so on the lanes too, as it does in the stage's chip table.
    ENGINE_STATE = {"stalled": "resetting", "resetting": "resetting", "recovering": "resetting", "warming": "warming"}

    def telemetry(self):
        snap = self.monitor.snapshot()
        engine = {c.chip: "resetting" if c.stalled or self.resetting(c) else self.ENGINE_STATE.get(c.state)
                  for c in self.chips}
        engine.update({n: "out_of_service" for n in self.args.out_of_service})
        for c in snap["chips"]:
            if engine.get(c["card"]):
                c["state"], c["folding"] = engine[c["card"]], None
        return snap

    def status(self):
        out = [{"chip": n, "state": "out_of_service"} for n in self.args.out_of_service]
        return {"type": "status", "models": self.args.models, "chips": [c.status() for c in self.chips] + out,
                "queue": len(self.visitors), "replays": len(self.replay.order), "t_wall": time.time()}

    async def watchdog(self):
        """A fold that stops producing events is stopped (SIGINT) and its chip restarted."""
        while not self.stopping:
            await asyncio.sleep(2)
            now = time.monotonic()
            for c in self.chips:
                quiet = now - c.last_event
                limit = self.args.stall_s if c.state == "busy" else self.args.warm_s
                if c.state in ("busy", "warming") and quiet > limit and not self.resetting(c):
                    self.hub.send({"type": "chip", "chip": c.chip, "state": "stalled", "quiet_s": round(quiet),
                                   "t_wall": time.time()})
                    c.stalled = True
                    c.stop()
                    c.last_event = now
                    asyncio.get_running_loop().call_later(self.args.term_s, self._term, c, c.proc)
            self.hub.send(self.status())
            # The unit's WatchdogSec: an event loop that is alive and stuck stops saying this, and
            # systemd restarts the engine. SIGINT could not have done it; its handler runs on this loop.
            sd_notify("WATCHDOG=1")

    def _term(self, c, proc):
        if proc and proc.returncode is None:  # SIGINT was not enough: SIGTERM, still never SIGKILL
            proc.terminate()
            asyncio.get_running_loop().call_later(self.args.term_s, self._wedged, c, proc)

    def _wedged(self, c, proc):
        """A worker that survives SIGINT and SIGTERM is stuck in the device: reset its board."""
        if proc and proc.returncode is None and self.reset_ok(c):
            asyncio.ensure_future(self.reset_board(c))

    # ---- HTTP + WebSocket ----
    async def handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            writer.close()
            return
        lines = head.decode("latin-1").split("\r\n")
        method, path, _ = (lines[0].split(" ") + ["", "", ""])[:3]
        hdr = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
        path, _, query = path.partition("?")
        if path == "/stream" and hdr.get("upgrade", "").lower() == "websocket":
            return await self.websocket(reader, writer, hdr)
        if method == "POST" and path == "/fold":
            body = await reader.readexactly(int(hdr.get("content-length", 0)))
            try:
                res = self.submit(json.loads(body).get("sequence", ""))
            except ValueError:
                res = {"type": "rejected", "reason": "json"}
            return self.reply(writer, 200, json.dumps(res).encode(), "application/json")
        if path.startswith("/fold/"):   # one finished fold's coordinates (PROTOCOL.md "GET /fold")
            e = self.folds.get(path[6:])
            if e is None:
                return self.reply(writer, 404, b"gone", "text/plain")
            every = dict(p.partition("=")[::2] for p in query.split("&")).get("every", "1")
            return self.reply(writer, 200, Folds.body(e, int(every) if every.isdigit() else 1),
                              "application/octet-stream")
        if path == "/telemetry" and self.monitor:  # the chip lanes poll this (hardware/README.md)
            return self.reply(writer, 200, json.dumps(self.telemetry()).encode(), "application/json")
        if path == "/status":
            return self.reply(writer, 200, json.dumps(self.status()).encode(), "application/json")
        if path == "/":  # the app's assets are relative to /app/, so send the bare origin there
            writer.write(b"HTTP/1.1 302 Found\r\nLocation: /app/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return writer.close()
        root = Path(self.args.static).resolve()
        f = (root / path.lstrip("/")).resolve()
        if f.is_dir():
            f = f / "index.html"
        if root not in f.parents and f != root or not f.is_file():
            return self.reply(writer, 404, b"not found", "text/plain")
        self.reply(writer, 200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))

    def reply(self, writer, code, body, ctype):
        writer.write(f"HTTP/1.1 {code} {'OK' if code == 200 else 'Not Found'}\r\nContent-Type: {ctype}\r\n"
                     f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n".encode()
                     + body)
        writer.close()

    async def websocket(self, reader, writer, hdr):
        accept = base64.b64encode(hashlib.sha1((hdr["sec-websocket-key"] + GUID).encode()).digest()).decode()
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        client = WSClient(writer)
        self.hub.clients.add(client)
        # what this page can pull at once: a page that has just loaded or reconnected fills its stage
        # from these, without waiting for the next fold to stream by
        client.send_text(json.dumps({**self.status(), "type": "hello", "protocol": 2, "folds": self.folds.summaries()}))
        try:
            while True:
                op, data = await client.recv(reader)
                if op == 8:
                    break
                if op == 9:
                    client.send(10, data)
                elif op == 1:
                    try:
                        msg = json.loads(data)
                    except ValueError:
                        continue
                    if msg.get("type") == "fold":
                        client.send_text(json.dumps(self.submit(msg.get("sequence", ""))))
                    elif msg.get("type") == "status":
                        client.send_text(json.dumps(self.status()))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.hub.clients.discard(client)
            writer.close()

    async def main(self):
        server = await asyncio.start_server(self.handle, self.args.host, self.args.port, limit=1 << 20)
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, self.shutdown)
        tasks = [asyncio.create_task(c.run()) for c in self.chips]
        tasks += [asyncio.create_task(self.watchdog()), asyncio.create_task(self.replay.loop())]
        if self.monitor:
            self.monitor.start()
        print(f"sc26 engine on http://{self.args.host}:{self.args.port}/  chips={self.args.chips} "
              f"replays={len(self.replay.order)}", flush=True)
        async with server:
            while not self.stopping:
                await asyncio.sleep(0.5)
            # Server.wait_closed() waits for every open connection, and a kiosk keeps its
            # WebSocket open forever: close them, or the service never exits on SIGINT/SIGTERM.
            server.close()
            for cl in list(self.hub.clients):
                cl.writer.close()
            for c in self.chips:
                c.stop()
            await asyncio.sleep(0)
            for c in self.chips:
                if c.proc:
                    try:
                        await asyncio.wait_for(c.proc.wait(), self.args.term_s)
                    except asyncio.TimeoutError:
                        c.proc.terminate()
                        await self._wait_exit(c.proc, self.args.term_s)
                    # close the worker's pipe while the loop still runs: left to the garbage
                    # collector it is closed after asyncio.run, which prints "Event loop is closed"
                    c.proc.stdin.close()
            await asyncio.sleep(0)

    def shutdown(self):
        self.stopping = True


class WSClient:
    """Server side of RFC 6455, enough for text messages, ping and close.

    A browser behind a slow or stalled link is never queued for: while more than BACKLOG bytes it
    has not taken yet are waiting, a message to it is dropped (the status every 2 s makes up for
    any of them), and a link that has taken nothing for STUCK_S seconds is closed. The page
    reconnects by itself and starts from a fresh hello."""

    BACKLOG, STUCK_S = 64 * 1024, 30.0

    def __init__(self, writer):
        self.writer = writer
        self.behind_since, self.dropped = None, 0

    def send(self, op, payload):
        if self.writer.is_closing():
            return
        if self.writer.transport.get_write_buffer_size() > self.BACKLOG:
            self.dropped += 1
            self.behind_since = self.behind_since or time.monotonic()
            if time.monotonic() - self.behind_since > self.STUCK_S:
                self.writer.transport.abort()
            return
        self.behind_since = None
        n = len(payload)
        head = bytes([0x80 | op]) + (bytes([n]) if n < 126 else
                                     bytes([126]) + struct.pack(">H", n) if n < 65536 else
                                     bytes([127]) + struct.pack(">Q", n))
        self.writer.write(head + payload)

    def send_text(self, s):
        self.send(1, s.encode())

    async def recv(self, reader):
        b0, b1 = await reader.readexactly(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", await reader.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", await reader.readexactly(8))[0]
        mask = await reader.readexactly(4) if b1 & 0x80 else b"\0\0\0\0"
        data = bytes(c ^ mask[i % 4] for i, c in enumerate(await reader.readexactly(n)))
        return b0 & 0x0F, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="", help="UMD chip ids, e.g. 0,1,2,3; empty for replay only")
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--worker", default=str(HERE / "chipworker.py"),
                    help="the chip worker program; tests/recorded_worker.py plays recorded folds without a chip")
    ap.add_argument("--models", default="openfold3",
                    help="what the booth folds with, comma-separated (openfold3, boltz2, esmfold2); with several the "
                    "chips take turns and the screen names each fold's model. Recordings of other models are not played")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8626)
    ap.add_argument("--static", default=str(DEMO / "web"))
    ap.add_argument("--replay", action="append", default=None,
                    help="directory of recorded folds (*.jsonl); repeatable")
    ap.add_argument("--record", default=str(HERE / "runs" / "recorded"),
                    help="where every finished live fold is saved in the replay format")
    ap.add_argument("--attract", default=str(HERE / "attract.json"),
                    help="sequences the chips fold when no visitor is waiting; empty to idle")
    ap.add_argument("--logdir", default=str(HERE / "runs" / "logs"))
    ap.add_argument("--fold-events", default=str(telemetry.EVENTS),
                    help="the fold ledger the telemetry reads (hardware/README.md)")
    ap.add_argument("--no-telemetry", action="store_true", help="do not sample the chips' sysfs counters")
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--stall-s", type=float, default=180, help="a busy chip silent this long is stopped")
    ap.add_argument("--warm-s", type=float, default=600, help="a warming chip silent this long is stopped")
    ap.add_argument("--term-s", type=float, default=30, help="SIGINT grace before SIGTERM")
    ap.add_argument("--reset-cmd", default="", help="board reset command; the chip ids are appended "
                    "(e.g. 'demo/sc26/ops/reset_board.sh'). Empty: never reset, only restart")
    ap.add_argument("--reset-after", type=int, default=2, help="consecutive unclean worker exits before a reset")
    ap.add_argument("--reset-timeout", type=float, default=180)
    ap.add_argument("--reset-min-gap", type=float, default=600, help="seconds between two resets of one chip")
    ap.add_argument("--boards", default="0,1 2,3", help="chips that share a board and reset together")
    ap.add_argument("--replay-gap", type=float, default=3.0)
    ap.add_argument("--replay-gap-live", type=float, default=20.0,
                    help="seconds between two recordings while a chip is live")
    ap.add_argument("--long-res", type=int, default=400,
                    help="an attract protein longer than this is a long fold; one chip never takes one")
    ap.add_argument("--out-of-service", default="", help="chips taken out of the demo on purpose, e.g. 2; "
                    "their lanes say so (ops/README.md)")
    ap.add_argument("--preempt-after", type=float, default=1.0,
                    help="seconds a visitor waits for a chip before an attract fold is dropped for it")
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.chips = [] if args.replay_only else [int(c) for c in args.chips.split(",") if c.strip()]
    args.out_of_service = [int(c) for c in args.out_of_service.split(",") if c.strip() and int(c) not in args.chips]
    args.replay = args.replay or [str(DEMO / "gallery" / "trajectories"), str(HERE / "recordings")]
    if not args.attract or not Path(args.attract).is_file():
        args.attract = None
    asyncio.run(Service(args).main())


if __name__ == "__main__":
    main()
