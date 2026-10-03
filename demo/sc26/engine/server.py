"""The SC26 fold service: warm workers on the chips, one stream to the browser.

    python3 demo/sc26/engine/server.py --chips 0,1,2,3          # live, four chips
    python3 demo/sc26/engine/server.py --replay-only            # no chip needed

Serves the app's static files and a WebSocket at /stream on http://127.0.0.1:8626/, the one
local origin the kiosk talks to. The messages are specified in demo/sc26/PROTOCOL.md.

Standard library only, so the booth box needs nothing installed beyond tt-bio itself.
"""
import argparse
import asyncio
import base64
import hashlib
import itertools
import json
import os
import signal
import struct
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO / "hardware"))
import telemetry  # noqa: E402  per-chip sysfs telemetry and the fold ledger, also stdlib only
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


class Hub:
    """Every connected browser; a message goes to all of them."""

    def __init__(self):
        self.clients = set()

    def send(self, msg):
        data = msg if isinstance(msg, str) else json.dumps(msg, separators=(",", ":"))
        for c in list(self.clients):
            c.send_text(data)


class Chip:
    """One chipworker.py process. Restarted when it exits; stopped with SIGINT, never SIGKILL."""

    def __init__(self, svc, chip, args):
        self.svc, self.chip, self.args = svc, chip, args
        self.node = chip
        self.state, self.job, self.proc = "starting", None, None
        self.folds, self.last_event, self.ready_at, self.restarts = 0, time.monotonic(), None, 0
        self.last_fold = None
        self.failures, self.stalled, self.last_reset = 0, False, None  # consecutive unclean exits

    def status(self):
        return {"chip": self.chip, "state": self.state, "job": self.job and self.job["id"],
                "aiclk_mhz": aiclk(self.node), "folds": self.folds, "restarts": self.restarts,
                "last_fold": self.last_fold}

    async def run(self):
        while not self.svc.stopping:
            await self.svc.board_free(self)
            if self.svc.stopping:
                break
            env = dict(os.environ, TT_VISIBLE_DEVICES=str(self.chip), TT_BIO_LEASE_CARDS=str(self.chip),
                       TT_BIO_LEASE_HOLDER=os.environ.get("TT_BIO_LEASE_HOLDER", f"sc26-demo:chip{self.chip}"))
            self.set_state("warming")
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", str(HERE / "chipworker.py"), "--chip", str(self.chip),
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
                self.svc.hub.send({"type": "fold_error", "id": job["id"], "chip": self.chip,
                                   "reason": "chip_lost", "t_wall": time.time()})
                self.svc.ledger("fail", self.chip)
                self.svc.requeue(job)
            if self.svc.stopping:
                self.set_state("recovering", rc=rc)
                break
            unclean, self.stalled = rc != 0 or self.stalled, False
            self.failures = self.failures + 1 if unclean else 0
            self.set_state("recovering", rc=rc)
            self.restarts += 1
            if unclean and self.failures >= self.args.reset_after and self.svc.reset_ok(self):
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
            elif st in ("warming", "stopped"):
                self.set_state(st)
            return
        job = self.job
        if job and ev.get("id") == job["id"]:
            ev["kind"] = job["kind"]
            if job.get("name"):
                ev.setdefault("name", job["name"])  # the chipworker never sees an attract fold's name
            raw = json.dumps(ev, separators=(",", ":"))
            self.svc.record(job, raw)
            if t == "fold_done":
                self.folds += 1
                self.failures = 0
                self.last_fold = {k: ev.get(k) for k in ("n_res", "seconds", "aiclk_mhz")}
                self.svc.ledger("done", self.chip, seconds=ev.get("seconds"))
            elif t == "fold_error":
                self.svc.ledger("fail", self.chip)
            if t in ("fold_done", "fold_error"):
                self.svc.finish(job, ev)
                self.job = None
                if t == "fold_error" and ev.get("reason") == "stopped":
                    self.svc.requeue(job)  # the chip was stopped under it: a visitor's fold moves on
        self.svc.hub.send(raw)

    def set_state(self, st, **kw):
        if self.svc.resetting(self) and st != "resetting":
            kw["worker"] = st  # the lane keeps saying "resetting" until the board is back
            st = "resetting"
        self.state = st
        self.svc.hub.send({"type": "chip", "chip": self.chip, "state": st, "aiclk_mhz": aiclk(self.node),
                           "t_wall": time.time(), **kw})

    def start(self, job):
        self.job, self.state = job, "busy"
        self.job["chip"], self.job["started"] = self.chip, time.monotonic()
        self.proc.stdin.write((json.dumps({k: job[k] for k in ("id", "sequence", "seed")}) + "\n").encode())
        self.svc.ledger("start", self.chip, model="esmfold2", name=job.get("name"), residues=len(job["sequence"]))

    def preempt(self):
        if self.proc and self.proc.returncode is None:
            self.proc.send_signal(signal.SIGUSR1)

    def stop(self):
        """SIGINT: the worker drops its fold and closes the chip cleanly. Never SIGKILL."""
        if self.proc and self.proc.returncode is None:
            self.proc.send_signal(signal.SIGINT)


class Replay:
    """Plays recorded folds through the same protocol, labelled source=replay."""

    def __init__(self, svc, dirs):
        self.svc = svc
        self.files = sorted(f for d in dirs for f in Path(d).glob("*.jsonl")
                            if any('"type":"fold_done"' in l for l in open(f)))
        self.task = None

    async def play(self, path, jid, kind="replay"):
        prev = None
        for line in open(path):
            ev = json.loads(line)
            if ev.get("type") not in ("fold_start", "stage", "frame", "fold_done"):
                continue
            if prev is not None and "t" in ev:
                await asyncio.sleep(max(0.0, min(2.0, ev["t"] - prev)))
            prev = ev.get("t", prev)
            ev.update(id=jid, chip=None, source="replay", kind=kind, t_wall=time.time(),
                      recorded=path.stem)
            self.svc.hub.send(ev)

    async def loop(self):
        """Attract with recordings when no chip can fold: the screen is never blank."""
        for path in itertools.cycle(self.files or [None]):
            if path is None:
                await asyncio.sleep(5)
                continue
            if any(c.state in ("ready", "busy") for c in self.svc.chips):
                await asyncio.sleep(1)
                continue
            await self.play(path, f"r{next(self.svc.ids)}")
            await asyncio.sleep(self.svc.args.replay_gap)


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
        self.attract = itertools.cycle(json.loads(Path(args.attract).read_text())) if args.attract else None
        self.replay = Replay(self, args.replay)
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
        """Stop every worker on the chip's board (SIGINT, then SIGTERM; never SIGKILL), reset the
        board with --reset-cmd, bounded by --reset-timeout, then let the workers start again. The
        other boards keep folding, and the replay loop fills the screen if no chip is left."""
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
            for c in mates:  # a worker that outlived SIGTERM usually exits once its chip is reset
                await self._wait_exit(c.proc, self.args.term_s)
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
                a = next(self.attract)
                c.start(self.job(a["sequence"], "attract", a.get("seed", 0), {"name": a.get("name")}))

    def _preempt(self):
        self._preempt_armed = False
        if not self.visitors:
            return
        busy = [c for c in self.chips if c.state == "busy" and c.job and c.job["kind"] == "attract"
                and not c.job.get("preempted")]
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

    def status(self):
        return {"type": "status", "chips": [c.status() for c in self.chips],
                "queue": len(self.visitors), "replays": len(self.replay.files), "t_wall": time.time()}

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
        path = path.split("?")[0]
        if path == "/stream" and hdr.get("upgrade", "").lower() == "websocket":
            return await self.websocket(reader, writer, hdr)
        if method == "POST" and path == "/fold":
            body = await reader.readexactly(int(hdr.get("content-length", 0)))
            try:
                res = self.submit(json.loads(body).get("sequence", ""))
            except ValueError:
                res = {"type": "rejected", "reason": "json"}
            return self.reply(writer, 200, json.dumps(res).encode(), "application/json")
        if path == "/telemetry" and self.monitor:  # the chip lanes poll this (hardware/README.md)
            return self.reply(writer, 200, json.dumps(self.monitor.snapshot()).encode(), "application/json")
        if path == "/status":
            return self.reply(writer, 200, json.dumps(self.status()).encode(), "application/json")
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
        client.send_text(json.dumps({"type": "hello", "protocol": 1, **self.status()}))
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
              f"replays={len(self.replay.files)}", flush=True)
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

    def shutdown(self):
        self.stopping = True


class WSClient:
    """Server side of RFC 6455, enough for text messages, ping and close."""

    def __init__(self, writer):
        self.writer = writer

    def send(self, op, payload):
        n = len(payload)
        head = bytes([0x80 | op]) + (bytes([n]) if n < 126 else
                                     bytes([126]) + struct.pack(">H", n) if n < 65536 else
                                     bytes([127]) + struct.pack(">Q", n))
        if not self.writer.is_closing():
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
    ap.add_argument("--preempt-after", type=float, default=1.0,
                    help="seconds a visitor waits for a chip before an attract fold is dropped for it")
    args = ap.parse_args()
    args.chips = [] if args.replay_only else [int(c) for c in args.chips.split(",") if c.strip()]
    args.replay = args.replay or [str(DEMO / "gallery" / "trajectories"), str(HERE / "recordings")]
    if not args.attract or not Path(args.attract).is_file():
        args.attract = None
    asyncio.run(Service(args).main())


if __name__ == "__main__":
    main()
