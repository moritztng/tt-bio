"""Inject booth failures on a schedule and record what the screen showed through each one.

    python3 demo/booth/ops/chaos.py --hours 8 --out ~/booth-logs/soak-<date>

Runs against the installed demo (booth-engine, booth-kiosk, booth-watchdog). Every --every
seconds it picks the next event, in rotation:

  browser_crash   SIGSEGV to the kiosk's main Firefox process
  browser_freeze  SIGSTOP to it (the watchdog must see a page that draws no frames)
  engine_kill     SIGTERM to the engine's main process (systemd must bring it back)
  worker_kill     SIGTERM to one chip worker (the engine restarts it)
  worker_wedge    SIGSTOP to one chip worker mid-fold, as a chip that took work and went quiet:
                  only the engine's stall limit and board reset bring the chip back. Continued
                  after --wedge-s if the engine has not ended it, so it cannot stay stopped
  board_reset     the board under a folding chip reset with ops/reset_board.sh, as a chip that
                  dies mid-fold: its worker's device calls fail or hang. Only when listed in
                  --events; it needs the chip ledger's go-ahead (state/bth/CHIPS.md). Skipped while
                  the board's other chip is lent out of the booth
  queue_flood     60 visitor folds at once, lengths 10-400
  engine_freeze   SIGSTOP to the engine for --freeze-s, then SIGCONT: alive and stuck. Its unit's
                  WatchdogSec must restart it; a SIGINT never lands on a stuck loop
  sway_freeze     SIGSTOP to the booth's compositor: the watchdog must see screenshots time out and
                  restart it (this script continues it after --freeze-s if nothing did)
  sway_crash      SIGTERM to the compositor: session.sh starts it again, the browser follows
  display_unplug  the compositor's output unplugged, as a screen cable pulled: session/display.sh
                  must give it another before a window maps onto nothing
  network_drop    every packet in or out dropped for --net-s seconds except ssh, which is how
                  this script is watched (needs sudo; a system timer lifts the rule even if
                  this script dies). One chip worker is restarted inside the drop, so a model
                  load with no network is part of the test

For each event: screenshots at +2, 5, 10, 20, 40 and 60 s (saved quarter size), whether the
screen was moving, flat or unchanged at each, the watchdog's page frame rate over the next
minute, and the time until the page drew 30+ fps again. One JSON line per event in events.jsonl,
then summary.json.
"""
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
from bench_live import HSA  # noqa: E402

EVENTS = ["browser_crash", "engine_kill", "queue_flood", "worker_kill", "browser_freeze",
          "network_drop", "worker_wedge", "sway_freeze", "engine_freeze", "display_unplug", "sway_crash"]


def sway_env():
    """SWAYSOCK and WAYLAND_DISPLAY as the booth session exported them to the user manager."""
    out = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, text=True).stdout
    env = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
    return {k: env[k] for k in ("SWAYSOCK", "WAYLAND_DISPLAY") if k in env}


def booth_sway():
    """The booth's compositor: a sway started by session/session.sh. Other sways on the box are not ours."""
    for p in pids(lambda c: c[0].endswith(b"sway") and b"-c" in c):
        try:
            ppid = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1]
            if b"session.sh" in Path(f"/proc/{ppid}/cmdline").read_bytes():
                return p
        except OSError:
            continue
    return None


def pids(match, env=None):
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
            if match(cmd) and (env is None or env.encode() in (p / "environ").read_bytes()):
                out.append(int(p.name))
        except OSError:
            continue
    return sorted(out)


def unit_pid(unit):
    r = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


class Chaos:
    def __init__(self, a):
        self.a = a
        self.out = Path(os.path.expanduser(a.out))
        (self.out / "shots").mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "events.jsonl", "a", buffering=1)
        self.wlog = Path(os.path.expanduser(a.watchdog_log))
        # health.py reads this so a restart the test caused is not counted against the booth
        self.marks = open(self.wlog.parent / "chaos.jsonl", "a", buffering=1)
        self.rng = random.Random(a.seed)
        self.frozen, self.hold = None, 0

    def shot(self, name):
        # the display as the booth session exported it, so a run started over ssh sees it too
        env = {**os.environ, **sway_env()}
        sock = Path(env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / env.get("WAYLAND_DISPLAY", "")
        if not env.get("WAYLAND_DISPLAY") or not sock.is_socket():
            print(f"DISPLAY LOST: {sock} is gone, this sample sees nothing", file=sys.stderr)
            return None, {"display": "lost"}
        try:
            img = subprocess.run(["grim", "-s", "0.25", "-t", "ppm", "-"], capture_output=True, timeout=10,
                                 env=env).stdout
        except subprocess.TimeoutExpired:
            return None, {"display": "stuck"}   # the compositor does not answer: the screen holds its last frame
        if not img:
            return None, {"display": "none"}
        px = img[img.index(b"255\n") + 4:]
        sub = px[::97]
        (self.out / "shots" / f"{name}.ppm").write_bytes(img)
        return hashlib.sha1(img).hexdigest(), {"flat": len(set(px[::7])) <= 3,
                                                "mean": round(sum(sub) / max(1, len(sub)), 1)}

    def fps_since(self, t0, t1):
        rows = []
        for line in self.wlog.read_text().splitlines()[-400:]:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if t0 <= r.get("t", 0) <= t1:
                rows.append(r)
        return rows

    def firefox_main(self):
        prof = self.a.profile.encode()
        return pids(lambda c: prof in b" ".join(c) and b"-contentproc" not in c)

    def workers(self):
        """The demo's chip workers, only on --worker-chips if given."""
        out = []
        found = pids(lambda c: any(x.endswith(b"chipworker.py") for x in c), env="worker:booth-demo")
        for p in found:
            try:
                env = Path(f"/proc/{p}/environ").read_bytes().split(b"\0")
                ppid = int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError):
                continue
            if ppid in found:
                continue  # a worker forks a helper with the same command line; signal the worker itself
            chip = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"TT_VISIBLE_DEVICES=")), "")
            if not self.a.worker_chips or chip in self.a.worker_chips.split(","):
                out.append(p)
        return out

    def visitors(self):
        """A visitor fold every --visitor-s seconds, the way people at the booth would type."""
        while True:
            time.sleep(self.a.visitor_s)
            n = self.rng.randint(10, 400)
            req = urllib.request.Request(f"{self.a.url_base}/fold", json.dumps({"sequence": HSA[:n]}).encode())
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    reply = json.loads(r.read()).get("type")
            except OSError as e:
                reply = f"error {e}"[:80]
            with open(self.out / "visitors.jsonl", "a") as f:
                f.write(json.dumps({"t": round(time.time(), 1), "n_res": n, "reply": reply}) + "\n")

    def act(self, ev):
        a = self.a
        if ev == "browser_crash":
            p = self.firefox_main()
            if not p:
                return "no browser"
            os.kill(p[0], signal.SIGSEGV)
            return f"SIGSEGV firefox {p[0]}"
        if ev == "browser_freeze":
            p = self.firefox_main()
            if not p:
                return "no browser"
            os.kill(p[0], signal.SIGSTOP)
            return f"SIGSTOP firefox {p[0]}"
        if ev in ("sway_freeze", "sway_crash"):
            p = booth_sway()
            if not p:
                return "skipped: no booth compositor"
            os.kill(p, signal.SIGSTOP if ev == "sway_freeze" else signal.SIGTERM)
            self.frozen, self.hold = (p, a.freeze_s) if ev == "sway_freeze" else (None, 0)
            return f"{'SIGSTOP' if ev == 'sway_freeze' else 'SIGTERM'} sway {p}"
        if ev == "engine_freeze":
            p = unit_pid("booth-engine.service")
            os.kill(p, signal.SIGSTOP)
            self.frozen, self.hold = p, a.freeze_s
            return f"SIGSTOP engine {p} for {a.freeze_s:.0f} s"
        if ev == "board_reset":
            st = json.loads(urllib.request.urlopen(f"{a.url_base}/status", timeout=5).read())
            busy = [c["chip"] for c in st["chips"] if c["state"] == "busy"]
            if not busy:
                return "skipped: no chip folding"
            chip = self.rng.choice(busy)
            board = "0,1" if chip in (0, 1) else "2,3"
            lent = sorted({int(c) for c in board.split(",")} - {c["chip"] for c in st["chips"]})
            if lent:
                return f"skipped: chip {lent[0]} on board {board} is lent out of the booth"
            r = subprocess.run([str(HERE / "reset_board.sh"), board], capture_output=True, text=True, timeout=200,
                               env=dict(os.environ, TT_BIO_LEASE_HOLDER="worker:booth-demo"))
            return f"reset board {board} under chip {chip} mid-fold: rc {r.returncode}, {r.stdout.strip().splitlines()[-1:]}"
        if ev == "display_unplug":
            env = dict(os.environ, **sway_env())
            outs = json.loads(subprocess.run(["swaymsg", "-t", "get_outputs", "-r"], capture_output=True,
                                             text=True, env=env).stdout or "[]")
            names = [o["name"] for o in outs if o.get("active")]
            for n in names:
                subprocess.run(["swaymsg", "output", n, "unplug" if n.startswith("HEADLESS") else "disable"],
                               env=env, capture_output=True)
            self.unplugged = [n for n in names if not n.startswith("HEADLESS")]
            return f"unplugged {', '.join(names) or 'nothing'}"
        if ev == "engine_kill":
            p = unit_pid("booth-engine.service")
            os.kill(p, signal.SIGTERM)
            return f"SIGTERM engine {p}"
        if ev in ("worker_kill", "worker_wedge"):
            w = self.workers()
            if not w:
                return "skipped: no chip worker"
            p = self.rng.choice(w)
            os.kill(p, signal.SIGTERM if ev == "worker_kill" else signal.SIGSTOP)
            self.frozen, self.hold = (p, a.wedge_s) if ev == "worker_wedge" else (None, 0)
            return f"{'SIGTERM' if ev == 'worker_kill' else 'SIGSTOP'} chipworker {p}"
        if ev == "queue_flood":
            ok = 0
            for _ in range(60):
                n = self.rng.randint(10, 400)
                req = urllib.request.Request(f"{a.url_base}/fold", json.dumps({"sequence": HSA[:n]}).encode())
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        ok += json.loads(r.read()).get("type") == "queued"
                except OSError:
                    pass
            return f"{ok} of 60 visitor folds queued"
        if ev == "network_drop":
            rules = ("table inet boothchaos { chain i { type filter hook input priority -10; policy drop; "
                     "iif lo accept; tcp dport 22 accept; }; chain o { type filter hook output priority -10; "
                     "policy drop; oif lo accept; tcp sport 22 accept; }; }")
            subprocess.run(["sudo", "systemd-run", "--quiet", f"--on-active={a.net_s + 120}",
                            "/usr/sbin/nft", "delete", "table", "inet", "boothchaos"], check=False)
            r = subprocess.run(["sudo", "nft", "-f", "-"], input=rules, text=True)
            # A worker that starts while the network is gone proves the model loads offline.
            w = self.workers()
            if w:
                time.sleep(3)
                os.kill(w[0], signal.SIGTERM)
            probe = subprocess.run(["curl", "-s", "-o", "/dev/null", "-m", "5", "-w", "%{http_code}",
                                    "https://huggingface.co"], capture_output=True, text=True).stdout
            return (f"network dropped for {a.net_s:.0f} s (nft rc {r.returncode}, "
                    f"huggingface.co from qb2 during the drop: HTTP {probe or '000'})"
                    + (f", chipworker {w[0]} restarted inside the drop" if w else ""))
        return "unknown"

    def undo(self, ev):
        if getattr(self, "frozen", None):
            try:
                os.kill(self.frozen, signal.SIGCONT)   # if the recovery has not already ended it
            except ProcessLookupError:
                pass
            self.frozen = None
        if ev == "display_unplug":
            env = dict(os.environ, **sway_env())
            for n in getattr(self, "unplugged", []):
                subprocess.run(["swaymsg", "output", n, "enable"], env=env, capture_output=True)
        if ev == "network_drop":
            subprocess.run(["sudo", "nft", "delete", "table", "inet", "boothchaos"], check=False)

    def run_event(self, i, ev):
        t0 = time.time()
        self.marks.write(json.dumps({"t": round(t0, 1), "event": ev}) + "\n")
        what = self.act(ev)
        samples, prev = [], None
        for dt in (2, 5, 10, 20, 40, 60, 90, 120, 180, 240, 300):
            if getattr(self, "frozen", None) and dt > self.hold:
                self.undo(ev)
            time.sleep(max(0, t0 + dt - time.time()))
            h, info = self.shot(f"{i:03d}-{ev}-{dt:03d}s")
            samples.append({"t": dt, "moving": h != prev if h else None, **info})
            prev = h
        if ev == "network_drop":
            time.sleep(max(0, t0 + self.a.net_s - time.time()))
        self.undo(ev)
        rows = self.fps_since(t0, time.time())
        fps = [r.get("fps") for r in rows if r.get("ev") == "tick"]
        back = next((round(r["t"] - t0) for r in rows if r.get("ev") == "tick" and (r.get("fps") or 0) >= 30
                     and r["t"] > t0 + 1), None)
        acts = [{k: r[k] for k in ("ev", "unit", "why") if k in r} for r in rows if r.get("ev") == "restart"]
        rec = {"i": i, "event": ev, "t_wall": round(t0), "action": what, "screen": samples, "fps": fps,
               "fps30_after_s": back, "watchdog_actions": acts}
        self.log.write(json.dumps(rec) + "\n")
        return rec

    def run(self):
        end = self.a.until or time.time() + self.a.hours * 3600
        evs = [e for e in (self.a.events.split(",") if self.a.events else EVENTS)]
        i = self.a.start_index
        if self.a.visitor_s:
            threading.Thread(target=self.visitors, daemon=True).start()
        while time.time() < end:
            nxt = time.time() + self.a.every
            self.run_event(i, evs[i % len(evs)])
            i += 1
            time.sleep(max(0, min(nxt, end) - time.time()))
        # every event of the soak, also those written before a reboot resumed it
        recs = [json.loads(x) for x in open(self.out / "events.jsonl") if x.strip()]
        bad = [r for r in recs if any(s.get("flat") for s in r["screen"])]
        summary = {"hours": round((end - recs[0]["t_wall"]) / 3600, 2) if recs else 0, "events": len(recs),
                   "by_event": {e: sum(r["event"] == e for r in recs) for e in evs},
                   "flat_screen_events": [r["i"] for r in bad],
                   # the app and the poster are dark (mean 12-27); a bright frame is a browser page
                   "bright_screen_events": [r["i"] for r in recs if any(s.get("mean", 0) > 120 for s in r["screen"])],
                   "blind_screen_events": [r["i"] for r in recs if any("display" in x for x in r["screen"])],
                   "fps30_after_s": {e: [r["fps30_after_s"] for r in recs if r["event"] == e] for e in evs}}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--until", type=float, default=0, help="unix time to stop at, instead of --hours from now")
    ap.add_argument("--start-index", type=int, default=0, help="first event number, to continue a rotation")
    ap.add_argument("--every", type=float, default=900, help="seconds between two events")
    ap.add_argument("--events", default="", help="comma list; default: all, in rotation")
    ap.add_argument("--net-s", type=float, default=300)
    ap.add_argument("--freeze-s", type=float, default=90, help="how long engine_freeze and sway_freeze hold their process")
    ap.add_argument("--wedge-s", type=float, default=270, help="how long worker_wedge holds a worker the engine has not ended")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url-base", default=f"http://127.0.0.1:{os.environ.get('BOOTH_PORT', '8626')}")
    ap.add_argument("--watchdog-log", default="~/booth-logs/watchdog.jsonl")
    ap.add_argument("--profile", default=os.environ.get("BOOTH_KIOSK_PROFILE", os.path.expanduser("~/boothkiosk/profile")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visitor-s", type=float, default=0, help="submit a visitor fold this often; 0: none")
    ap.add_argument("--worker-chips", default="", help="kill and wedge only workers on these chips")
    Chaos(ap.parse_args()).run()


if __name__ == "__main__":
    main()
