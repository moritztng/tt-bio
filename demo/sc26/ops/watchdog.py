"""Watch everything the booth screen depends on, recover what stops, and log the whole run.

    python3 demo/sc26/ops/watchdog.py --log ~/sc26-logs/watchdog.jsonl

Runs as the sc26-watchdog user unit next to sc26-engine and sc26-kiosk. Every --every seconds:

* engine   GET /status. No answer --engine-fails times in a row: restart sc26-engine. Chip
           workers are watched inside the engine itself (stall -> SIGINT -> SIGTERM -> board reset).
* page     through Firefox's Marionette port on 127.0.0.1 (the kiosk starts Firefox with
           MOZ_MARIONETTE=1): which URL is showing, and how many frames the page drew in one second.
           An error page or a foreign URL is navigated back to the app. A page that cannot run a
           script, or draws no frames, --page-fails times in a row is frozen: restart sc26-kiosk.
* screen   a small grim screenshot of the compositor. The same image for --freeze-s seconds is a
           frozen screen; one flat colour for --blank-s seconds is a blank one. Either restarts
           sc26-kiosk. With no display attached (no output) this check is skipped and logged.
* memory   RSS of the browser and of the engine, free memory and load, to the log, so a slow leak
           is visible long before day three.

Every check and every action is one JSON line in --log.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


class Marionette:
    """Just enough of Firefox's Marionette protocol (length-prefixed JSON over TCP)."""

    def __init__(self, port, timeout=10):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.n = 0
        self._read()  # the server's hello
        self.call("WebDriver:NewSession", {"capabilities": {}})

    def _read(self):
        buf = b""
        while b":" not in buf:
            c = self.s.recv(1)
            if not c:
                raise ConnectionError("marionette closed")
            buf += c
        n, rest = buf.split(b":", 1)
        n = int(n)
        while len(rest) < n:
            c = self.s.recv(n - len(rest))
            if not c:
                raise ConnectionError("marionette closed")
            rest += c
        return json.loads(rest)

    def call(self, cmd, params=None):
        self.n += 1
        body = json.dumps([0, self.n, cmd, params or {}]).encode()
        self.s.sendall(str(len(body)).encode() + b":" + body)
        while True:
            msg = self._read()
            if msg[0] == 1 and msg[1] == self.n:
                if msg[2]:
                    raise RuntimeError(msg[2].get("message", msg[2]))
                return msg[3]

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


FPS_JS = """
const done = arguments[arguments.length - 1];
let n = 0; const t0 = performance.now();
function tick(t) { n++; if (t - t0 < 1000) requestAnimationFrame(tick); else done({fps: n * 1000 / (t - t0),
  href: location.href, title: document.title, visible: document.visibilityState}); }
requestAnimationFrame(tick);
setTimeout(() => done({fps: n, href: location.href, title: document.title, timeout: true}), 4000);
"""


def rss_mb(pattern):
    tot = 0
    for p in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ")
            if pattern.encode() in cmd:
                for line in (p / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1])
        except OSError:
            continue
    return round(tot / 1024)


def unit_mb(unit):
    p = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}/memory.current")
    try:
        return round(int(p.read_text()) / 2**20)
    except (OSError, ValueError):
        return None


def meminfo():
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        m[k] = int(v.split()[0])
    return round(m["MemAvailable"] / 1024 / 1024, 1)


class Watch:
    def __init__(self, a):
        self.a = a
        self.log = open(os.path.expanduser(a.log), "a", buffering=1)
        self.engine_fail = self.page_fail = 0
        self.shot_hash, self.shot_since, self.blank_since = None, time.monotonic(), None
        self.mn = None
        self.last_mem = 0.0
        self.last_restart = {}

    def emit(self, **kw):
        kw.setdefault("t", round(time.time(), 1))
        self.log.write(json.dumps(kw, separators=(",", ":")) + "\n")

    def restart(self, unit, why):
        now = time.monotonic()
        if now - self.last_restart.get(unit, -1e9) < self.a.restart_gap:
            self.emit(ev="restart_skipped", unit=unit, why=why)
            return
        self.last_restart[unit] = now
        rc = subprocess.call(["systemctl", "--user", "restart", f"{unit}.service"], timeout=150)
        self.emit(ev="restart", unit=unit, why=why, rc=rc)
        if unit == "sc26-kiosk":
            self.close_mn()
            self.page_fail, self.shot_hash, self.shot_since, self.blank_since = 0, None, now, None

    def close_mn(self):
        if self.mn:
            self.mn.close()
        self.mn = None

    def check_engine(self):
        try:
            with urllib.request.urlopen(f"{self.a.url_base}/status", timeout=5) as r:
                st = json.loads(r.read())
            self.engine_fail = 0
            return {"chips": [[c["chip"], c["state"], c["aiclk_mhz"], c["folds"], c["restarts"]]
                              for c in st["chips"]], "queue": st["queue"]}
        except (OSError, ValueError) as e:
            self.engine_fail += 1
            self.emit(ev="engine_fail", n=self.engine_fail, err=str(e)[:200])
            if self.engine_fail >= self.a.engine_fails:
                self.restart("sc26-engine", f"/status unanswered {self.engine_fail}x")
                self.engine_fail = 0
            return None

    def check_page(self):
        try:
            if self.mn is None:
                self.mn = Marionette(self.a.marionette)
            r = self.mn.call("WebDriver:ExecuteAsyncScript", {"script": FPS_JS, "args": [],
                                                               "scriptTimeout": 6000})["value"]
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            self.close_mn()
            self.page_fail += 1
            self.emit(ev="page_fail", n=self.page_fail, err=str(e)[:200])
            if self.page_fail >= self.a.page_fails:
                self.restart("sc26-kiosk", f"page unreachable {self.page_fail}x")
            return None
        href = r.get("href", "")
        if not href.startswith(self.a.url_base):
            self.emit(ev="page_wrong", href=href[:200], title=r.get("title", "")[:100])
            try:
                self.mn.call("WebDriver:Navigate", {"url": self.a.app_url})
            except (OSError, RuntimeError) as e:
                self.close_mn()
                self.emit(ev="navigate_fail", err=str(e)[:200])
        if r.get("timeout") or r.get("fps", 0) < 1:
            self.page_fail += 1
            self.emit(ev="page_frozen", n=self.page_fail, fps=r.get("fps"))
            if self.page_fail >= self.a.page_fails:
                self.restart("sc26-kiosk", f"page drew no frames {self.page_fail}x")
        else:
            self.page_fail = 0
        return round(r.get("fps", 0), 1)

    def check_screen(self):
        try:
            img = subprocess.run(["grim", "-s", "0.125", "-t", "ppm", "-"], capture_output=True,
                                 timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            img = b""
        if not img:
            return "none"
        now = time.monotonic()
        h = hashlib.sha1(img).hexdigest()
        px = img[img.index(b"255\n") + 4:] if b"255\n" in img else img
        flat = len(set(px[::7])) <= 3
        if h != self.shot_hash:
            self.shot_hash, self.shot_since = h, now
        if flat:
            self.blank_since = self.blank_since or now
        else:
            self.blank_since = None
        if self.blank_since and now - self.blank_since > self.a.blank_s:
            self.restart("sc26-kiosk", f"screen one flat colour for {now - self.blank_since:.0f} s")
            return "blank"
        if now - self.shot_since > self.a.freeze_s:
            self.restart("sc26-kiosk", f"screen unchanged for {now - self.shot_since:.0f} s")
            return "frozen"
        return "flat" if flat else "moving"

    def run(self):
        self.emit(ev="start", args=vars(self.a))
        while True:
            t0 = time.monotonic()
            eng = self.check_engine()
            fps = self.check_page()
            scr = self.check_screen()
            row = {"ev": "tick", "engine": eng, "fps": fps, "screen": scr}
            if t0 - self.last_mem > self.a.mem_every:
                self.last_mem = t0
                row.update(rss_browser_mb=rss_mb(self.a.profile), engine_mb=unit_mb("sc26-engine.service"),
                           mem_avail_gb=meminfo(), load1=os.getloadavg()[0])
            self.emit(**row)
            time.sleep(max(0.5, self.a.every - (time.monotonic() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="~/sc26-logs/watchdog.jsonl")
    port = os.environ.get("SC26_PORT", "8626")
    ap.add_argument("--url-base", default=f"http://127.0.0.1:{port}")
    ap.add_argument("--app-url", default=f"http://127.0.0.1:{port}/app/")
    ap.add_argument("--marionette", type=int, default=2828)
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--engine-fails", type=int, default=3)
    ap.add_argument("--page-fails", type=int, default=3)
    ap.add_argument("--freeze-s", type=float, default=60)
    ap.add_argument("--blank-s", type=float, default=30)
    ap.add_argument("--restart-gap", type=float, default=90, help="minimum seconds between two restarts of one unit")
    ap.add_argument("--mem-every", type=float, default=60)
    ap.add_argument("--profile", default=os.environ.get("SC26_KIOSK_PROFILE", os.path.expanduser("~/sc26kiosk/profile")),
                    help="the kiosk's Firefox profile path; its processes are the browser's memory")
    a = ap.parse_args()
    Path(os.path.expanduser(a.log)).parent.mkdir(parents=True, exist_ok=True)
    Watch(a).run()


if __name__ == "__main__":
    main()
