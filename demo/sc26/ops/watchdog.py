"""Watch everything the booth screen depends on, recover what stops, and log the whole run.

    python3 demo/sc26/ops/watchdog.py --log ~/sc26-logs/watchdog.jsonl

Runs as the sc26-watchdog user unit next to sc26-engine and sc26-kiosk. Every --every seconds:

* engine   GET /status. No answer --engine-fails times in a row: restart sc26-engine. Chip
           workers are watched inside the engine itself (stall -> SIGINT -> SIGTERM -> board reset).
* page     through Firefox's Marionette port on 127.0.0.1 (the kiosk starts Firefox with
           MOZ_MARIONETTE=1): which URL is showing, and how many frames the page drew in one second.
           An error page or a foreign URL is navigated back to the app once the app answers; an
           error page while it does not restarts sc26-kiosk (poster). A page that cannot run a
           script, or draws no frames, --page-fails times in a row is frozen: restart sc26-kiosk.
* screen   a small grim screenshot of the compositor. The same image for --freeze-s seconds is a
           frozen screen; one flat colour for --blank-s seconds is a blank one. Either restarts
           sc26-kiosk. With no output this check is skipped and logged as none; a WAYLAND_DISPLAY
           socket that has disappeared is logged as lost.
* memory   RSS of the browser and of the engine, free memory and load, to the log, so a slow leak
           is visible long before day three.

Every check and every action is one JSON line in --log.
"""
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


class Marionette:
    """Just enough of Firefox's Marionette protocol (length-prefixed JSON over TCP)."""

    def __init__(self, port, timeout=5):
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
const page = () => { const c = document.querySelector('canvas'), s = window.sc26?.stream;
  return {href: location.href, doc: document.documentURI, title: document.title, visible: document.visibilityState,
    gl_lost: c ? !!c.getContext('webgl2')?.isContextLost() : null,
    stream_age_s: s ? (performance.now() - s.lastMessage) / 1000 : null}; };
function tick(t) { n++; if (t - t0 < 1000) requestAnimationFrame(tick); else done({fps: n * 1000 / (t - t0), ...page()}); }
requestAnimationFrame(tick);
setTimeout(() => done({fps: n, timeout: true, ...page()}), 4000);
"""


def proc_usage(pid):
    """(RSS in MB, open file descriptors) of one process, or None once it is gone."""
    try:
        rss = next(int(l.split()[1]) for l in Path(f"/proc/{pid}/status").read_text().splitlines()
                   if l.startswith("VmRSS:"))
        return rss / 1024, len(os.listdir(f"/proc/{pid}/fd"))
    except (OSError, StopIteration, ValueError):
        return None


def resources(profile, engine_pid, paths):
    """Everything that can leak over a booth week, one sample: memory and descriptors of the engine,
    each chip worker and the browser, the GPU's memory, the disk, the logs. The cgroup figure
    systemd prints counts page cache too, so it is not a leak figure; RSS and fds are."""
    r = {"mem_avail_gb": meminfo(), "load1": round(os.getloadavg()[0], 2)}
    # The browser is every process under the one started on the kiosk profile: its content, GPU and
    # socket processes come from a fork server and do not carry the profile on their command line.
    parent, roots = {}, []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            parent[int(p.name)] = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
            if profile.encode() in (p / "cmdline").read_bytes():
                roots.append(int(p.name))
        except (OSError, ValueError, IndexError):
            continue
    kids = [q for q, pp in parent.items() if engine_pid and pp == engine_pid]
    tree, todo = set(), list(roots)
    while todo:
        q = todo.pop()
        if q not in tree:
            tree.add(q)
            todo += [c for c, pp in parent.items() if pp == q]
    browser = [0.0, 0]
    for u in filter(None, map(proc_usage, tree)):
        browser[0] += u[0]
        browser[1] += u[1]
    eng = proc_usage(engine_pid) if engine_pid else None
    work = [u for u in map(proc_usage, kids) if u]
    r.update(engine_rss_mb=eng and round(eng[0]), engine_fds=eng and eng[1],
             worker_rss_mb=[round(u[0]) for u in work], worker_fds=[u[1] for u in work],
             rss_browser_mb=round(browser[0]), browser_fds=browser[1],
             engine_mb=unit_mb("sc26-engine.service"), **unit_stat("sc26-engine.service"))
    for k, f in (("vram_mb", "mem_info_vram_used"), ("gtt_mb", "mem_info_gtt_used")):
        try:
            r[k] = round(int(next(Path("/sys/class/drm").glob(f"card*/device/{f}")).read_text()) / 2**20)
        except (OSError, StopIteration, ValueError):
            pass
    st = os.statvfs(os.path.expanduser("~"))
    r["disk_free_gb"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
    r["logs_mb"] = round(sum(f.stat().st_size for f in paths if f.exists()) / 2**20, 1)
    return r


def unit_mb(unit):
    p = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}/memory.current")
    try:
        return round(int(p.read_text()) / 2**20)
    except (OSError, ValueError):
        return None


def unit_stat(unit):
    """The unit's memory split into anon (what its processes allocated) and file (page cache it
    touched, which the kernel gives back under pressure)."""
    p = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}/memory.stat")
    try:
        st = dict(l.split() for l in p.read_text().splitlines())
        return {"engine_anon_mb": round(int(st["anon"]) / 2**20), "engine_file_mb": round(int(st["file"]) / 2**20)}
    except (OSError, ValueError, KeyError):
        return {}


def meminfo():
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        m[k] = int(v.split()[0])
    return round(m["MemAvailable"] / 1024 / 1024, 1)


def bound(path, cap, keep):
    """Keep a log under `cap` bytes by cutting it, in place, to its last `keep` bytes from a line
    start. In place, because the writers (the engine's chip workers, sway, Firefox) hold the file
    open with O_APPEND: a renamed file would keep growing unseen. A line or two written during the
    cut can land out of order; nothing is lost that a booth needs."""
    try:
        if path.stat().st_size <= cap:
            return 0
        with open(path, "rb") as f:
            f.seek(-keep, os.SEEK_END)
            tail = f.read()
        tail = tail[tail.find(b"\n") + 1:]
        with open(path, "r+b") as f:
            f.truncate(0)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, tail)
        finally:
            os.close(fd)
        return 1
    except OSError:
        return 0


def booth_sway(pids):
    """The sway under the booth session's loop (session/session.sh), not the first one pgrep lists: a
    second sway on the box (a test rig, a desktop) has a lower pid as often as not, and then the
    booth's would never be restarted. None if no candidate runs under the loop."""
    for p in pids:
        try:
            ppid = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1]
            if b"session.sh" in Path(f"/proc/{ppid}/cmdline").read_bytes():
                return int(p)
        except (OSError, IndexError, ValueError):
            continue
    return None


class Watch:
    def __init__(self, a):
        self.a = a
        self.log = open(os.path.expanduser(a.log), "a", buffering=1)
        self.engine_fail = self.page_fail = 0
        self.shot_hash, self.shot_since, self.blank_since = None, time.monotonic(), None
        self.mn = None
        self.last_mem = 0.0
        self.last_restart = {}
        self.stuck = 0   # consecutive screen checks the compositor did not answer
        self.bounded = [Path(os.path.expanduser(x)) for x in a.bound]

    def emit(self, **kw):
        kw.setdefault("t", round(time.time(), 1))
        self.log.write(json.dumps(kw, separators=(",", ":")) + "\n")

    def restart(self, unit, why):
        now = time.monotonic()
        if now - self.last_restart.get(unit, -1e9) < self.a.restart_gap:
            self.emit(ev="restart_skipped", unit=unit, why=why)
            return
        self.last_restart[unit] = now
        # --no-block: a unit that is slow to stop must not stall every other check behind it
        rc = subprocess.call(["systemctl", "--user", "--no-block", "restart", f"{unit}.service"], timeout=30)
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
            return {"chips": [[c["chip"], c["state"], c.get("aiclk_mhz"), c.get("folds"), c.get("restarts")]
                              for c in st["chips"]], "queue": st["queue"]}
        except (OSError, ValueError) as e:
            self.engine_fail += 1
            self.emit(ev="engine_fail", n=self.engine_fail, err=str(e)[:200])
            if self.engine_fail >= self.a.engine_fails:
                self.restart("sc26-engine", f"/status unanswered {self.engine_fail}x")
                self.engine_fail = 0
            return None

    def load_app(self):
        """Navigate to the app, but only if it answers. Navigating while the engine is down puts
        Firefox's "Unable to connect" page on the booth screen (soak 10-06: 5 times, up to 40 s);
        a live page left alone keeps playing what it has and reconnects by itself."""
        try:
            with urllib.request.urlopen(self.a.app_url, timeout=3) as r:
                r.read(1)
        except OSError as e:
            self.emit(ev="navigate_held", err=str(e)[:200])
            return False
        try:
            self.mn.call("WebDriver:Navigate", {"url": self.a.app_url})
        except (OSError, RuntimeError) as e:
            self.close_mn()
            self.emit(ev="navigate_fail", err=str(e)[:200])
        return True

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
            return None, None
        href = r.get("href", "")
        # An error page keeps the app's URL in location.href; only documentURI says about:neterror.
        if not href.startswith(self.a.url_base) or r.get("doc", href).startswith("about:"):
            self.emit(ev="page_wrong", href=href[:200], doc=r.get("doc", "")[:200], title=r.get("title", "")[:100])
            if not self.load_app():
                # the app is down and Firefox shows its own error: the launcher shows the poster
                # and waits for the app instead
                self.restart("sc26-kiosk", "error page while the app is down")
        # The page reconnects a silent stream itself after 6 s. One that has heard nothing for
        # --stream-s is stuck in a way its own code cannot see: load it again.
        elif (r.get("stream_age_s") or 0) > self.a.stream_s:
            self.emit(ev="stream_stuck", age_s=round(r["stream_age_s"]))
            self.load_app()
        if r.get("gl_lost"):
            self.emit(ev="gl_lost")   # the page reloads itself on webglcontextlost; logged as evidence
        if r.get("timeout") or r.get("fps", 0) < 1:
            self.page_fail += 1
            self.emit(ev="page_frozen", n=self.page_fail, fps=r.get("fps"))
            if self.page_fail >= self.a.page_fails:
                self.restart("sc26-kiosk", f"page drew no frames {self.page_fail}x")
        else:
            self.page_fail = 0
        return round(r.get("fps", 0), 1), r.get("stream_age_s") and round(r["stream_age_s"], 1)

    def check_screen(self):
        sock = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / os.environ.get("WAYLAND_DISPLAY", "")
        if not sock.is_socket():
            return "lost"
        try:
            img = subprocess.run(["grim", "-s", "0.125", "-t", "ppm", "-"], capture_output=True,
                                 timeout=self.a.grim_s).stdout
        except subprocess.TimeoutExpired:
            return self.compositor_stuck()
        except OSError:
            img = b""
        self.stuck = 0
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

    def compositor_stuck(self):
        """The socket is there and a screenshot does not come back: sway itself is stuck. After
        --sway-fails checks in a row it gets SIGTERM, and only if it runs under the booth session's
        loop (session/session.sh), which starts it again; the browser's launcher then reopens on
        it. Anywhere else (a desktop someone is using) this only logs."""
        self.stuck += 1
        if self.stuck < self.a.sway_fails:
            return "stuck"
        pid = booth_sway(subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", "sway"], capture_output=True,
                                        text=True).stdout.split())
        looped = pid is not None
        now = time.monotonic()
        if looped and now - self.last_restart.get("sway", -1e9) > self.a.restart_gap:
            self.last_restart["sway"] = now
            os.kill(pid, signal.SIGCONT)   # a stopped process cannot act on SIGTERM
            os.kill(pid, signal.SIGTERM)
            self.emit(ev="restart", unit="sway", why=f"no screenshot for {self.stuck} checks", pid=pid)
            self.stuck = 0
            self.close_mn()
            self.page_fail, self.shot_hash, self.shot_since, self.blank_since = 0, None, now, None
        else:
            self.emit(ev="sway_stuck", n=self.stuck, pid=pid, looped=looped)
        return "stuck"

    def run(self):
        self.emit(ev="start", args=vars(self.a))
        while True:
            t0 = time.monotonic()
            eng = self.check_engine()
            scr = self.check_screen()
            # the page cannot draw under a stuck compositor, and its 4 s probe would only delay the next screenshot
            fps, stream_age = self.check_page() if scr != "stuck" else (None, None)
            row = {"ev": "tick", "engine": eng, "fps": fps, "stream_age_s": stream_age, "screen": scr}
            if t0 - self.last_mem > self.a.mem_every:
                self.last_mem = t0
                pid = int(subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value",
                                          "sc26-engine.service"], capture_output=True, text=True).stdout.strip() or 0)
                row.update(resources(self.a.profile, pid, self.bounded))
                cut = [str(f) for f in self.bounded if bound(f, self.a.log_cap_mb << 20, self.a.log_keep_mb << 20)]
                if cut:
                    row["bounded"] = cut
            self.emit(**row)
            # a screenshot that did not come back is checked again at once, not in 10 s: a stuck
            # compositor leaves its last frame on the glass until it is restarted
            time.sleep(1 if 0 < self.stuck < self.a.sway_fails else max(0.5, self.a.every - (time.monotonic() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="~/sc26-logs/watchdog.jsonl")
    port = os.environ.get("SC26_PORT", "8626")
    ap.add_argument("--url-base", default=f"http://127.0.0.1:{port}")
    ap.add_argument("--app-url", default=f"http://127.0.0.1:{port}/app/")
    ap.add_argument("--marionette", type=int, default=2828)
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--engine-fails", type=int, default=3)
    ap.add_argument("--page-fails", type=int, default=2)
    ap.add_argument("--freeze-s", type=float, default=30)
    ap.add_argument("--blank-s", type=float, default=30)
    ap.add_argument("--restart-gap", type=float, default=90, help="minimum seconds between two restarts of one unit")
    ap.add_argument("--mem-every", type=float, default=60)
    ap.add_argument("--stream-s", type=float, default=60, help="a page that has heard nothing from the stream this long is reloaded")
    ap.add_argument("--grim-s", type=float, default=2,
                    help="a screenshot slower than this means the compositor is stuck (measured p95 0.21 s on qb2)")
    ap.add_argument("--sway-fails", type=int, default=3, help="stuck screenshots in a row before sway is restarted")
    ap.add_argument("--log-cap-mb", type=int, default=64, help="a log past this is cut to its last --log-keep-mb")
    ap.add_argument("--log-keep-mb", type=int, default=16)
    ap.add_argument("--bound", nargs="*", default=[
        "~/sc26-logs/watchdog.jsonl", "~/sc26-logs/sway.log", "~/sc26-logs/engine/reset.log",
        *(f"~/sc26-logs/engine/chip{i}.stderr" for i in range(4)), "~/sc26kiosk/firefox.log",
        "~/.local/state/sc26/folds.jsonl"], help="every file the demo appends to")
    ap.add_argument("--profile", default=os.environ.get("SC26_KIOSK_PROFILE", os.path.expanduser("~/sc26kiosk/profile")),
                    help="the kiosk's Firefox profile path; its processes are the browser's memory")
    a = ap.parse_args()
    Path(os.path.expanduser(a.log)).parent.mkdir(parents=True, exist_ok=True)
    Watch(a).run()


if __name__ == "__main__":
    main()
