"""The watchdog never navigates the kiosk to the app while the app does not answer.
Navigating then is what put Firefox's "Unable to connect" page on the booth screen in the 10-06 soak.
A fake Marionette records what the watchdog asks of the browser; the app is a local HTTP server,
up or down. Control: the watchdog before this rule (git 7744e58cd) navigates in case A.
Touches no unit, browser or chip. Run from the repo with any python3."""
import http.server, importlib.util, socket, subprocess, sys, tempfile, threading
from pathlib import Path
from types import SimpleNamespace

OPS = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(f"wd{abs(hash(path))}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeMn:
    def __init__(self, page):
        self.page, self.calls = page, []

    def call(self, cmd, params):
        self.calls.append(cmd)
        return {"value": self.page} if cmd == "WebDriver:ExecuteAsyncScript" else {"value": None}

    def close(self):
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(mod, port, page):
    w = mod.Watch.__new__(mod.Watch)
    base = f"http://127.0.0.1:{port}"
    w.a = SimpleNamespace(url_base=base, app_url=base + "/app/", stream_s=60, page_fails=2)
    w.log, w.page_fail, w.restarts = open("/dev/null", "w"), 0, []
    w.mn = FakeMn(dict(page, href=base + "/app/"))
    w.restart = lambda unit, why: w.restarts.append(unit)
    w.check_page()
    return "WebDriver:Navigate" in w.mn.calls, w.restarts


live_stuck = {"doc": "", "fps": 60, "stream_age_s": 70}
error_page = {"doc": "about:neterror?e=connectionFailure", "fps": 60, "stream_age_s": None}
new = load(OPS / "watchdog.py")
old_src = subprocess.run(["git", "show", "7744e58cd:demo/sc26/ops/watchdog.py"], cwd=OPS,
                         capture_output=True, text=True, check=True).stdout
old_file = Path(tempfile.mkdtemp()) / "watchdog_old.py"
old_file.write_text(old_src)
old = load(old_file)

down = free_port()
up = free_port()
root = Path(tempfile.mkdtemp())
(root / "app").mkdir()
(root / "app" / "index.html").write_text("app")


class Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(root), **k)

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", up), Quiet)
threading.Thread(target=srv.serve_forever, daemon=True).start()

cases = [  # name, module, port, page, expected (navigated, restarts)
    ("A app down, live page, stream silent 70 s", new, down, live_stuck, (False, [])),
    ("B app down, error page showing", new, down, error_page, (False, ["booth-kiosk"])),
    ("C app up, live page, stream silent 70 s", new, up, live_stuck, (True, [])),
    ("D app up, error page showing", new, up, error_page, (True, [])),
]
rc = 0
for name, mod, port, page, want in cases:
    got = run(mod, port, page)
    ok = got == want
    rc |= not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: navigated={got[0]} restarts={got[1]}")
ctl = run(old, down, live_stuck)
print(f"control, old watchdog, case A: navigated={ctl[0]} "
      f"({'navigates into the error page, as the soak saw' if ctl[0] else 'UNEXPECTED'})")
rc |= not ctl[0]
srv.shutdown()
sys.exit(rc)
