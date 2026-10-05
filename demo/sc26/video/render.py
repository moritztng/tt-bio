"""Render a composed session (video/compose.py) frame by frame into a lossless video.

    python3 demo/sc26/video/render.py --session ~/sc26-video/s1 --size 3840x2160 \
        --from 30001 --to 37800 --out ~/sc26-video/s1/frames.mkv
    python3 demo/sc26/video/render.py --session ~/sc26-video/s1 --size 1280x720 --end 40000   # timing run

Serves the demo's own web/ directory, unchanged, on its own port, with one addition to the app's
index.html: video/shim.js, which runs the page on a virtual clock and feeds it the session in place
of the engine. The page runs full screen in Firefox on a private headless sway output on qb2's GPU,
as web/app/tools/look.sh does; after each frame the page has finished, grim takes the output and
ffmpeg stores it losslessly (FFV1). Every slot change on the stage is logged to <session>/slots.jsonl,
which is how compose.py finds the loop.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"
sys.path.insert(0, str(HERE.parent / "engine"))
from server import MIME, Folds  # noqa: E402
from compose import load_fold  # noqa: E402


def compositor(tag, size):
    """sway on the headless backend and GLES2, under its own name and socket (web/app/tools/look.sh)."""
    run = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    base = Path.home() / f"sc26{tag}"
    (base / "bin").mkdir(parents=True, exist_ok=True)
    exe = base / "bin" / f"sway-{tag}"
    if not exe.exists():
        exe.symlink_to(shutil.which("sway"))
    cfg = base / f"sway-{size}"
    cfg.write_text(f"output HEADLESS-1 mode {size}@60Hz\ndefault_border none\nseat * hide_cursor 1\n")
    subprocess.run(["pkill", "-INT", "-u", str(os.getuid()), "-x", f"sway-{tag}"])
    time.sleep(1)
    sock = f"wayland-9{tag}"
    before = {p.name for p in Path(run).glob("wayland-*")}
    env = dict(os.environ, XDG_RUNTIME_DIR=run, WLR_BACKENDS="headless", WLR_RENDERER="gles2",
               WLR_LIBINPUT_NO_DEVICES="1", WLR_RENDER_DRM_DEVICE="/dev/dri/renderD128")
    proc = subprocess.Popen([str(exe), "-c", str(cfg)], env=env, stdout=open(base / "sway.log", "w"),
                            stderr=subprocess.STDOUT)
    for _ in range(100):
        new = [p for p in Path(run).glob("wayland-[0-9]*") if p.name not in before and not p.name.endswith(".lock")
               and p.name[8:].isdigit()]
        if new:
            break
        time.sleep(0.1)
    os.rename(new[0], Path(run) / sock)
    if Path(f"{new[0]}.lock").exists():
        os.rename(f"{new[0]}.lock", Path(run) / f"{sock}.lock")
    time.sleep(1)
    return proc, dict(os.environ, XDG_RUNTIME_DIR=run, WAYLAND_DISPLAY=sock), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--size", default="3840x2160")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--from", dest="first", type=int, default=0, help="first frame to keep")
    ap.add_argument("--to", type=int, default=-1, help="last frame to keep")
    ap.add_argument("--end", type=int, default=0, help="last frame to run (default --to)")
    ap.add_argument("--out", default="")
    ap.add_argument("--port", type=int, default=8646)
    ap.add_argument("--tag", default="vid")
    args = ap.parse_args()
    ses = Path(args.session).expanduser()
    session = json.loads((ses / "session.json").read_text())
    folds = {fid: path for fid, path in session.pop("folds").items()}
    entries = {}
    config = json.dumps({"fps": args.fps, "from": args.first, "to": args.to, "end": args.end or args.to,
                         "t0": session["t0"]}).encode()
    body_session = json.dumps({"messages": session["messages"]}, separators=(",", ":")).encode()
    shim = (HERE / "shim.js").read_bytes()
    log = open(ses / "slots.jsonl", "w", buffering=1)
    done = threading.Event()
    enc = None
    if args.out:
        enc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate",
                                str(args.fps), "-c:v", "ppm", "-i", "-", "-c:v", "ffv1", "-level", "3", "-slices", "16",
                                "-threads", "4", "-g", "1", args.out], stdin=subprocess.PIPE)
    state = {"frames": 0, "t0": time.time()}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path)
            p = u.path
            if p == "/__video/config.json":
                return self.send(config, "application/json")
            if p == "/__video/session.json":
                return self.send(body_session, "application/json")
            if p.startswith("/fold/"):
                fid = p[6:]
                if fid not in folds:
                    return self.send(b"", "text/plain", 404)
                if fid not in entries:
                    entries[fid] = load_fold(folds[fid], session["rewrite"][fid])
                every = int(parse_qs(u.query).get("every", ["1"])[0])
                return self.send(Folds.body(entries[fid], every), "application/octet-stream")
            f = (WEB / p.lstrip("/")).resolve()
            if p.endswith("/"):
                f = f / "index.html"
            if not f.is_file() or WEB not in f.parents:
                return self.send(b"", "text/plain", 404)
            body = f.read_bytes()
            if f == WEB / "app" / "index.html":   # the one addition: the recorder's clock and stream
                body = body.replace(b"<head>", b"<head>\n<script>" + shim + b"</script>", 1)
            self.send(body, MIME.get(f.suffix, "application/octet-stream"))

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(n) if n else b""
            if u.path == "/__video/log":
                log.write(data.decode() + "\n")
            elif u.path == "/__video/frame":
                shot = subprocess.run(["grim", "-t", "ppm", "-"], env=env, capture_output=True).stdout
                enc.stdin.write(shot)
                state["frames"] += 1
                if state["frames"] % 300 == 0:
                    print(f"{state['frames']} frames, {(time.time() - state['t0']) / state['frames']:.3f} s each",
                          flush=True)
            elif u.path == "/__video/end":
                done.set()
            self.send(b"{}", "application/json")

    sway, env, base = compositor(args.tag, args.size)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    prof = base / "profile"
    shutil.rmtree(prof, ignore_errors=True)
    prof.mkdir(parents=True)
    shutil.copy(WEB / "app" / "kiosk" / "user.js", prof / "user.js")
    fx = subprocess.Popen(["firefox", "--no-remote", "--profile", str(prof), "--kiosk",
                           f"http://127.0.0.1:{args.port}/app/"],
                          env=env | {"GTK_USE_PORTAL": "0", "MOZ_ENABLE_WAYLAND": "1", "MOZ_CRASHREPORTER_DISABLE": "1"},
                          stdout=open(base / "firefox.log", "w"), stderr=subprocess.STDOUT)
    print(f"serving on {args.port}, firefox {fx.pid}, sway {sway.pid}", flush=True)
    done.wait()
    print(f"end: {state['frames']} frames in {time.time() - state['t0']:.0f} s", flush=True)
    if enc:
        enc.stdin.close()
        enc.wait()
    subprocess.run(["pkill", "-TERM", "-u", str(os.getuid()), "-f", f"[f]irefox.*sc26{args.tag}/profile"])
    time.sleep(2)
    sway.send_signal(2)
    srv.shutdown()


if __name__ == "__main__":
    main()
