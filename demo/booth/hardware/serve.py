"""Stand-alone server for the hardware screens, for development and for screenshots.

    python3 demo/booth/hardware/serve.py            # http://127.0.0.1:8627/app/lanes/

Serves ``demo/booth/web/`` and the current telemetry snapshot at ``/telemetry``. In the booth the
engine's server on :8626 carries the same snapshot as a ``chips`` message on ``/stream``; the
screens accept either. ``--fixture`` serves a recorded snapshot instead of live sysfs, and the
message says so in its ``source`` field, which the screen prints.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import telemetry  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "web"


class Handler(SimpleHTTPRequestHandler):
    snapshot = staticmethod(lambda: {})

    def do_GET(self):
        if self.path.split("?")[0] != "/telemetry":
            return super().do_GET()
        body = json.dumps(self.snapshot(), separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8627)
    ap.add_argument("--events", type=Path, default=telemetry.EVENTS)
    ap.add_argument("--fixture", type=Path, help="serve this recorded snapshot instead of live sysfs")
    a = ap.parse_args()
    if a.fixture:
        rec = json.loads(a.fixture.read_text()) | {"source": "fixture"}
        Handler.snapshot = staticmethod(lambda: rec)
    else:
        Handler.snapshot = telemetry.Monitor(events=a.events).start().snapshot
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=str(WEB)))
    print(f"http://127.0.0.1:{a.port}/app/lanes/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
