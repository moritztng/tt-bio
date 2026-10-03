#!/usr/bin/env python3
"""Static server for the renderer harness, plus POST /bench to record measurements.

    serve.py [--port 8636] [--out bench.jsonl]
Development only; the booth origin (127.0.0.1:8626) is engine's server.
"""
import argparse
import json
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    out = None

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        rec = json.loads(body)
        rec["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8636)
    ap.add_argument("--out", default=str(ROOT / "bench.jsonl"))
    a = ap.parse_args()
    Handler.out = a.out
    Handler.extensions_map[".js"] = "text/javascript"
    ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=str(ROOT))).serve_forever()


if __name__ == "__main__":
    main()
