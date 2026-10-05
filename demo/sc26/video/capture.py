"""Listen to the running booth engine and keep what the video replays: every stream message as it
arrived, and each finished live fold's coordinates.

    python3 demo/sc26/video/capture.py --out ~/sc26-video/tap --minutes 60

A passive page: it only reads, so the booth carries on exactly as before. Each line of
tap.jsonl is {"at": <local receive time>, "m": <the message>}; a finished live fold is pulled once
with GET /fold/<id>?every=1 (all its sampler states) into folds/<at>-<id>.bin. Reconnects when the
engine restarts.
"""
import argparse
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from client import connect, recv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8626)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    (out / "folds").mkdir(parents=True, exist_ok=True)
    log = open(out / "tap.jsonl", "a", buffering=1)
    end = time.time() + 60 * args.minutes
    while time.time() < end:
        try:
            s = connect(args.host, args.port)
        except OSError:
            time.sleep(2)
            continue
        log.write(json.dumps({"at": time.time(), "m": {"type": "_connected"}}) + "\n")
        s.settimeout(1.0)
        try:
            while time.time() < end:
                try:
                    op, data = recv(s)
                except socket.timeout:
                    continue
                if op == 8:
                    break
                if op != 1:
                    continue
                at, m = time.time(), json.loads(data)
                log.write(json.dumps({"at": at, "m": m}, separators=(",", ":")) + "\n")
                if m.get("type") == "fold_done" and m.get("source") == "live" and m.get("n_frames"):
                    url = f"http://{args.host}:{args.port}/fold/{m['id']}?every=1"
                    try:
                        body = urllib.request.urlopen(url, timeout=30).read()
                        (out / "folds" / f"{at:.3f}-{m['id']}.bin").write_bytes(body)
                    except OSError as e:
                        log.write(json.dumps({"at": time.time(), "m": {"type": "_pull_failed", "id": m["id"],
                                                                       "error": str(e)}}) + "\n")
        except (OSError, ConnectionError, ValueError):
            pass
        finally:
            s.close()
        log.write(json.dumps({"at": time.time(), "m": {"type": "_disconnected"}}) + "\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
