"""A minimal stream client: connects to /stream, optionally asks for a fold, logs what arrives.

    python3 demo/sc26/engine/client.py --fold MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE \
        --seconds 60 --log runs/client.jsonl

Standard library only. Prints one line per message (frames abbreviated) and writes the full
messages to --log. Exits after --seconds, or after --folds fold_done messages.
"""
import argparse
import base64
import json
import os
import socket
import struct
import sys
import time


def connect(host, port):
    s = socket.create_connection((host, port))
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /stream HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(1)
    assert b" 101 " in head.split(b"\r\n")[0], head
    return s


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def recv(s):
    b0, b1 = recv_exact(s, 2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", recv_exact(s, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", recv_exact(s, 8))[0]
    return b0 & 0x0F, recv_exact(s, n)


def send_text(s, text):
    data, mask = text.encode(), os.urandom(4)
    n = len(data)
    head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n))
    s.sendall(head + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8626)
    ap.add_argument("--fold", action="append", default=[])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--folds", type=int, default=0, help="exit after this many fold_done")
    ap.add_argument("--log")
    args = ap.parse_args()
    s = connect(args.host, args.port)
    for seq in args.fold:
        send_text(s, json.dumps({"type": "fold", "sequence": seq}))
    log = open(args.log, "w") if args.log else None
    end, done = time.time() + args.seconds, 0
    s.settimeout(1.0)
    while time.time() < end:
        try:
            op, data = recv(s)
        except socket.timeout:
            continue
        if op == 8:
            break
        msg = json.loads(data)
        if log:
            log.write(data.decode() + "\n")
        t = msg["type"]
        if t == "frame":
            n = len(base64.b64decode(msg["xyz"])) // 12
            print(f"frame  {msg['id']} chip={msg['chip']} step={msg['step']}/{msg['of']} t={msg['t']} atoms={n}")
        elif t == "status":
            print("status " + " ".join(f"{c['chip']}:{c['state']}@{c['aiclk_mhz']}" for c in msg["chips"])
                  + f" queue={msg['queue']}")
        elif t in ("fold_start", "fold_done"):
            keep = {k: v for k, v in msg.items() if k not in ("xyz", "atoms", "plddt", "sequence")}
            print(f"{t:10s} {json.dumps(keep)}")
            done += t == "fold_done"
            if args.folds and done >= args.folds:
                break
        else:
            print(f"{t:10s} {json.dumps(msg)[:200]}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
