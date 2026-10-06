"""What the stream costs on the wire: bytes per message type, per fold, per second and per minute.

    python3 demo/booth/engine/wire.py --seconds 600 --out runs/wire-before.json

A passive client of /stream (it sends nothing), so it can watch the live booth. Counts every
WebSocket frame as it arrives, header included, which is what a link carries above TCP.
"""
import argparse
import json
import struct
import time
from collections import defaultdict

from client import connect, recv_exact


def frames(s):
    """(opcode, payload, bytes on the wire) for every WebSocket frame."""
    while True:
        b0, b1 = recv_exact(s, 2)
        n, head = b1 & 0x7F, 2
        if n == 126:
            n, head = struct.unpack(">H", recv_exact(s, 2))[0], 4
        elif n == 127:
            n, head = struct.unpack(">Q", recv_exact(s, 8))[0], 10
        yield b0 & 0x0F, recv_exact(s, n), head + n


def kind(op, data):
    if op == 2:   # binary: the first byte names it (PROTOCOL.md)
        return {1: "frame"}.get(data[0], f"bin{data[0]}"), None
    m = json.loads(data)
    return m.get("type"), m


def per_protein(folds):
    """One line per protein: how many folds of it crossed the wire, and the bytes of the median one."""
    by = defaultdict(list)
    for f in folds:
        by[(f["n_res"], f["name"])].append(f)
    out = []
    for (n_res, name), fs in sorted(by.items()):
        f = sorted(fs, key=lambda f: f["bytes"])[len(fs) // 2]
        out.append({"name": name, "n_res": n_res, "n_atoms": f["n_atoms"], "folds": len(fs),
                    "live": sum(x["source"] == "live" for x in fs), "frames": f["frames"],
                    "MB": round(f["bytes"] / 1e6, 2), "bytes_per_frame": f["bytes"] // max(1, f["frames"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8626)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--out")
    args = ap.parse_args()
    s = connect(args.host, args.port)
    s.settimeout(args.seconds + 30)
    t0 = time.time()
    by_type = defaultdict(lambda: [0, 0])
    per_s = defaultdict(int)
    folds = {}
    frame_sizes = []
    total = 0
    for op, data, n in frames(s):
        now = time.time() - t0
        if now > args.seconds:
            break
        total += n
        per_s[int(now)] += n
        t, m = kind(op, data)
        by_type[t][0] += 1
        by_type[t][1] += n
        fid = m.get("id") if m else None
        if t == "fold_start":
            folds[fid] = {"n_res": m["n_res"], "n_atoms": m.get("n_atoms"), "source": m.get("source"),
                          "chip": m.get("chip"), "name": m.get("name"), "bytes": 0, "frames": 0, "t0": now}
        if fid in folds:
            f = folds[fid]
            f["bytes"] += n
            if t == "frame":
                f["frames"] += 1
                frame_sizes.append((f["n_atoms"], n))
            if t == "fold_done":
                f["done"], f["seconds_on_wire"] = True, round(now - f["t0"], 1)
    dur = time.time() - t0
    secs = [per_s.get(i, 0) for i in range(int(dur))]
    mins = [sum(secs[i:i + 60]) for i in range(0, len(secs) - 59, 60)]
    done = {k: v for k, v in folds.items() if v.get("done")}
    rep = {
        "seconds": round(dur, 1), "bytes": total, "mean_kbit_s": round(total * 8 / dur / 1e3, 1),
        "peak_1s_kbit_s": round(max(secs or [0]) * 8 / 1e3, 1),
        "per_minute_MB": [round(x / 1e6, 2) for x in mins],
        "by_type": {k: {"n": v[0], "bytes": v[1], "mean": v[1] // max(1, v[0])} for k, v in sorted(by_type.items())},
        "folds": per_protein(done.values()),
    }
    print(json.dumps(rep, indent=1))
    if args.out:
        open(args.out, "w").write(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
