"""Sustained folds per second of the booth's chips on one sequence, through the booth's own stream.

    python3 demo/sc26/engine/throughput.py --seconds 180 --depth 8 --out runs/throughput-300.json

Keeps --depth folds of the same sequence queued as visitor folds (they go before the attract
loop), lets every chip fold it --warm times first (the first fold of a new length compiles), then
counts the folds that finish in the next --seconds. Rate = folds finished in the window / window,
the same estimator as scripts/gpu_vs_tt/gpu_concurrency.py on the GPU side. Every fold carries the
chip's AICLK sampled while it ran. Same model, recycles and steps as every booth fold (the
engine's --models; chipworker.py runs each model with tt-bio's own defaults).
"""
import argparse
import json
import socket
import statistics
import time

from bench_live import HSA
from client import connect, recv, send_text

HSA300 = HSA[:300]   # human serum albumin, residues 1-300



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8626)
    ap.add_argument("--sequence", default=HSA300)
    ap.add_argument("--depth", type=int, default=8, help="folds kept queued")
    ap.add_argument("--warm", type=int, default=2, help="untimed folds per chip first")
    ap.add_argument("--chips", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = connect("127.0.0.1", a.port)
    s.settimeout(1.0)
    mine, done, outstanding = set(), [], 0
    t_start = t_stop = None
    warm_left = {c: a.warm for c in range(a.chips)}

    def submit():
        nonlocal outstanding
        send_text(s, json.dumps({"type": "fold", "sequence": a.sequence}))
        outstanding += 1

    for _ in range(a.depth):
        submit()
    while True:
        now = time.time()
        if t_stop and now >= t_stop:
            break
        try:
            op, data = recv(s)
        except socket.timeout:
            continue
        if op == 8:
            raise SystemExit("stream closed")
        m = json.loads(data)
        t = m.get("type")
        if t == "queued":
            mine.add(m["id"])
        elif t == "rejected":
            raise SystemExit(f"rejected: {m}")
        elif t == "fold_done" and m.get("id") in mine:
            outstanding -= 1
            rec = {k: m.get(k) for k in ("id", "chip", "model", "n_res", "seconds", "stages", "aiclk_mhz")}
            rec["t_wall"] = now
            if t_start is None and warm_left.get(m["chip"], 0) > 0:
                warm_left[m["chip"]] -= 1
                rec["warm"] = True
                if not any(warm_left.values()):
                    t_start, t_stop = now, now + a.seconds
            done.append(rec)
            print(json.dumps({k: rec[k] for k in ("chip", "seconds", "aiclk_mhz")} | {"warm": rec.get("warm", False)}),
                  flush=True)
            if not t_stop or now < t_stop:
                submit()
    timed = [r for r in done if not r.get("warm") and t_start < r["t_wall"] <= t_stop]
    span = t_stop - t_start
    out = {"sequence_len": len(a.sequence), "depth": a.depth, "window_s": span, "folds": len(timed),
           "folds_per_s": len(timed) / span, "median_fold_s": statistics.median(r["seconds"] for r in timed),
           "per_chip": {c: sum(r["chip"] == c for r in timed) for c in range(a.chips)},
           "t_start": t_start, "t_stop": t_stop, "events": done}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    main()
