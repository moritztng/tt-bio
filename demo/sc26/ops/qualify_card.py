"""Qualify one chip on the demo's own workload: the real fold service, for hours.

    TT_BIO_LEASE_HOLDER=worker:sc26-robust python3 demo/sc26/ops/qualify_card.py --chip 2 \
        --hours 6 --out demo/sc26/ops/runs/card2-qual

Starts demo/sc26/engine/server.py on one chip with its attract loop (the booth's idle workload)
and, every --visitor-s seconds, submits a visitor fold of a random length from --min-len to
--max-len (prefixes of human serum albumin, so every length is a real sequence). Logs every
chip, fold_done and fold_error event and a 1/30 s status line, and writes summary.json at the end.

A chip passes if no fold errored, no fold stalled, and the worker never restarted.

--server-args go to the engine as they are. One chip alone never takes a long fold in the booth's
rotation (server.py --long-res), so to qualify a chip on every pick give it `--long-res 100000`.
"""
import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"
sys.path.insert(0, str(ENGINE))
import client  # noqa: E402  (the engine's stdlib WebSocket client)
from bench_live import HSA  # noqa: E402


def post(port, seq):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/fold", json.dumps({"sequence": seq}).encode())
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8629)
    ap.add_argument("--visitor-s", type=float, default=20)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-args", default="", help="more engine flags, e.g. '--long-res 100000'")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-u", str(ENGINE / "server.py"), "--chips", str(args.chip),
                            "--port", str(args.port), "--record", "", "--logdir", str(out), *args.server_args.split()],
                           stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    for _ in range(120):
        try:
            s = client.connect("127.0.0.1", args.port)
            break
        except OSError:
            time.sleep(1)
    s.settimeout(1.0)
    rng = random.Random(args.seed)
    log = open(out / "events.jsonl", "w")
    t0 = time.time()
    end, next_visit, next_status = t0 + args.hours * 3600, t0 + 60, 0.0
    done, errors, stalls, aiclk, last = [], [], [], [], {}
    try:
        while time.time() < end and srv.poll() is None:
            now = time.time()
            if now >= next_visit:
                n = rng.randint(args.min_len, args.max_len)
                log.write(json.dumps({"type": "submit", "n_res": n, "reply": post(args.port, HSA[:n]),
                                      "t_wall": now}) + "\n")
                next_visit = now + args.visitor_s
            try:
                op, data = client.recv(s)
            except TimeoutError:
                continue
            if op == 8:
                break
            m = json.loads(data)
            t = m["type"]
            if t == "status":
                c = last = m["chips"][0]
                if c["state"] == "busy" and c["aiclk_mhz"]:
                    aiclk.append(c["aiclk_mhz"])
                if now >= next_status:
                    log.write(json.dumps(m) + "\n")
                    next_status = now + 30
            elif t == "chip":
                log.write(json.dumps(m) + "\n")
                if m["state"] == "stalled":
                    stalls.append(m)
            elif t == "fold_done" and m.get("source") != "replay":
                log.write(json.dumps({k: m.get(k) for k in ("type", "id", "kind", "n_res", "seconds",
                                                             "aiclk_mhz", "t_wall")}) + "\n")
                done.append(m)
            elif t == "fold_error":
                log.write(json.dumps(m) + "\n")
                errors.append(m)
            log.flush()
    finally:
        srv.send_signal(2)  # SIGINT: the worker closes the chip cleanly
        try:
            rc = srv.wait(120)
        except subprocess.TimeoutExpired:
            srv.terminate()
            rc = srv.wait(60)
    secs = lambda k: [d["seconds"] for d in done if d.get("kind") == k]
    real_err = [e for e in errors if e.get("reason") not in ("preempted",)]
    summary = {
        "chip": args.chip, "hours": round((time.time() - t0) / 3600, 2), "server_rc": rc,
        "folds_done": len(done), "visitor_folds": len(secs("visitor")), "attract_folds": len(secs("attract")),
        "fold_errors": len(real_err), "preempted": len(errors) - len(real_err), "stalls": len(stalls), "worker_restarts": last.get("restarts"),
        "max_len_folded": max((d.get("n_res") or 0 for d in done), default=0),
        "visitor_s_median": statistics.median(secs("visitor")) if secs("visitor") else None,
        "aiclk_busy_mhz": {"n": len(aiclk), "min": min(aiclk, default=None),
                           "median": statistics.median(aiclk) if aiclk else None, "max": max(aiclk, default=None)},
        "error_reasons": sorted({e.get("reason") for e in real_err}),
    }
    summary["clean"] = not real_err and not stalls and not last.get("restarts") and rc == 0
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
