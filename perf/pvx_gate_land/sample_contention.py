#!/usr/bin/env python3
"""Per-card AICLK + host contention, sampled on a settable interval.

Two things this exists to record, because neither can be reconstructed afterwards:

  AICLK, DURING the fold. On this box the governor alone swings a 512 aa fold 1.27-1.41x, so a
  ladder timed at an unrecorded clock is not a measurement. Sampled PER CARD -- a snapshot that
  greps the first AICLK it finds reports whichever card tt-smi happened to list first, and on this
  host three cards idle at 800 while one folds at 1350.

  Host contention. benchlock samples load ONCE, at acquisition, and is blind to a co-tenant that
  arrives afterwards; on 2026-09-20 this arm acquired at loadavg 2.72 and had a 565 % job land on
  it 12 seconds later. size-ladder scores RUNTIME against scaling exponents and runtime_s carries a
  size-independent host term, so contention inflates the small rungs most and pushes the exponent
  down. A red judged without this record cannot be told from a co-tenant.

The interval is an argument because 30 s was calibrated for size-ladder folds that run for minutes
and is useless against a short one. A 25 s fold can contain ZERO samples, and the reader then has
nothing to intersect and falls back to the whole-run min/max -- which on this host mixes the 1350
of the fold with the 800 of the idle gap beside it and reports "min 800 max 1350" for a fold that
never left 1350. Size the interval so the SHORTEST scored fold contains at least two samples.
"""
import argparse, json, os, subprocess, time

TT = os.path.expanduser("~/.local/bin/tt-smi")


def aiclk():
    try:
        d = json.loads(subprocess.check_output([TT, "-s"], text=True, timeout=60))
    except Exception as e:
        return {"error": str(e)[:80]}
    out = {}
    for i, c in enumerate(d.get("device_info", [])):
        out[str(i)] = c.get("telemetry", {}).get("aiclk")
    return out


def top(n=4):
    try:
        raw = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,comm,args", "--sort=-pcpu"], text=True, timeout=30)
    except Exception:
        return []
    rows = []
    for line in raw.splitlines()[1:n + 1]:
        f = line.split(None, 3)
        if len(f) == 4 and float(f[1]) >= 10.0:
            rows.append({"pid": int(f[0]), "pcpu": float(f[1]), "cmd": f[3][:110]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get(
        "CONTENTION_OUT", "/home/ttuser/pvx_arms/gateland_sizeladder_contention.jsonl"),
        help="jsonl to append to. Defaults to $CONTENTION_OUT, then the size-ladder path.")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between samples. Must be short enough that the shortest scored "
                         "fold contains at least two samples; 30 s cannot see a 25 s fold.")
    a = ap.parse_args()
    if a.interval <= 0:
        ap.error("--interval must be positive")
    while True:
        # `t` is the epoch second the sample was taken. The reader intersects it with each fold's
        # own [t_start, t_end], so it has to be a number, not only the human-readable ts.
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "t": time.time(),
               "loadavg": os.getloadavg(), "ncpu": os.cpu_count(),
               "aiclk": aiclk(), "top": top()}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
