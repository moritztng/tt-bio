"""A chip worker that opens no device: it plays back a fold recorded on this box with the event
times the chip measured, through the same stdin/stdout protocol as chipworker.py. For testing the
server and the page without taking a chip (server.py --worker).

Each job plays the gallery recording of the nearest length (gallery/store/*.json), so an attract
pick plays its own recorded fold. No frame carries coordinates (none does, PROTOCOL.md); when the
job is the recording's own protein, fold_start carries its atoms and fold_done its packed states,
from the built gallery (gallery/trajectories/<id>.jsonl), so a page can pull it as a live fold.

Test conditions, by environment, each "<chip>:<value>" and comma-separated across chips:
  REC_PACE   1.5      every fold on that chip runs 1.5x slower than recorded
  REC_COLD   1        that chip's first fold carries its recording's measured compile time
                      (seconds_compile_fold - seconds), half before the first trunk recycle and half
                      before the first sampler step, where a cold fold compiles
  REC_HOLD   30:20    on that chip's first fold over 30 s, go quiet at 30 s for 20 s
  REC_FAIL   12       that chip's first fold over 12 s fails at 12 s
REC_AFTER  45      (all chips) apply the conditions only to folds that start 45 s or more after the
                      worker did, so a page that is still loading sees them
SIGUSR1 drops the fold (preempted), SIGINT ends the worker, as in chipworker.py.
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

STORE = Path(__file__).resolve().parents[2] / "gallery" / "store"
BUILT = STORE.parent / "trajectories"


def coordinates(r, seq):
    """The recording's atoms and packed states, if it is a fold of `seq` and the gallery is built."""
    try:
        msgs = [json.loads(l) for l in (BUILT / f"{r['id']}.jsonl").read_text().splitlines() if l.strip()]
    except OSError:
        return {}, {}
    start = next(m for m in msgs if m["type"] == "fold_start")
    done = next(m for m in msgs if m["type"] == "fold_done")
    if start.get("sequence") != seq or "frames" not in done:
        return {}, {}
    return ({k: start[k] for k in ("atoms", "chains") if k in start},
            {k: done[k] for k in ("frames", "xyz", "plddt", "ptm") if k in done})


def per_chip(name, chip):
    for part in os.environ.get(name, "").split(","):
        c, _, v = part.partition(":")
        if c.strip() == str(chip):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--warm", default=None)
    args = ap.parse_args()
    recs = [json.loads(f.read_text()) for f in STORE.glob("*.json")]
    recs = [r for r in recs if r.get("model") == args.model and r.get("stage_events")]
    pace = float(per_chip("REC_PACE", args.chip) or 1)
    cold = per_chip("REC_COLD", args.chip) is not None
    hold = per_chip("REC_HOLD", args.chip)
    fail = per_chip("REC_FAIL", args.chip)
    after = time.monotonic() + float(os.environ.get("REC_AFTER", 0))
    drop = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *a: drop.set())
    lock = threading.Lock()

    def emit(**ev):
        ev.setdefault("t_wall", round(time.time(), 3))
        with lock:
            print(json.dumps(ev, separators=(",", ":")), flush=True)

    emit(type="chip", chip=args.chip, state="warming")
    time.sleep(1)
    emit(type="chip", chip=args.chip, state="ready", aiclk_mhz=1350, model=args.model)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            job = json.loads(line)
            seq = job["sequence"].strip().upper()
            n = len(seq.replace(":", ""))
            r = min(recs, key=lambda r: abs(r["n_res"] - n))
            at_start, at_done = coordinates(r, seq)
            emit(type="chip", chip=args.chip, state="busy", job=job["id"])
            drop.clear()
            armed = time.monotonic() >= after
            extra = (r["seconds_compile_fold"] - r["seconds"]) / 2 if cold and armed else 0.0
            cold = cold and not armed
            ev = [("fold_start", None, 0, 0, r["stages"]["prep"])]
            ev += [("stage", s, k, K, t) for s, k, K, t in r["stage_events"]]
            # a cold fold compiles at its first trunk recycle and its first sampler step
            shift, timed = 0.0, []
            for e in ev:
                if e[0] == "stage" and e[2] == 0 and e[1] in ("trunk", "diffusion"):
                    shift += extra
                timed.append((*e[:4], (e[4] + shift) * pace))
            total = (r["seconds"] + 2 * extra) * pace
            t_hold = t_fail = None
            if armed and hold and total > float(hold.split(":")[0]):
                t_hold, hold = (float(hold.split(":")[0]), float(hold.split(":")[1])), None
            if armed and fail and total > float(fail):
                t_fail, fail = float(fail), None
            t0, dropped = time.monotonic(), False
            held = 0.0
            for kind, stage, k, K, t in timed + [("fold_done", None, 0, 0, total)]:
                if t_fail is not None and t >= t_fail:
                    time.sleep(max(0.0, t0 + t_fail + held - time.monotonic()))
                    emit(type="fold_error", id=job["id"], chip=args.chip, reason="RuntimeError: recorded test failure")
                    dropped = True
                    break
                if t_hold is not None and t >= t_hold[0]:
                    held, t_hold = t_hold[1], None
                while time.monotonic() < t0 + t + held and not drop.is_set():
                    time.sleep(0.005)
                if drop.is_set():
                    emit(type="fold_error", id=job["id"], chip=args.chip, reason="preempted")
                    dropped = True
                    break
                if kind == "fold_start":
                    emit(type="fold_start", id=job["id"], chip=args.chip, model=args.model, sequence=seq,
                         n_res=n, n_atoms=r["n_atoms"], steps=200, loops=3, seed=0, source="live", t=round(t + held, 3),
                         **at_start)
                elif kind == "stage":
                    emit(type="stage", id=job["id"], chip=args.chip, stage=stage, step=k, total=K, t=round(t + held, 3))
            if not dropped:
                emit(type="fold_done", id=job["id"], chip=args.chip, model=args.model, n_res=n,
                     seconds=round(total + held, 3), stages={}, aiclk_mhz={"median": 1350}, source="live",
                     **({"plddt": []} | at_done))
            emit(type="chip", chip=args.chip, state="ready", aiclk_mhz=1350)
    except KeyboardInterrupt:
        pass
    emit(type="chip", chip=args.chip, state="stopped")


if __name__ == "__main__":
    main()
