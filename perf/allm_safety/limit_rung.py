#!/usr/bin/env python3
"""Does OpenDDE still fold at its published Blackhole cap? Scored on COMPLETION, not on time.

`size_limits.CEILINGS['opendde']['blackhole']` is `binds=freeze, pass_at=1024, fail_at=1536,
mechanism=trunk_freeze`. Its evidence: at 1536 "the trunk walks nine of its ten recycles at a
steady 91-93 s each and stops at `trunk 9/10` forever" -- not an OOM, not idle, not slow -- and a
freeze "leaves the chip refusing every device open at risc_firmware_initializer.cpp:1115, so an
unguarded attempt costs the next job on that card too". So this is run on THIS ROW'S OWN GRANTED
CARD and the card is smoke-tested before it is released.

Why not `perf/bh1536/run_rung.py`, which encodes this protocol already: it hardcodes
`TT_VISIBLE_DEVICES="0", TT_BIO_LEASE_CARDS="0"` at :451 and `env.update` overwrites anything the
caller exports, so it cannot be pointed at another card. Card 0 is another row's dispatch grant.
The protocol is reproduced here rather than the script edited, because that file is not this row's.

Protocol copied from that runner's defaults: `--single_sequence`, `--sampling_steps 20`, the
model's own recycling. The verdict is the rung's, not a ratio:

  FOLDED   the process exited 0 and wrote a structure -- the cap holds
  STALLED  the log stopped growing for `--stall` seconds -- the freeze signature
  FAILED   it exited non-zero -- an exception, which is NOT the freeze this cell guards
"""
from __future__ import annotations

import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "allm_safety"))
import card_guard  # noqa: E402


def descendants(pid):
    """Every live descendant of `pid`, walked from /proc. Explicit pids, never a pattern kill."""
    kids, seen = [], set()
    stack = [pid]
    while stack:
        p = stack.pop()
        try:
            for t in Path(f"/proc/{p}/task").iterdir():
                for c in (t / "children").read_text().split():
                    c = int(c)
                    if c not in seen:
                        seen.add(c); kids.append(c); stack.append(c)
        except OSError:
            pass
    return kids


def stop(proc, log):
    """SIGINT the engine so it can unwind, then SIGKILL whatever is left, parent last."""
    chain = [proc.pid] + descendants(proc.pid)
    for p in chain:
        try: os.kill(p, signal.SIGINT)
        except OSError: pass
    t0 = time.time()
    while time.time() - t0 < 45 and proc.poll() is None:
        time.sleep(1)
    for p in reversed(chain):
        try: os.kill(p, signal.SIGKILL)
        except OSError: pass
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opendde")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--stall", type=int, default=420)
    ap.add_argument("--sampling_steps", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    card_guard.preflight(hold=False)   # the fold is a CHILD process; it acquires its own lease

    fixture = ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{a.size}.yaml"
    assert fixture.is_file(), fixture
    work = a.out.parent / f".rung-{a.model}-{a.size}"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "fold.log"
    log.write_text("")
    outdir = work / "structures"

    cmd = [sys.executable, "-u", "-m", "tt_bio.main", "predict", str(fixture),
           "--model", a.model, "--single_sequence", "--sampling_steps", str(a.sampling_steps),
           "--seed", "0", "--out_dir", str(outdir)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    print(f"=== {a.model} {a.size} aa, card {env.get('TT_VISIBLE_DEVICES')}, "
          f"budget {a.budget}s stall {a.stall}s ===\n  {' '.join(cmd[-10:])}", flush=True)

    t0 = time.time()
    with open(log, "ab", buffering=0) as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        last_size, last_grow = 0, time.time()
        verdict = None
        while True:
            rc = proc.poll()
            if rc is not None:
                verdict = "FOLDED" if rc == 0 else "FAILED"
                break
            sz = log.stat().st_size
            if sz != last_size:
                last_size, last_grow = sz, time.time()
            if a.stall and time.time() - last_grow > a.stall:
                verdict = "STALLED"; stop(proc, log); rc = proc.returncode
                break
            if time.time() - t0 > a.budget:
                verdict = "BUDGET"; stop(proc, log); rc = proc.returncode
                break
            time.sleep(5)
    wall = round(time.time() - t0, 1)

    tail = log.read_text(errors="replace").splitlines()[-25:]
    cifs = sorted(outdir.rglob("*.cif")) if outdir.is_dir() else []
    res = {"model": a.model, "size": a.size, "verdict": verdict, "rc": rc, "wall_s": wall,
           "sampling_steps": a.sampling_steps, "single_sequence": True,
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "structures": [c.name for c in cifs],
           "quiet_seconds_at_end": round(time.time() - last_grow, 1),
           "tail": tail}
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n--- {a.model} {a.size}: {verdict} rc={rc} wall={wall}s "
          f"structures={len(cifs)}", flush=True)
    for line in tail[-8:]:
        print("   ", line[:150], flush=True)
    return 0 if verdict == "FOLDED" else 1


if __name__ == "__main__":
    sys.exit(main())
