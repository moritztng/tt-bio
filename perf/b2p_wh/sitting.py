#!/usr/bin/env python3
"""Run a list of card-gated commands on ONE chip of a Wormhole Galaxy, under one lease.

    sitting.py --chip 30 --params ~/bwx/af2_params --out OUT -- 'pytest ...' 'python round.py ...'

The ladder's chip handling (`ladder.py`: the real lease and the device node held open for the
whole sitting, released on every catchable signal) for commands that are not rungs: the
card-gated half of the test suites and `perf/bgx_inputs/round.py`.

`--params` is made resolvable through a private `BOLTZ_CACHE` under OUT, so `weights.resolve`
finds the AlphaFold parameters for the commands in this sitting and nowhere else. Linking them
into the box's `~/.boltz` instead makes every later card-gated run on the box try to open chips
its agent owns.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from ladder import device_node, sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds a single command may take")
    ap.add_argument("commands", nargs="+", help="each run with bash -c, in order")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "boltz_cache" / "af2" / "params"
    cache.mkdir(parents=True, exist_ok=True)
    for npz in pathlib.Path(args.params).expanduser().glob("params_*.npz"):
        link = cache / npz.name
        if not link.exists():
            link.symlink_to(npz)

    os.environ["TT_VISIBLE_DEVICES"] = str(args.chip)
    from tt_bio.device_lease import DeviceLease

    lease = DeviceLease(card=args.chip, timeout=600).acquire()
    node = device_node(args.chip)
    held = os.open(node, os.O_RDWR)
    say = lambda m: print(f"{time.strftime('%FT%TZ', time.gmtime())} {m}", flush=True)  # noqa: E731
    say(f"holding {lease.path} and {node}; {len(args.commands)} commands")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: (lease.release(), sys.exit(128 + s)))

    stop = threading.Event()
    threading.Thread(target=sample, args=(out, stop), daemon=True).start()
    env = dict(os.environ, TT_BIO_LEASE_DIR=str(out / "leases"),
               TT_BIO_LEASE_CARDS=str(args.chip), BOLTZ_CACHE=str(out / "boltz_cache"))
    try:
        for i, cmd in enumerate(args.commands):
            say(f"=== [{i}] {cmd}")
            t0 = time.time()
            with open(out / f"cmd{i}.log", "w") as log:
                try:
                    rc = subprocess.call(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                                         env=env, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            row = {"i": i, "cmd": cmd, "rc": rc, "seconds": round(time.time() - t0, 1),
                   "finished_utc": time.strftime("%FT%TZ", time.gmtime())}
            with open(out / "sitting.jsonl", "a", buffering=1) as f:
                f.write(json.dumps(row) + "\n")
            say(f"    [{i}] rc={rc} {row['seconds']:.0f}s")
    finally:
        stop.set()
        os.close(held)
        lease.release()
        say("sitting done, lease released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
