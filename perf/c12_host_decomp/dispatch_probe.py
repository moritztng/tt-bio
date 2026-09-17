#!/usr/bin/env python3
"""Can this chip run a program at all? Answer in bounded time, in a process you can kill.

Written because pass 3 lost its whole turn to a chip that could not. `decomp.py` opened
`/dev/tenstorrent/2`, issued the 32x32 bf16 add that `tt_bio.tenstorrent._assert_local_dispatch`
uses as its bring-up probe, and never came back: twelve minutes blocked in
`ttnn.synchronize_device` with the board flat at 32-35 W and no kernel compile running. tt_bio's
own probe is the right test and it is already in the production path; what it lacks is a deadline,
because it runs inline in the process that then holds the device for the rest of the capture.

So run the same test FIRST, in a child process, under a timeout. A wedged chip then costs the
timeout instead of the pass, and the capture never starts against a card that cannot serve it.
The child is expendable: if it hangs it keeps the chip, which a wedged chip does anyway, and the
parent exits without having spent a model load on it.

    python3 dispatch_probe.py --node 2 [--timeout 240]

    exit 0  the chip dispatched a program and synchronised. Timing read may proceed.
    exit 1  it did not, within the deadline. Do NOT measure on this node; it needs a reset.
    exit 2  usage or environment problem.

Deliberately NOT part of `decomp.py`: a check that runs inside the process it is protecting cannot
bound that process. This is also why the deadline is on the parent's `subprocess` call and not on
a signal inside the child, which is exactly the handler that did not fire on 2026-09-17 because
the thread was blocked below Python.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import json, os, sys, time
node = os.environ["TT_VISIBLE_DEVICES"]
t = {}
t0 = time.monotonic()
import torch, ttnn
t["import_s"] = round(time.monotonic() - t0, 3)
sys.path.insert(0, os.environ["TT_BIO_ROOT"])
import tt_bio.tenstorrent as T
t1 = time.monotonic()
dev = T.get_device()
t["open_and_probe_s"] = round(time.monotonic() - t1, 3)
t2 = time.monotonic()
a = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev)
b = ttnn.add(a, a)
ttnn.synchronize_device(dev)
host = ttnn.to_torch(b)
t["add_s"] = round(time.monotonic() - t2, 3)
t["answer_ok"] = bool((host == 0).all())
t["node"] = node
print("PROBE " + json.dumps(t))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    clk = Path(f"/sys/class/tenstorrent/tenstorrent!{a.node}/tt_aiclk")
    if not clk.exists():
        print(f"dispatch_probe: node {a.node} has no {clk} -- not on the bus", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": str(a.node), "TT_BIO_LEASE_CARDS": str(a.node),
                "TT_BIO_ROOT": str(root)})
    env.setdefault("TT_BIO_LEASE_HOLDER", "worker:c12-host-decomp")

    t0 = time.monotonic()
    try:
        p = subprocess.run([a.python, "-c", CHILD], env=env, timeout=a.timeout,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        # The child keeps the chip. That is not a leak this probe can fix: a chip that will not
        # complete a 32x32 add does not release on SIGTERM either, because the thread is blocked
        # below Python. Report it by pid so the caller can name it in its own doc.
        print(json.dumps({"node": a.node, "dispatched": False,
                          "reason": f"no completion within {a.timeout:.0f} s",
                          "verdict": "NEEDS A RESET -- do not measure on this node"}, indent=2))
        return 1
    wall = round(time.monotonic() - t0, 2)

    line = next((l for l in p.stdout.splitlines() if l.startswith("PROBE ")), None)
    if p.returncode != 0 or line is None:
        print(json.dumps({"node": a.node, "dispatched": False, "rc": p.returncode,
                          "wall_s": wall, "stderr_tail": (p.stderr or "")[-600:],
                          "verdict": "did not dispatch"}, indent=2))
        return 1
    got = json.loads(line[len("PROBE "):])
    got.update(dispatched=True, wall_s=wall,
               aiclk_MHz_after=int(clk.read_text()),
               verdict="dispatched and synchronised; timing read may proceed")
    if not got.get("answer_ok"):
        got["verdict"] = "dispatched but returned the WRONG answer -- do not measure"
        print(json.dumps(got, indent=2))
        return 1
    print(json.dumps(got, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
