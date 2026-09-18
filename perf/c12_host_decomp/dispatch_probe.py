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



def upstream_port_state(node) -> dict:
    """Read the bus-master bit of this chip's UPSTREAM PCIe port. Costs about a millisecond.

    Why this is now the FIRST check and not a footnote. The "host-spin wedge" that has cost this
    campaign several passes has a mechanical cause, measured on qb2 on 2026-09-17: the kernel's
    containment handler clears bus-mastering on the upstream port ("QB quarantine: port
    0000:00:01.4 COMMAND 0407 -> 0405; reboot required"). A chip whose port cannot master the bus
    still OPENS -- the device node is there, `fuser` shows a holder, benchlock is happy -- and then
    the first DMA completion never arrives, so the host polls it forever at 100 % CPU with
    `syscr +0`. That is the entire wedge signature, and every instrument built for it (a 60 s
    dispatch probe, a minutes-long forward-progress check) is slower and less specific than
    reading one config-space register.

    The bit is MEMORY SPACE ENABLE, bit 1, and getting that wrong is the easy mistake: the
    quarantine goes 0x0407 -> 0x0405, and both values still carry bit 2 (Bus Master Enable), so a
    bus-master test passes a quarantined port. 0x0407 = I/O + memory + bus master; 0x0405 drops
    memory. With memory-space decoding off, the port stops forwarding MMIO to the endpoint, so
    every register read the driver makes returns nothing: ARC telemetry reads fail with ENODATA
    and a completion-queue poll never advances. Checked both bits here and named separately.

    Note that `tt_aiclk.exists()` does NOT catch this state -- the sysfs file is still there and
    only READING it fails with ENODATA -- which is why an existence check let a quarantined chip
    through three passes of this row.
    """
    out = {"node": str(node)}
    try:
        dev = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/device").resolve()
        port = dev.parent
        out["endpoint"], out["port"] = dev.name, port.name
        cmd = int.from_bytes((port / "config").read_bytes()[4:6], "little")
    except OSError as e:
        out.update(error=repr(e), bus_master=None, ok=False)
        return out
    out["port_COMMAND"] = f"0x{cmd:04x}"
    out["memory_space"] = bool(cmd & 0x2)
    out["bus_master"] = bool(cmd & 0x4)
    out["ok"] = out["memory_space"] and out["bus_master"]
    if not out["ok"]:
        missing = [n for n, b in (("MEMORY SPACE", 0x2), ("BUS MASTER", 0x4)) if not cmd & b]
        out["verdict"] = (f"upstream port has {' and '.join(missing)} CLEARED -- this chip is "
                          "quarantined and needs a REBOOT, not a reset. An open will succeed and "
                          "then host-spin forever, because MMIO to the endpoint goes nowhere.")
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    port = upstream_port_state(a.node)
    print(json.dumps(port, indent=2), flush=True)
    if not port["ok"]:
        print(f"dispatch_probe: node {a.node} upstream port cannot master the bus "
              f"({port.get('port_COMMAND')}) -- REBOOT required, refusing to open it",
              file=sys.stderr)
        return 3
    clk = Path(f"/sys/class/tenstorrent/tenstorrent!{a.node}/tt_aiclk")
    try:
        int(clk.read_text())
    except OSError as e:
        # Existence is not readability: a quarantined or ARC-dead chip keeps the sysfs file and
        # fails the READ with ENODATA, so `clk.exists()` passed this through for three passes.
        print(f"dispatch_probe: node {a.node} cannot read {clk}: {e!r} -- not measurable",
              file=sys.stderr)
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
