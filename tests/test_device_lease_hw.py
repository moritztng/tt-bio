"""On-hardware proof: two tt_bio device opens racing for ONE physical TT card serialize.

Reproduces the flagship-vs-RFD3 incident (two `tt_bio.main predict`-style opens landing on
the same card) on a real TT card. The HOLDER opens card 0, runs a tiny op, and holds the
device for HOLD seconds; the WAITER, launched once the holder is up, calls the same
get_device() and must BLOCK on the physical-card lease until the holder closes -- never
opening the card concurrently (which would collide at the fd level). Afterwards the WAITER
runs a tiny op to confirm the card is healthy.

Uses the DEFAULT lease dir (~/.coworker/state/leases, <host>-card<N>.json) -- the exact path
and naming the fleet dispatcher uses -- so this also exercises the real shared-view path.

Run on a host with a free TT card 0:  TT_VISIBLE_DEVICES=0 python3 tests/test_device_lease_hw.py
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import os, sys, time
sys.path.insert(0, os.environ["REPO"])
role = os.environ["ROLE"]
t0 = time.time()
# Match the real `tt_bio.main predict` bring-up: on a P300 (Blackhole QuietBox) each
# single-chip worker needs the 1x1 mesh-graph descriptor or ttnn.open_device aborts.
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
from tt_bio import tenstorrent as tt
dev = tt.get_device()                      # <-- acquires the physical-card lease, then opens
opened = time.time()
import torch, ttnn
x = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
ttnn.synchronize_device(dev)
print(f"{role} OPENED_AT={opened:.3f} WAITED={opened - t0:.2f}s", flush=True)
open(os.environ["MARKER"], "w").write(str(opened))
time.sleep(float(os.environ.get("HOLD", "0")))
tt.cleanup()                               # <-- closes device + releases the lease
print(f"{role} CLOSED_AT={time.time():.3f}", flush=True)
"""


def _run(role, hold, marker):
    e = dict(os.environ)
    e.update(REPO=REPO, ROLE=role, HOLD=str(hold), MARKER=marker, TT_VISIBLE_DEVICES="0")
    e.pop("TT_BIO_LEASE_DIR", None)   # use the real default (~/.coworker/state/leases)
    e.pop("TT_BIO_LEASE_HOST", None)
    return subprocess.Popen([sys.executable, "-c", CHILD], env=e,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


if __name__ == "__main__":
    hm, wm = "/tmp/hw_holder_opened", "/tmp/hw_waiter_opened"
    for m in (hm, wm):
        if os.path.exists(m):
            os.remove(m)

    holder = _run("HOLDER", hold=25, marker=hm)
    # wait until the holder actually has the card open
    for _ in range(1200):
        if os.path.exists(hm):
            break
        if holder.poll() is not None:
            print("HOLDER exited early:\n" + holder.communicate()[0])
            sys.exit(1)
        time.sleep(0.1)
    holder_opened = float(open(hm).read())
    print(f"holder opened card 0 at {holder_opened:.3f}; launching waiter (must block on lease)")

    waiter = _run("WAITER", hold=0, marker=wm)
    wout = waiter.communicate()[0]
    hout = holder.communicate()[0]
    print("--- holder output ---\n" + hout)
    print("--- waiter output ---\n" + wout)

    assert waiter.returncode == 0, "waiter failed to open the card after the holder released"
    assert os.path.exists(wm), "waiter never opened the card"
    waiter_opened = float(open(wm).read())
    # The waiter must have opened AFTER the holder closed -> serialized, not concurrent.
    holder_closed = None
    for line in hout.splitlines():
        if "CLOSED_AT=" in line:
            holder_closed = float(line.split("CLOSED_AT=")[1])
    assert holder_closed is not None, "holder never reported CLOSED_AT"
    gap = waiter_opened - holder_closed
    assert waiter_opened >= holder_closed, (
        f"COLLISION: waiter opened the card ({waiter_opened:.3f}) BEFORE the holder "
        f"closed it ({holder_closed:.3f})")
    print(f"\nSERIALIZED: waiter opened {gap:+.2f}s relative to holder close "
          f"(>=0 => no concurrent open). Card healthy after (tiny op ran). PASS")
