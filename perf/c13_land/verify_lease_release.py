#!/usr/bin/env python3
"""Does the gate driver hand the card back after its one in-process fold leg?

Runs release_gate's own run_pxdesign() in THIS process, then opens the card from a
subprocess exactly as every later fold leg does. Pre-fix that second open exits 75 against
the driver's own lease, which is what failed opendde-abag, nesso1 and both capacity legs on
2026-09-18.

  --in-driver   negative control: runs the design in THIS process, reproducing the pre-fix
                leg. That arm must fail. It already has: on 2026-09-18 the probe below hung
                5m39s at 0.5 % CPU in futex_wait behind the driver's own UMD fds.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def device_fds():
    """Device nodes this process has open, read from its own /proc — the ground truth the
    lease file only describes."""
    out = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            t = os.readlink(fd)
        except OSError:
            continue
        if "tenstorrent" in t:
            out.append(t)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-driver", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    import release_gate as rg

    # main() does this before any leg, and every child inherits it. A lone p300 chip is a
    # CUSTOM cluster and refuses to open without it, which has nothing to do with the leases
    # under test here.
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    rec = {"arm": "in-driver" if a.in_driver else "fixed", "aiclk": None}
    rec["aiclk"] = Path("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk").read_text().strip()

    t0 = time.monotonic()
    if a.in_driver:
        print("NEGATIVE CONTROL: folding in this process, the probe below must fail", flush=True)
        rg._pxdesign_child(rg.REPO_ROOT / "pxdesign_gate_designs",
                           rg.REPO_ROOT / "pxdesign_gate.json")
        row = {"gate": True, "fit_rmsd": None, "seconds": None, "error": None}
    else:
        row = rg.run_pxdesign(keep=False)
    rec["pxdesign"] = {k: row.get(k) for k in ("gate", "fit_rmsd", "seconds", "error")}
    print(f"pxdesign leg: gate={row['gate']} fit_rmsd={row['fit_rmsd']} "
          f"err={row['error']} ({time.monotonic()-t0:.1f}s)", flush=True)

    # Did the DRIVER actually let go? Two independent reads: its own open fds, and the lease.
    rec["driver_fds_after"] = device_fds()
    tt = sys.modules.get("tt_bio.tenstorrent")
    rec["driver_lease_after"] = None if tt is None else (tt._device_lease is not None)
    rec["driver_device_after"] = None if tt is None else (tt._device is not None)
    print(f"driver after leg: fds={rec['driver_fds_after']} "
          f"lease_held={rec['driver_lease_after']} device_open={rec['driver_device_after']}",
          flush=True)

    # The thing that actually broke: a later leg's subprocess opening the same card.
    try:
        grid = rg._l1_budget_physical_grid()
        rec["subprocess_open"] = {"ok": True, "grid": list(grid)}
        print(f"subprocess device open: OK, compute grid {grid[0]}x{grid[1]}", flush=True)
    except Exception as e:
        rec["subprocess_open"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"subprocess device open: FAILED {type(e).__name__}: {e}", flush=True)

    ok = (row["gate"] and not rec["driver_fds_after"]
          and rec["driver_lease_after"] in (None, False)
          and rec["driver_device_after"] in (None, False)
          and rec["subprocess_open"]["ok"])
    rec["verdict"] = "PASS" if ok else "FAIL"
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rec, indent=2) + "\n")
    print(f"VERDICT {rec['verdict']} (arm={rec['arm']}, aiclk={rec['aiclk']})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
