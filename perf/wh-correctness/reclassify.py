#!/usr/bin/env python3
"""Split fleet faults out of the matrix so a cell records its input's verdict, not its luck.

Some production workers hold remote-only MeshDevices and fail every job they take in ~15 s
(state doc 0.2). A cell that landed on one has measured the fleet, not the input, exactly
as a 429 measures the runner. Those rows move to `fleet_faults.jsonl` -- which is the
evidence for 0.2 and must not be lost -- and leave `matrix.jsonl`, so the resumable runner
picks the cell up again.

    reclassify.py            # report what would move
    reclassify.py --apply

Three signatures move, and only three. An OOM, an L1/CB clash or a bad structure is a real
answer about the input and stays where it is.

  mesh_device_remote_only  the 0.2 throw, ~15 s into the job
  server_restart           the job died with "Interrupted by server restart": the pool was
                           bounced under it, so the input was never answered
  pool_offline             submit got a 503 saying the accelerators are offline. That is
                           the service being down, and a cell that was never submitted has
                           measured nothing about its input
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "results" / "matrix.jsonl"
FAULTS = HERE / "results" / "fleet_faults.jsonl"
API = "https://api.japanfold.com"
SIG = "SubDeviceManagerTracker is not initialized"
RESTART = "Interrupted by server restart"


def job_log(job: str) -> str:
    return subprocess.run(["curl", "-s", "-m", "40", f"{API}/v1/jobs/{job}/logs"],
                          capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in MATRIX.read_text().splitlines() if l.strip()]
    keep, moved = [], []
    for r in rows:
        if r.get("pass"):
            keep.append(r)
            continue
        if r.get("submit_status") == 503:
            r["fleet_fault"] = "pool_offline"
            moved.append(r)
            continue
        if r.get("status") == "failed" and RESTART in str(r.get("error", "")):
            r["fleet_fault"] = "server_restart"
            moved.append(r)
            continue
        if r.get("status") == "failed" and r.get("job"):
            log = job_log(r["job"])
            if SIG in log:
                dev = next((ln.split("MeshDevice")[1].split(".")[0].strip()
                            for ln in log.splitlines() if SIG in ln), "?")
                r["fleet_fault"] = "mesh_device_remote_only"
                r["mesh_device"] = dev
                moved.append(r)
                continue
        keep.append(r)

    for r in moved:
        detail = (f"MeshDevice {r['mesh_device']}" if r.get("mesh_device")
                  else r["fleet_fault"])
        print(f"  fleet fault: {r['cell']:32s} {detail}  ({r.get('wall_s')}s)")
    print(f"{len(moved)} fleet faults, {len(keep)} real cells")
    if a.apply and moved:
        with FAULTS.open("a") as f:
            for r in moved:
                f.write(json.dumps(r) + "\n")
        MATRIX.write_text("".join(json.dumps(r) + "\n" for r in keep))
        print(f"moved to {FAULTS.name}; those cells will re-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
