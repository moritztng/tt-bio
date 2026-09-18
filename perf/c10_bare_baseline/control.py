"""Shared evidence helpers for the bounded card-0 calibration."""
from __future__ import annotations
import hashlib, json, os, select, subprocess, sys, threading, time
from pathlib import Path
from audit_prerequisites import HOLDERS

ROOT = Path(__file__).resolve().parents[2]
METAL = Path("/home/ttuser/tt-metal-k10")
CLOCK_REF = "5c60137c2:tt_bio/aiclk.py"

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n")

def digest(path):
    p = Path(path).resolve()
    return {"path":str(p), "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes":p.stat().st_size}

def holders():
    return json.loads(subprocess.check_output(["sudo","-n","python3","-c",HOLDERS],text=True))

def own_nodes(pid=None):
    nodes = set()
    for fd in Path(f"/proc/{pid or os.getpid()}/fd").iterdir():
        try:
            target = os.readlink(fd)
            if target.startswith("/dev/tenstorrent/"): nodes.add(target)
        except OSError: pass
    return sorted(nodes)

def snapshot():
    return {"monotonic_ns":time.monotonic_ns(), "utc_ns":time.time_ns(),
            "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "module_srcversion":Path("/sys/module/tenstorrent/srcversion").read_text().strip(),
            "containment":subprocess.check_output(["systemctl","is-active","qb2-endpoint-containment.service"],text=True).strip(),
            "holders":holders(), "own_nodes":own_nodes()}

def validate_snapshot(s, opened=False):
    if s["containment"] != "active" or s["module_srcversion"] != "A10759A24565BC5BBE903C5":
        raise RuntimeError("Containment or driver prerequisite failed")
    for h in s["holders"]:
        if "/dev/tenstorrent/0" in h["nodes"] and h["pid"] != os.getpid():
            raise RuntimeError(f"Assigned node busy: {h}")
    if opened and s["own_nodes"] != ["/dev/tenstorrent/0"]:
        raise RuntimeError(f"Unexpected device opens: {s['own_nodes']}")

def coverage(samples, interval):
    start,end=interval["start_monotonic_ns"],interval["end_monotonic_ns"]
    during=[s for s in samples if s.get("read_start_ns",-1)>=start and s.get("read_end_ns",end+1)<=end]
    valid=[s for s in during if "MHz" in s]
    centers=[(s["read_start_ns"]+s["read_end_ns"])//2 for s in valid]
    points=[start]+centers+[end]
    result={"samples":len(valid),"min_MHz":min((s["MHz"] for s in valid),default=None),
      "max_MHz":max((s["MHz"] for s in valid),default=None),
      "max_gap_ns":max(b-a for a,b in zip(points,points[1:])),
      "first_offset_ns":centers[0]-start if centers else None,
      "last_offset_ns":end-centers[-1] if centers else None,
      "sample_span_fraction":(centers[-1]-centers[0])/(end-start) if centers else 0,
      "errors":[s for s in during if "error" in s]}
    result["pass"]=(len(valid)>=3 and result["min_MHz"]==result["max_MHz"]==1350
                    and result["max_gap_ns"]<=10_000_000 and not result["errors"])
    return result

def clock_worker(path, owner):
    stop=threading.Event()
    def observe():
        with Path(path).with_name("holders.jsonl").open("w") as out:
            while not stop.is_set():
                row={"monotonic_ns":time.monotonic_ns(),"utc_ns":time.time_ns()}
                try:
                    row.update(holders=holders(),owner_nodes=own_nodes(owner))
                except BaseException as e: row["error"]=repr(e)
                out.write(json.dumps(row)+"\n"); out.flush()
                stop.wait(0.1)
    monitor=threading.Thread(target=observe)
    monitor.start()
    try:
        with Path(path).open("w") as out:
            while not select.select([sys.stdin],[],[],0.001)[0]:
                row={"read_start_ns":time.monotonic_ns(),"utc_ns":time.time_ns(),
                     "node":"/sys/class/tenstorrent/tenstorrent!0/tt_aiclk"}
                try: row["MHz"]=int(Path(row["node"]).read_text())
                except (OSError,ValueError) as e: row["error"]=repr(e)
                row["read_end_ns"]=time.monotonic_ns()
                out.write(json.dumps(row)+"\n"); out.flush()
    finally:
        stop.set(); monitor.join(timeout=10)

if __name__=="__main__":
    clock_worker(sys.argv[1],int(sys.argv[2]))
