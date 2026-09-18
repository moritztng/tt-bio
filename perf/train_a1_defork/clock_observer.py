#!/usr/bin/env python3
"""Independent clock observer: every card's AICLK, and which node the fold actually opened.

The first pass sampled `/sys/class/tenstorrent/tenstorrent!0` because the fold was launched
with TT_VISIBLE_DEVICES=0, and read 800 MHz flat. `tt-smi` and the sysfs nodes do NOT share
an index on this host -- tt-smi's device 2 is the card holding /dev/tenstorrent/3 -- so a
sampler keyed to the launch flag can be watching a different chip than the one computing.
This samples ALL FOUR nodes and separately records which device nodes the folding process has
open, so the attribution is read off the process rather than inferred from an env var.

Nothing here opens a device: sysfs reads and /proc/<pid>/fd readlinks only.
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path("/sys/class/tenstorrent")
NODES = sorted(int(p.name.split("!")[1]) for p in ROOT.glob("tenstorrent!*"))


def clocks():
    out = {}
    for n in NODES:
        try:
            out[n] = int((ROOT / f"tenstorrent!{n}" / "tt_aiclk").read_text().strip())
        except (OSError, ValueError):
            out[n] = None
    return out


def power():
    out = {}
    for n in NODES:
        p = next(iter((ROOT / f"tenstorrent!{n}").glob("device/hwmon/hwmon*/power1_input")), None)
        try:
            out[n] = round(int(p.read_text().strip()) / 1e6, 1) if p else None
        except (OSError, ValueError):
            out[n] = None
    return out


def holders():
    """pid -> the device nodes it has open, for every process holding one."""
    out = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        nodes = set()
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    t = os.readlink(fd)
                except OSError:
                    continue
                if "/dev/tenstorrent/" in t:
                    nodes.add(int(t.rsplit("/", 1)[1]))
        except (PermissionError, FileNotFoundError, ProcessLookupError, NotADirectoryError):
            continue
        if nodes:
            try:
                cmd = (proc / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
            except OSError:
                cmd = "?"
            out[int(proc.name)] = {"nodes": sorted(nodes), "cmd": cmd[:110]}
    return out


def main():
    out_path, hz, stop_file = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    rec = {"nodes": NODES, "samples": [], "holder_log": []}
    last_h = None
    t0 = time.time()
    while not stop_file.exists():
        c = clocks()
        rec["samples"].append({"t": round(time.time() - t0, 2), "aiclk": c})
        h = holders()
        key = json.dumps({str(k): v["nodes"] for k, v in h.items()}, sort_keys=True)
        if key != last_h:
            rec["holder_log"].append({"t": round(time.time() - t0, 2), "holders": h,
                                      "power_w": power()})
            last_h = key
        if len(rec["samples"]) % 50 == 0:
            out_path.write_text(json.dumps(rec))
        time.sleep(1.0 / hz)
    out_path.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
