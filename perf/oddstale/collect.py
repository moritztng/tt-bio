#!/usr/bin/env python3
"""One JSON row per leg: what it ran, how long it took, and the clock it took it at.

The clock samples are trimmed to the fold itself -- the 10 s sampler starts with the process and
the first and last samples straddle device open and close, where AICLK reads its 500 MHz idle
value. Reporting those with the fold would understate the clock the work actually ran at, so the
window is START..END from the leg's own .run file and nothing outside it.
"""
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parent


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def leg(tag, root):
    run = (root / "logs" / f"{tag}.run").read_text().splitlines()
    head = dict(re.findall(r"(\w+)=(\S+)", run[0]))
    row = {"tag": tag, "rung": head["rung"], "model": head["model"],
           "card_umd": int(head["card"]), "node": int(head["node"]), "sha": head["sha"],
           "start": run[0].split()[-1]}
    if len(run) > 1:
        tail = dict(re.findall(r"(\w+)=(\S+)", run[1]))
        row |= {"rc": int(tail["rc"]), "wall_s": int(tail["wall"].rstrip("s")),
                "cifs": int(tail["cifs"]), "end": run[1].split()[-1]}
    else:
        row["rc"] = None
    clk = (root / "logs" / f"{tag}.clk").read_text().splitlines()
    t0, t1 = _ts(row["start"]), _ts(row.get("end") or clk[-1].split()[0])
    a, p, l = [], [], []
    for line in clk:
        f = dict(x.split("=", 1) for x in line.split()[1:])
        t = _ts(line.split()[0])
        if not (t0 <= t <= t1):
            continue
        a.append(int(f["A"])); p.append(int(f["P"])); l.append(float(f["L"]))
    # drop the open/close edges: AICLK idles at 500 MHz and the sampler catches it there
    row |= {"aiclk_n": len(a), "aiclk_min": min(a), "aiclk_med": statistics.median(a),
            "aiclk_max": max(a),
            "aiclk_med_infold": statistics.median([x for x in a if x > 500] or a),
            "power_w_med": statistics.median(p) / 1e6, "power_w_max": max(p) / 1e6,
            "host_load_med": statistics.median(l), "host_load_max": max(l)}
    cif = sorted((root / "out" / tag).rglob("*.cif"))
    if cif:
        row["cif"] = str(cif[0].relative_to(root))
        row["cif_sha256"] = hashlib.sha256(cif[0].read_bytes()).hexdigest()
    res = sorted((root / "out" / tag).rglob("results.json"))
    if res:
        r = json.loads(res[0].read_text())
        row["results"] = r
    log = (root / "logs" / f"{tag}.log").read_text()
    row["l1_refusals"] = len(re.findall(r"TT_THROW.*circular buffers", log))
    row["dram_refusals"] = len(re.findall(r"TT_FATAL: Out of Memory", log))
    # forward progress, not just exit code: the longest gap between two engine log lines
    stamps = [datetime.strptime(m, "%H:%M:%S") for m in re.findall(r"^(\d\d:\d\d:\d\d)  \[", log, re.M)]
    row["max_quiet_s"] = max((int((b - a_).total_seconds()) for a_, b in zip(stamps, stamps[1:])), default=0)
    return row


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for tag in sys.argv[2:]:
        print(json.dumps(leg(tag, root)))
