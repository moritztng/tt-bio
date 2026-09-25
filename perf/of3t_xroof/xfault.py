"""What the resident set actually costs: page supply, or memory bandwidth. NO DEVICE OPENED.

`xmem.py` held a balloon in its own process with the loadavg flat at 2.1-2.4 and watched the
906 MB score-block crossing go 0.348 s -> 0.765 s at 4 GiB held. Both halves of that crossing
allocate and first-touch a fresh 906 MB host destination, so there are two candidate mechanisms
and they have opposite fixes:

  page supply    a fresh 906 MB mapping needs 906 MB of zeroed pages, and with a large resident
                 set the kernel reclaims and zeroes rather than handing over a warm free block.
                 Cost lands in the FAULT, and a copy between two already-faulted buffers is
                 unaffected.
  bandwidth      the resident set evicts cache and the copy itself runs slower. Cost lands in
                 the COPY, and an already-faulted buffer is just as slow.

So measure both at each balloon level. Nothing here touches a card.

    xfault.py --out perf/of3t_xroof/out/xfault.json
"""
import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

NS = time.perf_counter_ns
N_BYTES = 384 * 4 * 384 * 384 * 4
FLOOR_GIB = 6.0


def mem_avail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / (1024 * 1024)
    return 0.0


def meminfo(*keys):
    r = {}
    for ln in open("/proc/meminfo"):
        k = ln.split(":")[0]
        if k in keys:
            r[k] = int(ln.split()[1])
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--balloons", type=float, nargs="+", default=[0, 4, 8])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch

    rep = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(), "no_device_opened": True,
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thp_enabled": Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text().strip()
        if Path("/sys/kernel/mm/transparent_hugepage/enabled").exists() else None,
        "loadavg_start": [round(x, 2) for x in os.getloadavg()]},
        "payload_gb": round(N_BYTES / 1e9, 6), "sweep": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(rep, indent=1, default=str))   # noqa: E731

    balloon, held = [], 0.0
    warm_src = torch.ones(N_BYTES // 4, dtype=torch.float32)
    warm_dst = torch.ones(N_BYTES // 4, dtype=torch.float32)
    for want in a.balloons:
        while held < want and mem_avail() > FLOOR_GIB + 1.0:
            b = torch.empty(int(1 << 30), dtype=torch.uint8)
            b.fill_(7)
            balloon.append(b)
            held += 1.0
        fault, copy = [], []
        for _ in range(a.reps):
            t0 = NS()
            x = torch.empty(N_BYTES // 4, dtype=torch.float32)
            x.fill_(3.0)                                   # allocate + first touch
            fault.append((NS() - t0) / 1e9)
            del x
            gc.collect()
            t0 = NS()
            warm_dst.copy_(warm_src)                       # both already faulted in
            copy.append((NS() - t0) / 1e9)
        mi = meminfo("AnonHugePages", "MemFree", "MemAvailable")
        rep["sweep"].append({
            "balloon_gib_held": held,
            "mem_available_gib": round(mem_avail(), 2),
            "loadavg1": round(os.getloadavg()[0], 2),
            "alloc_and_first_touch_906mb_s": round(statistics.median(fault), 6),
            "alloc_and_first_touch_gb_s": round(N_BYTES / 1e9 / statistics.median(fault), 4),
            "warm_copy_906mb_s": round(statistics.median(copy), 6),
            "warm_copy_traffic_gb_s": round(2 * N_BYTES / 1e9 / statistics.median(copy), 4),
            "anon_hugepages_kb": mi.get("AnonHugePages"),
            "mem_free_kb": mi.get("MemFree"),
            "fault_all_s": [round(x, 5) for x in fault],
            "copy_all_s": [round(x, 5) for x in copy]})
        dump()
    balloon.clear()
    gc.collect()
    rep["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    rep["ok"] = True
    dump()
    print(json.dumps([{k: v for k, v in r.items() if not k.endswith("_all_s")}
                      for r in rep["sweep"]], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
