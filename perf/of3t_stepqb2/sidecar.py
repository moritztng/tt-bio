#!/usr/bin/env python3
"""AICLK, RSS and MemAvailable for a run this row does not own, sampled from OUTSIDE it.

`bwprof.py` writes its clock on the last line of `main()`, the same shape that cost this
campaign five exactness-ON arms their AICLK. This row's own harness solved that by sampling
into a flushed log from inside; a file this row does not own gets the same guarantee from a
separate process instead, which also cannot perturb the subject's memory.

    sidecar.py <pid> <out.jsonl> [--period 1.0]
"""
import json, os, sys, time
from pathlib import Path

CLASS = Path("/sys/class/tenstorrent")


def rd(p):
    try:
        return p.read_text().strip()
    except OSError:
        return None


def main():
    pid, out = int(sys.argv[1]), Path(sys.argv[2])
    period = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    card = os.environ.get("TT_VISIBLE_DEVICES", "0")
    node = CLASS / f"tenstorrent!{card}"
    page = os.sysconf("SC_PAGE_SIZE")
    gib = 1024.0 ** 3
    with out.open("w", buffering=1) as f:
        f.write(json.dumps({"header": True, "pid": pid, "card": card,
                            "board": rd(node / "tt_card_type"), "serial": rd(node / "tt_serial"),
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        t0 = time.perf_counter()
        while True:
            try:
                rss = int(open(f"/proc/{pid}/statm").read().split()[1]) * page
            except OSError:
                break                       # the subject is gone; the samples are already durable
            avail = 0
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
            clk = rd(node / "tt_aiclk")
            f.write(json.dumps({"t": round(time.perf_counter() - t0, 2),
                                "rss_gib": round(rss / gib, 4),
                                "avail_gib": round(avail / gib, 4),
                                "aiclk_mhz": int(clk) if clk else None,
                                "load1": round(os.getloadavg()[0], 2)}) + "\n")
            time.sleep(period)
        f.write(json.dumps({"footer": True, "subject_exited": True,
                            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


if __name__ == "__main__":
    main()
