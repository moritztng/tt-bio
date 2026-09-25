"""Separate the two contention stories: other processes, or the subject's own resident set.

`xload.py` showed the score-block crossing is host-memory-bandwidth bound -- two saturating
threads take it from 0.556 s to 0.784 s, which lands on the banked 0.712-0.763 s. But the banked
runs were not measured beside two memcpy threads; they were measured inside a taped backward
holding 10-15 GiB resident on a 30.5 GiB box. A large resident set is a different limiter from a
noisy neighbour, and only one of them is the sprint's to fix.

So: hold a balloon in THIS process, re-measure the same legs, and keep a MemAvailable floor so
the probe cannot become the incident. No other process is touched.

    xmem.py --out perf/of3t_xroof/out/xmem_384.json
"""
import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402

NS = time.perf_counter_ns
SCORE = (384, 4, 384, 384)
FLOOR_GIB = 6.0


def mem_avail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / (1024 * 1024)
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--balloons", type=float, nargs="+", default=[0, 4, 8])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rep = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mem_total_gib": round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                               / (1 << 30), 2),
        "mem_available_gib_start": round(mem_avail(), 2),
        "loadavg_start": [round(x, 2) for x in os.getloadavg()]},
        "config": {"reps": a.reps, "balloons_gib": a.balloons, "floor_gib": FLOOR_GIB}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(rep, indent=1, default=str))   # noqa: E731
    dump()

    with during() as clk:
        balloon = []
        try:
            import torch
            import ttnn
            from tt_bio.tenstorrent import get_device

            dev = get_device()
            ne = 1
            for d in SCORE:
                ne *= d
            gb = ne * 4 / 1e9
            ht = torch.randn(SCORE, dtype=torch.float32)
            t_tile = ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
            host_tile = ttnn.from_device(t_tile)
            ttnn.synchronize_device(dev)

            LEGS = {
                "A_to_torch_from_TILE": lambda: ttnn.to_torch(t_tile),
                "D_from_torch_to_TILE": lambda: ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT,
                                                                device=dev, dtype=ttnn.float32),
                "G_from_device_DMA_only": lambda: ttnn.from_device(t_tile),
                "H_host_untilize_no_DMA": lambda: ttnn.to_torch(host_tile),
            }
            table = []
            held = 0.0
            for want in a.balloons:
                while held < want:
                    if mem_avail() < FLOOR_GIB + 1.0:
                        rep.setdefault("guard", []).append(
                            f"stopped growing at {held} GiB: MemAvailable "
                            f"{mem_avail():.2f} GiB near the {FLOOR_GIB} GiB floor")
                        break
                    b = torch.empty(int(1 << 30), dtype=torch.uint8)
                    b.fill_(7)
                    balloon.append(b)
                    held += 1.0
                acc = {k: [] for k in LEGS}
                for _ in range(a.reps):
                    for k, fn in LEGS.items():
                        if mem_avail() < FLOOR_GIB:
                            raise MemoryError(f"MemAvailable {mem_avail():.2f} GiB under floor")
                        ttnn.synchronize_device(dev)
                        t0 = NS()
                        x = fn()
                        ttnn.synchronize_device(dev)
                        acc[k].append((NS() - t0) / 1e9)
                        del x
                        gc.collect()
                row = {"balloon_gib_held": held,
                       "mem_available_gib": round(mem_avail(), 2),
                       "loadavg1": round(os.getloadavg()[0], 2)}
                for k in LEGS:
                    m = statistics.median(acc[k])
                    row[k + "_s"] = round(m, 6)
                    row[k + "_gb_s"] = round(gb / m, 4)
                    row[k + "_all_s"] = [round(x, 5) for x in acc[k]]
                table.append(row)
                rep["sweep"] = table
                dump()
            rep["payload_gb"] = round(gb, 6)
            rep["ok"] = True
        except Exception:                                                # noqa: BLE001
            rep["error"] = traceback.format_exc()[-4000:]
            rep["ok"] = False
        finally:
            balloon.clear()
            gc.collect()
            rep["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
            rep["env"]["mem_available_gib_end"] = round(mem_avail(), 2)
            dump()
    rep["env"]["aiclk_during"] = clk.summary()
    keys = list(clk.summary())
    rep["env"]["aiclk_line"] = clk.line(keys[0]) if keys else clk.line()
    dump()
    print(json.dumps({"ok": rep.get("ok"), "guard": rep.get("guard"),
                      "aiclk": rep["env"].get("aiclk_line"),
                      "sweep": [{k: v for k, v in r.items() if not k.endswith("_all_s")}
                                for r in rep.get("sweep", [])]}, indent=1, default=str))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
