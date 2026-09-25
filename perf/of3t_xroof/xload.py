"""Name the limiter: sweep HOST CONTENTION over the same crossing, one process, one card.

`xroof.py` read the shipped score-block crossing at 0.442 s (2.05 GB/s down) and 0.204 s
(4.44 GB/s up) on a quiet box. Three banked runs read 0.712-0.763 s and 0.450-0.491 s on a box
at loadavg 4.6-5.8. Same board, same shape, same layout path -- so either the banked figures
carry a limiter nobody named, or this one does.

The crossing is DMA plus a single-threaded host tile-shuffle over 906 MB. That second term is
host memory bandwidth, which is shared, so the falsifiable prediction is that contention alone
reproduces the banked seconds. Arms are INTERLEAVED round-robin over the load levels rather than
run in blocks, so a drift in the box during the sweep cannot alias onto the axis.

    xload.py --out perf/of3t_xroof/out/xload_384.json
"""
import argparse
import ctypes
import gc
import json
import os
import socket
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402

NS = time.perf_counter_ns
SCORE = (384, 4, 384, 384)


class Load:
    """N threads each streaming a private 256 MiB buffer. Memory bandwidth, not ALU: a spin
    loop would move the loadavg without touching the resource the untilize is bound by."""

    def __init__(self, n, mib=256):
        self.n, self.mib, self._stop, self._t = n, mib, threading.Event(), []

    def _work(self):
        import numpy as np
        a = np.ones(self.mib * 1024 * 1024 // 8, dtype=np.float64)
        b = np.empty_like(a)
        while not self._stop.is_set():
            np.copyto(b, a)

    def __enter__(self):
        for _ in range(self.n):
            t = threading.Thread(target=self._work, daemon=True)
            t.start()
            self._t.append(t)
        if self.n:
            time.sleep(2.0)
        return self

    def __exit__(self, *e):
        self._stop.set()
        for t in self._t:
            t.join(timeout=20)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rep = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nproc": os.cpu_count(),
        "loadavg_start": [round(x, 2) for x in os.getloadavg()]},
        "config": {"reps": a.reps, "levels": a.levels}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(rep, indent=1, default=str))   # noqa: E731
    dump()

    with during() as clk:
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
            host_built = ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
            ttnn.synchronize_device(dev)

            LEGS = {
                "A_to_torch_from_TILE": lambda: ttnn.to_torch(t_tile),
                "D_from_torch_to_TILE": lambda: ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT,
                                                                device=dev, dtype=ttnn.float32),
                "G_from_device_DMA_only": lambda: ttnn.from_device(t_tile),
                "H_host_untilize_no_DMA": lambda: ttnn.to_torch(host_tile),
                "I_host_tilize_no_DMA": lambda: ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT,
                                                                dtype=ttnn.float32),
                "J_to_device_DMA_only": lambda: ttnn.to_device(host_built, dev),
            }
            acc = {lv: {k: [] for k in LEGS} for lv in a.levels}
            load_seen = {lv: [] for lv in a.levels}

            for r in range(a.reps):
                for lv in a.levels:
                    with Load(lv):
                        for k, fn in LEGS.items():
                            ttnn.synchronize_device(dev)
                            t0 = NS()
                            x = fn()
                            ttnn.synchronize_device(dev)
                            acc[lv][k].append((NS() - t0) / 1e9)
                            del x
                            gc.collect()
                        load_seen[lv].append(round(os.getloadavg()[0], 2))
                    rep["progress"] = f"rep {r + 1}/{a.reps} level {lv} done"
                    dump()

            table = []
            for lv in a.levels:
                row = {"contending_threads": lv,
                       "loadavg1_observed": load_seen[lv]}
                for k in LEGS:
                    m = statistics.median(acc[lv][k])
                    row[k + "_s"] = round(m, 6)
                    if k in ("A_to_torch_from_TILE", "D_from_torch_to_TILE",
                             "G_from_device_DMA_only", "J_to_device_DMA_only",
                             "H_host_untilize_no_DMA", "I_host_tilize_no_DMA"):
                        row[k + "_gb_s"] = round(gb / m, 4)
                    row[k + "_all_s"] = [round(x, 5) for x in acc[lv][k]]
                table.append(row)
            rep["sweep"] = table
            rep["payload_gb"] = round(gb, 6)
            rep["ok"] = True
        except Exception:                                                # noqa: BLE001
            rep["error"] = traceback.format_exc()[-4000:]
            rep["ok"] = False
        finally:
            rep["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
            dump()
    rep["env"]["aiclk_during"] = clk.summary()
    keys = list(clk.summary())
    rep["env"]["aiclk_line"] = clk.line(keys[0]) if keys else clk.line()
    dump()
    print(json.dumps({"ok": rep.get("ok"), "aiclk": rep["env"].get("aiclk_line"),
                      "sweep": [{k: v for k, v in r.items() if not k.endswith("_all_s")}
                                for r in rep.get("sweep", [])]}, indent=1, default=str))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
