"""What `to_host` actually spends its time on, over the real gradient census.

After the single-read fix AdamW still pays 1.167 s to read 1.525 GB of gradient back, which is
1.3 GB/s. A Gen4 x16 link does about twenty times that, so most of that second is not the link.
`to_host` is three things in a row -- `ttnn.to_torch` (the read, and the untilize if the tensor
is tiled), `.to(torch.float32)` (a host cast that doubles 0.76 GB to 1.53 GB), and `.numpy()`
-- and which of them is the second decides whether there is a lever here at all.

The same question applies to the backward, which is 56.6 % of the step and reads tensors back
by the same path, so this is measured at the census rather than on one tensor.

    read_split.py --tokens 384 --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402
from perf.of3t_stepfloor.fullstep import declare_all                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(), "ncpu": os.cpu_count(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "loadavg_start": [round(x, 2) for x in os.getloadavg()],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    _clk = [None]

    def dump():
        if _clk[0] is not None:
            out["env"]["aiclk_during"] = _clk[0].summary()
        out["env"]["loadavg_now"] = [round(x, 2) for x in os.getloadavg()]
        a.out.write_text(json.dumps(out, indent=1, default=str))

    dump()
    with during() as clk:
        _clk[0] = clk
        import numpy as np
        import torch
        import ttnn
        from tt_bio.tenstorrent import get_device
        from tt_bio.train.tensors import to_device

        held, _ = S.capture(a.tokens, out)
        dev = get_device()
        params = declare_all(held["trunk"][0], held["sampler"][0], out)
        rng = np.random.default_rng(20260926)
        grads = []
        for n, t in params.items():
            host = np.asarray(ttnn.to_torch(t.value).to(torch.float32))
            grads.append(to_device((rng.standard_normal(host.shape) * 1e-3).astype(np.float32),
                                   dev, dtype=t.value.dtype))
        ttnn.synchronize_device(dev)
        out["census"] = {"tensors": len(grads),
                         "elements": int(sum(int(g.volume()) for g in grads)),
                         "layouts": sorted({str(g.layout) for g in grads}),
                         "dtypes": sorted({str(g.dtype) for g in grads})}
        dump()

        rows = []
        out["stages"] = rows
        for rep in range(a.reps):
            t_read = t_cast = t_numpy = t_contig = 0.0
            pc = time.perf_counter
            t_all = pc()
            for g in grads:
                t0 = pc()
                th = ttnn.to_torch(g)
                t1 = pc()
                th32 = th.to(torch.float32)
                t2 = pc()
                arr = th32.numpy()
                t3 = pc()
                np.ascontiguousarray(arr)
                t4 = pc()
                t_read += t1 - t0
                t_cast += t2 - t1
                t_numpy += t3 - t2
                t_contig += t4 - t3
            rows.append({"rep": rep, "total_s": round(pc() - t_all, 4),
                         "to_torch_s": round(t_read, 4), "to_float32_s": round(t_cast, 4),
                         "numpy_s": round(t_numpy, 4), "ascontiguous_s": round(t_contig, 4),
                         "loadavg": round(os.getloadavg()[0], 2)})
            print(f"[rep {rep}] to_torch {t_read:6.3f}s  to_float32 {t_cast:6.3f}s  "
                  f"numpy {t_numpy:6.3f}s  ascontiguous {t_contig:6.3f}s  "
                  f"total {rows[-1]['total_s']:6.3f}s", flush=True)
            dump()

        # THE CONTROL. Same bytes, same dtype, same layout, one call instead of 3,152. If the
        # census is slow because of the link this reads the same seconds; if it is slow because
        # of what each call costs, this reads almost none of them.
        one = {}
        try:
            elems = out["census"]["elements"]
            side = 32 * int((elems ** 0.5) // 32)
            big = to_device(np.zeros((side, side), np.float32), dev, dtype=ttnn.bfloat16)
            ttnn.synchronize_device(dev)
            times = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                ttnn.to_torch(big)
                times.append(round(time.perf_counter() - t0, 4))
            gib1 = side * side * 2 / 2 ** 30
            one = {"elements": side * side, "gib_bf16": round(gib1, 3), "times_s": times,
                   "median_s": round(statistics.median(times), 4),
                   "gib_per_s": round(gib1 / statistics.median(times), 2)}
            print(f"[one tensor] {side}x{side} bf16 {gib1:.3f} GiB in "
                  f"{one['median_s']:.4f}s = {one['gib_per_s']:.1f} GiB/s", flush=True)
        except Exception as e:                                   # noqa: BLE001
            one = {"error": repr(e)}
            print(f"[one tensor] {e!r}", flush=True)
        out["one_tensor_control"] = one
        dump()

        warm = rows[1:] or rows
        out["summary"] = {k: round(statistics.median([r[k] for r in warm]), 4)
                          for k in ("total_s", "to_torch_s", "to_float32_s", "numpy_s",
                                    "ascontiguous_s")}
        gib = out["census"]["elements"] * 2 / 2 ** 30
        out["summary"]["device_bytes_gib_bf16"] = round(gib, 3)
        out["summary"]["census_gib_per_s"] = round(gib / out["summary"]["to_torch_s"], 2)
        out["summary"]["us_per_tensor"] = round(
            out["summary"]["to_torch_s"] * 1e6 / out["census"]["tensors"], 1)
        if "median_s" in one:
            out["summary"]["one_tensor_gib_per_s"] = one["gib_per_s"]
            out["summary"]["overhead_ratio"] = round(
                one["gib_per_s"] / out["summary"]["census_gib_per_s"], 1)
        out["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
        dump()
        print(json.dumps(out["summary"], indent=1))
    dump()
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
