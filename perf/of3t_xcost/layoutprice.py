"""How much of a `to_torch` / `from_torch` is the HOST tilize, and no device is opened.

`of3t-bwattrib`'s per-shape-class table prices the triangle attention score block
`[384,4,384,384]` FLOAT32 at 251.07 s of a 705.78 s backward, 216 crossings each way:

    to_torch    216 calls   153.854 s   0.712 s/call
    from_torch  216 calls    97.214 s   0.450 s/call

The logical payload is 384*4*384*384*4 = 905.97 MB, so the effective rate is 1.27 GB/s down and
2.01 GB/s up on a PCIe link an order of magnitude faster than that. (The `gb` column in that
table is OPERAND traffic -- output bytes plus every ttnn argument -- not PCIe bytes, so its
GB/s is not a link rate and is not used here.)

`ttnn/operations/core.py:398` is why. `to_torch` is `from_device(tensor)` -- a DMA of the
tensor still in TILE layout -- and then `tensor.to_torch()`, which untilizes on the HOST.
`from_torch` is the mirror: `ttnn.Tensor(...)` tilizes the torch buffer on the host and uploads
the result. A 32x32 tile shuffle over 906 MB is a strided single-threaded pass, and the
device's own `to_layout` does the same work at 19.21 GB/s (`by_verb`, same run).

So the question is how many of those seconds are the tilize rather than the link, and it is
answerable with NO DEVICE. `ttnn.from_torch(..., device=None)` builds a host tensor: with
`layout=TILE_LAYOUT` it runs exactly the host tilize `from_torch` runs, and with
`ROW_MAJOR_LAYOUT` it does not. `ttnn.to_torch` on a host tensor runs the host untilize with no
DMA at all. The difference between the two arms is the term a device-side `to_layout` would
move off the host.

NO DEVICE IS OPENED. `ttnn.open_device` is never called; every tensor here lives in host
memory. That makes the number board-insensitive -- a property of this host's memory system --
which is the half of the question `of3t/BACKWARD.md` 4b says can run without the one card.

    layoutprice.py --out perf/of3t_xcost/out/LAYOUTPRICE.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time

import torch

import ttnn


def loadavg():
    return [round(x, 2) for x in os.getloadavg()]


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
        del r
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="perf/of3t_xcost/out/LAYOUTPRICE.json")
    a = ap.parse_args()

    n, h = a.tokens, a.heads
    el = n * h * n * n
    payload_gb = el * 4 / 1e9
    rep = {"doc": __doc__.strip(), "argv": vars(a),
           "env": {"host": platform.node(), "torch": torch.__version__,
                   "threads": torch.get_num_threads(), "cpus": os.cpu_count(),
                   "loadavg_start": loadavg(),
                   "device_opened": False,
                   "board_insensitive": "yes -- host memory only, ttnn.open_device is never "
                                        "called and no tensor has a device"},
           "shape": [n, h, n, n], "elements": el, "payload_gb": payload_gb,
           "banked": {"to_torch_s_per_call": 0.7122887, "from_torch_s_per_call": 0.4500656,
                      "source": "perf/of3t_bwattrib/out/hist_384_base.json by_shape_class, "
                                "216 calls each, arm base, pc card 0, AICLK 1350 median "
                                "sampled DURING"},
           "arms": {}}

    x = torch.randn(n, h, n, n, dtype=torch.float32)

    # --- the way OUT of torch: what `from_torch` does on the host before the upload -------
    rep["arms"]["from_torch_host_TILE"] = {
        "what": "ttnn.from_torch(x, dtype=float32, layout=TILE_LAYOUT, device=None) -- the "
                "host tilize `from_torch` runs before its upload",
        "s": timed(lambda: ttnn.from_torch(x, dtype=ttnn.float32,
                                           layout=ttnn.TILE_LAYOUT), a.reps)}
    rep["arms"]["from_torch_host_ROW_MAJOR"] = {
        "what": "the same call in ROW_MAJOR: no tilize, the buffer copy only",
        "s": timed(lambda: ttnn.from_torch(x, dtype=ttnn.float32,
                                           layout=ttnn.ROW_MAJOR_LAYOUT), a.reps)}

    # --- the way IN to torch: what `to_torch` does after the DMA -------------------------
    t_tile = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    t_rm = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    rep["arms"]["to_torch_host_TILE"] = {
        "what": "ttnn.to_torch on a host TILE tensor -- the host untilize `to_torch` runs "
                "after its DMA, with no DMA in the number",
        "s": timed(lambda: ttnn.to_torch(t_tile), a.reps)}
    rep["arms"]["to_torch_host_ROW_MAJOR"] = {
        "what": "the same on a ROW_MAJOR host tensor: no untilize",
        "s": timed(lambda: ttnn.to_torch(t_rm), a.reps)}

    # bit-identity of the route the lever proposes: TILE -> ROW_MAJOR -> torch must equal
    # TILE -> torch. On the host this is `ttnn.to_layout`; on the device it is the same op.
    y_direct = ttnn.to_torch(t_tile)
    y_via_rm = ttnn.to_torch(ttnn.to_layout(t_tile, ttnn.ROW_MAJOR_LAYOUT))
    rep["bit_identity"] = {
        "route": "to_torch(to_layout(t, ROW_MAJOR)) vs to_torch(t) on a TILE float32 tensor",
        "bit_identical": bool(torch.equal(y_direct, y_via_rm)),
        "max_abs_diff": float((y_direct - y_via_rm).abs().max()),
        "note": "a layout conversion moves bytes; for FLOAT32 at a shape that is a multiple "
                "of the 32x32 tile there is no pad and no rounding. Checked, not assumed."}
    del y_direct, y_via_rm, t_tile, t_rm
    gc.collect()

    for k, v in rep["arms"].items():
        v["median_s"] = statistics.median(v["s"])
        v["min_s"] = min(v["s"])
        v["gb_s_at_payload"] = payload_gb / v["median_s"]

    out_tilize = (rep["arms"]["from_torch_host_TILE"]["median_s"]
                  - rep["arms"]["from_torch_host_ROW_MAJOR"]["median_s"])
    in_untilize = (rep["arms"]["to_torch_host_TILE"]["median_s"]
                   - rep["arms"]["to_torch_host_ROW_MAJOR"]["median_s"])
    rep["split"] = {
        "host_tilize_s_per_call": out_tilize,
        "host_untilize_s_per_call": in_untilize,
        "banked_from_torch_s_per_call": 0.4500656,
        "banked_to_torch_s_per_call": 0.7122887,
        "host_tilize_share_of_from_torch": out_tilize / 0.4500656,
        "host_untilize_share_of_to_torch": in_untilize / 0.7122887,
        "calls_each_way_per_backward": 216,
        "seconds_on_the_host_per_backward": (out_tilize + in_untilize) * 216,
        "backward_s": 705.78,
        "share_of_backward": (out_tilize + in_untilize) * 216 / 705.78,
        "caveat": "an UPPER bound on what a device-side to_layout could move, and by two "
                  "terms: the device to_layout is not free (19.21 GB/s on the same run's "
                  "by_verb, so ~0.047 s per 906 MB) and the ROW_MAJOR DMA is larger than the "
                  "TILE one by nothing at all but is still a DMA. It is also measured on an "
                  "UNCONTENDED-ish host at the loadavg recorded here, while the banked "
                  "seconds were taken at loadavg 4.58.",
    }
    rep["env"]["loadavg_end"] = loadavg()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({"arms": {k: {"median_s": v["median_s"], "gb_s": v["gb_s_at_payload"]}
                               for k, v in rep["arms"].items()},
                      "split": rep["split"], "bit_identity": rep["bit_identity"]}, indent=1))


if __name__ == "__main__":
    main()
