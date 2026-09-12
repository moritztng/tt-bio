#!/usr/bin/env python3
"""F1-direct: the fused trimul tail packs its matmul straight to the result CB.

`trimul_tail`'s kernel stages every matmul output block in `intermediate_cb` and then copies it
back out, which is an unpack and a pack of the whole block, twice per tail call. The stage is the
running-sum buffer for a multi-K-block contraction, and `trimul_tail.eligible` refuses any call
that has one. So on every call the kernel serves, the stage hands back the number it was given.

Three phases, one process, one device open:

  parity -- `torch.equal` of the whole TriangleMultiplication output with the fused tail on, direct
            pack against staged. The claim is bit-exactness; anything short of equal fails.
  bytes  -- ttnn.graph capture per arm. The deleted traffic is L1, not DRAM, so this is expected to
            read ZERO and is recorded to say so: the pass-deletion thesis pays inside an op.
  time   -- interleaved A/B/A/B, medians, staged arm run under two labels for its own A/A floor.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_algebra"))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.trimul_tail as F1                                               # noqa: E402
from real_traffic import counts                                               # noqa: E402
from ledger import trimul_weights, CZ, Z_MB                                   # noqa: E402


def arm(on):
    F1.DIRECT_PACK = 1 if on else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--host", default="whglx")
    ap.add_argument("--card", type=int, default=5)
    ap.add_argument("--cz", type=int, default=CZ)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    W = lambda s: trimul_weights(s, a.cz)
    mods = {False: T.TriangleMultiplication(False, W(0), ck),
            True: T.TriangleMultiplication(True, W(3), ck)}
    g = torch.Generator().manual_seed(7)
    z = ttnn.from_torch(torch.randn(1, a.n, a.n, a.cz, generator=g), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    m = ttnn.from_torch(torch.ones(1, a.n, a.n), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    run = lambda mod: mod(z, m)

    res = {"n": a.n, "cz": a.cz, "host": a.host, "card": a.card, "arch": "WH" if a.host == "whglx" else "BH",
           "grid": list(T.COMPUTE_GRID_MAIN), "reps": a.reps,
           "tail_f1_on": bool(T._TRIMUL_TAIL_F1),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    parity = {}
    for ending, mod in mods.items():
        got = {}
        for on in (False, True):
            arm(on)
            o = run(mod)
            ttnn.synchronize_device(dev)
            got[on] = ttnn.to_torch(o)
            ttnn.deallocate(o)
        parity["ending" if ending else "starting"] = {
            "equal": bool(torch.equal(got[False], got[True])),
            "max_abs": float((got[False].float() - got[True].float()).abs().max()),
        }
    res["parity"] = parity
    res["tail_served"] = list(F1.STATS)
    res["tail_rejects"] = {str(k): v for k, v in getattr(F1, "REJECTS", {}).items()}
    print("parity:", json.dumps(parity), "tail served/declined:", F1.STATS, flush=True)

    byts = {}
    for ending, mod in mods.items():
        cell = {}
        for on in (False, True):
            arm(on)
            ttnn.deallocate(run(mod))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            o = run(mod)
            ttnn.synchronize_device(dev)
            gr = ttnn.graph.end_graph_capture()
            ttnn.deallocate(o)
            c = counts({"sig": "trimul", "nodes": gr})
            cell["on" if on else "off"] = {k: round(c[k], 3) for k in
                                           ("real_MB", "real_w_MB", "real_r_MB")} | {
                                               "n_ops": c["n_ops"]}
        cell["deleted_MB"] = round(cell["off"]["real_MB"] - cell["on"]["real_MB"], 3)
        byts["ending" if ending else "starting"] = cell
        print("bytes", "ending" if ending else "starting", json.dumps(cell), flush=True)
    res["bytes"] = byts

    labels = [("off", False), ("on", True), ("off2", False)]
    times = {k: {"starting": [], "ending": []} for k, _ in labels}
    for on in (False, True):
        arm(on)
        for mod in mods.values():
            ttnn.deallocate(run(mod))
    ttnn.synchronize_device(dev)
    for _ in range(a.reps):
        for lbl, on in labels:
            arm(on)
            for ending, mod in mods.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = run(mod)
                ttnn.synchronize_device(dev)
                times[lbl]["ending" if ending else "starting"].append(
                    (time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)
    med = {lbl: {k: round(statistics.median(v), 4) for k, v in d.items()}
           for lbl, d in times.items()}
    res["time_ms"] = med
    res["time_raw"] = times
    res["ratio"] = {k: round(med["off"][k] / med["on"][k], 5) for k in ("starting", "ending")}
    res["aa_floor"] = {k: round(med["off"][k] / med["off2"][k], 5) for k in ("starting", "ending")}
    print("median ms:", json.dumps(med))
    print("ratio off/on:", res["ratio"], " A/A floor:", res["aa_floor"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
