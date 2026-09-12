#!/usr/bin/env python3
"""K2: the triangle-attention gate rides in the qkv projection. Parity, bytes and time.

One process, one device open, three phases:

  parity -- `torch.equal` of the whole TriangleAttention output, fused against shipped, both
            directions. The claim is bit-exactness, so anything short of equal is a failure and
            not a tolerance question.
  bytes  -- ttnn.graph capture per arm, counted with the buffer-address rule
            (perf/b2x_difflayer/real_traffic.py). The predicted delta is one operand read.
  time   -- interleaved A/B/A/B, medians, with the shipped arm run under two labels so the
            session reports its own A/A floor.
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

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.triatt_qkv as QK                                                # noqa: E402
from real_traffic import counts                                               # noqa: E402

sys.path.insert(0, str(ROOT / "perf" / "b2z2_algebra"))
from ledger import triatt_weights, CZ, HEADS, HEAD_DIM, Z_MB                  # noqa: E402


def arm(on):
    QK._FUSED_ENABLED = bool(on)


def run(mod, z):
    return mod(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--host", default="whglx")
    ap.add_argument("--card", type=int, default=5)
    ap.add_argument("--cz", type=int, default=CZ)
    ap.add_argument("--heads", type=int, default=HEADS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    W = lambda s: triatt_weights(s, a.cz, a.heads, HEAD_DIM)
    mods = {False: T.TriangleAttention(HEAD_DIM, a.heads, False, W(1), ck),
            True: T.TriangleAttention(HEAD_DIM, a.heads, True, W(4), ck)}
    g = torch.Generator().manual_seed(7)
    z = ttnn.from_torch(torch.randn(1, a.n, a.n, a.cz, generator=g), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    res = {"n": a.n, "cz": a.cz, "heads": a.heads, "host": a.host, "card": a.card, "arch": "WH" if a.host == "whglx" else "BH",
           "grid": list(T.COMPUTE_GRID_MAIN), "reps": a.reps,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- parity -------------------------------------------------------------------
    parity = {}
    for ending, mod in mods.items():
        got = {}
        for on in (False, True):
            arm(on)
            o = run(mod, z)
            ttnn.synchronize_device(dev)
            got[on] = ttnn.to_torch(o)
            ttnn.deallocate(o)
        parity["ending" if ending else "starting"] = {
            "equal": bool(torch.equal(got[False], got[True])),
            "max_abs": float((got[False].float() - got[True].float()).abs().max()),
        }
    res["parity"] = parity
    res["fused_served"] = list(QK.FUSED_STATS)
    res["fused_rejects"] = {str(k): v for k, v in QK.FUSED_REJECTS.items()}
    print("parity:", json.dumps(parity), "served/declined:", QK.FUSED_STATS, flush=True)

    # ---- bytes --------------------------------------------------------------------
    byts = {}
    for ending, mod in mods.items():
        cell = {}
        for on in (False, True):
            arm(on)
            ttnn.deallocate(run(mod, z))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            o = run(mod, z)
            ttnn.synchronize_device(dev)
            gr = ttnn.graph.end_graph_capture()
            ttnn.deallocate(o)
            c = counts({"sig": "triatt", "nodes": gr})
            cell["on" if on else "off"] = {k: round(c[k], 3) for k in
                                           ("real_MB", "real_w_MB", "real_r_MB")} | {
                                               "n_ops": c["n_ops"]}
        d = cell["off"]["real_MB"] - cell["on"]["real_MB"]
        cell["deleted_MB"] = round(d, 3)
        cell["deleted_Z"] = round(d / Z_MB, 3)
        byts["ending" if ending else "starting"] = cell
        print("bytes", "ending" if ending else "starting", json.dumps(cell), flush=True)
    res["bytes"] = byts

    # ---- time ---------------------------------------------------------------------
    # Three labels, two arms: `off` twice so the run reports its own A/A floor.
    labels = [("off", False), ("on", True), ("off2", False)]
    times = {k: {"starting": [], "ending": []} for k, _ in labels}
    for ending, mod in mods.items():
        arm(False)
        ttnn.deallocate(run(mod, z))
        arm(True)
        ttnn.deallocate(run(mod, z))
        ttnn.synchronize_device(dev)
    for _ in range(a.reps):
        for lbl, on in labels:
            arm(on)
            for ending, mod in mods.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = run(mod, z)
                ttnn.synchronize_device(dev)
                times[lbl]["ending" if ending else "starting"].append(
                    (time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)
    med = {lbl: {k: round(statistics.median(v), 4) for k, v in d.items()}
           for lbl, d in times.items()}
    res["time_ms"] = med
    res["time_raw"] = {lbl: d for lbl, d in times.items()}
    res["ratio"] = {k: round(med["off"][k] / med["on"][k], 5) for k in ("starting", "ending")}
    res["aa_floor"] = {k: round(med["off"][k] / med["off2"][k], 5) for k in ("starting", "ending")}
    print("median ms:", json.dumps(med))
    print("ratio off/on:", res["ratio"], " A/A floor:", res["aa_floor"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
