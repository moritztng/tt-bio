#!/usr/bin/env python3
"""Served fused triangle-attention SDPA at HiFi4 (Mol-1 BH serves it so: triatt_hifi2 is not in MOL1_BH) vs HiFi3
and HiFi2 on the same captured operands (stream_probe.capture). math_approx off and fp32 accumulation kept, as the
triatt_hifi2 lever does. Prints rel RMS of each arm vs the HiFi4 output, then interleaved timings with the AICLK
sampled during each arm.

Usage: fid_probe.py --cells 1024x10,1024x14,512x10 --reps 5 --rounds 3
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn
import tt_bio.tenstorrent as T
import clocksample
from stream_probe import capture, timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="1024x10,1024x14,512x10")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    dev = T.get_device()
    for cell in a.cells.split(","):
        S, H = map(int, cell.split("x"))
        sargs, skw = capture(dev, S, H * 32, H)
        print(json.dumps({"S": S, "H": H, "served_ckc": str(sargs[5]), "served_kw": str(skw)}), flush=True)
        arms = {}
        for name, fid in (("hifi4", ttnn.MathFidelity.HiFi4), ("hifi3", ttnn.MathFidelity.HiFi3),
                          ("hifi2", ttnn.MathFidelity.HiFi2)):
            args = list(sargs)
            args[5] = (fid,) + tuple(sargs[5][1:])
            arms[name] = (lambda args=args: T._tri_att_sdpa_at(*args, **skw))
        ref = ttnn.to_torch(arms["hifi4"]()).float()
        for name in ("hifi3", "hifi2"):
            got = ttnn.to_torch(arms[name]()).float()
            print(json.dumps({"S": S, "H": H, "arm": name, "rel_rms_vs_hifi4": round(
                ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(), 6),
                "finite": bool(got.isfinite().all())}), flush=True)
        for name, fn in [kv for _ in range(a.rounds) for kv in arms.items()]:
            try:
                with clocksample.during(period=0.5) as clk:
                    ms = timed(fn, a.reps, dev)
                print(json.dumps({"S": S, "H": H, "arm": name, "ms": round(ms, 3), "aiclk": clk.summary()}), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"S": S, "H": H, "arm": name, "error": str(e)[:300]}), flush=True)
        for t in sargs:
            if isinstance(t, ttnn.Tensor):
                t.deallocate()


if __name__ == "__main__":
    main()
