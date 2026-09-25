"""The traced trunk step in bf16 against bfp8 activations, on the loop's own callback path.

`round_ab.py` cannot time the bfp8 arm: its design loss goes non-finite on the second round and
BindCraft 2 ends the stage, so no timed round is ever collected. This drives the same
`EvoformerOnDevice._taped` / `._backward` the loop calls, with the same shapes the capture key
records, so the step it reports is the same quantity as `bcx-tracewire`'s 6.93 s traced trunk
step -- without needing the trajectory to survive.

One arm per process: two captures would not fit the trace region and `TraceWire` would evict and
re-capture on every switch.
"""
import json, os, pathlib, sys, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_predictor")):
    if p not in sys.path:
        sys.path.insert(0, p)

import afgrad as A
import stack as S
import trace_wire

N = int(os.environ.get("BFP8_N", "186"))
REPS = int(os.environ.get("BFP8_REPS", "8"))
B8 = bool(int(os.environ.get("BFP8_B8", "0")))
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "3"))

trace_wire.open_traced_device(768)
from splice import EvoformerOnDevice                                    # noqa: E402
from tt_bio import tenstorrent as tn                                    # noqa: E402

lv = S.Levers()
dm, _ = A.load_models(A.DEFAULT_PARAMS, refs=("bf16",))
dev = A.Dev(dm.to_device())
lv.arm("stack")
if B8:
    tn.set_fast_mode(True)
clock = S.Clock(dt=0.25)
evo = EvoformerOnDevice(dev, k_evo=48, trace=True)

rng = np.random.default_rng(0)
msa = (rng.standard_normal((2, N, 256)) * 0.5).astype(np.float32)
pair = (rng.standard_normal((N, N, 128)) * 0.5).astype(np.float32)
mask = np.ones((2, N), dtype=np.float32)
pmask = np.ones((N, N), dtype=np.float32)

rows, spans = [], []
for r in range(REPS + 1):
    t0 = time.perf_counter()
    out = evo._taped(msa, pair, mask, pmask)
    t1 = time.perf_counter()
    g = evo._backward(out[2], rng.standard_normal(out[0].shape).astype(np.float32) * 1e-3,
                      rng.standard_normal(out[1].shape).astype(np.float32) * 1e-3)
    t2 = time.perf_counter()
    rows.append({"rep": r, "fwd": t1 - t0, "bwd": t2 - t1, "step": t2 - t0,
                 "load1": round(os.getloadavg()[0], 1),
                 "finite": [bool(np.isfinite(x).all()) for x in (out[0], out[1], g[0], g[1])],
                 "absmax": [float(np.abs(x[np.isfinite(x)]).max()) if np.isfinite(x).any() else None
                            for x in (out[0], out[1], g[0], g[1])]})
    if r:                                        # rep 0 carries the capture
        spans.append((t0, t2))
    print(json.dumps(rows[-1]), flush=True)

body = [r for r in rows if r["rep"] >= 1]
blob = {"stamp": A.stamp(CARD) | {"pci": S.sysfs_node()[1], "argv": sys.argv, "n": N,
                                  "b8": B8, "reps": REPS, "aiclk_node": clock.path},
        "rows": rows,
        "fwd": S.dist([r["fwd"] for r in body]), "bwd": S.dist([r["bwd"] for r in body]),
        "step": S.dist([r["step"] for r in body]),
        "aiclk": clock.window(spans), "wire": evo.wire.stats()}
clock.stop()
out_path = ROOT / "perf" / "bcx_bfp8" / f"trunk_{'b8' if B8 else 'bf16'}_n{N}_{os.environ.get('BFP8_LEG','a')}.json"
out_path.write_text(json.dumps(blob, indent=1, default=str))
print(json.dumps({k: blob[k] for k in ("fwd", "bwd", "step", "aiclk")}, default=str), flush=True)
print("wrote", out_path, flush=True)
