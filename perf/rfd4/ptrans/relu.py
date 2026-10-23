"""RFD4's pair transition program (`pair_relu` with the float32 residual, WS_RES32) alone: ms per call, a digest of
both outputs for an A/B of two trees run as separate processes, and time attribution.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/ptrans/relu.py --out <json> [--n 512 256 128] [--attr]

Inputs per n: seeded random z float32 [1, n, n, 384], its bf16 copy, LayerNorm and transition weights at c_z 384 /
hidden 768 (bfp8, as the trunk builds them), HiFi4. One call on fresh inputs gives the digest of (z, zb); time is the
median over `--reps` of `--inner` back-to-back in-place calls between syncs. `--attr` adds arms with the kernels'
WS_NO_XREAD / WS_NO_WRITE / WS_NO_FC3 switches (wrong outputs, time only): what each part costs when it is skipped
shows which floor binds. AICLK and host load sampled during the run.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, nargs="+", default=[512, 256, 128])
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--inner", type=int, default=10)
ap.add_argument("--attr", action="store_true")
a = ap.parse_args()
sys.path.append(str(Path(__file__).resolve().parents[3]))   # after PYTHONPATH, so an A/B tree wins
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # perf/: clocksample

import torch  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import pair_swiglu_ws as W  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

C, H, EPS = 384, 768, 1e-5
BASE = (("WS_RESID", "1"), ("WS_RES32", "1"))
ATTR = {"no_xread": (("WS_NO_XREAD", "1"),), "no_write": (("WS_NO_WRITE", "1"),), "no_fc3": (("WS_NO_FC3", "1"),)}


def main():
    dev = get_device()
    up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gen = torch.Generator().manual_seed(21)
    w1, w2 = torch.randn(C, H, generator=gen) * C ** -0.5, torch.randn(H, C, generator=gen) * 0.02 * H ** -0.5
    b1, b2 = torch.randn(H, generator=gen) * 0.1, torch.randn(C, generator=gen) * 0.01
    ws = W.relu_weights(w1, b1, w2, b2, dev, dtype=ttnn.bfloat8_b)
    ln = (up(1 + 0.1 * torch.randn(1, C, generator=gen), ttnn.bfloat16),
          up(0.1 * torch.randn(1, C, generator=gen), ttnn.bfloat16), EPS)
    rows = []
    with clocksample.during() as clk:
        for n in a.n:
            z = torch.randn(1, n, n, C, generator=torch.Generator().manual_seed(n)) * 2.0
            z_t, x_t = up(z, ttnn.float32), up(z, ttnn.bfloat16)
            arms = {"full": BASE}
            if a.attr:
                arms.update({k: BASE + v for k, v in ATTR.items()})
            for arm, defines in arms.items():
                call = lambda: W._run(x_t, ws[0], ws[2], ttnn.MathFidelity.HiFi4, defines, W.SILU_POLY,  # noqa: E731
                                      ln, z_t, (ws[1], ws[3]))
                r = dict(n=n, arm=arm)
                if arm == "full":
                    call()
                    ttnn.synchronize_device(dev)
                    z32, zb = ttnn.to_torch(z_t), ttnn.to_torch(x_t)
                    r["digest"] = hashlib.sha256(z32.contiguous().numpy().tobytes()
                                                 + zb.view(torch.int16).numpy().tobytes()).hexdigest()[:16]
                    r["update_rms"] = float((z32.double() - z.double()).pow(2).mean().sqrt())
                ts = []
                for _ in range(a.reps):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(a.inner):
                        call()
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / a.inner)
                ts.sort()
                r.update(ms=round(1e3 * ts[len(ts) // 2], 4), ms_min=round(1e3 * ts[0], 4),
                         gb=round(n * n * C * 12 / 1e9, 4), t=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(json.dumps(r), flush=True)
            for t in (z_t, x_t):
                ttnn.deallocate(t)
    out = dict(rows=rows, tt_bio=str(Path(W.__file__).parents[1]), aiclk=clk.summary(), load=clk.load and max(clk.load),
               script="perf/rfd4/ptrans/relu.py")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("aiclk", clk.summary(), "load", out["load"])


if __name__ == "__main__":
    main()
