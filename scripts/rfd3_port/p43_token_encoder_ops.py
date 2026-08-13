"""S2 -- itemise the RFD3 diffusion token encoder, the largest object in the step.

`perf/p42/r4_b2_drains2.json` puts `DiffusionTokenEncoder.run_device` at 699.4 ms/step, 39.9 % of
the step at the pinned rfd3_R4 fixture, with no mechanism. A per-callsite `to_torch` timer cannot go
inside it, because the encoder does not drain: it hands z to the DiT on the card.

So this uses the other instrument. Every ttnn op issued inside `run_device` is bracketed by a
`ttnn.synchronize_device`, which is the wrong tool for a host/device split and the right tool for
RANKING device ops (memory `ttnn-sync-before-every-timed-region`). Ops outside the region run
unwrapped, so the rest of the step is untouched.

Each row reports ms/step, the bytes its output tensor occupies ON DEVICE (padded shape x dtype, so a
65-wide row costs its 96-wide tile padding), and the implied GB/s against that. A row above the
measured 385 GB/s clone roof for this card is a byte-model error to be caught here, not published.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.model import DiffusionTokenEncoder  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

ACC: dict[str, float] = defaultdict(float)
CNT: dict[str, int] = defaultdict(int)
BYT: dict[str, float] = defaultdict(float)
SNAP: list[tuple[float, dict, dict, dict]] = []
ON = [False]
DEV = [None]

# every ttnn entry point the encoder's chain can reach, directly or through Transition /
# PairformerBlock. Anything not here is invisible, so the row total is checked against the region
# wall below and the gap is reported.
OPS = ["concat", "rms_norm", "layer_norm", "linear", "matmul", "embedding", "to_layout", "reshape",
       "add", "multiply", "subtract", "typecast", "permute", "transpose", "sigmoid", "silu",
       "softmax", "mul", "gelu", "relu", "zeros", "ones", "from_torch", "to_torch", "clone",
       "repeat", "repeat_interleave", "experimental", "pad", "slice", "unsqueeze", "squeeze",
       "transformer"]

_DT_BYTES = {}


def _bytes_of(x) -> float:
    """Device footprint of a returned tensor: padded shape x element size."""
    try:
        shp = tuple(x.padded_shape) if hasattr(x, "padded_shape") else tuple(x.shape)
    except Exception:
        return 0.0
    try:
        dt = str(x.dtype)
    except Exception:
        return 0.0
    if dt not in _DT_BYTES:
        _DT_BYTES[dt] = 4.0 if ("float32" in dt or "int32" in dt or "uint32" in dt) else \
                        (1.0 if ("8" in dt and "16" not in dt) else 2.0)
    n = 1.0
    for d in shp:
        n *= float(d)
    return n * _DT_BYTES[dt]


def _wrap(name, fn):
    def w(*a, **k):
        if not ON[0]:
            return fn(*a, **k)
        fr = traceback.extract_stack(limit=3)[-2]
        lbl = f"{name:<10s} {Path(fr.filename).name}:{fr.lineno}"
        t0 = time.perf_counter()
        r = fn(*a, **k)
        ttnn.synchronize_device(DEV[0])
        ACC[lbl] += time.perf_counter() - t0
        CNT[lbl] += 1
        BYT[lbl] += _bytes_of(r)
        return r
    return w


def instrument() -> None:
    for n in OPS:
        f = getattr(ttnn, n, None)
        if callable(f):
            setattr(ttnn, n, _wrap(n, f))


REGION = [0.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="perf/dsfix/targets/R4_9q6y_A.pdb")
    ap.add_argument("--contig", default="A1-585,100")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=8)
    ap.add_argument("--designs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--roof", type=float, default=385.0, help="measured clone roof, GB/s")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    instrument()
    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    dev_ti = build_token_initializer(
        torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
    dev_dm = build_diffusion_module(
        torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)

    enc_call = DiffusionTokenEncoder.run_device

    def enc(self, *ar, **kw):
        DEV[0] = self.device
        ttnn.synchronize_device(DEV[0])
        ON[0] = True
        t0 = time.perf_counter()
        try:
            return enc_call(self, *ar, **kw)
        finally:
            ttnn.synchronize_device(DEV[0])
            REGION[0] += time.perf_counter() - t0
            ON[0] = False

    DiffusionTokenEncoder.run_device = enc

    cls = type(dev_dm)
    dm_call = cls.__call__

    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            SNAP.append((time.perf_counter() - t0, dict(ACC), dict(CNT),
                         {"__region__": REGION[0], **dict(BYT)}))

    cls.__call__ = stepped

    ACC.clear(); CNT.clear(); BYT.clear()
    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    with torch.no_grad():
        sampler.sample(dev_dm, a.designs, L, coord0, f, init,
                       f["is_motif_atom_with_fixed_coord"],
                       generator=[torch.Generator().manual_seed(a.seed + i)
                                  for i in range(a.designs)])

    def med(d_idx, key, scale=1.0):
        return statistics.median([(SNAP[i][d_idx].get(key, 0.0) - SNAP[i - 1][d_idx].get(key, 0.0))
                                  * scale for i in range(2, len(SNAP))])

    keys = set().union(*[s[1].keys() for s in SNAP])
    ms = {k: med(1, k, 1e3) for k in keys}
    cnt = {k: med(2, k) for k in keys}
    gb = {k: med(3, k, 1e-9) for k in keys}
    step = statistics.median([s[0] * 1e3 for s in SNAP[2:]])
    region = med(3, "__region__", 1e3)
    total = sum(ms.values())

    I = int(f["atom_to_token_map"].max()) + 1
    print(f"\n[S2] L={L} atoms  I={I} tokens  designs={a.designs}  steps={a.num_timesteps}",
          flush=True)
    print(f"[S2] median warm step {step:.1f} ms; token-encoder region {region:.1f} ms "
          f"({100 * region / step:.1f} % of step); itemised {total:.1f} ms "
          f"({100 * total / max(region, 1e-9):.1f} % of region)", flush=True)
    print(f"\n{'op / callsite':40s} {'ms/step':>9s} {'%reg':>6s} {'calls':>6s} {'out GB':>8s} "
          f"{'GB/s':>8s} {'x roof':>7s}")
    for k, v in sorted(ms.items(), key=lambda kv: -kv[1]):
        if v < 0.20:
            continue
        g = gb.get(k, 0.0)
        rate = g / (v * 1e-3) if v > 0 else 0.0
        print(f"{k:40s} {v:9.2f} {100 * v / max(region, 1e-9):5.1f}% {cnt.get(k, 0):6.1f} "
              f"{g:8.4f} {rate:8.1f} {rate / a.roof:6.2f}x")
    print(f"\n{'(rows < 0.20 ms/step hidden)':40s} "
          f"{sum(v for v in ms.values() if v < 0.20):9.2f}")
    print(f"{'UNSEEN (region - itemised)':40s} {region - total:9.2f}", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "atoms": L, "tokens": I, "designs": a.designs, "num_timesteps": a.num_timesteps,
            "median_warm_step_ms": step, "region_ms_per_step": region,
            "itemised_ms_per_step": total, "roof_gbs": a.roof,
            "ms_per_step": ms, "calls_per_step": cnt, "out_gb_per_step": gb}, indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
