"""p40 — the end-to-end bound on device residency across the encoder/DiT/decoder boundary.

p39's census says three `ttnn.to_torch` callsites carry 96 of the step's 101 ms of blocking drain:

    model.py:2793  R_upd from the decoder   34.67 ms/step  ->  scale_positions_out, HOST fp32
    model.py:2781  S_I  from the token enc  33.26 ms/step  ->  re-uploaded unchanged
    model.py:1574  Q_L  from the atom enc   28.05 ms/step  ->  re-uploaded unchanged

The last two cross to host and come straight back with no host arithmetic on them: the modules are
written host-in/host-out for the parity scripts, and only `z` (p19) and the decoder's two outputs
(p20) ever got the residency treatment. Removing them cannot change any arithmetic (bf16 -> fp32 ->
bf16 is lossless), but it can change the WALL, because a `to_torch` is a barrier: the host stops
dispatching until the device drains, so nothing downstream can overlap the device work upstream.

This measures the bound the same way §2 bounded `attn_indices`: with a deliberately WRONG design
that removes the barrier without changing the op stream. `to_torch` at those two lines returns the
value it returned on the first step instead of reading the device. The design is garbage after
step 0 and is never shippable. What it measures is exactly how much wall those two barriers cost,
which is the cap on the residency refactor (L10) before anyone builds it.

Arms:
  --plain            the honest wall
  --stale 1574       drop the encoder barrier only
  --stale 2781       drop the token-encoder barrier only
  --stale 1574,2781  drop both -- the bound on L10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

SNAP: list[float] = []


def stale_to_torch(lines: set[int]):
    """Return the first step's value at the named callsites, and never drain there again."""
    orig = ttnn.to_torch
    cache: dict = {}
    hits = {ln: 0 for ln in lines}

    def w(*a, **k):
        fr = sys._getframe(1)
        ln = fr.f_lineno
        if ln in lines:
            key = (ln, fr.f_code.co_name)
            if key in cache:
                hits[ln] += 1
                return cache[key]
            out = orig(*a, **k)
            cache[key] = out
            return out
        return orig(*a, **k)

    ttnn.to_torch = w
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb")
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stale", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    lines = {int(x) for x in a.stale.split(",") if x.strip()}
    hits = stale_to_torch(lines) if lines else {}

    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    ti_w = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_w)
    dev_dm = build_diffusion_module(dm_w)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)

    cls = type(dev_dm)
    dm_call = cls.__call__
    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            SNAP.append((time.perf_counter() - t0) * 1e3)
    cls.__call__ = stepped

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    t0 = time.perf_counter()
    with torch.no_grad():
        sampler.sample(dev_dm, 1, L, coord0, f, init,
                       f["is_motif_atom_with_fixed_coord"],
                       generator=torch.Generator().manual_seed(a.seed))
    wall = time.perf_counter() - t0
    med = statistics.median(SNAP[2:])
    print(f"\n[p40] stale={sorted(lines) or 'none'}  L={L}  wall={wall:.2f}s  "
          f"first={SNAP[0]:.0f} second={SNAP[1]:.0f}  MEDIAN WARM STEP={med:.1f} ms  "
          f"min={min(SNAP[2:]):.1f} max={max(SNAP[2:]):.1f}  stale_hits={hits}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"stale": sorted(lines), "atoms": L, "wall_s": wall,
                                     "median_warm_step_ms": med, "step_walls_ms": SNAP,
                                     "stale_hits": hits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
