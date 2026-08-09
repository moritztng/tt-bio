#!/usr/bin/env python3
"""RFD3 design A/B: the fused atom-block attention against the shipped unfused chain.

One leg per process (the fused flag is read when the block is built, and the bias
projection weight is pre-scaled at that point), same seed, same contig, hot card. Prints
median ms/step and writes the sampled coordinates so the two legs can be compared for
parity. Structure follows scripts/rfd3_port/p32_trace_ab.py, including its warmup rule --
a cold read is 13x.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

ap = argparse.ArgumentParser()
ap.add_argument("--leg", choices=("fused", "unfused"), required=True)
ap.add_argument("--contig", default="A1-10,278,A31-40")
ap.add_argument("--pdb", type=Path, default=PDB)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--timesteps", type=int, default=6)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

os.environ["RFD3_FUSED_ATTENTION"] = "1" if args.leg == "fused" else "0"

import tt_bio.rfd3.model as R  # noqa: E402
import tt_bio.tenstorrent as TTd  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

TTd.get_device(trace_region_size=1 << 30)
data = {"input": str(args.pdb), "contig": args.contig}
spec = InputSpecification.from_dict(data)
spec.validate()
f = featurize(data["input"], spec)
f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
L = int(f["ref_pos"].shape[0])
n_atom = int(f["ref_pos"].shape[0])

ti = R.build_token_initializer(torch.load(
    GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
dm = R.build_diffusion_module(torch.load(
    GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
fused = [b._fused_attention for b in dm.decoder.atom_blocks]
print(f"leg={args.leg} L={L} fused_flags={fused}", flush=True)

coord0 = f["motif_pos"].float().unsqueeze(0)
with torch.no_grad():
    init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

    def run(timesteps, seed):
        g = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        out = RFD3Sampler(num_timesteps=timesteps).sample(
            dm, args.batch, L, coord0, f, init,
            f["is_motif_atom_with_fixed_coord"], generator=g)
        wall = time.perf_counter() - t0
        x = out[0] if isinstance(out, tuple) else out
        return wall, x.detach().clone().float()

    run(args.warmup, 7)
    walls, x = [], None
    for i in range(args.reps):
        w, x = run(args.timesteps, 42)
        walls.append(w)
        print(f"  rep {i+1} {w*1e3:9.1f} ms  ({w/(args.timesteps-1)*1e3:7.2f} ms/step)", flush=True)

med = statistics.median(walls)
rec = {"leg": args.leg, "L": L, "contig": args.contig, "batch": args.batch,
       "timesteps": args.timesteps, "steps": args.timesteps - 1,
       "walls_s": walls, "median_s": med,
       "ms_per_step": med / (args.timesteps - 1) * 1e3,
       "fused_flags": fused}
args.out.write_text(json.dumps(rec, indent=1))
torch.save(x, args.out.with_suffix(".coords.pt"))
print(f"MEDIAN {med*1e3:.1f} ms  {rec['ms_per_step']:.2f} ms/step  -> {args.out}", flush=True)
