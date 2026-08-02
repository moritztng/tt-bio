"""p32: interleaved A/B over the two RFD3 trace levers, with p30's bias reuse wired in.

Four legs, all in one process on one hot card so drift and thermals hit them equally (p14):

  eager  decoder eager,  encoder eager   -- shipped
  dec    decoder traced, encoder eager
  enc    decoder eager,  encoder traced
  both   decoder traced, encoder traced

Both trace flags are read from the environment in `RFD3DiffusionModule.__init__`, but they
are plain attributes afterwards (`decoder.trace`, `dm._trace_encoder`), so one process can
alternate them and the legs differ only in which code path runs -- no cross-tree harness
difference (p31 had to use two trees because the head-merge change rewrote the call graph;
this one does not).

The decoder's traced path now captures the core loop TWICE against one bias-cache dict, so
the pair-bias scatter is baked into the recycle-1 trace only (p32; see
`CompactStreamingDecoder._capture_sparse_trace`). Without that, tracing would TRADE p30's
+7% for trace's dispatch win instead of compounding with it, and this A/B would be measuring
"trace alone" against "cache alone".

Every leg re-seeds and compares its sampled coordinates against the first eager leg's, so
the two-gate check (`ttnn-trace-interleaved-eager-corruption`) runs in the same process that
measures the speedup: a trace that passes an isolated component PCC can still corrupt or
hang once it is interleaved with eager allocation in the real loop.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... TT_BIO_TRACE_REGION_SIZE=$((1<<30)) \
       PYTHONPATH=$PWD python3 scripts/rfd3_port/p32_trace_ab.py \
         --contig "A1-10,230,A31-40" --batch 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

LEGS = {"eager": (False, False), "dec": (True, False),
        "enc": (False, True), "both": (True, True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--alternations", type=int, default=3)
    ap.add_argument("--legs", nargs="+", default=list(LEGS))
    ap.add_argument("--json_out", type=Path)
    args = ap.parse_args()

    import tt_bio.rfd3.model as R
    import tt_bio.tenstorrent as TTd
    from tt_bio.rfd3.featurize import featurize
    from tt_bio.rfd3.input import InputSpecification
    from tt_bio.rfd3.sampler import RFD3Sampler

    TTd.get_device(trace_region_size=1 << 30)

    data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(data)
    spec.validate()
    f = featurize(data["input"], spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = int(f["ref_pos"].shape[0])

    ti = R.build_token_initializer(torch.load(
        GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
    dm = R.build_diffusion_module(torch.load(
        GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
    steps = args.timesteps - 1          # the sampler walks consecutive schedule pairs

    coord0 = f["motif_pos"].float().unsqueeze(0)
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

        def run(name, timesteps, seed):
            dm.decoder.trace, dm._trace_encoder = LEGS[name]
            g = torch.Generator().manual_seed(seed)
            t0 = time.perf_counter()
            out = RFD3Sampler(num_timesteps=timesteps).sample(
                dm, args.batch, L, coord0, f, init,
                f["is_motif_atom_with_fixed_coord"], generator=g)
            ms = (time.perf_counter() - t0) / (timesteps - 1) * 1e3
            x = out[0] if isinstance(out, tuple) else out    # sample() -> (X_L, traj)
            return ms, x.detach().clone().float()

        # p17: a cold read is 13x, and a trace capture is a one-time cost on top. Warm
        # every leg (kernel cache, resident pair table, captured traces) before timing.
        for name in args.legs:
            run(name, args.warmup, 7)

        ms = {name: [] for name in args.legs}
        ref, maxabs = None, {name: 0.0 for name in args.legs}
        for i in range(args.alternations):
            for name in args.legs:
                leg_ms, x = run(name, args.timesteps, 42)
                if ref is None:
                    ref = x
                ms[name].append(leg_ms)
                maxabs[name] = max(maxabs[name], (x - ref).abs().max().item())
                print(f"  alternation {i + 1} {name:<5} {leg_ms:7.2f} ms/step  "
                      f"maxabs {maxabs[name]:.3e}", flush=True)

    med = {name: statistics.median(v) for name, v in ms.items()}
    base = med[args.legs[0]]
    print(f"\n=== p32 trace A/B  L={L} atoms  batch={args.batch}  "
          f"steps/leg={steps}  alternations={args.alternations} ===")
    for name in args.legs:
        print(f"{name:<6} median {med[name]:8.2f} ms/step  "
              f"{base / med[name]:6.4f}x  ({(base / med[name] - 1) * 100:+6.1f}%)  "
              f"legs {[round(v, 1) for v in ms[name]]}  "
              f"maxabs {maxabs[name]:.3e} "
              f"{'BIT-EXACT' if maxabs[name] == 0.0 else 'NOT BIT-EXACT -- do not ship'}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"L": L, "batch": args.batch, "steps_per_leg": steps, "ms": ms,
             "median": med, "maxabs": maxabs}, indent=1))


if __name__ == "__main__":
    main()
