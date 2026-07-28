"""Time a real full-length RFD3 design, one-time costs included.

`bench_batch_designs_per_sec.py` warms up with a sampler that shares the timed run's
`initial` dict. The resident pair-gather table p26 added is keyed on the P_LL storage
address and lives on the diffusion module, not on the sampler, so that warmup builds
the table and the timed region measures pure steady state -- a 20-step projection then
credits the change with a per-step saving it never paid the setup for. Same shape as
`rfd3-p17-bench-bypasses-production-clamp`: a harness isolating one code path from the
plumbing around it describes a run nobody makes.

This evicts the table between the warmup and the timed run, so the timed region pays
the build exactly once, the way a real `tt-bio design --num_timesteps 200` does. The
kernel cache and the -1e4 mask template stay warm: both are untouched by p26 and
identical in either tree, and warming them keeps the comparison out of the noise.

`--steady-pass` repeats the run with the table already resident, which prices the
one-time build as the difference. That is what lets a shorter run be composed into a
200-step number honestly (`build + 199 * steady_per_step`) instead of quoted bare.

The pre-change tree has no table; the eviction is a no-op there and the same script
runs unmodified in both, which is what makes the A/B a code diff and not a harness diff.

Usage (see run_p27_sweep.sh for the interleaved A/B driver):
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$TREE \
    python3 $TREE/scripts/rfd3_port/p27_real_design_timing.py \
      --timesteps 200 --batches 1 8 --contig "A1-10,230,A31-40"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--warmup-timesteps", type=int, default=4)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--contig", default="A1-10,20,A31-40")
    ap.add_argument("--spec", type=Path, help="JSON InputSpecification; overrides --contig")
    ap.add_argument("--steady-pass", action="store_true",
                    help="also time the run with the table resident, to price the build")
    ap.add_argument("--tag", default="", help="label echoed on every result line")
    args = ap.parse_args()
    if any(batch < 1 for batch in args.batches):
        ap.error("--batches values must be at least 1")
    return args


def caches_of(diffusion_module) -> list[dict]:
    """The distinct step/table caches the diffusion module holds.

    The encoder and the decoder are handed the same dict (rfd3.py, RFD3DiffusionModule),
    so dedupe by identity rather than trusting that to stay true.
    """
    found = {}
    for part in (getattr(diffusion_module, "encoder", None),
                 getattr(diffusion_module, "decoder", None)):
        cache = getattr(part, "_mask_cache", None)
        if cache is not None:
            found[id(cache)] = cache
    return list(found.values())


def evict_tables(diffusion_module) -> int:
    """Drop the resident gather table and the per-step cache; return tables freed.

    Returns 0 on the pre-change tree, which has neither.
    """
    import ttnn

    freed = 0
    for cache in caches_of(diffusion_module):
        cache.pop("step", None)
        for _, table in cache.pop("tables", {}).values():
            ttnn.deallocate(table)
            freed += 1
    return freed


def live_tables(diffusion_module) -> int:
    return sum(len(cache.get("tables", {})) for cache in caches_of(diffusion_module))


def main() -> None:
    args = parse_args()
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        input_path = Path(spec_data["input"])
        if not input_path.is_absolute():
            input_path = args.spec.parent / input_path
        spec_data["input"] = str(input_path.resolve())
        fixture = f"spec={args.spec.name}"
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
        fixture = f"contig={args.contig!r}"
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {
        key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in features.items()
    }
    token_weights = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                              map_location="cpu", weights_only=True)
    diffusion_weights = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                                   map_location="cpu", weights_only=True)
    token_initializer = build_token_initializer(token_weights)
    diffusion_module = build_diffusion_module(diffusion_weights)
    with torch.no_grad():
        initial = token_initializer({
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in features.items()
        })

    length = features["ref_pos"].shape[0]
    fixed = features["is_motif_atom_with_fixed_coord"]
    coord = features["motif_pos"].float().unsqueeze(0)
    print(f"fixture: {fixture} L={length} timesteps={args.timesteps} "
          f"trace_decoder={os.environ.get('RFD3_TRACE_DECODER') == '1'}", flush=True)

    for batch in args.batches:
        sample_kwargs = dict(features=features, initial=initial, fixed=fixed)
        with torch.no_grad():
            RFD3Sampler(num_timesteps=args.warmup_timesteps).sample(
                diffusion_module, batch, length, coord,
                sample_kwargs["features"], sample_kwargs["initial"], sample_kwargs["fixed"],
                generator=torch.Generator().manual_seed(1000 + batch))
            freed = evict_tables(diffusion_module)
            measured = RFD3Sampler(num_timesteps=args.timesteps)
            start = time.perf_counter()
            output, _ = measured.sample(
                diffusion_module, batch, length, coord,
                sample_kwargs["features"], sample_kwargs["initial"], sample_kwargs["fixed"],
                generator=torch.Generator().manual_seed(2000 + batch))
            cold = time.perf_counter() - start
            built = live_tables(diffusion_module)
            steady = float("nan")
            if args.steady_pass:
                start = time.perf_counter()
                output2, _ = measured.sample(
                    diffusion_module, batch, length, coord,
                    sample_kwargs["features"], sample_kwargs["initial"], sample_kwargs["fixed"],
                    generator=torch.Generator().manual_seed(2000 + batch))
                steady = time.perf_counter() - start
                del output2
        steps = measured.num_timesteps - 1
        row = {
            "tag": args.tag, "L": length, "D": batch, "steps": steps,
            "evicted": freed, "tables_built": built,
            "run_s": round(cold, 4),
            "ms_per_step": round(cold / steps * 1000, 3),
            "designs_per_sec": round(batch / cold, 6),
            "finite": bool(torch.isfinite(output).all().item()),
        }
        if args.steady_pass:
            row["steady_s"] = round(steady, 4)
            row["steady_ms_per_step"] = round(steady / steps * 1000, 3)
            row["build_s"] = round(cold - steady, 4)
        print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
