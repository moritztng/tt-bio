"""Measure RFD3 in-forward batching throughput.

Uses the parity fixture and production decoder-trace setting. Each shape gets a
short warmup before a timed sampler run; the 200-timestep rate is derived from
the measured per-step time when another step count is requested.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--pdb", type=Path, default=PDB)
    parser.add_argument("--contig", default="A1-10,20,A31-40")
    parser.add_argument(
        "--spec", type=Path,
        help="JSON InputSpecification; overrides --pdb/--contig",
    )
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches):
        parser.error("--batches values must be at least 1")
    return args


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
        fixture = f"spec={args.spec}"
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
        fixture = f"pdb={args.pdb} contig={args.contig!r}"
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {
        key: value.float()
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in features.items()
    }
    token_weights = torch.load(
        GOLDEN_DIR / "token_initializer.real_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    diffusion_weights = torch.load(
        GOLDEN_DIR / "diffusion_module.real_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    token_initializer = build_token_initializer(token_weights)
    diffusion_module = build_diffusion_module(diffusion_weights)
    with torch.no_grad():
        initial = token_initializer(
            {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in features.items()
            }
        )

    length = features["ref_pos"].shape[0]
    fixed = features["is_motif_atom_with_fixed_coord"]
    coord = features["motif_pos"].float().unsqueeze(0)
    print(
        f"fixture: {fixture} I={features['restype'].shape[0]} L={length} "
        f"trace_decoder={os.environ.get('RFD3_TRACE_DECODER') == '1'}"
    )
    print(
        f"D  sample_{args.timesteps}_s  ms_per_step  "
        f"measured_designs_per_sec  projected_designs_per_sec_200"
    )
    for batch in args.batches:
        warmup = RFD3Sampler(num_timesteps=4)
        measured = RFD3Sampler(num_timesteps=args.timesteps)
        with torch.no_grad():
            warmup.sample(
                diffusion_module,
                batch,
                length,
                coord,
                features,
                initial,
                fixed,
                generator=torch.Generator().manual_seed(1000 + batch),
            )
            start = time.perf_counter()
            output, _ = measured.sample(
                diffusion_module,
                batch,
                length,
                coord,
                features,
                initial,
                fixed,
                generator=torch.Generator().manual_seed(2000 + batch),
            )
            elapsed = time.perf_counter() - start
        steps = measured.num_timesteps - 1
        per_step = elapsed / steps
        measured_rate = batch / elapsed
        rate_200 = batch / (per_step * 199)
        print(
            f"{batch:<2d} {elapsed:11.4f} {per_step * 1000:12.3f} "
            f"{measured_rate:18.4f} {rate_200:20.4f} "
            f"finite={torch.isfinite(output).all().item()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
