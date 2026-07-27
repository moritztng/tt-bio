"""Run the real RFD3 sampler loop under the Tracy device profiler.

Companion to ``summarize_fpu_utilization.py``.  Intended to be launched as::

    python -m tracy -r --profiler-capture-perf-counters fpu -o <out> \
        scripts/rfd3_port/profile_fpu_utilization.py --batch 1 --timesteps 3

Every ttnn op the sampler issues is captured with its hardware FPU/MATH
counters, so the resulting report answers "is this step compute-saturated?"
with silicon counters rather than a wall-clock inference.

Keep ``--timesteps`` small: the op count per diffusion step is in the
thousands and the profiler writes one CSV row per op instance.  Op shapes are
identical across steps, so two steady-state steps are enough.
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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=3)
    parser.add_argument("--warmup-timesteps", type=int, default=3)
    parser.add_argument("--pdb", type=Path, default=PDB)
    parser.add_argument("--contig", default="A1-10,230,A31-40")
    parser.add_argument("--spec", type=Path)
    return parser.parse_args()


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
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
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
        f"PROFILE batch={args.batch} I={features['restype'].shape[0]} L={length} "
        f"steps={args.timesteps - 1} warmup_steps={args.warmup_timesteps - 1} "
        f"trace_decoder={os.environ.get('RFD3_TRACE_DECODER') == '1'}",
        flush=True,
    )

    with torch.no_grad():
        if args.warmup_timesteps > 1:
            RFD3Sampler(num_timesteps=args.warmup_timesteps).sample(
                diffusion_module, args.batch, length, coord, features, initial,
                fixed, generator=torch.Generator().manual_seed(1000 + args.batch),
            )
        sampler = RFD3Sampler(num_timesteps=args.timesteps)
        start = time.perf_counter()
        output, _ = sampler.sample(
            diffusion_module, args.batch, length, coord, features, initial,
            fixed, generator=torch.Generator().manual_seed(2000 + args.batch),
        )
        elapsed = time.perf_counter() - start

    steps = sampler.num_timesteps - 1
    print(
        f"PROFILED batch={args.batch} steps={steps} "
        f"ms_per_step={elapsed / steps * 1000:.3f} "
        f"finite={torch.isfinite(output).all().item()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
