"""p25: peak device DRAM per design batch size, to decide the batch clamp.

`_BATCH_ATOM_PAIR_BUDGET` (rfd3_design.py) is the only thing stopping `tt-bio design`
from honouring `--batch_size` on a large design. Its value (8*419*419) is the largest
configuration the batching commit happened to measure, not a memory measurement, so it
pins D=1 for anything past ~1185 atoms. This script measures the thing the constant is
supposed to encode: the peak DRAM a real sampler step actually occupies at a given
(atom count, batch size).

Instrumentation: the ttnn allocator is host-side bookkeeping updated at op-dispatch time,
so reading `ttnn.get_memory_view` from the calling thread is synchronous and does not
sync the device (same instrument as `tt_bio.tenstorrent.dram_peak`). Sampling only at step
boundaries would miss the intra-step transients that are what actually OOMs, so every
allocating ttnn op used by the RFD3 path is wrapped and the peak is sampled after each
call. That costs wall clock, so this script measures memory only -- throughput is a
separate, uninstrumented A/B.

Usage:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... \
    python3 scripts/rfd3_port/p25_dram_headroom.py --contig "A1-10,230,A31-40" \
      --batches 1 2 4 8 --timesteps 3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path(
    os.environ.get("RFD3_GOLDEN_DIR", "~/.coworker/artifacts/rfd3-goldens/capture")
).expanduser()

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", type=Path, default=PDB)
ap.add_argument("--contig", default="A1-10,20,A31-40")
ap.add_argument("--spec", type=Path, help="JSON InputSpecification; overrides --pdb/--contig")
ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--timesteps", type=int, default=3)
ap.add_argument("--json", type=Path, help="append one JSON record per (fixture, D)")
ap.add_argument("--no-instrument", action="store_true",
                help="sample only at step boundaries (cheap; misses intra-step peaks)")
args = ap.parse_args()

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402
from tt_bio.rfd3_sampler import RFD3Sampler  # noqa: E402

# Every ttnn entry point the RFD3 diffusion path allocates through (p24's op survey
# enumerated the producer chains; this is that list plus the allocating helpers).
WRAPPED = [
    "matmul", "linear", "softmax", "concat", "reshape", "to_layout", "from_torch",
    "to_torch", "permute", "add", "subtract", "multiply", "div", "pad", "slice",
    "embedding", "scatter", "full", "typecast", "rms_norm", "layer_norm", "sigmoid",
    "silu", "gelu", "transpose", "sum", "mean", "max", "min", "exp", "sqrt", "rsqrt",
    "clone", "repeat", "repeat_interleave", "squeeze", "unsqueeze", "zeros", "ones",
    "empty", "arange", "where", "eq", "ne", "gt", "lt", "logical_and", "logical_or",
    "bitwise_and", "neg", "reciprocal", "abs",
]

_peak = 0
_banks = None
_total = 0


def sample_dram() -> int:
    """Current DRAM bytes in use across all banks; updates the running peak."""
    global _peak, _banks, _total
    view = ttnn.get_memory_view(get_device(), ttnn.BufferType.DRAM)
    _banks = view.num_banks
    _total = view.total_bytes_per_bank * view.num_banks
    used = (view.total_bytes_per_bank - view.total_bytes_free_per_bank) * view.num_banks
    if used > _peak:
        _peak = used
    return used


def instrument() -> int:
    """Wrap the allocating ttnn ops so the peak includes intra-step transients."""
    count = 0
    for name in WRAPPED:
        original = getattr(ttnn, name, None)
        if original is None or not callable(original):
            continue

        def wrapper(*a, _f=original, **kw):
            out = _f(*a, **kw)
            sample_dram()
            return out

        try:
            setattr(ttnn, name, wrapper)
        except Exception:
            continue
        count += 1
    return count


def main() -> None:
    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        input_path = Path(spec_data["input"])
        if not input_path.is_absolute():
            input_path = args.spec.parent / input_path
        spec_data["input"] = str(input_path.resolve())
        fixture = args.spec.stem
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
        fixture = args.contig
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {
        k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in features.items()
    }
    token_initializer = build_token_initializer(
        torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                   map_location="cpu", weights_only=True))
    diffusion_module = build_diffusion_module(
        torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                   map_location="cpu", weights_only=True))
    with torch.no_grad():
        initial = token_initializer(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in features.items()})

    L = features["ref_pos"].shape[0]
    I = features["restype"].shape[0]
    fixed = features["is_motif_atom_with_fixed_coord"]
    coord = features["motif_pos"].float().unsqueeze(0)

    wrapped = 0 if args.no_instrument else instrument()
    resident = sample_dram()
    print(f"fixture: {fixture} I={I} L={L} timesteps={args.timesteps} "
          f"wrapped_ops={wrapped}")
    print(f"after weight load + token initializer: {resident / 2**30:.3f} GiB used "
          f"of {_total / 2**30:.1f} GiB ({_banks} banks)")
    print("D   peak_GiB  free_at_peak_GiB  headroom_x  status  seconds")

    global _peak
    records = []
    for D in args.batches:
        _peak = resident          # count the steady-state residency in the peak
        status, elapsed = "ok", float("nan")
        try:
            sampler = RFD3Sampler(num_timesteps=args.timesteps)
            start = time.perf_counter()
            with torch.no_grad():
                out, _ = sampler.sample(
                    diffusion_module, D, L, coord, features, initial, fixed,
                    generator=[torch.Generator().manual_seed(7000 + i) for i in range(D)],
                )
            elapsed = time.perf_counter() - start
            if not torch.isfinite(out).all().item():
                status = "NONFINITE"
        except Exception as exc:                       # an OOM is a result, not a crash
            status = type(exc).__name__ + ": " + str(exc).split("\n")[0][:120]
        peak = _peak
        gc.collect()
        free = _total - peak
        print(f"{D:<3d} {peak / 2**30:8.3f} {free / 2**30:16.3f} "
              f"{(_total / peak) if peak else 0:10.2f}  {status}  {elapsed:.1f}",
              flush=True)
        records.append({"fixture": fixture, "I": I, "L": L, "D": D,
                        "peak_bytes": peak, "total_bytes": _total,
                        "resident_bytes": resident, "status": status,
                        "seconds": elapsed, "timesteps": args.timesteps})
    if args.json:
        with open(args.json, "a") as fp:
            for record in records:
                fp.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
