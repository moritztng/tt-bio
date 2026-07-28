"""p31: one timed leg of the head-merge A/B, plus the trajectory it produced.

The change under test (gate applied after the head merge instead of before, and
`nlp_concat_heads` for the merge wherever head_dim is a tile multiple) rewrites the call
graph, so it cannot be toggled in-process the way p30's `_bias_cache` could. Instead each
leg is its own process against its own `tt_bio` tree -- the shipped one via
`git archive HEAD`, the new one via the worktree -- and the driver alternates them on one
hot card so drift and thermals hit both legs equally (p14: a non-interleaved first read
said +13.3% against an honest +5.2%).

Each leg warms the kernel cache with a throwaway trajectory before the timed ones (p17: a
cold read is 13x), then writes its per-alternation ms/step and its final coordinates to a
.pt, so the driver can prove bit-exactness at TRAJECTORY level from the same runs that
produced the timing (p25: an op-level win can wash out in the loop, and an op-level PCC can
hide a loop-level divergence).

Run via scripts/rfd3_port/run_p31_headmerge_ab.sh -- not directly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import tt_bio.rfd3 as R
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

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

        RFD3Sampler(num_timesteps=args.warmup).sample(
            dm, args.batch, L, coord0, f, init,
            f["is_motif_atom_with_fixed_coord"],
            generator=torch.Generator().manual_seed(7))

        ms_list, x = [], None
        for _ in range(args.reps):
            g = torch.Generator().manual_seed(42)
            t0 = time.perf_counter()
            out = RFD3Sampler(num_timesteps=args.timesteps).sample(
                dm, args.batch, L, coord0, f, init,
                f["is_motif_atom_with_fixed_coord"], generator=g)
            ms_list.append((time.perf_counter() - t0) / steps * 1e3)
            x = (out[0] if isinstance(out, tuple) else out).detach().clone().float()

    torch.save({"L": L, "batch": args.batch, "steps": steps, "ms": ms_list, "x": x}, args.out)
    print(f"L={L} batch={args.batch} steps/rep={steps} ms/step={[round(v, 2) for v in ms_list]}",
          flush=True)


if __name__ == "__main__":
    main()
