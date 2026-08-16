#!/usr/bin/env python3
"""Reproduce the esmfold2 9j4c WH OOM with the full python traceback, which the
predict worker swallows into results.json.

The AbAg-XM campaign folds esmfold2 at `--diffusion_samples 64 --recycling_steps 10
--sampling_steps 100`, single-sequence, fast mode (auto-enabled on Wormhole by the CLI).
This script used to hard-code `num_sampling_steps=6, num_diffusion_samples=1` while its
docstring called that "campaign config", so the fix it was written to verify
(a0c009764, row-tiling the pair init) was signed off against 1/64 of the sample load and
1/16 of the step count. Measured 2026-08-11 on the WH Galaxy at the REAL campaign config,
9j4c still refuses:

    Out of Memory: Not enough space to allocate 627916800 B DRAM buffer across 12 banks,
    where each bank needs to store 52326400 B, but bank size is 1073741792 B
    (allocated: 1039460160 B, free: 34281632 B, largest free block: 13993888 B)

That is 96.8 pct bank occupancy -- genuine capacity, not fragmentation -- and it lands
AFTER all 68 diffusion steps, so it is not the pair-init allocation a0c009764 moved.

Defaults are now the campaign's own numbers. Override for a quick smoke run, but do not
sign off a capacity fix on an overridden config.

    TT_VISIBLE_DEVICES=31 python3 esm_9j4c_repro.py [target]
    TT_VISIBLE_DEVICES=31 python3 esm_9j4c_repro.py 9j4c --samples 1 --steps 6
"""
import argparse
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="9j4c")
    ap.add_argument("--loops", type=int, default=10, help="recycling_steps")
    ap.add_argument("--steps", type=int, default=100, help="sampling_steps")
    ap.add_argument("--samples", type=int, default=64, help="diffusion_samples")
    ap.add_argument("--seed", type=int, default=50000)
    a = ap.parse_args()

    from pathlib import Path

    from tt_bio.esmfold2_runtime import load_ttnn_esmfold2, fold_complex
    from tt_bio.main import _read_protein_chains

    chains = [(cid, seq, None) for cid, seq, *_ in
              _read_protein_chains(Path(f"examples/abag_xm/{a.target}.yaml"))]
    print(f"chains: {[(c, len(s)) for c, s, _ in chains]}", flush=True)
    print(f"config: loops={a.loops} steps={a.steps} samples={a.samples} seed={a.seed}",
          flush=True)
    model = load_ttnn_esmfold2(esmfold2_repo="biohub/ESMFold2", fast=True)
    try:
        fold_complex(model, chains, num_loops=a.loops, num_sampling_steps=a.steps,
                     num_diffusion_samples=a.samples, seed=a.seed)
        print("FOLDED OK", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
