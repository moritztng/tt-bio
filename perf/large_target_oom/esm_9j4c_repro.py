#!/usr/bin/env python3
"""Reproduce the esmfold2 9j4c WH OOM with the full python traceback, which the
predict worker swallows into results.json. Campaign config: single-sequence
(no MSA), num_loops=10, fast mode (auto-enabled on Wormhole by the CLI).

    TT_VISIBLE_DEVICES=24 python3 esm_9j4c_repro.py [target]
"""
import sys
import traceback


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "9j4c"
    from pathlib import Path

    from tt_bio.esmfold2_runtime import load_ttnn_esmfold2, fold_complex
    from tt_bio.main import _read_protein_chains

    chains = [(cid, seq, None) for cid, seq, _ in
              _read_protein_chains(Path(f"examples/abag_xm/{target}.yaml"))]
    print(f"chains: {[(c, len(s)) for c, s, _ in chains]}", flush=True)
    model = load_ttnn_esmfold2(esmfold2_repo="biohub/ESMFold2", fast=True)
    try:
        fold_complex(model, chains, num_loops=10, num_sampling_steps=6,
                     num_diffusion_samples=1, seed=50000)
        print("FOLDED OK", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
