#!/usr/bin/env python3
"""Add the newly-named host stages to the on-card instrument's region tree.

usage: extend_instrument.py <path-to-perf/b2x_host_residual/host_residual.py>

The first version of the instrument bracketed the four top-level steps of `predict_one` and the
three device phases. The pc legs then named what is inside `predict_step`'s exclusive time, and
`diffusion_conditioning` turned out to be 60.7 % of the whole host path, so it needs its own row
on the card rather than being part of a glue remainder. Anchor-checked and idempotent.
"""
import argparse
import sys
from pathlib import Path

ANCHOR = '''    ok["confidence"] = reg.patch(boltz2.ConfidenceModule, "forward", "confidence")
    return ok'''

NEW = '''    ok["confidence"] = reg.patch(boltz2.ConfidenceModule, "forward", "confidence")
    # --- the host stages between the trunk and the sampler ----------------------------------
    # `predict_step` exclusive turned out to be the biggest single block of the fold's host
    # path, and `DiffusionConditioning` is 60.7 % of it: 120 GFLOP of dense fp32 matmul with no
    # ttnn implementation at all, running between the trunk and the first denoiser call. Give
    # each of these its own row so the residual table has no glue remainder to hide in.
    ok["rel_pos"] = reg.patch(boltz2.RelativePositionEncoder, "forward", "rel_pos")
    ok["input_embedder"] = reg.patch(boltz2.InputEmbedder, "forward", "input_embedder")
    ok["diffusion_cond"] = reg.patch(boltz2.DiffusionConditioning, "forward", "diffusion_cond")
    ok["pairwise_cond"] = reg.patch(boltz2.PairwiseConditioning, "forward", "pairwise_cond")
    ok["atom_encoder"] = reg.patch(boltz2.AtomEncoder, "forward", "atom_encoder")
    return ok'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    a = ap.parse_args()
    s = a.path.read_text()
    if "diffusion_cond" in s:
        print("already extended")
        return 0
    assert s.count(ANCHOR) == 1, f"anchor x{s.count(ANCHOR)} -- host_residual.py has drifted"
    a.path.write_text(s.replace(ANCHOR, NEW))
    print(f"extended {a.path}: rel_pos, input_embedder, diffusion_cond, pairwise_cond, "
          f"atom_encoder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
