"""Dump the upstream RFD4 feature dict for one annotated CIF, as the parity golden.

Runs ONLY in the upstream reference env (/home/moritz/rfd4-ref-env/venv), never in tt-bio's:
it imports rfproteina and atomworks 3.0.4. tt-bio's own featurizer is compared against the
.npz this writes, field by field.

    rfd4-ref-env/venv/bin/python -I dump_upstream_features.py <in.cif[.zst]> <out.npz> [seed]

THE SEED IS NOT OPTIONAL FOR PARITY. `RealizeExpandableSegments` samples the length of every
expandable segment, so an input with a variable-length binder featurizes to a DIFFERENT NUMBER
OF TOKENS on each run: the ppi insulinr example came back 205, 206 and 248 tokens on three
unseeded runs of the same file. Upstream's own host entry point, `rfd4_proteina_prep.prepare_cif`,
takes a `seed` and passes `expandable_segment_rng=np.random.default_rng(seed)` for this reason.
We do the same, and additionally seed `random`, `numpy` and `torch` so any other draw in the
pipeline is pinned too. Parity is defined at a stated seed; a golden without one is noise.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_SEED = 0


def dump(cif: Path, out: Path, seed: int = DEFAULT_SEED) -> dict:
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.inference_utils.datasets import DesignProblemDataset
    from rfproteina.utils.constants import LOCAL_LATENT_RESIDUES

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Same construction as rfd4_proteina_prep.prepare_cif: na_ss_conditioning on (RFD4's own
    # inference default of no tags silently drops nucleic-acid secondary-structure conditioning
    # carried in the file), the checkpoint's local-latent alphabet, and the seeded segment rng.
    pipeline = build_design_validation_pipeline(
        metrics_tags=["na_ss_conditioning"],
        residue_names_supporting_local_latents=list(LOCAL_LATENT_RESIDUES),
        path_length_search_cutoff=64,
        expandable_segment_rng=np.random.default_rng(seed),
    )
    ds = DesignProblemDataset([cif], transform=pipeline)
    example = ds[0]
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}/{k}" if prefix else str(k), v)
        elif isinstance(obj, torch.Tensor):
            flat[prefix] = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            flat[prefix] = obj
        elif isinstance(obj, (int, float, bool, str)):
            flat[prefix] = np.array(obj)

    walk("", example)
    flat["_seed"] = np.array(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **flat)
    return flat


if __name__ == "__main__":
    cif, out = Path(sys.argv[1]), Path(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SEED
    flat = dump(cif, out, seed)
    print(f"{len(flat)} arrays (seed {seed}) -> {out}")
    for k in sorted(flat):
        a = flat[k]
        print(f"  {k}: {a.shape} {a.dtype}")
