"""How many alignment rows actually reach BoltzGen's MSA module on the design path.

The shipped `tt-bio design --model boltzgen` CLI has no `--msa` / `--use_msa_server` option and
never searches an alignment, so the depth is decided inside the featurizer rather than by the
user. `featurizer.dummy_msa` builds ONE sequence per chain out of that chain's own residues,
which reads as "depth 1" -- but reading a depth off the code is exactly the mistake this
coverage work is meant to avoid (a 2048-row tiled MSA once arrived as 35 rows). So this patches
`prepare_msa_arrays`, the single place the per-chain MSAs become the tensor, and prints the
shape it returns on a real design run.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 perf/bgcov/msa_depth.py <spec.yaml> <out>

Prints one MSA_DEPTH line per featurization, then lets the run continue.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio.boltzgen.data import featurizer  # noqa: E402

_orig = featurizer.prepare_msa_arrays


def patched(tokens, pairing, is_paired, deletions, msa):
    out = _orig(tokens, pairing, is_paired, deletions, msa)
    rows = {cid: len(m.sequences) for cid, m in msa.items()}
    print(f"MSA_DEPTH chains={rows} msa_array_shape={getattr(out[0], 'shape', None)}",
          flush=True)
    return out


featurizer.prepare_msa_arrays = patched

if __name__ == "__main__":
    spec, out_dir = sys.argv[1], sys.argv[2]
    sys.argv = ["tt-bio", "design", spec, "--model", "boltzgen", "--out_dir", out_dir,
                "--num_designs", "1", "--steps", "design", "--debug"]
    from tt_bio.main import cli
    cli()
