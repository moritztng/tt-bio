#!/usr/bin/env python3
"""Does the same input score to the same number twice?

The scoring half of the question. It runs `tt_bio.interface_scores` over the same files N times
in N fresh interpreters, each with a different PYTHONHASHSEED and a different BLAS thread count,
and compares the serialised output byte for byte. Dict iteration order, thread count and
reduction order are the three things that would move a float here without anyone editing code.

    python scripts/interface_scores_determinism.py --runs 8

The other half is the FOLD: whether two runs of the model produce the same coordinates and PAE.
That needs a device and is not measured here. A caller that wants the fold's own spread asks for
K diffusion samples and reads `interface_score_distribution`, which is what that field is for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_WORKER = """
import json, sys
from tt_bio import interface_scores as isc
d = sys.argv[1]
out = isc.score_files(d + "/m_model_0.cif", d + "/pae_m_model_0.npz",
                      d + "/plddt_m_model_0.npz", d + "/confidence_m_model_0.json",
                      float(sys.argv[2]), float(sys.argv[3]))
sys.stdout.write(json.dumps(out, sort_keys=True))
"""


def build_case(tmp: Path, seed: int, lengths) -> Path:
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT))
    from test_interface_scores import _case
    tmp.mkdir(parents=True, exist_ok=True)
    return _case(tmp, seed, lengths, ligand_atoms=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--pae_cutoff", type=float, default=15.0)
    ap.add_argument("--dist_cutoff", type=float, default=15.0)
    a = ap.parse_args()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ipsae-det-") as td:
        d = build_case(Path(td), 4242, (48, 31, 19))
        digests = []
        for i in range(a.runs):
            env = {**os.environ, "PYTHONHASHSEED": str(i * 7919 + 1),
                   "OMP_NUM_THREADS": str(1 + i % 4), "MKL_NUM_THREADS": str(1 + i % 4),
                   "OPENBLAS_NUM_THREADS": str(1 + i % 4)}
            r = subprocess.run([sys.executable, "-c", _WORKER, str(d),
                                str(a.pae_cutoff), str(a.dist_cutoff)],
                               check=True, capture_output=True, text=True, cwd=ROOT, env=env)
            digests.append((hashlib.sha256(r.stdout.encode()).hexdigest(), r.stdout))
    uniq = {h for h, _ in digests}
    print(f"{a.runs} runs, {len(uniq)} distinct result(s)")
    print(f"sha256 {digests[0][0]}")
    ex = json.loads(digests[0][1])["pairs"]
    for pair, v in ex.items():
        print(f"  {pair}: ipsae {v['ipsae']!r} ipsae_min {v['ipsae_min']!r} "
              f"pdockq2 {v['pdockq2']!r} interface_pae {v['interface_pae']!r}")
    if len(uniq) == 1:
        print("identical: every run returned the same bytes")
        return 0
    print("NOT identical -- the scoring step is not reproducible, which is the product")
    return 1


if __name__ == "__main__":
    sys.exit(main())
