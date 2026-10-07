#!/usr/bin/env python3
"""Build every set member's MSA once with tt-bio's own search calls, network only (no device).

    PYTHONPATH=. python perf/pfm_accuracy/make_msa.py <data dir>

The same calls and settings as perf/jgp_gpu/make_msa.py (ColabFold API, no envdb, greedy pairing,
no cap): per-chain unpaired a3m from `_generate_esmfold2_a3m`, species-paired a3m from
`_paired_msa`. Writes <data>/msa/<pdb>/<chain>.{unpaired,paired}.a3m, the files upstream Protenix
is handed on the GPU box and tt-bio reads on ours, and appends depths to <data>/msa/depth.tsv.
"""
import sys
from pathlib import Path

import yaml

from tt_bio.cache import seq_hash
from tt_bio.main import _generate_esmfold2_a3m
from tt_bio.worker import _paired_msa

URL = "https://api.colabfold.com"
data = Path(sys.argv[1])
cache = data / "msa" / "cache"
cache.mkdir(parents=True, exist_ok=True)
cfg = dict(use_msa_server=True, msa_server_url=URL, msa_pairing_strategy="greedy", use_envdb=False)
for yml in sorted((data / "inputs").glob("*.yaml")):
    out = data / "msa" / yml.stem
    if (out / "B.paired.a3m").exists():
        continue
    out.mkdir(parents=True, exist_ok=True)
    seqs = [(s["protein"]["id"], s["protein"]["sequence"]) for s in yaml.safe_load(yml.read_text())["sequences"]]
    _generate_esmfold2_a3m({seq_hash(s): s for _c, s in seqs}, yml.stem, cache, None, False,
                           URL, "greedy", None, None, None)
    paired = _paired_msa(yml, [(c, s, None, "protein", None) for c, s in seqs], cache, cfg)
    row = [yml.stem]
    for c, s in seqs:
        un = (cache / f"{seq_hash(s)}.a3m").read_text()
        (out / f"{c}.unpaired.a3m").write_text(un)
        (out / f"{c}.paired.a3m").write_text(paired[seq_hash(s)])
        row += [c, len(s), un.count(">"), paired[seq_hash(s)].count(">")]
    with open(data / "msa" / "depth.tsv", "a") as f:
        f.write("\t".join(map(str, row)) + "\n")
    print(*row, flush=True)
print("MSA-DONE")
