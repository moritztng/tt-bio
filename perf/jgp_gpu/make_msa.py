#!/usr/bin/env python3
"""Rebuild the complex730 MSA with tt-bio's own search calls, network only (no device).

    PYTHONPATH=. python perf/jgp_gpu/make_msa.py state/jfb/complex730.yaml perf/jgp_gpu/msa

Same calls, same server, same settings as the 280.57 s Blackhole run (ColabFold API, no envdb,
greedy pairing, no msa_cap): per-chain unpaired a3m from `_generate_esmfold2_a3m` and the
species-paired a3m from `_paired_msa`. Writes <out>/<chain>.unpaired.a3m and
<out>/<chain>.paired.a3m, the files upstream Protenix is handed on the GPU.
"""
import sys
from pathlib import Path

import yaml

from tt_bio.cache import seq_hash
from tt_bio.main import _generate_esmfold2_a3m
from tt_bio.worker import _paired_msa

URL = "https://api.colabfold.com"
yml, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
msa_dir = out / "cache"
msa_dir.mkdir(exist_ok=True)
seqs = [(s["protein"]["id"], s["protein"]["sequence"]) for s in yaml.safe_load(yml.read_text())["sequences"]]
cfg = dict(use_msa_server=True, msa_server_url=URL, msa_pairing_strategy="greedy", use_envdb=False)

_generate_esmfold2_a3m({seq_hash(s): s for _c, s in seqs}, yml.stem, msa_dir, None, False,
                       URL, "greedy", None, None, None)
chains = [(c, s, None, "protein", None) for c, s in seqs]
paired = _paired_msa(yml, chains, msa_dir, cfg)
for c, s in seqs:
    un = (msa_dir / f"{seq_hash(s)}.a3m").read_text()
    (out / f"{c}.unpaired.a3m").write_text(un)
    (out / f"{c}.paired.a3m").write_text(paired[seq_hash(s)])
    print(c, len(s), "unpaired", un.count(">"), "paired", paired[seq_hash(s)].count(">"))
