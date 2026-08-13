#!/usr/bin/env python3
"""Screen 2 of `opendde-beat-b200`: price the rest of the 3.86 s "trunk glue" row. CPU ONLY.

§5.1 named 0.37 s of that row (the relp one-hot) and landed it. §9 left ~3.49 s with NO mechanism,
and the biggest named candidate was `StructuralTokenExpander._pair_features_rows` -- 8 row chunks of
(128 x 995) host `torch` per fold, never timed by anyone. This times it, on the shipped function.

The cost of `_pair_features_rows` is shape-driven, not value-driven: every term is an elementwise
compare/AND over (chunk x Ns) or a boolean-mask assignment into a (chunk x Ns) int64. So synthetic
indices of the right shape price it exactly, and no device, no weights and no checkpoint are needed.

GATE, written before the run:
  * >= 0.30 s/fold summed over the 8 chunks -> it is a real row in the ledger, screen a rewrite next
  * < 0.30 s/fold -> it is NOT where the 3.49 s is; say so, name what is left, and stop looking here

Predicted BEFORE the run, from the byte model: the masks are (128 x 995) bool = 127 KB each and
there are ~12 of them, plus one (128 x 995) int64 = 1.02 MB written 8 times by the seven
`rpt[mask] = k` assignments. ~30 MB of host traffic per chunk, ~240 MB per fold, which at the 8-12
GB/s single-threaded host bandwidth §5.1 measured is 0.02-0.03 s -- i.e. the prediction is that this
row is worth ~25 ms and the gate FAILS. The seven boolean-mask assignments are the one term that
could break that: each is a `nonzero` + `index_put`, not a bandwidth-bound pass.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time

sys.path.insert(0, "/home/ttuser/.coworker/wt/opendde-beat-b200")

import torch

from tt_bio.opendde import StructuralTokenExpander, _BACKBONE, _SIDECHAIN

NT, NS, CHUNK = 512, 995, 128


def synth():
    """One protein chain of NT residues expanded to NS structural tokens: every residue gets a
    backbone token, and the first NS-NT get a sidechain token too. That is the shape cdk2x2_512
    presents (995 structural tokens over 512 residues) with the role mix the pair features branch on.
    """
    parent = torch.cat([torch.arange(NT), torch.arange(NS - NT)])
    role = torch.cat([torch.full((NT,), _BACKBONE[0]), torch.full((NS - NT,), _SIDECHAIN)])
    ifd = {"asym_id": torch.zeros(NT, dtype=torch.long),
           "prev_parent_residue_idx": (parent - 1).clamp(min=-1),
           "next_parent_residue_idx": parent + 1}
    return ifd, role, parent


def main():
    ifd, role, parent = synth()
    ex = StructuralTokenExpander({}, None)
    n_chunks = (NS + CHUNK - 1) // CHUNK
    rows = [torch.arange(i * CHUNK, min((i + 1) * CHUNK, NS)) for i in range(n_chunks)]

    # warm the allocator once, then 5 full passes over all chunks
    ex._pair_features_rows(ifd, role, parent, rows[0])
    per_pass, per_chunk = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        for ri in rows:
            tc = time.perf_counter()
            out = ex._pair_features_rows(ifd, role, parent, ri)
            per_chunk.append((int(ri.numel()), time.perf_counter() - tc))
            del out
        per_pass.append(time.perf_counter() - t0)

    med = st.median(per_pass)
    bytes_per_chunk = (12 * CHUNK * NS * 1 + 8 * CHUNK * NS * 8) / 1048576   # 12 bool + 8 int64 passes
    rec = {"host": __import__("socket").gethostname(),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "NT": NT, "Ns": NS, "chunk": CHUNK, "n_chunks": n_chunks,
           "per_fold_s_median": round(med, 4),
           "per_fold_s_min": round(min(per_pass), 4),
           "per_fold_s_max": round(max(per_pass), 4),
           "per_chunk_ms_median": round(st.median([t for _n, t in per_chunk]) * 1e3, 3),
           "modelled_MB_per_chunk": round(bytes_per_chunk, 1),
           "implied_host_GBs": round(bytes_per_chunk * n_chunks / 1024 / med, 2),
           "gate_030_pass": bool(med >= 0.30)}
    print(json.dumps(rec, indent=1))
    out = sys.argv[1] if len(sys.argv) > 1 else "screen_expander_host.json"
    open(out, "w").write(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()
