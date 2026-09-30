#!/usr/bin/env python3
"""Card-free: the fused-HiFi chunk CANDIDATES per length. Pure arithmetic, no device, no arch
input, so whatever it says holds on Wormhole exactly as on Blackhole. What it cannot say is
which candidate fits L1 -- that needs the card."""
import json, sys
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/b2p-ceiling"); sys.path.insert(0, str(ROOT))
import tt_bio.tenstorrent as T
out = {}
for S in range(192, 993, 32):
    pad = T._padded_sdpa_len(S)
    kc = list(T._dividing_k_chunks(S, S))
    qc = list(T._tri_att_q_chunks(S, S))
    div = [c for c in qc if pad % c == 0]
    out[S] = {"padded": pad, "k_chunks": kc, "q_chunks": qc, "q_dividing": div,
              "n_k": len(kc), "aligned_divisors": [d for d in range(32, pad + 1, 32) if pad % d == 0]}
for S, r in out.items():
    print(f"{S:4d} pad={r['padded']:4d} aligned_divisors={len(r['aligned_divisors']):3d} "
          f"k_chunks={r['k_chunks']} q_dividing={r['q_dividing']}")
json.dump(out, open(ROOT / "perf/b2p_ceiling/chunkset_cardfree.json", "w"), indent=1)
