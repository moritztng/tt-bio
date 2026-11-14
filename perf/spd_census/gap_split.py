# Per-call idle gap for one signature, reusing analyze.py's join and unwrap exactly.
import json, sys
import numpy as np
from pathlib import Path
RUN = Path(sys.argv[1]); FOLD = sys.argv[2]; SIG = int(sys.argv[3])
calls = np.array([json.loads(l) for l in open(RUN / f"ops_{FOLD}.jsonl")], dtype=np.int64)
calls = calls[np.argsort(calls[:, 1], kind="stable")]
P = np.array([[x if x is not None else -1 for x in json.loads(l)] for l in open(RUN / f"progs_{FOLD}.jsonl")],
             dtype=np.float64)
RID, BAT, CORES, K, FW, T0, T1, T2, BR, NC, ST, EN = range(12)
P[:, RID] = np.floor(P[:, RID] / 1024)
T44 = 2.0 ** 44
_, inv = np.unique(P[:, BAT], return_inverse=True)
top = np.full(inv.max() + 1, -np.inf); np.maximum.at(top, inv, P[:, ST]); top = top[inv]
for c in (ST, EN): P[:, c] += np.where(top - P[:, c] > T44 / 2, T44, 0.0)
for c in (K, FW, T0, T1, T2, BR, NC): P[:, c] += np.where(P[:, c] > 2.0 ** 62, T44 - 2.0 ** 64, 0.0)
WRAP = 2.0 ** 32
wrap = np.floor((P[:, EN] - P[:, ST]) / WRAP)
if (wrap > 0).any():
    sp = P[:, EN] - P[:, ST]; good = (wrap == 0) & (sp > 0) & (P[:, K] > 0)
    wrap_ns = WRAP * float(np.median(P[good, K] / sp[good]))
    for i in np.flatnonzero(wrap > 0):
        P[i, ST] += wrap[i] * WRAP
        for c in (K, FW, T0, T1, T2, BR, NC):
            if P[i, c] >= wrap_ns * 0.999: P[i, c] -= wrap[i] * wrap_ns
idx = np.searchsorted(calls[:, 1], P[:, RID], side="right") - 1
ok = (idx >= 0) & (P[:, RID] < calls[np.clip(idx, 0, None), 2])
P = P[ok]; idx = idx[ok]
o = np.lexsort((P[:, ST], P[:, BAT]))
same = P[o[1:], BAT] == P[o[:-1], BAT]
gap = np.where(same, (P[o[1:], ST] - np.maximum.accumulate(P[o, EN])[:-1]), 0.0)
gap_before = np.zeros(len(P)); gap_before[o[1:]] = np.clip(gap, 0, None)
sel = calls[idx, 0] == SIG
print(f"sig {SIG}: {int(sel.sum())} programs, total gap {gap_before[sel].sum()/1e9:.4f} s")
rows = sorted(zip(idx[sel], gap_before[sel] / 1e6), key=lambda r: -r[1])
for call_i, ms in rows[:20]:
    print(f"  call index {int(call_i):6d}  gap {ms:9.2f} ms")
rest = sorted(r[1] for r in rows)[:-1]
print(f"excluding the single largest: {sum(rest)/1000:.4f} s over {len(rest)} programs")
