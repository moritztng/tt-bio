#!/usr/bin/env python3
"""AF2 structure module on CPU alone, at the AF2-IG shape (195 residues), by host thread count.

Card-free. The structure module runs in host torch between recycles (AF2DeviceModel keeps only the two
block stacks on the card), and the WH device profile shows the card idle ~380 ms at each recycle
boundary. This prices the structure module's share of that. Its cost does not depend on weight values,
so random weights and placeholder atom maps are enough.

    python perf/spd/af2_sm_threads.py <tt-bio root> 2,1,4,2

pc, 2026-10-12, load ~6: 2 threads 102.6 / 113.5 ms, 1 thread 145.5, 4 threads 85.0. Two to four threads
buys ~17 %, so it is per-op overhead over eight small layers, not thread starvation.
"""
import sys, time, statistics as st
import torch
sys.path.insert(0, sys.argv[1])
from tt_bio import af2_reference as R
N = 195
torch.manual_seed(0)
sm = R.AF2StructureModule().eval()
single = torch.randn(N, R.C_S); pair = torch.randn(N, N, R.C_Z)
feats = {"seq_mask": torch.ones(N), "aatype": torch.randint(0, 20, (N,)), "atom14_atom_exists": torch.ones(N, 14), "atom37_atom_exists": torch.ones(N, 37), "residx_atom37_to_atom14": torch.zeros(N, 37, dtype=torch.long)}
def timeit(fn, reps=5, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return st.median(ts), min(ts), max(ts)
with torch.no_grad():
    for th in [int(x) for x in sys.argv[2].split(",")]:
        torch.set_num_threads(th)
        m, lo, hi = timeit(lambda: sm(single, pair, feats))
        act_2d = sm.pair_norm(pair)
        pl, _, _ = timeit(lambda: sm.ipa.pair_logits(act_2d))
        pn, _, _ = timeit(lambda: sm.pair_norm(pair))
        print(f"threads {th:2d}: structure module {m*1e3:7.1f} ms (min {lo*1e3:.1f} max {hi*1e3:.1f}) | "
              f"pair_norm {pn*1e3:6.1f} ms | ipa.pair_logits {pl*1e3:6.1f} ms", flush=True)
