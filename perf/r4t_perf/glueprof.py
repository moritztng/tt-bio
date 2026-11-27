#!/usr/bin/env python3
"""CPU-only profile of trunk_step's host glue (FlowGlue.initial_states) on a dumped batch: the K-candidate call
(grad off) and the winner call (grad on, then a backward through it), at a given torch thread count.
    glueprof.py BATCH.npz THREADS [N_REPEAT]"""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t"))
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E

path, th = sys.argv[1], int(sys.argv[2]); rep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
torch.set_num_threads(th)
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
glue = N.FlowGlue(state, config); glue.requires_grad_(True)
b, m, _r, _l = load_batch(path)
K = int(b["padding_mask"].shape[0])
take = lambda d: {k: take(v) for k, v in d.items()} if isinstance(d, dict) else (d[0:1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d)
bw = take(b)
for i in range(rep):
    t0 = time.perf_counter()
    with torch.no_grad():
        glue.initial_states(b)
    t1 = time.perf_counter()
    s, z = glue.initial_states(bw)
    t2 = time.perf_counter()
    torch.autograd.backward([s, z], [torch.ones_like(s), torch.ones_like(z)])
    t3 = time.perf_counter()
    print(f"rep {i} threads {th} K={K} cand_nograd {t1-t0:.3f} winner_fwd {t2-t1:.3f} winner_bwd {t3-t2:.3f}", flush=True)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as p:
    with torch.no_grad():
        glue.initial_states(b)
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
