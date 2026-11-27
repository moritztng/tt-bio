import sys, time, torch
sys.path.insert(0, "perf/r4t")
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
torch.set_num_threads(2)
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
g = N.FlowGlue(state, config); b = load_batch(sys.argv[1])[0]
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    g.pair_split.constant(b)
    t0 = time.perf_counter(); g.pair_split.constant(b); print("constant_s", round(time.perf_counter() - t0, 3))
    with profile(activities=[ProfilerActivity.CPU]) as p:
        g.pair_split.constant(b)
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
