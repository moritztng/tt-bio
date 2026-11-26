"""Host cost of the K-candidate glue.initial_states with the card pair path (constant + inputs), 2 threads."""
import sys, time, torch
sys.path.insert(0, "perf/r4t")
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
torch.set_num_threads(2)
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
g = N.FlowGlue(state, config); b = load_batch(sys.argv[1])[0]
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    for _ in range(2):
        t0 = time.perf_counter(); c = g.pair_split.constant(b); t1 = time.perf_counter()
        g.initial_states(b, pair_constant=c); t2 = time.perf_counter()
        print("constant", round(t1 - t0, 3), "initial_states", round(t2 - t1, 3))
    with profile(activities=[ProfilerActivity.CPU]) as p:
        g.initial_states(b, pair_constant=g.pair_split.constant(b))
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=14))
