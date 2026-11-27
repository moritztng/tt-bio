import os, sys, time, torch
sys.path.insert(0, "perf/r4t")
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
torch.set_num_threads(int(sys.argv[2]))
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
g = N.FlowGlue(state, config); b = load_batch(sys.argv[1])[0]
with torch.no_grad():
    for i in range(3):
        t0 = time.perf_counter(); g.pair_repr_builder(b); t1 = time.perf_counter()
        c = g.pair_split.constant(b); t2 = time.perf_counter()
        p = g.pair_split.inputs(b, c); t3 = time.perf_counter()
        s = g.init_repr_factory(b); t4 = time.perf_counter()
        print(f"host_pair {t1-t0:.3f} constant {t2-t1:.3f} step_inputs {t3-t2:.3f} seq_feats {t4-t3:.3f}", flush=True)
