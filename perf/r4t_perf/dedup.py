"""PairSplit.constant on the K-candidate batch: old (all K rows) vs dedup (one row, expanded). Bit-identity + time.
   dedup.py BATCH.npz [threads]"""
import sys, time, hashlib, torch
sys.path.insert(0, "perf/r4t")
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
g = N.FlowGlue(state, config); b = load_batch(sys.argv[1])[0]
sp = g.pair_split
print("K", b["padding_mask"].shape)
dig = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]
with torch.no_grad():
    for arm in ("old", "new", "old", "new"):
        sp.DEDUP = arm == "new"
        t0 = time.perf_counter(); c = sp.constant(b); dt = time.perf_counter() - t0
        print(arm, round(dt, 3), tuple(c.proj.shape), dig(c.proj), dig(c.scales))
# Broken arm: one candidate's constant feature differs -> the guard must decline (dedup == full compute, row 1 != row 0).
cr = sp.const_creators[0]
class Bump:
    def __init__(s, c): s.c = c
    def get_dim(s): return s.c.get_dim()
    def __call__(s, batch):
        f = s.c(batch).clone(); f.view(f.shape[0], -1)[1, 0] += 1.0; return f
sp.const_creators[0] = Bump(cr)
with torch.no_grad():
    sp.DEDUP = False; o = sp.constant(b)
    sp.DEDUP = True; d = sp.constant(b)
same = torch.equal(o.proj, d.proj) and torch.equal(o.scales, d.scales)
rows_differ = not torch.equal(d.proj[0], d.proj[1])
print("BROKEN-ARM", "declined (PASS)" if same and rows_differ else "SERVED (FAIL)", dig(o.proj), dig(d.proj))
