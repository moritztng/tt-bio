"""K3 probe: the K-candidate grad-off trunk pass as one batch-K call vs K sequential batch-1 calls, same inputs,
both with all-ones masks (so mask handling is equal), warm, synced, AICLK sampled from sysfs during. Timing only:
runs inside trunk_step at the 2nd batch-K grad-off Trunk call, writes $KSPLIT_OUT and exits. Usage:
  KSPLIT_OUT=x.json python3 perf/r4t_perf/ksplit.py perf/r4t_step/trunk_step.py <trunk_step args>"""
import sys, os, time, json, runpy, threading
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio.rfd4 import trunk as TR
from tt_bio import autograd as ag, tenstorrent as T
DEV, SEEN, CLK = [], [0], []
_gd = T.get_device
def gd(*a, **k):
    d = _gd(*a, **k); DEV[:] = DEV or [d]; return d
T.get_device = gd
card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
def sampler():
    p = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
    while True:
        try: CLK.append(int(open(p).read().split()[0]))
        except Exception: pass
        time.sleep(0.05)
v = lambda t: getattr(t, "value", t)
def outs_free(o):
    for t in (o if isinstance(o, (tuple, list)) else [o]):
        try: ttnn.deallocate(v(t))
        except Exception: pass
_call = TR.Trunk.__call__
def call(self, s, z, conds, masks):
    S = v(s); K = int(S.shape[0])
    if ag.is_grad_enabled() or K == 1:
        return _call(self, s, z, conds, masks)
    SEEN[0] += 1
    if SEEN[0] < 3:
        return _call(self, s, z, conds, masks)
    dev = DEV[0]; Z = v(z); C = [v(c) for c in conds]; n = int(S.shape[1])
    mk = TR.trunk_masks(torch.ones(K, n), dev, dtype=self.dtype, residual_dtype=self.residual_dtype)
    m1 = TR.trunk_masks(torch.ones(1, n), dev, dtype=self.dtype, residual_dtype=self.residual_dtype)
    zd = self.pair_dtype(m1)
    def sl(t, i):
        sh = [int(x) for x in t.shape]
        return ttnn.slice(t, [i] + [0] * (len(sh) - 1), [i + 1] + sh[1:])
    def run(ss, zz, cc, mm):
        if zz.dtype != zd: zz = ttnn.typecast(zz, zd)
        ttnn.synchronize_device(dev); c0 = len(CLK); t0 = time.perf_counter()
        o = _call(self, ss, zz, cc, mm); ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0; outs_free(o); return dt, CLK[c0:]
    threading.Thread(target=sampler, daemon=True).start()
    res = dict(K=K, n=n, zdtype_in=str(Z.dtype), zdtype=str(zd), batchK=[], split=[], clk=[])
    for rep in range(6):
        dt, ck = run(ttnn.clone(S), ttnn.clone(Z), [ttnn.clone(c) for c in C], mk)
        res["batchK"].append(round(dt, 4)); res["clk"] += ck
        tot = 0.0
        for i in range(K):
            dt, ck = run(sl(S, i), sl(Z, i), [sl(c, i) for c in C], m1); tot += dt; res["clk"] += ck
        res["split"].append(round(tot, 4))
    c = res.pop("clk"); res["aiclk"] = [min(c or [0]), max(c or [0]), len(c)]
    json.dump(res, open(os.environ["KSPLIT_OUT"], "w"), indent=1); print("KSPLIT", json.dumps(res), flush=True)
    os._exit(0)
TR.Trunk.__call__ = call
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
