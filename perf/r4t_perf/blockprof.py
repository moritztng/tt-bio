"""r4t-perf: where the time of ONE taped RFD4 PairUpdate block goes (crop 256), each verb priced against its roof.

Same production module, weights and inputs as r4t-step's `perf/r4t_step/pairblock_mem.py` (that row measures the block's
MEMORY; this measures its TIME). One process, one device context, arms run back to back:

  wall      instrument off: forward taped, backward, each synchronized. The production number.
  untaped   grad-off forward (inference path) for the taped/untaped ratio.
  drain     every ttnn verb (tt-bio modules, autograd included) wrapped with a stopwatch AND a synchronize inside it, so
            each verb's SELF time is its drained device time plus one dispatch. Sums to the leg; the drain's own cost
            shows as drain-sum minus wall.
  wall_aa   the wall arm again after the instrument is removed: the A/A floor.
Host crossings are counted as to_torch/from_torch verbs inside each leg (OF3T's backward was 73.5 % host crossings,
from exact_training's float64 round trips; that instrument is OFF by default now and the harness records its state).

Roofs (state/of3t/BACKWARD.md, measured on Blackhole): compute 115.685 TFLOP/s dense bf16, DRAM ~440 GB/s.
Per verb: roof_s = max(flops / compute roof, bytes / DRAM roof); bytes = every ttnn operand read + output written.

Run: TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf python3 perf/r4t_perf/blockprof.py --out X.json
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
COMPUTE_ROOF = 115.685e12
DRAM_ROOF = 440e9
NS = time.perf_counter_ns
HOST_VERBS = ("to_torch", "from_torch")


def _desc(t):
    try:
        shp = tuple(int(d) for d in t.shape)
        dt = str(t.dtype)
    except Exception:  # noqa: BLE001
        return None
    n = 1
    for d in shp:
        n *= d
    return shp, dt, n * (4 if ("float32" in dt or "int32" in dt) else 1 if "8" in dt else 2)


class Rec:
    def __init__(self):
        self.verb, self.node, self.cur, self.on, self.ch = {}, {}, None, False, []

    def add(self, name, out, args, ns, sns):
        d = _desc(out)
        ins = [x for x in (_desc(a) for a in args) if x is not None]
        if d is None and ins:
            d = ins[0]
        shp = d[0] if d else ()
        nb = (d[2] if d else 0) + sum(x[2] for x in ins)
        fl = 0
        if d and len(ins) >= 2 and ("matmul" in name or "linear" in name):
            k = ins[0][0][-1] if ins[0][0] else 0
            o = 1
            for s in shp:
                o *= s
            fl = 2 * o * k
        key = (name, shp)
        r = self.verb.setdefault(key, [0, 0, 0, 0])
        r[0] += 1
        r[1] += sns
        r[2] += nb
        r[3] += fl
        if self.cur is not None and self.cur in self.node:
            self.node[self.cur][2] += sns


REC = Rec()


class W:
    """Rebinds `ttnn` inside tt-bio modules (bwprof._W's walk), timing SELF ns; with drain, synchronize inside the timer
    so the device work lands on the verb that issued it, not on whichever verb next blocks."""
    drain_dev = None

    def __init__(self, real, prefix=""):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        real = object.__getattribute__(self, "_real")
        attr = getattr(real, name)
        qual = object.__getattribute__(self, "_prefix") + name
        if isinstance(attr, (types.ModuleType, W)) or type(attr).__name__ == "_Ttnn":
            out = W(attr, qual + ".")
        elif callable(attr) and not isinstance(attr, type) and name != "synchronize_device":
            def call(*a, _f=attr, _q=qual, **k):
                rec = REC
                if not rec.on:
                    return _f(*a, **k)
                rec.ch.append(0)
                t0 = NS()
                try:
                    r = _f(*a, **k)
                    if W.drain_dev is not None:
                        _SYNC(W.drain_dev)
                finally:
                    ns = NS() - t0
                    kids = rec.ch.pop()
                    if rec.ch:
                        rec.ch[-1] += ns
                rec.add(_q, r, a, ns, max(ns - kids, 0))
                return r
            out = call
        else:
            return attr
        object.__setattr__(self, name, out)
        return out


_SYNC = None
_SAVED = []


def swap(on):
    if on:
        for nm, mod in list(sys.modules.items()):
            if nm.startswith("tt_bio") and mod is not None:
                cur = getattr(mod, "ttnn", None)
                if cur is not None and not isinstance(cur, W):
                    _SAVED.append((mod, cur))
                    mod.ttnn = W(cur)
        REC.on = True
    else:
        REC.on = False
        for mod, cur in _SAVED:
            mod.ttnn = cur
        _SAVED.clear()


def instrument_nodes(ag):
    orig = ag._Node.__init__

    def node_init(self, fn, parents, group=None):
        qn = getattr(fn, "__qualname__", repr(fn))

        def timed(g, _f=fn, _q=qn):
            prev, REC.cur = REC.cur, _q
            n = REC.node.setdefault(_q, [0, 0, 0])
            n[0] += 1
            t0 = NS()
            try:
                return _f(g)
            finally:
                n[1] += NS() - t0
                REC.cur = prev
        orig(self, timed, parents, group)
    ag._Node.__init__ = node_init


def table(rec, wall_s):
    agg = {}
    for (name, shp), (n, sns, nb, fl) in rec.verb.items():
        verb = name.split(".")[-1]
        a = agg.setdefault(verb, [0, 0, 0, 0, set()])
        a[0] += n
        a[1] += sns
        a[2] += nb
        a[3] += fl
        a[4].add(shp)
    rows = []
    for v, (n, sns, nb, fl, shps) in agg.items():
        roof = max(fl / COMPUTE_ROOF, nb / DRAM_ROOF)
        s = sns / 1e9
        rows.append(dict(verb=v, n=n, self_s=round(s, 5), share=round(s / wall_s, 4) if wall_s else None,
                         us_per_call=round(sns / n / 1e3, 1), gb=round(nb / 1e9, 3), tflop=round(fl / 1e12, 3),
                         roof_s=round(roof, 5), pct_of_roof=round(100 * roof / s, 1) if s else None,
                         bound="compute" if fl / COMPUTE_ROOF > nb / DRAM_ROOF else "dram", shapes=len(shps)))
    rows.sort(key=lambda r: -r["self_s"])
    nodes = sorted(({"closure": k, "fired": v[0], "s": round(v[1] / 1e9, 5)} for k, v in rec.node.items()),
                   key=lambda r: -r["s"])
    tot = sum(r["self_s"] for r in rows)
    roof = sum(r["roof_s"] for r in rows)
    host = sum(r["self_s"] for r in rows if r["verb"] in HOST_VERBS)
    return dict(verb_sum_s=round(tot, 5), roof_sum_s=round(roof, 5), host_verb_s=round(host, 5),
                host_verb_calls=sum(r["n"] for r in rows if r["verb"] in HOST_VERBS), by_verb=rows[:40],
                by_closure=nodes[:30])


def main():
    global _SYNC
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import subprocess
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.train import lora
    import rfd4_reference as REF

    _SYNC = ttnn.synchronize_device
    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    clk, stop = [], threading.Event()

    def sampler():
        p = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
        while not stop.is_set():
            try:
                clk.append((time.time(), int(open(p).read().split()[0])))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)
    threading.Thread(target=sampler, daemon=True).start()

    def clk_between(t0, t1):
        v = [c for t, c in clk if t0 - 0.06 <= t <= t1 + 0.06]
        return [min(v), max(v), len(v)] if v else None

    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    x, z, _, _ = REF.block_inputs(1, a.n, 768, 384, 256, seed=1)
    mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                        TR.compute_kernel_config())

    def put(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xt, zt = put(x), put(z)
    ag.install()
    params = lora.walked_weights(mod, None, mod, xt, zt, None, None, None)
    leaves = list(params.values()) if hasattr(params, "values") else list(params)
    instrument_nodes(ag)
    from tt_bio import lnbw as _lnbw
    _elig, _seen = _lnbw.eligible, {}

    def _eligible(x, g, gamma, **k):  # which LN backward calls the kernel serves, by shape, dtype and form
        r = _elig(x, g, gamma, **k)
        key = f"{list(x.shape)} x={x.dtype} g={g.dtype} gamma={gamma is not None} weights={k.get('weights', False)} -> {r}"
        _seen[key] = _seen.get(key, 0) + 1
        return r
    _lnbw.eligible = _eligible

    def step(prof=None):
        """One taped fwd + bwd. prof: None (wall) or 'drain' (per-verb drained attribution)."""
        recs = {}
        ttnn.synchronize_device(dev)
        global REC
        if prof:
            REC = Rec()
            W.drain_dev = dev
        t0 = time.time()
        with ag.tape():
            # Inside the tape: tape() shims only modules whose `ttnn` is the real module, so the instrument
            # must wrap the shim, not stand in front of it.
            if prof:
                swap(True)
            out = mod(xt, zt, None, None, None)
            ttnn.synchronize_device(dev)
            if prof:
                swap(False)
        t1 = time.time()
        if prof:
            recs["fwd"] = REC
            REC = Rec()
            swap(True)
        out_t = out if isinstance(out, ag.Tensor) else ag.Tensor(out)
        seed = ttnn.ones_like(out_t.value)
        t1b = time.time()
        ag.backward([out_t], [seed])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        if prof:
            swap(False)
            recs["bwd"] = REC
            W.drain_dev = None
        grads = [ttnn.to_torch(p.grad).float() for p in leaves if p.grad is not None]
        gd = hashlib.sha256(b"".join(g.numpy().tobytes() for g in grads)).hexdigest()[:16]
        nz = sum(1 for g in grads if float(g.abs().sum()) > 0)
        for p in leaves:
            p.grad = None
        ag.release_pins()
        row = dict(fwd_s=round(t1 - t0, 5), bwd_s=round(t2 - t1b, 5), aiclk_fwd=clk_between(t0, t1),
                   aiclk_bwd=clk_between(t1b, t2), grad_digest=gd, nonzero=f"{nz}/{len(leaves)}")
        return row, recs

    def untaped():
        """Grad-off forward through the dispatching attach point (tape() + no_grad: production op, no node).
        A bare call fails: walked_weights leaves the module holding autograd leaves, which plain ttnn.reshape
        (trunk._write) refuses, installed or not."""
        ttnn.synchronize_device(dev)
        t0 = time.time()
        with ag.tape(), ag.no_grad():
            o = mod(xt, zt, None, None, None)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        d = hashlib.sha256(ttnn.to_torch(o.value if isinstance(o, ag.Tensor) else o).float().numpy().tobytes()
                           ).hexdigest()[:16]
        return dict(fwd_s=round(t1 - t0, 5), aiclk=clk_between(t0, t1), out_digest=d)

    res = dict(n=a.n, card=card, host=os.uname().nodename, sha=sha, board="p300c", roofs=dict(
        compute_tflops=115.685, dram_gbs=440, src="state/of3t/BACKWARD.md"), arms={})
    res["exact_ops"] = list(ag.exact_training_ops())  # production: () (exact_training is OFF by default)
    for _ in range(2):
        step()  # warm: program cache, JIT
    res["arms"]["wall"] = [step()[0] for _ in range(a.reps)]
    untaped()
    res["arms"]["untaped"] = [untaped() for _ in range(a.reps)]
    row, recs = step("drain")
    wall_f = min(r["fwd_s"] for r in res["arms"]["wall"])
    wall_b = min(r["bwd_s"] for r in res["arms"]["wall"])
    res["arms"]["drain"] = dict(step=row, fwd=table(recs["fwd"], wall_f), bwd=table(recs["bwd"], wall_b))
    res["arms"]["wall_aa"] = [step()[0] for _ in range(a.reps)]  # A/A: same arm after the instrument is removed
    from tt_bio import lnbw
    res["lnbw"] = dict(weights=lnbw.WEIGHTS, reach=dict(lnbw.REACH), calls=_seen)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["arms"]["wall"]), json.dumps(res["arms"]["wall_aa"]), flush=True)
    stop.set()
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
