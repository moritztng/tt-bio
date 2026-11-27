"""r4t-perf L2: LayerNorm backward with dx, dgamma, dbeta from one kernel (tt_bio.lnbw.layer_norm_bw_weights) vs the
composed path (autograd._layer_norm_bw), graded against float64. Bar pre-registered in state/r4t/rows/perf.md.
"""
import argparse, hashlib, json, os, subprocess, threading, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,256,256,256")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import tenstorrent as T, autograd as ag, lnbw
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
            time.sleep(0.01)
    threading.Thread(target=sampler, daemon=True).start()

    shp = [int(v) for v in a.shape.split(",")]
    K, eps = shp[-1], 1e-5
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(*shp, generator=gen) * 2 + 0.5).bfloat16()
    g = torch.randn(*shp, generator=gen).bfloat16()
    gm = (1 + 0.3 * torch.randn(K, generator=gen)).bfloat16()
    bt = (0.1 * torch.randn(K, generator=gen)).bfloat16()
    # float64 reference on the same rounded inputs
    xd = x.double().requires_grad_(); gd = gm.double().requires_grad_(); bd = bt.double().requires_grad_()
    y = torch.nn.functional.layer_norm(xd, (K,), gd, bd, eps)
    y.backward(g.double())
    ref = dict(dx=xd.grad, dgamma=gd.grad, dbeta=bd.grad)

    put = lambda t: ttnn.from_torch(t.reshape(1, -1) if t.dim() == 1 else t, dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
    xt, gt, gmt, btt = put(x), put(g), put(gm), put(bt)

    def run(weights):
        lnbw.WEIGHTS = weights
        X = ag.Tensor(xt, requires_grad=True); G = ag.Tensor(gmt, requires_grad=True); Bt = ag.Tensor(btt, requires_grad=True)
        ag._layer_norm_bw(X, G, Bt, eps, None)(gt)
        return X, G, Bt

    def grads(weights):
        X, G, Bt = run(weights)
        out = {}
        for k, t in (("dx", X), ("dgamma", G), ("dbeta", Bt)):
            out[k] = ttnn.to_torch(t.grad).double().reshape(ref[k].shape)
        return out

    def timed(weights):
        for _ in range(3):
            run(weights)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.reps):
            t0 = time.time(); X, G, Bt = run(weights); ttnn.synchronize_device(dev); ts.append((t0, time.time()))
            del X, G, Bt
        ck = [c for t, c in clk if any(t0 <= t <= t1 for t0, t1 in ts)]
        d = [t1 - t0 for t0, t1 in ts]
        return dict(s_min=round(min(d), 6), s_med=round(sorted(d)[len(d) // 2], 6),
                    aiclk=[min(ck), max(ck), len(ck)] if ck else None)

    res = dict(shape=shp, host="qb2", chip=card, board="p300c", sha=sha, broken=a.broken, arms={})
    for name, w in (("composed", False), ("fused", True)):
        gr = grads(w)
        if a.broken and w:
            gr["dgamma"] = gr["dgamma"] * 1.10
        m = {}
        for k in ref:
            r, o = ref[k].flatten(), gr[k].flatten()
            m[k] = dict(rel_l2=float((o - r).norm() / r.norm()), cos=float(torch.dot(o, r) / (o.norm() * r.norm())))
        h = hashlib.sha256(b"".join(gr[k].float().numpy().tobytes() for k in ref)).hexdigest()[:16]
        res["arms"][name] = dict(metrics=m, digest=h, time=timed(w))
    stop.set()
    res["reach"] = dict(lnbw.REACH)
    ok = True
    for k in ref:
        f, c = res["arms"]["fused"]["metrics"][k], res["arms"]["composed"]["metrics"][k]
        ok &= f["rel_l2"] <= 5e-2 and f["cos"] >= 0.9999 and f["rel_l2"] <= max(1.5 * c["rel_l2"], 1e-3)
    res["L2_accuracy"] = "PASS" if ok else "FAIL"
    tc, tf = res["arms"]["composed"]["time"]["s_min"], res["arms"]["fused"]["time"]["s_min"]
    res["speedup"] = round(tc / tf, 3)
    res["arms"]["fused"]["served"] = res["reach"].get("served: weights", 0) > 0
    if not res["arms"]["fused"]["served"]:
        res["L2_accuracy"] = "FAIL (kernel never served: the composed path was graded twice)"
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
