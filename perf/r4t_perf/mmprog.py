"""r4t-perf: PairUpdate crop-256 matmul VJP, program choice (fidelity is not the lever, L3). dX = G W^T: stock
ag._matmul(transpose_b) vs pair_mm lever vs minimal_matmul / ttnn.linear on a pre-transposed W (W^T is weight-sized).
dW = X^T G: stock ag._matmul(transpose_a, fp32 out) vs explicit transpose of X then ttnn.matmul / minimal_matmul.
HiFi4 precise_config everywhere. ms = min of reps, device-synchronized; rel L2 vs float64."""
import argparse, json, os, subprocess, threading, time
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import tenstorrent as T, autograd as ag, pair_mm
    dev = T.get_device(); card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    clk, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            try: clk.append((time.time(), int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())))
            except Exception: pass
            time.sleep(0.005)
    threading.Thread(target=sampler, daemon=True).start()
    cfg = ag.precise_config(); K = 65536
    res = dict(host="qb2", chip=card, board="p300c", sha=sha, rows=[])
    def timeit(fn, ref):
        try:
            o = fn()
            if o is None: return dict(declined=True)
            ttnn.synchronize_device(dev); ts = []
            for _ in range(a.reps):
                t0 = time.time(); r = fn(); ttnn.synchronize_device(dev); t1 = time.time(); ts.append((t0, t1)); ttnn.deallocate(r)
            od = ttnn.to_torch(o).double().reshape(ref.shape); ttnn.deallocate(o)
            ck = [c for t, c in clk if any(t0 <= t <= t1 for t0, t1 in ts)]
            return dict(ms=round(1e3 * min(t1 - t0 for t0, t1 in ts), 3), rel=float((od - ref).norm() / ref.norm()),
                        aiclk=[min(ck), max(ck), len(ck)] if ck else None)
        except Exception as e:  # noqa: BLE001
            return dict(error=str(e)[:160])
    gen = torch.Generator().manual_seed(0)
    for p, q in ((1024, 384), (384, 384), (768, 384), (384, 768), (384, 256)):   # dX: g [K,p], W [q,p] -> [K,q]
        G = torch.randn(K, p, generator=gen).bfloat16(); Wt = (torch.randn(q, p, generator=gen) / p ** 0.5).bfloat16()
        ref = G.double() @ Wt.double().T
        dg = ttnn.from_torch(G, layout=ttnn.TILE_LAYOUT, device=dev); dw = ttnn.from_torch(Wt, layout=ttnn.TILE_LAYOUT, device=dev)
        dwT = ttnn.from_torch(Wt.T.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev)
        arms = dict(stock=lambda: ag._matmul(dg, dw, transpose_b=True, compute_kernel_config=cfg),
                    pair_mm=lambda: pair_mm.matmul(dg, dw, compute_kernel_config=cfg, transpose_b=True),
                    minimal_wT=lambda: ttnn.experimental.minimal_matmul(dg, dwT, compute_kernel_config=cfg, dtype=ttnn.bfloat16),
                    linear_wT=lambda: ttnn.linear(dg, dwT, compute_kernel_config=cfg),
                    matmul_wT=lambda: ttnn.matmul(dg, dwT, compute_kernel_config=cfg))
        for n, f in arms.items():
            row = dict(kind="dX", p=p, q=q, arm=n, **timeit(f, ref)); res["rows"].append(row); print(json.dumps(row), flush=True)
        for t in (dg, dw, dwT): ttnn.deallocate(t)
    for m, n in ((384, 1024), (256, 384), (384, 384), (768, 384), (384, 768)):   # dW: x [K,m], g [K,n] -> [m,n] fp32
        X = torch.randn(K, m, generator=gen).bfloat16(); G = torch.randn(K, n, generator=gen).bfloat16()
        ref = X.double().T @ G.double()
        dx = ttnn.from_torch(X, layout=ttnn.TILE_LAYOUT, device=dev); dg = ttnn.from_torch(G, layout=ttnn.TILE_LAYOUT, device=dev)
        def tr_mm():
            xt = ttnn.transpose(dx, -2, -1); o = ttnn.matmul(xt, dg, compute_kernel_config=cfg, dtype=ttnn.float32); ttnn.deallocate(xt); return o
        def tr_min():
            xt = ttnn.transpose(dx, -2, -1); o = ttnn.experimental.minimal_matmul(xt, dg, compute_kernel_config=cfg, dtype=ttnn.float32); ttnn.deallocate(xt); return o
        def tr_only():
            return ttnn.transpose(dx, -2, -1)
        arms = dict(stock=lambda: ag._matmul(dx, dg, transpose_a=True, compute_kernel_config=cfg, dtype=ttnn.float32),
                    transpose_then_matmul=tr_mm, transpose_then_minimal=tr_min)
        for nm, f in arms.items():
            row = dict(kind="dW", p=m, q=n, arm=nm, **timeit(f, ref)); res["rows"].append(row); print(json.dumps(row), flush=True)
        row = dict(kind="dW", p=m, q=n, arm="transpose_only", **timeit(tr_only, X.double().T)); res["rows"].append(row); print(json.dumps(row), flush=True)
        for t in (dx, dg): ttnn.deallocate(t)
    stop.set(); json.dump(res, open(a.out, "w"), indent=1)
if __name__ == "__main__":
    main()
