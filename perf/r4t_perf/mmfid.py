"""r4t-perf L3: matmul VJP at PairUpdate crop-256 shapes (dW = X^T G, K = 65536, fp32 out; dX = G W^T, bf16 out) per
math fidelity (HiFi4 = production precise_config, HiFi3, HiFi2, LoFi = the broken arm). fp32 dest acc + packer_l1_acc
as production. Min-of-reps device-synchronized ms, rel L2 and cos vs float64. Bar: rows/perf.md L3 (pre-registered)."""
import argparse, json, os, subprocess, threading, time
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    dev = T.get_device(); card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    clk, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            try: clk.append((time.time(), int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())))
            except Exception: pass
            time.sleep(0.005)
    threading.Thread(target=sampler, daemon=True).start()
    cfgs = {f: ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, f), math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=True)
            for f in ("HiFi4", "HiFi3", "HiFi2", "LoFi")}
    K = 65536
    cases = [("dW", K, m, n) for m, n in ((384, 1024), (256, 384), (384, 384), (768, 384), (384, 768))] + \
            [("dX", K, n, m) for m, n in ((384, 1024), (384, 384), (384, 768), (768, 384), (256, 384))]
    res = dict(host="qb2", chip=card, board="p300c", sha=sha, rows=[])
    gen = torch.Generator().manual_seed(0)
    for kind, T_, p, q in cases:
        if kind == "dW":   # x [T, p], g [T, q] -> [p, q] fp32
            A = torch.randn(T_, p, generator=gen).bfloat16(); B = torch.randn(T_, q, generator=gen).bfloat16()
            ref = A.double().T @ B.double(); kw = dict(transpose_a=True, dtype=ttnn.float32)
        else:              # g [T, p], w [q, p] -> [T, q] bf16
            A = torch.randn(T_, p, generator=gen).bfloat16(); B = (torch.randn(q, p, generator=gen) / p ** 0.5).bfloat16()
            ref = A.double() @ B.double().T; kw = dict(transpose_b=True)
        da = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev); db = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
        for f, cfg in cfgs.items():
            run = lambda: ag._matmul(da, db, compute_kernel_config=cfg, **kw)
            o = run(); ttnn.synchronize_device(dev); ts = []
            for _ in range(a.reps):
                t0 = time.time(); r = run(); ttnn.synchronize_device(dev); t1 = time.time(); ts.append((t0, t1)); ttnn.deallocate(r)
            od = ttnn.to_torch(o).double(); ttnn.deallocate(o)
            rel = float((od - ref).norm() / ref.norm())
            cos = float((od.flatten() @ ref.flatten()) / (od.norm() * ref.norm()))
            ck = [c for t, c in clk if any(t0 <= t <= t1 for t0, t1 in ts)]
            row = dict(kind=kind, T=T_, p=p, q=q, fid=f, ms=round(1e3 * min(t1 - t0 for t0, t1 in ts), 3), rel=rel, cos=cos,
                       aiclk=[min(ck), max(ck), len(ck)] if ck else None)
            res["rows"].append(row); print(json.dumps(row), flush=True)
        ttnn.deallocate(da); ttnn.deallocate(db)
    stop.set()
    ok = lambda r: r["rel"] <= 1e-3 and r["cos"] >= 0.99999
    res["verdict_a"] = {f: all(ok(r) for r in res["rows"] if r["fid"] == f) for f in cfgs}
    res["ms_sum"] = {f: round(sum(r["ms"] for r in res["rows"] if r["fid"] == f), 3) for f in cfgs}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(dict(verdict_a=res["verdict_a"], ms_sum=res["ms_sum"])))
if __name__ == "__main__":
    main()
