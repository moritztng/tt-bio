"""r4t-perf: dW = X^T G at PairUpdate crop-256 shapes (K = 65536 tokens): current tt_bio path (ttnn.matmul transpose_a,
fp32 out; _split_k declines because the output exceeds SPLIT_K_OUT_TILES=32) vs _split_k forced at P chunks.
Time (device-synchronized, warm, min of reps) and rel L2 vs float64 per arm."""
import argparse, json, os, subprocess, sys, threading, time
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
            time.sleep(0.02)
    threading.Thread(target=sampler, daemon=True).start()
    cfg = ag.precise_config()
    shapes = [(65536, 384, 1024), (65536, 256, 384), (65536, 384, 384), (65536, 768, 384), (65536, 384, 768)]
    res = dict(host="qb2", chip=card, board="p300c", sha=sha, rows=[])
    gen = torch.Generator().manual_seed(0)
    for K, M, N in shapes:
        x = torch.randn(K, M, generator=gen).bfloat16(); g = torch.randn(K, N, generator=gen).bfloat16()
        ref = x.double().T @ g.double()
        dx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev); dg = ttnn.from_torch(g, layout=ttnn.TILE_LAYOUT, device=dev)
        arms = [("current", None)] + [(f"splitk{P}", P) for P in (8, 16, 32, 64)]
        for name, P in arms:
            def run():
                if P is None:
                    old = ag.SPLIT_K_OUT_TILES; ag.SPLIT_K_OUT_TILES = 0  # force decline = what production does here
                else:
                    old = (ag.SPLIT_K_OUT_TILES, ag.SPLIT_K_CHUNKS); ag.SPLIT_K_OUT_TILES = 10**9; ag.SPLIT_K_CHUNKS = P
                try:
                    return ag._matmul(dx, dg, transpose_a=True, compute_kernel_config=cfg, dtype=ttnn.float32)
                finally:
                    if P is None: ag.SPLIT_K_OUT_TILES = old
                    else: ag.SPLIT_K_OUT_TILES, ag.SPLIT_K_CHUNKS = old
            try:
                o = run(); ttnn.synchronize_device(dev)
                ts = []
                for _ in range(a.reps):
                    t0 = time.time(); r = run(); ttnn.synchronize_device(dev); t1 = time.time(); ts.append((t0, t1)); ttnn.deallocate(r)
                od = ttnn.to_torch(o).double(); ttnn.deallocate(o)
                rel = float((od - ref).norm() / ref.norm())
                ck = [c for t, c in clk if any(t0 <= t <= t1 for t0, t1 in ts)]
                row = dict(K=K, M=M, N=N, arm=name, ms=round(1e3 * min(t1 - t0 for t0, t1 in ts), 3), rel=rel,
                           aiclk=[min(ck), max(ck), len(ck)] if ck else None)
            except Exception as e:  # noqa: BLE001
                row = dict(K=K, M=M, N=N, arm=name, error=str(e)[:200])
            res["rows"].append(row); print(json.dumps(row), flush=True)
        ttnn.deallocate(dx); ttnn.deallocate(dg)
    stop.set(); res["split_k_served"] = ag.SPLIT_K_STATS
    json.dump(res, open(a.out, "w"), indent=1)
if __name__ == "__main__":
    main()
