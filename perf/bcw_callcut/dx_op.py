#!/usr/bin/env python3
"""The 2-D dX products of the Evoformer backward, [82944,K] @ W[128,K]^T, as the tape issues
them (transpose_b, precise config, the default plan) against the same product on a
pre-transposed weight. Synced walls, arms alternated, AICLK during."""
import json, pathlib, statistics as st, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    card, out = sys.argv[1], sys.argv[2]
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as ag
    dev = get_device()
    clk = pathlib.Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ag.precise_config()
    res, clks = {}, []
    for K in (128, 384, 512):
        torch.manual_seed(K)
        g = up(torch.randn(82944, K) * 1e-2)
        w = up(torch.randn(128, K) * 0.05)
        wt = up(torch.randn(128, K).t().contiguous() * 0.05)
        arms = {"tb": lambda: ttnn.matmul(g, w, transpose_b=True, compute_kernel_config=cfg),
                "pre_t": lambda: ttnn.matmul(g, wt, compute_kernel_config=cfg),
                "tb_default_cfg": lambda: ttnn.matmul(g, w, transpose_b=True)}
        t = {k: [] for k in arms}
        for rep in range(12):
            for k in (list(arms) if rep % 2 == 0 else list(reversed(arms))):
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(10):
                    ttnn.deallocate(arms[k]())
                ttnn.synchronize_device(dev); t[k].append((time.perf_counter() - t0) / 10 * 1e3)
                clks.append(int(clk.read_text().split()[0]))
        res[f"K={K}"] = {k: round(st.median(v), 4) for k, v in t.items()}
        print(K, res[f"K={K}"], flush=True)
    res["aiclk"] = {"median": st.median(clks), "min": min(clks), "n": len(clks)}
    pathlib.Path(out).write_text(json.dumps(res, indent=1)); print(res["aiclk"])


if __name__ == "__main__":
    main()
