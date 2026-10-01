#!/usr/bin/env python3
"""C1 at the op: the pair transition ReLU, composed (as the taped recompute runs it) against
fused. Forward: linear then relu vs linear(activation="relu"). Backward: ttnn.relu_bw vs one
multiply gating on the output (b activation GTZ). Equality first, then synced walls, arms
alternated, AICLK sampled during."""
import json, pathlib, sys, time
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
    torch.manual_seed(0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x = up(torch.randn(1, 288, 288, 128))
    w = up(torch.randn(128, 512) * 0.09)
    b = up(torch.randn(512) * 0.1)
    g = up(torch.randn(1, 288, 288, 512) * 1e-3)
    cfg = ag.precise_config() if hasattr(ag, "precise_config") else None
    kw = dict(compute_kernel_config=cfg) if cfg is not None else {}
    lin = lambda act: ttnn.linear(x, w, bias=b, activation=act, **kw)
    arms = {
        "fwd_composed": lambda: ttnn.relu(lin(None)),
        "fwd_fused": lambda: lin("relu"),
    }
    h = arms["fwd_fused"]()
    hc = arms["fwd_composed"]()
    res = {"fwd_equal": bool(torch.equal(ttnn.to_torch(h), ttnn.to_torch(hc)))}
    arms["bwd_relu_bw"] = lambda: ttnn.relu_bw(g, h)[0]
    arms["bwd_gtz_mul"] = lambda: ttnn.multiply(g, h, input_tensor_b_activations=[ttnn.UnaryOpType.GTZ])
    a1, a2 = ttnn.to_torch(arms["bwd_relu_bw"]()), ttnn.to_torch(arms["bwd_gtz_mul"]())
    ref = ttnn.to_torch(g).double() * (ttnn.to_torch(h).double() > 0)
    res["bwd_equal"] = bool(torch.equal(a1, a2))
    res["bwd_maxabs_vs_f64"] = [float((a1.double() - ref).abs().max()), float((a2.double() - ref).abs().max())]
    t = {k: [] for k in arms}
    clks = []
    for rep in range(30):
        order = list(arms) if rep % 2 == 0 else list(reversed(arms))
        for k in order:
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(10):
                y = arms[k]()
                ttnn.deallocate(y)
            ttnn.synchronize_device(dev); t[k].append((time.perf_counter() - t0) / 10 * 1e3)
            clks.append(int(clk.read_text().split()[0]) if clk.exists() else -1)
    import statistics as st
    res["ms_median"] = {k: round(st.median(v), 4) for k, v in t.items()}
    res["ms_min"] = {k: round(min(v), 4) for k, v in t.items()}
    res["aiclk"] = {"median": st.median(clks), "min": min(clks), "n": len(clks)}
    print(json.dumps(res, indent=1))
    pathlib.Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
