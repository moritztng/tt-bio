#!/usr/bin/env python3
"""C4 at the op: `AF2MaskedOuterProductMean._sum_rows` on the real block's OPM weights, rows summed
after S products (shipped) against rows joined along the contraction (`rows_in_k`). Both graded
against float64 `sum_s a_sic b_sjd W_cdk + bias` on the same bf16 operands; synced walls with the
arms alternated; AICLK during. Device programs per call from ttnn.graph capture."""
import json, pathlib, statistics as st, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    card, out = sys.argv[1], sys.argv[2]
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    dev = tr.device
    opm = tr.model.device_evoformer[0].opm
    clk = pathlib.Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    torch.manual_seed(0)
    S, n = 2, 288
    C = int(opm.a_weight.shape[-1]); D = int(opm.b_weight.shape[-1])
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    a = up(torch.randn(S, n, C)); b = up(torch.randn(S, n, D))
    A64, B64 = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
    W = ttnn.to_torch(opm.o_weight).double().reshape(C, D, -1)
    bias = ttnn.to_torch(opm.o_bias).double().reshape(-1)
    ref = torch.einsum("sic,sjd,cdk->ijk", A64, B64, W) + bias
    res = {"C": C, "D": D, "c_z": int(W.shape[-1])}
    arms = {}
    for k, v in (("shipped", False), ("rows_in_k", True)):
        def f(v=v):
            type(opm).rows_in_k = v
            return opm._sum_rows(a, b)
        arms[k] = f
        y = ttnn.to_torch(f()).double().reshape(ref.shape)
        res[f"rel_l2_vs_f64_{k}"] = float((y - ref).norm() / ref.norm())
        res[f"maxabs_vs_f64_{k}"] = float((y - ref).abs().max())
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        f(); g = ttnn.graph.end_graph_capture()
        res[f"programs_{k}"] = sum(1 for nd in g if nd.get("node_type") == "function_start"
                                   and "DeviceOperation" in nd.get("params", {}).get("name", ""))
    t = {k: [] for k in arms}; clks = []
    for rep in range(20):
        for k in (list(arms) if rep % 2 == 0 else list(reversed(arms))):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(10):
                ttnn.deallocate(arms[k]())
            ttnn.synchronize_device(dev); t[k].append((time.perf_counter() - t0) / 10 * 1e3)
            clks.append(int(clk.read_text().split()[0]))
    res["ms_median"] = {k: round(st.median(v), 4) for k, v in t.items()}
    res["aiclk"] = {"median": st.median(clks), "min": min(clks), "n": len(clks)}
    print(json.dumps(res, indent=1))
    pathlib.Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
