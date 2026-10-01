"""pair_mm with relu at pack vs minimal_matmul + relu vs ttnn.linear(activation=relu): equality and time."""
import json, time, sys
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import pair_mm, tenstorrent
pair_mm.PAIR_MM_FUSED = True
dev = tenstorrent.get_device()
torch.manual_seed(0)
res = []
for (M, K, N) in [(288, 128, 512), (256, 128, 512), (512, 128, 512)]:
    xt = torch.randn(1, M, M, K); wt = torch.randn(K, N) / K**0.5; bt = torch.randn(N) * 0.1
    x = ttnn.from_torch(xt, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(wt, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(bt.reshape(1, N), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    arms = {
        "fused": lambda: pair_mm.matmul(x, w, b, cfg, activation="relu"),
        "mm_then_relu": lambda: ttnn.relu(pair_mm.matmul(x, w, b, cfg)),
        "ttnn_linear_relu": lambda: ttnn.linear(x, w, bias=b, activation="relu", compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=10, x=11)),
    }
    outs = {k: ttnn.to_torch(f()) for k, f in arms.items()}
    ms = {}
    for rep in range(2):
        for k, f in arms.items():
            ttnn.synchronize_device(dev); t = time.perf_counter()
            for _ in range(20): o = f(); ttnn.deallocate(o)
            ttnn.synchronize_device(dev); ms[k] = (time.perf_counter() - t) / 20 * 1e3
    ref = torch.relu(xt.double() @ wt.double() + bt.double())
    rel = {k: float((v.double() - ref).norm() / ref.norm()) for k, v in outs.items()}
    r = dict(shape=[M, K, N], equal_fused_vs_mm_relu=bool(torch.equal(outs["fused"], outs["mm_then_relu"])), ms=ms, rel_l2_vs_f64=rel)
    print(json.dumps(r)); res.append(r)
json.dump(res, open("perf/bcp_evo/out/relu_mm_bench.json", "w"), indent=1)
