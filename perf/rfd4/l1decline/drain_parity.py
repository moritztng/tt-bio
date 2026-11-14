"""Only the drain block moves: ttnn's core_grid plan against explicit 1D plans with ttnn's in0_block_w (1)
and other drain blocks, on the trimul gate projection's operands. Bit-equality expected.

    TT_VISIBLE_DEVICES=0 python perf/rfd4/l1decline/drain_parity.py 4 8
"""
import json
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402

dev = tt.get_device()
gx, gy = tt.COMPUTE_GRID_MAIN
torch.manual_seed(0)
w = ttnn.from_torch(torch.randn(384, 384) / 20, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
report = []
for b in [int(a) for a in sys.argv[1:]]:
    x = ttnn.from_torch(torch.randn(b, 192, 192, 384), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    pm = -(-b * 192 * 6 // (gx * gy))
    try:
        ref = ttnn.to_torch(ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                        compute_kernel_config=ckc, core_grid=tt.CORE_GRID_MAIN))
    except Exception as e:
        report.append({"b": b, "ttnn_plan": "refused", "why": str(e).splitlines()[2][:160]})
        ttnn.deallocate(x)
        continue
    picked = tt._core_grid_1d_plan(x, w, ckc, ttnn.bfloat16)
    drains = [(h, v) for h in (1, 2, 3, 4, 6, 9, pm) for v in (1, 3, 4, 6) if pm % h == 0 and 12 % v == 0]
    for h, v in drains:
        sh = max(s for s in range(min(4, h), 0, -1) if h % s == 0)
        sw = max(s for s in range(min(4 // sh, v), 0, -1) if v % s == 0)
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=1, out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=h, out_block_w=v, per_core_M=pm, per_core_N=12, fuse_batch=True, mcast_in0=False)
        try:
            y = ttnn.to_torch(ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                          compute_kernel_config=ckc, program_config=cfg))
            report.append({"b": b, "drain": [h, v], "equal": bool(torch.equal(y, ref))})
        except Exception as e:
            report.append({"b": b, "drain": [h, v], "refused": str(e).splitlines()[2][:120]})
        print(report[-1], flush=True)
    report.append({"b": b, "gate_on_fitting_plan": None if picked is None else "re-blocked"})
    ttnn.deallocate(x)
print(json.dumps(report))
