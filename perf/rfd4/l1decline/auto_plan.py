"""ttnn's own plan for the trimul gate projection at widths 16 and 32, read before any program build.

    TT_VISIBLE_DEVICES=0 python perf/rfd4/l1decline/auto_plan.py
"""
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402

dev = tt.get_device()
print("grid", dev.compute_with_storage_grid_size(), "CORE_GRID_MAIN", tt.CORE_GRID_MAIN, flush=True)
w = ttnn.from_torch(torch.randn(384, 384), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
for b in [int(a) for a in sys.argv[1:]] or [16, 32]:
    x = ttnn.from_torch(torch.randn(b, 192, 192, 384), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    p = ttnn._ttnn.operations.matmul.MatmulParams()
    p.output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    p.output_dtype = ttnn.bfloat16
    p.compute_kernel_config = ckc
    try:
        q = ttnn._ttnn.operations.matmul.create_matmul_attributes(x, w, p, [None])
        print(b, "PLAN", q.program_config, flush=True)
    except Exception as e:
        print(b, "PLAN-ERR", str(e).splitlines()[:3], flush=True)
    ttnn.deallocate(x)
