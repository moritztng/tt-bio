"""The failing call alone: ttnn.linear((B,192,192,384) bfp8, (384,384) bf16, core_grid=13x10), DRAM out.

    TT_VISIBLE_DEVICES=0 python perf/rfd4/l1decline/standalone.py 16 32
"""
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402

dev = tt.get_device()
w = ttnn.from_torch(torch.randn(384, 384), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for b in [int(a) for a in sys.argv[1:]]:
    x = ttnn.from_torch(torch.randn(b, 192, 192, 384), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    for fp32 in (True, False):
        for l1acc in (True, False):
            ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                         fp32_dest_acc_en=fp32, packer_l1_acc=l1acc)
            try:
                y = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc, core_grid=tt.CORE_GRID_MAIN)
                ttnn.deallocate(y)
                print(b, fp32, l1acc, "OK", flush=True)
            except Exception as e:
                print(b, fp32, l1acc, "REFUSED", str(e).splitlines()[2][:200], flush=True)
    ttnn.deallocate(x)
