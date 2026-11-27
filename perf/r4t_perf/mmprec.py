"""Precision of ttnn.matmul on bf16 inputs with an fp32 output, against the float64 product of the SAME bf16 values."""
import json, torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as T
dev = T.get_device()
gen = torch.Generator().manual_seed(0)
res = []
for m, n in ((256, 768), (768, 6144), (256, 32)):
    x = (torch.randn(4, m, n, generator=gen) / (m * n) ** 0.5).bfloat16()
    ref = x.double() @ x.double().transpose(-1, -2)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for fid in ("HiFi4", "HiFi3", "HiFi2"):
        for l1 in (True, False):
            ck = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                                         fp32_dest_acc_en=True, packer_l1_acc=l1)
            o = ttnn.to_torch(ttnn.matmul(xt, xt, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=ck)).double()
            e = float((o - ref).norm() / ref.norm())
            # fp32-accumulate CPU control: exact products, float32 sums
            res.append(dict(m=m, n=n, fid=fid, l1acc=l1, rel=e))
    cpu32 = float(((x.float() @ x.float().transpose(-1, -2)).double() - ref).norm() / ref.norm())
    res.append(dict(m=m, n=n, fid="cpu_fp32", rel=cpu32))
for r in res: print(r)
json.dump(res, open("/home/ttuser/r4t/perf/b1p5/mmprec.json", "w"), indent=1)
