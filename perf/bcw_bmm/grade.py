"""Float64 grade of the product the narrowing changes, and the plan at axes it must not touch.

At each axis n the dA product of the attention VJP, [2,8,n,n]^T @ [2,8,n,32] under the precise
config, is run three ways: `autograd.bmm` (the priced plan), the plan as it was before pricing
(in0_block_w = largest divisor of Kt up to 8, issued only where it fits), and ttnn's own plan.
Each is graded against float64 on the same bf16 inputs.
"""
import json, sys
import torch, ttnn
from tt_bio import tenstorrent as T, autograd as ag

dev = T.get_device()
cfg = ag.precise_config()
rows = []
for n in [int(x) for x in sys.argv[1].split(",")]:
    torch.manual_seed(n)
    a = ttnn.from_torch(torch.randn(2, 8, n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(2, 8, n, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = ttnn.to_torch(a).double().transpose(-1, -2) @ ttnn.to_torch(b).double()
    ag.BMM_NARROWED.clear()
    pc = ag.bmm_program_config(a, b, True, False, compute_kernel_config=cfg)
    Kt = n // 32
    w0 = max(d for d in range(1, 9) if Kt % d == 0)
    row = {"n": n, "Kt": Kt, "asked": w0, "given": pc.in0_block_w if pc else None,
           "narrowed": dict((str(k), v) for k, v in ag.BMM_NARROWED.items())}
    def grade(y):
        y = ttnn.to_torch(y).double()
        return {"rel_l2": float((y - ref).norm() / ref.norm()), "max_abs": float((y - ref).abs().max()),
                "finite": bool(torch.isfinite(y).all())}, y
    row["priced"], yp = grade(ag.bmm(a, b, True, False, compute_kernel_config=cfg))
    row["ttnn_own"], _ = grade(ag._matmul(a, b, transpose_a=True, compute_kernel_config=cfg))
    if pc is not None and pc.in0_block_w == w0:
        old = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), in0_block_w=w0,
            out_subblock_h=pc.out_subblock_h, out_subblock_w=pc.out_subblock_w,
            per_core_M=pc.per_core_M, per_core_N=pc.per_core_N)
        g, yo = grade(ag._matmul(a, b, transpose_a=True, program_config=old, compute_kernel_config=cfg))
        row["unpriced"] = g
        row["priced_equals_unpriced_bitwise"] = bool(torch.equal(yp, yo))
    else:
        row["unpriced"] = "refused: CBs exceed the core"
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
