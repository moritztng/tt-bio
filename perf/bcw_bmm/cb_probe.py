"""Read the circular-buffer bytes the batched reuse matmul asks for, off the device's own refusal.

Each case issues one `autograd._matmul` under an explicit MatmulMultiCoreReuseProgramConfig
with the whole output matrix per block (the only region `bmm_program_config` plans) and records
whether it ran and, if refused, the CB end the refusal quotes. A case is
[B, H, M, K, N, transpose_a, transpose_b, in0_block_w, a_dtype, b_dtype, cfg, out_dtype] with
dtypes "bf16"/"f32", cfg "precise" (autograd.precise_config) or "none", out_dtype "bf16"/"f32"/"-".
"""
import json, re, sys
import torch, ttnn
from tt_bio import tenstorrent as T, autograd as ag

DT = {"bf16": ttnn.bfloat16, "f32": ttnn.float32}
dev = T.get_device()
out = {"unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
       "bank": int(ttnn.get_memory_view(dev, ttnn.BufferType.L1).total_bytes_per_bank),
       "arch": str(dev.arch()), "rows": []}
RE = re.compile(r"grow to (\d+) B which is beyond max L1 size of (\d+) B")
for B, H, M, K, N, ta, tb, w, da, db, cfg, do in json.loads(sys.argv[1]):
    a = ttnn.from_torch(torch.randn([B, H, K, M] if ta else [B, H, M, K]), dtype=DT[da],
                        layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn([B, H, N, K] if tb else [B, H, K, N]), dtype=DT[db],
                        layout=ttnn.TILE_LAYOUT, device=dev)
    Mt, Nt = M // 32, N // 32
    sw = max(d for d in range(1, min(Nt, 4) + 1) if Nt % d == 0)
    sh = max(d for d in range(1, min(Mt, max(1, 4 // sw)) + 1) if Mt % d == 0)
    kw = {"program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), in0_block_w=w,
        out_subblock_h=sh, out_subblock_w=sw, per_core_M=Mt, per_core_N=Nt)}
    if cfg == "precise":
        kw["compute_kernel_config"] = ag.precise_config()
    if do != "-":
        kw["dtype"] = DT[do]
    row = dict(zip("B H M K N ta tb w da db cfg do".split(), (B, H, M, K, N, ta, tb, w, da, db, cfg, do)))
    try:
        y = ag._matmul(a, b, transpose_a=bool(ta), transpose_b=bool(tb), **kw)
        ttnn.synchronize_device(dev)
        row["ok"] = True
        ttnn.deallocate(y)
    except Exception as e:
        m = RE.search(str(e))
        row["ok"] = False
        row["cb_end"] = int(m.group(1)) if m else None
        if not m:
            row["err"] = str(e)[:300]
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    print(json.dumps(row), flush=True)
    out["rows"].append(row)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("unreserved", out["unreserved"], "bank", out["bank"], out["arch"])
