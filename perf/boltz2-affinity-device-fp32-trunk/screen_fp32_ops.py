#!/usr/bin/env python3
"""Screen: measured fp32 vs bf16 cost of every op class a composed fp32 pairformer
block would run on the p150a, at the affinity targets' real shape.

Both committed affinity targets pad to L=192 (fkbp12 L=141, dhfr L=187;
PAIRFORMER_PAD_MULTIPLE=64). d_z=128, d_s=384, tri-attn 4 heads x 32, s-attn 16 x 24.

Method (PLAYBOOKS measurement rules): each arm warmed 3x (JIT+program cache), timed
reps interleaved bf16/fp32, ttnn.synchronize_device immediately before and after each
timed call. Fidelity check: fp32 arm output vs torch fp32 reference (PCC) on one call,
to prove the device really computed fp32 (a silent bf16 downcast reads PCC ~0.99x,
fp32 reads > 0.99999).
"""
import json
import time

import torch
import ttnn

L = 192
DZ = 128
DS = 384
NH = 4
HD = 32
REPS = 8
INNER = 5  # calls per timed sample: amortizes launch latency, measures pipelined rate

device = ttnn.open_device(device_id=0)
_kernel_cls = (
    ttnn.types.WormholeComputeKernelConfig
    if device.arch() == ttnn.Arch.WORMHOLE_B0
    else ttnn.types.BlackholeComputeKernelConfig
)
CKC = _kernel_cls(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def mk(shape, dtype):
    t = torch.randn(*shape, dtype=torch.float32) / (shape[-1] ** 0.5)
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


def mkw(shape, dtype):
    t = torch.randn(*shape, dtype=torch.float32) / (shape[0] ** 0.5)
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


# op registry: (name, build_fn(dtype) -> callable, flops)
def lin(rows, cin, cout):
    def build(dt):
        a, w = mk((rows, cin), dt), mkw((cin, cout), dt)
        return lambda: ttnn.linear(a, w, compute_kernel_config=CKC)
    return build, 2 * rows * cin * cout


def bmm(b, m, k, n):
    def build(dt):
        a, w = mk((b, m, k), dt), mkw((b, k, n), dt)
        return lambda: ttnn.matmul(a, w, compute_kernel_config=CKC)
    return build, 2 * b * m * k * n


def lnorm(*shape):
    def build(dt):
        x = mk(shape, dt)
        w = ttnn.from_torch(torch.ones(shape[-1]), layout=ttnn.TILE_LAYOUT, device=device, dtype=dt)
        b = ttnn.from_torch(torch.zeros(shape[-1]), layout=ttnn.TILE_LAYOUT, device=device, dtype=dt)
        return lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=CKC)
    return build, 5 * torch.Size(shape).numel()


def smax(b, m, n):
    def build(dt):
        x = mk((b, m, n), dt)
        return lambda: ttnn.softmax(x, dim=-1, compute_kernel_config=CKC) if _softmax_ckc else ttnn.softmax(x, dim=-1)
    return build, 5 * b * m * n


OPS = [
    ("lin_z256  [36864,128]x[128,256]  trimul g/p_in", *lin(L * L, DZ, 2 * DZ), 4),
    ("lin_z128  [36864,128]x[128,128]  trimul out+triatt o/g", *lin(L * L, DZ, DZ), 8),
    ("lin_z512  [36864,128]x[128,512]  transition_z up", *lin(L * L, DZ, 4 * DZ), 1),
    ("lin_512z  [36864,512]x[512,128]  transition_z dn", *lin(L * L, 4 * DZ, DZ), 1),
    ("lin_qkv   [36864,128]x[128,384]  triatt qkv", *lin(L * L, DZ, 3 * NH * HD), 2),
    ("bmm_tri   [128,192,192]@[128,192,192]  trimul contract", *bmm(DZ, L, L, L), 2),
    ("bmm_qk    [768,192,32]@[768,32,192]  triatt QK^T", *bmm(NH * L, L, HD, L), 2),
    ("bmm_av    [768,192,192]@[768,192,32]  triatt AV", *bmm(NH * L, L, L, HD), 2),
    ("softmax   [768,192,192]  triatt", *smax(NH * L, L, L), 2),
    ("layernorm [1,192,192,128]  z norms", *lnorm(1, L, L, DZ), 7),
    ("lin_s1536 [192,384]x[384,1536]  transition_s up", *lin(L, DS, 4 * DS), 1),
    ("lin_s384  [192,1536]x[1536,384]  transition_s dn", *lin(L, 4 * DS, DS), 1),
    ("lin_sqkv  [192,384]x[384,1152]  s-attn qkv", *lin(L, DS, 3 * DS), 1),
    ("lin_zbias [36864,128]x[128,16]  s-attn pair bias", *lin(L * L, DZ, 16), 1),
]

# softmax compute_kernel_config kwarg support varies; probe once.
_softmax_ckc = True
try:
    _x = mk((32, 32, 32), ttnn.bfloat16)
    ttnn.softmax(_x, dim=-1, compute_kernel_config=CKC)
except TypeError:
    _softmax_ckc = False

results = {}
for name, build, flops, per_block in OPS:
    arms = {}
    for dt, tag in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
        fn = build(dt)
        for _ in range(3):
            out = fn()
        ttnn.synchronize_device(device)
        arms[tag] = {"fn": fn, "times": [], "out": out}
    for _ in range(REPS):
        for tag in ("bf16", "fp32"):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(INNER):
                arms[tag]["out"] = arms[tag]["fn"]()
            ttnn.synchronize_device(device)
            arms[tag]["times"].append((time.perf_counter() - t0) / INNER)
    row = {}
    for tag in ("bf16", "fp32"):
        ts = sorted(arms[tag]["times"])
        med = ts[len(ts) // 2]
        row[tag + "_ms"] = round(med * 1e3, 4)
        row[tag + "_tflops"] = round(flops / med / 1e12, 3)
    # fidelity probe on the fp32 arm output (matmul/linear rows only)
    row["per_block"] = per_block
    results[name] = row
    print(f"{name}: bf16 {row['bf16_ms']:.3f} ms ({row['bf16_tflops']} TF/s) | "
          f"fp32 {row['fp32_ms']:.3f} ms ({row['fp32_tflops']} TF/s) | x{per_block}/block",
          flush=True)

# fidelity: one fp32 linear vs torch fp32
a_t = torch.randn(1024, DZ) / (DZ ** 0.5)
w_t = torch.randn(DZ, DZ) / (DZ ** 0.5)
ref = a_t @ w_t
a_d = ttnn.from_torch(a_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
w_d = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
out = ttnn.to_torch(ttnn.linear(a_d, w_d, compute_kernel_config=CKC))
diff = (out - ref).abs()
pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
print(f"fidelity fp32 linear vs torch fp32: maxabs {diff.max():.3e} "
      f"meanabs {diff.mean():.3e} PCC {pcc:.8f}", flush=True)
a_d16 = ttnn.from_torch(a_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
w_d16 = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
out16 = ttnn.to_torch(ttnn.linear(a_d16, w_d16, compute_kernel_config=CKC)).float()
diff16 = (out16 - ref).abs()
pcc16 = torch.corrcoef(torch.stack([out16.flatten(), ref.flatten()]))[0, 1].item()
print(f"fidelity bf16 linear vs torch fp32: maxabs {diff16.max():.3e} "
      f"meanabs {diff16.mean():.3e} PCC {pcc16:.8f}", flush=True)

# per-block cost model (fp32 and bf16), then x 64 blocks x 6 recycles
block_ms = {"bf16": 0.0, "fp32": 0.0}
for name, build, flops, per_block in OPS:
    r = results[name]
    block_ms["bf16"] += r["bf16_ms"] * per_block
    block_ms["fp32"] += r["fp32_ms"] * per_block
print(f"per-block device-busy: bf16 {block_ms['bf16']:.2f} ms | fp32 {block_ms['fp32']:.2f} ms",
      flush=True)
print(f"trunk projection (x64 blocks x6 recycles, device-busy only, no dispatch/glue): "
      f"bf16 {block_ms['bf16'] * 384 / 1e3:.2f} s | fp32 {block_ms['fp32'] * 384 / 1e3:.2f} s",
      flush=True)

with open("screen_fp32_ops.json", "w") as f:
    json.dump({"L": L, "ops": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
               "block_ms": block_ms}, f, indent=2)
ttnn.close_device(device)
