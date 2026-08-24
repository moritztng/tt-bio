"""Measure the card's achievable roof for the two kernels X-Cell is made of, then place the model."""
import time, torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN
dev = get_device()
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
tt = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

def bench(fn, n=5):
    fn(); ttnn.synchronize_device(dev)
    t0=time.perf_counter()
    for _ in range(n): fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter()-t0)/n

print("=== matmul roof, the projection kernel (ttnn.linear, CORE_GRID_MAIN) ===")
print(f"{'M':>7} {'K':>6} {'N':>6} {'ms':>8} {'TFLOP/s':>9}")
best_mm = 0.0
for M,K,N in [(4096,512,512),(8192,512,512),(32768,512,512),(4096,2048,2048),(4096,4096,4096)]:
    a=tt(torch.randn(1,M,K)); b=tt(torch.randn(K,N))
    s=bench(lambda: ttnn.linear(a,b,compute_kernel_config=ck,dtype=ttnn.bfloat16,
                                core_grid=CORE_GRID_MAIN))
    tf=2*M*K*N/s/1e12; best_mm=max(best_mm,tf)
    print(f"{M:>7} {K:>6} {N:>6} {s*1e3:>8.2f} {tf:>9.2f}")

print("\n=== SDPA roof at X-Cell's own attention shape (H=8, d_head=64) ===")
print(f"{'rows':>5} {'S':>6} {'ms':>8} {'TFLOP/s':>9}")
best_sdpa=0.0
for Nr,S in [(1,513),(1,2049),(1,4001),(8,4001)]:
    q=tt(torch.randn(Nr,8,S,64)); k=tt(torch.randn(Nr,8,S,64)); v=tt(torch.randn(Nr,8,S,64))
    s=bench(lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,
                                                                  scale=64**-0.5),n=3)
    f=Nr*8*(2*S*S*64*2)   # scores + AV
    tf=f/s/1e12; best_sdpa=max(best_sdpa,tf)
    print(f"{Nr:>5} {S:>6} {s*1e3:>8.2f} {tf:>9.2f}")

print(f"\nMEASURED roofs on this card: matmul {best_mm:.1f} TFLOP/s, "
      f"SDPA(H=8,dh=64) {best_sdpa:.1f} TFLOP/s")
print(f"X-Cell Mini measured 8.0-8.5 TFLOP/s end to end.")
print(f"  vs matmul roof: {8.46/best_mm*100:.0f}%   vs SDPA roof: {8.46/best_sdpa*100:.0f}%")
ttnn.close_device(dev)
