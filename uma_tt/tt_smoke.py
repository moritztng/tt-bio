"""ttnn smoke test: open device, do a tall-skinny GEMM like the SO2 conv workload,
check PCC vs torch and measure throughput."""
import os, time, numpy as np, torch
import ttnn

def pcc(a,b):
    a=a.flatten().float(); b=b.flatten().float()
    return torch.corrcoef(torch.stack([a,b]))[0,1].item()

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        E, IN, OUT = 8424, 768, 640
        torch.manual_seed(0)
        x = torch.randn(E, IN)
        w = torch.randn(OUT, IN) * 0.02
        b = torch.randn(OUT) * 0.01
        ref = x @ w.T + b

        kcfg = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)

        def pad32(n): return ((n+31)//32)*32
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        wt = ttnn.from_torch(w.T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        bt = ttnn.from_torch(b.reshape(1,-1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        yt = ttnn.linear(xt, wt, bias=bt, compute_kernel_config=kcfg)
        y = ttnn.to_torch(yt)
        print(f"GEMM [{E},{IN}]@[{IN},{OUT}] PCC={pcc(ref,y):.5f}")

        # perf: warm then time device-resident matmul
        for _ in range(3):
            yt = ttnn.linear(xt, wt, bias=bt, compute_kernel_config=kcfg); ttnn.synchronize_device(dev)
        N=50; t0=time.time()
        for _ in range(N):
            yt = ttnn.linear(xt, wt, bias=bt, compute_kernel_config=kcfg)
        ttnn.synchronize_device(dev); dt=(time.time()-t0)/N
        print(f"ttnn GEMM warm: {dt*1000:.3f} ms/call  ({E*IN*OUT*2/dt/1e9:.1f} GFLOP/s)")

        # cpu ref timing
        t0=time.time()
        for _ in range(N): _ = x@w.T+b
        cdt=(time.time()-t0)/N
        print(f"torch CPU GEMM: {cdt*1000:.3f} ms/call")
    finally:
        ttnn.close_device(dev)

if __name__=="__main__":
    main()
