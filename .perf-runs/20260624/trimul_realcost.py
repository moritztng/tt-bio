"""Measure a REAL TriangleMultiplication.__call__ synced on-device time at L=512
--fast, to see how much of a trimul call is parallelizable compute (the part TP
could shard /4) vs fixed overhead. Decides whether full-trimul channel-TP could
net-win against the ~1.2ms all_gather of the pair output."""
import os, time
import torch, ttnn
import tt_bio.tenstorrent as T

L = int(os.environ.get("L", "512"))
CZ = 128
HID = 128  # n_pairs(4)*C(32)
REPS = 20

T.set_fast_mode(True)
dev = T.get_device()
ckc = T._default_compute_kernel_config() if hasattr(T, "_default_compute_kernel_config") else ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True)

torch.manual_seed(0)
sd = {
    "norm_in.weight": torch.ones(CZ), "norm_in.bias": torch.zeros(CZ),
    "norm_out.weight": torch.ones(HID), "norm_out.bias": torch.zeros(HID),
    "g_in.weight": torch.randn(2*HID, CZ)*0.02, "p_in.weight": torch.randn(2*HID, CZ)*0.02,
    "g_out.weight": torch.randn(CZ, HID)*0.02, "p_out.weight": torch.randn(CZ, HID)*0.02,
}
tm = T.TriangleMultiplication(ending=False, state_dict=sd, compute_kernel_config=ckc)
print("n_pairs =", tm.n_pairs)

x_t = torch.randn(1, L, L, CZ)*0.1
def mk(): return ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)

x = mk()
o = tm(x)  # warmup/compile
ttnn.synchronize_device(dev)
ttnn.deallocate(o)

t0 = time.perf_counter()
for _ in range(REPS):
    o = tm(x)
    ttnn.deallocate(o)
ttnn.synchronize_device(dev)
dt = (time.perf_counter()-t0)/REPS
print(f"[trimul] real __call__ @L={L} fast: {dt*1000:.2f} ms/call (synced device, no host gaps)")
print(f"  if channel-TP /4: compute ~{dt*1000/4:.2f} ms + ~1.2ms gather(ring,L512) = ~{dt*1000/4+1.2:.2f} ms  -> {dt*1000/(dt*1000/4+1.2):.2f}x")
ttnn.close_device(dev)
