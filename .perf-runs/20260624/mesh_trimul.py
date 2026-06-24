"""Validated full-trimul channel-TP across a 1x4 mesh vs the real single-device
TriangleMultiplication. Shards the n_pairs channel chunks one-per-device, runs
the ENTIRE per-chunk pipeline (proj + gate + permute + cubic matmul + permute) in
parallel, then ONE all_gather of the hidden, then replicated output projection.

Reports per-call time (single vs mesh) and PCC/maxdiff (expect bit-identical).
"""
import os, time
import torch, ttnn
import tt_bio.tenstorrent as T

L = int(os.environ.get("L", "512")); L = ((L+31)//32)*32
CZ = 128; C = T.TRIANGLE_MULT_CHUNK_SIZE; HID = 128; NP = HID // C  # 4
REPS = 30
ENDING = False
T.set_fast_mode(True)

ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True)

torch.manual_seed(0)
sd = {
    "norm_in.weight": torch.randn(CZ)*0.1+1, "norm_in.bias": torch.randn(CZ)*0.05,
    "norm_out.weight": torch.randn(HID)*0.1+1, "norm_out.bias": torch.randn(HID)*0.05,
    "g_in.weight": torch.randn(2*HID, CZ)*0.05, "p_in.weight": torch.randn(2*HID, CZ)*0.05,
    "g_out.weight": torch.randn(CZ, HID)*0.05, "p_out.weight": torch.randn(CZ, HID)*0.05,
}
x_t = torch.randn(1, L, L, CZ)*0.1

def pcc(x,y):
    x=x.flatten().float(); y=y.flatten().float()
    xm,ym=x-x.mean(),y-y.mean()
    return (xm@ym/(xm.norm()*ym.norm()+1e-12)).item()

# ---------- single device reference ----------
dev = ttnn.open_device(device_id=0)
T._device = dev  # get_device() returns this
tm = T.TriangleMultiplication(ending=ENDING, state_dict=sd, compute_kernel_config=ckc)
xs = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)
o = tm(xs); ttnn.synchronize_device(dev); ref = ttnn.to_torch(o); ttnn.deallocate(o)
t0=time.perf_counter()
for _ in range(REPS): o=tm(xs); ttnn.deallocate(o)
ttnn.synchronize_device(dev); t_single=(time.perf_counter()-t0)/REPS
print(f"[single] trimul/call @L={L}: {t_single*1000:.2f} ms")
ttnn.close_device(dev)

# ---------- 1x4 mesh, channel-sharded ----------
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,4))
T._device = mesh

H = L; mem = T._triangle_mul_memory_config(H); pc = T._triangle_mul_program_config((H+31)//32)
rep = ttnn.ReplicateTensorToMesh(mesh)
# per-device fused in-weight: concat 4 chunks along cols, shard dim=1 (one chunk/dev)
g_in_t, p_in_t = sd["g_in.weight"].t(), sd["p_in.weight"].t()  # [CZ, 2*HID]
fused_cols = []
for i in range(NP):
    fused_cols.append(torch.cat([
        g_in_t[:, i*C:(i+1)*C], g_in_t[:, (i+NP)*C:(i+NP+1)*C],
        p_in_t[:, i*C:(i+1)*C], p_in_t[:, (i+NP)*C:(i+NP+1)*C]], dim=1))  # [CZ,4C]
W_fused = torch.cat(fused_cols, dim=1)  # [CZ, NP*4C]
W_in = ttnn.from_torch(W_fused, layout=ttnn.TILE_LAYOUT, device=mesh, dtype=ttnn.bfloat16,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
def mk(key, transform=lambda x:x):
    return ttnn.from_torch(transform(sd[key]), layout=ttnn.TILE_LAYOUT, device=mesh, dtype=ttnn.bfloat16, mesh_mapper=rep)
nin_w, nin_b = mk("norm_in.weight"), mk("norm_in.bias")
non_w, non_b = mk("norm_out.weight"), mk("norm_out.bias")
g_out_w = mk("g_out.weight", lambda x:x.t()); p_out_w = mk("p_out.weight", lambda x:x.t())

def transform_chunk(chunk, perm):
    old = chunk
    for op,*args in [(ttnn.typecast, ttnn.bfloat16),(ttnn.permute,perm),(ttnn.typecast,ttnn.bfloat8_b),(ttnn.reallocate,)]:
        chunk = op(chunk,*args,memory_config=mem); ttnn.deallocate(old); old=chunk
    return chunk

def mesh_trimul(x, gather=True):
    xn = ttnn.layer_norm(x, weight=nin_w, bias=nin_b, epsilon=1e-5, compute_kernel_config=ckc)
    gp = ttnn.experimental.minimal_matmul(xn, W_in, memory_config=mem, dtype=T._dtype(), compute_kernel_config=ckc)
    g_a,g_b,p_a,p_b = ttnn.chunk(gp,chunks=4,dim=-1); ttnn.deallocate(gp)
    a = ttnn.multiply_(p_a,g_a,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    b = ttnn.multiply_(p_b,g_b,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(g_a); ttnn.deallocate(g_b)
    a = transform_chunk(a,(0,3)+((2,1) if ENDING else (1,2)))
    b = transform_chunk(b,(0,3)+((1,2) if ENDING else (2,1)))
    xc = ttnn.matmul(a,b,compute_kernel_config=ckc,memory_config=mem,program_config=pc,dtype=ttnn.bfloat16)
    ttnn.deallocate(a); ttnn.deallocate(b)
    xc = ttnn.permute(xc,(0,2,3,1),memory_config=mem)   # [1,L,L,C] sharded hidden
    xg = ttnn.all_gather(xc, dim=-1) if gather else xc   # [1,L,L,HID] replicated
    ttnn.deallocate(xc)
    xg = ttnn.layer_norm(xg, weight=non_w, bias=non_b, epsilon=1e-5, compute_kernel_config=ckc)
    p_out = ttnn.linear(xg, p_out_w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(), compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    g_out = ttnn.linear(xn, g_out_w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(), compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(xg); ttnn.deallocate(xn)
    out = ttnn.multiply_(p_out,g_out,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    return out

xm = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=mesh, dtype=ttnn.bfloat8_b, mesh_mapper=rep)
o = mesh_trimul(xm); ttnn.synchronize_device(mesh)
got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]  # device0 copy
ttnn.deallocate(o)
print(f"[pcc  ] mesh vs single: {pcc(got, ref):.6f}  maxdiff={(got-ref).abs().max().item():.3e}")

for gather in (True, False):
    o=mesh_trimul(xm, gather=gather); ttnn.synchronize_device(mesh); ttnn.deallocate(o)
    t0=time.perf_counter()
    for _ in range(REPS): o=mesh_trimul(xm, gather=gather); ttnn.deallocate(o)
    ttnn.synchronize_device(mesh); t=(time.perf_counter()-t0)/REPS
    tag = "mesh+gather" if gather else "mesh nogather"
    print(f"[{tag:13s}] trimul/call @L={L}: {t*1000:.2f} ms  ({t_single/t:.2f}x vs single)")

ttnn.close_mesh_device(mesh); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
