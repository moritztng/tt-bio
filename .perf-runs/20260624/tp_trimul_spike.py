"""Feasibility spike: tensor-parallel the trimul cubic matmul across a 1x4 mesh.

The dominant trunk cost is TriangleMultiplication's per-channel-chunk matmul
(tenstorrent.py:538): for each of n_pairs=4 channel chunks (C=32), a batched
[1,C,L,L] @ [1,C,L,L] -> [1,C,L,L] matmul (the O(L^3) term), concatenated over
channels at the end. The chunks are INDEPENDENT -> shard the 128-channel dim
across 4 cards, then all_gather the hidden.

This measures: does (cubic/4 on mesh + all_gather) beat (cubic on 1 device)?
PCC-checks the sharded result == single-device result (must be bit-identical:
same kernel, same data, just split across devices).
"""
import os, time
import torch
import ttnn

L = int(os.environ.get("SPIKE_L", "512"))
L = ((L + 31) // 32) * 32  # tile-align
HIDDEN = 128          # trimul pair hidden dim
C = 32                # TRIANGLE_MULT_CHUNK_SIZE
N = HIDDEN // C       # 4 channel chunks == 4 devices
REPS = int(os.environ.get("SPIKE_REPS", "20"))
DT = ttnn.bfloat8_b   # --fast dtype

ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
    fp32_dest_acc_en=False, packer_l1_acc=True,
)

torch.manual_seed(0)
a_t = torch.randn(1, HIDDEN, L, L, dtype=torch.float32) * 0.1
b_t = torch.randn(1, HIDDEN, L, L, dtype=torch.float32) * 0.1

def pcc(x, y):
    x = x.flatten().float(); y = y.flatten().float()
    xm, ym = x - x.mean(), y - y.mean()
    return (xm @ ym / (xm.norm() * ym.norm() + 1e-12)).item()

# ---------- single device ----------
dev = ttnn.open_device(device_id=0)
def sync_dev(): ttnn.synchronize_device(dev)
a1 = ttnn.from_torch(a_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=DT)
b1 = ttnn.from_torch(b_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=DT)
# warmup + compile
out1 = ttnn.matmul(a1, b1, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
sync_dev()
ref = ttnn.to_torch(out1)
t0 = time.perf_counter()
for _ in range(REPS):
    o = ttnn.matmul(a1, b1, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
sync_dev()
t_single = (time.perf_counter() - t0) / REPS
print(f"[single] full [1,{HIDDEN},{L},{L}] matmul: {t_single*1000:.2f} ms/rep")
ttnn.deallocate(a1); ttnn.deallocate(b1); ttnn.deallocate(out1)
ttnn.close_device(dev)

# ---------- 1x4 mesh, channel-sharded ----------
FABRIC = os.environ.get("SPIKE_FABRIC", "FABRIC_1D")
ttnn.set_fabric_config(getattr(ttnn.FabricConfig, FABRIC))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
def sync_mesh(): ttnn.synchronize_device(mesh)
# shard the HIDDEN (dim=1) across the 4 devices: each gets [1,32,L,L]
shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=1)
am = ttnn.from_torch(a_t, layout=ttnn.TILE_LAYOUT, device=mesh, dtype=DT, mesh_mapper=shard_mapper)
bm = ttnn.from_torch(b_t, layout=ttnn.TILE_LAYOUT, device=mesh, dtype=DT, mesh_mapper=shard_mapper)

# (a) pure sharded compute, no gather
om = ttnn.matmul(am, bm, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
sync_mesh()
t0 = time.perf_counter()
for _ in range(REPS):
    o = ttnn.matmul(am, bm, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
sync_mesh()
t_mesh_compute = (time.perf_counter() - t0) / REPS
print(f"[mesh ] sharded [1,32,{L},{L}] matmul (no gather): {t_mesh_compute*1000:.2f} ms/rep")

# (b) compute + all_gather hidden back to full [1,128,L,L] on every device
og = ttnn.all_gather(om, dim=1)
sync_mesh()
t0 = time.perf_counter()
for _ in range(REPS):
    o = ttnn.matmul(am, bm, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    o = ttnn.all_gather(o, dim=1)
sync_mesh()
t_mesh_gather = (time.perf_counter() - t0) / REPS
print(f"[mesh ] sharded matmul + all_gather: {t_mesh_gather*1000:.2f} ms/rep")

# accuracy: gathered result on device 0 vs single-device ref
got = ttnn.to_torch(og, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
# all_gather replicates: each device holds full; ConcatMeshToTensor would stack 4x.
# Instead read shard outputs and concat manually for a clean compare:
shards = ttnn.to_torch(om, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
print(f"[pcc  ] sharded-concat vs single-device: {pcc(shards, ref):.6f}  maxdiff={ (shards-ref).abs().max().item():.3e}")

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

print("\n=== SUMMARY ===")
print(f"L={L} HIDDEN={HIDDEN} reps={REPS}")
print(f"single full        : {t_single*1000:.2f} ms")
print(f"mesh compute only  : {t_mesh_compute*1000:.2f} ms  ({t_single/t_mesh_compute:.2f}x vs single)")
print(f"mesh + all_gather  : {t_mesh_gather*1000:.2f} ms  ({t_single/t_mesh_gather:.2f}x vs single)")
print(f"all_gather cost    : {(t_mesh_gather-t_mesh_compute)*1000:.2f} ms")
