"""Trunk tensor-parallel feasibility spike (2026-06-30), built on current main.

The 2026-06-24 TP work sharded ONLY TriangleMultiplication (channel-TP, 1.75x@512,
bit-identical) and found the whole-model mesh REGRESSED e2e. The open question it
left: can the trunk WIN if we also shard the SECOND dominant op-type, triangle
ATTENTION? Boltz-2 trunk triangle-attn has exactly tri_att_n_heads=4 -> one head
per Blackhole card, a clean head-parallel split. trimul(31.5%)+triattn(25%)=56.5%
of trunk device time. This spike measures, on the REAL primitives at production
config (Blackhole HiFi4, --fast bf8):

  1. trimul channel-shard : single vs 1x4 mesh  (reproduce + confirm infra)
  2. triattn-start head-shard : single vs 1x4 mesh  (NEW measurement)
  3. replicated-op mesh tax : Transition single vs replicated-on-mesh (overhead)
  4. all_gather cost @ each size

From (1)+(2)+(3)+(4) and the known op shares we project the full-trunk outcome.
Standalone: NO edits to the live model, no cache-fragile permute changes.
"""
import os, time
import torch, ttnn
import tt_bio.tenstorrent as T

L = int(os.environ.get("L", "512")); L = ((L + 31) // 32) * 32
CZ = 128                     # Boltz-2 trunk pair channel (token_z)
N_HEADS = 4                  # tri_att_n_heads
HEAD_DIM = 32                # tri_att_head_dim
HID = 128                    # trimul hidden
C = T.TRIANGLE_MULT_CHUNK_SIZE
NP = HID // C                # 4 channel chunks
REPS = int(os.environ.get("REPS", "20"))
T.set_fast_mode(True)

torch.manual_seed(0)


def pcc(x, y):
    x = x.flatten().float(); y = y.flatten().float()
    xm, ym = x - x.mean(), y - y.mean()
    return (xm @ ym / (xm.norm() * ym.norm() + 1e-12)).item()


def ckc_for(dev):
    cls = (ttnn.types.WormholeComputeKernelConfig
           if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


# ---- synthetic weights (random; we test SPEED + sharded-vs-single PCC) ----
tm_sd = {
    "norm_in.weight": torch.randn(CZ) * 0.1 + 1, "norm_in.bias": torch.randn(CZ) * 0.05,
    "norm_out.weight": torch.randn(HID) * 0.1 + 1, "norm_out.bias": torch.randn(HID) * 0.05,
    "g_in.weight": torch.randn(2 * HID, CZ) * 0.05, "p_in.weight": torch.randn(2 * HID, CZ) * 0.05,
    "g_out.weight": torch.randn(CZ, HID) * 0.05, "p_out.weight": torch.randn(CZ, HID) * 0.05,
}
ta_sd = {
    "layer_norm.weight": torch.randn(CZ) * 0.1 + 1, "layer_norm.bias": torch.randn(CZ) * 0.05,
    "linear.weight": torch.randn(N_HEADS, CZ) * 0.05,           # bias proj CZ->n_heads
    "linear_q.weight": torch.randn(N_HEADS * HEAD_DIM, CZ) * 0.05,
    "linear_k.weight": torch.randn(N_HEADS * HEAD_DIM, CZ) * 0.05,
    "linear_v.weight": torch.randn(N_HEADS * HEAD_DIM, CZ) * 0.05,
    "linear_g.weight": torch.randn(N_HEADS * HEAD_DIM, CZ) * 0.05,
    "linear_o.weight": torch.randn(CZ, N_HEADS * HEAD_DIM) * 0.05,
}
x_t = torch.randn(1, L, L, CZ) * 0.1


def bench(fn, dev, label):
    o = fn(); ttnn.synchronize_device(dev)
    if isinstance(o, ttnn.Tensor):
        ttnn.deallocate(o)
    t0 = time.perf_counter()
    for _ in range(REPS):
        o = fn()
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPS


results = {}

# ============================ SINGLE DEVICE ============================
dev = ttnn.open_device(device_id=int(os.environ.get("DEV", "0")))
dev.enable_program_cache()
T._configure_active_compute_grid(dev)
T._device = dev
ckc = ckc_for(dev)

tm = T.TriangleMultiplication(ending=False, state_dict=tm_sd, compute_kernel_config=ckc)
ta = T.TriangleAttention(HEAD_DIM, N_HEADS, ending=False, state_dict=ta_sd,
                         compute_kernel_config=ckc)
trans_sd = {
    "norm.weight": torch.randn(CZ) * 0.1 + 1, "norm.bias": torch.randn(CZ) * 0.05,
    "fc1.weight": torch.randn(4 * CZ, CZ) * 0.05, "fc2.weight": torch.randn(4 * CZ, CZ) * 0.05,
    "fc3.weight": torch.randn(CZ, 4 * CZ) * 0.05,
}
trans = T.Transition(state_dict=trans_sd, compute_kernel_config=ckc)

xs = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)
xs4 = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)  # [1,L,L,CZ]

ref_tm = ttnn.to_torch(tm(xs))
ref_ta = ttnn.to_torch(ta(xs4))

results["single_trimul"] = bench(lambda: tm(xs), dev, "trimul")
results["single_triattn"] = bench(lambda: ta(xs4), dev, "triattn")
results["single_transition"] = bench(lambda: trans(xs4), dev, "transition")
print(f"[single] L={L}  trimul={results['single_trimul']*1000:.2f}ms  "
      f"triattn={results['single_triattn']*1000:.2f}ms  "
      f"transition={results['single_transition']*1000:.2f}ms", flush=True)
ttnn.close_device(dev)
T._device = None

# ============================ 1x4 MESH ============================
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
mesh.enable_program_cache()
T._configure_active_compute_grid(mesh)
T._device = mesh
ckc = ckc_for(mesh)
rep = ttnn.ReplicateTensorToMesh(mesh)
H = L
mem = T._triangle_mul_memory_config(H)
pc = T._triangle_mul_program_config((H + 31) // 32)


def mk(t, transform=lambda x: x, dim=None):
    mm = ttnn.ShardTensorToMesh(mesh, dim=dim) if dim is not None else rep
    return ttnn.from_torch(transform(t), layout=ttnn.TILE_LAYOUT, device=mesh,
                           dtype=ttnn.bfloat16, mesh_mapper=mm)


# ---------- mesh trimul (channel-shard, from 06-24 spike) ----------
g_in_t, p_in_t = tm_sd["g_in.weight"].t(), tm_sd["p_in.weight"].t()
fused_cols = [torch.cat([g_in_t[:, i * C:(i + 1) * C], g_in_t[:, (i + NP) * C:(i + NP + 1) * C],
                         p_in_t[:, i * C:(i + 1) * C], p_in_t[:, (i + NP) * C:(i + NP + 1) * C]], dim=1)
              for i in range(NP)]
W_in = mk(torch.cat(fused_cols, dim=1), dim=1)
tm_nin_w, tm_nin_b = mk(tm_sd["norm_in.weight"]), mk(tm_sd["norm_in.bias"])
tm_non_w, tm_non_b = mk(tm_sd["norm_out.weight"]), mk(tm_sd["norm_out.bias"])
tm_g_out, tm_p_out = mk(tm_sd["g_out.weight"], lambda x: x.t()), mk(tm_sd["p_out.weight"], lambda x: x.t())


def _tchunk(chunk, perm):
    old = chunk
    for op, *a in [(ttnn.typecast, ttnn.bfloat16), (ttnn.permute, perm),
                   (ttnn.typecast, ttnn.bfloat8_b), (ttnn.reallocate,)]:
        chunk = op(chunk, *a, memory_config=mem); ttnn.deallocate(old); old = chunk
    return chunk


def mesh_trimul(x):
    xn = ttnn.layer_norm(x, weight=tm_nin_w, bias=tm_nin_b, epsilon=1e-5, compute_kernel_config=ckc)
    gp = ttnn.experimental.minimal_matmul(xn, W_in, memory_config=mem, dtype=T._dtype(), compute_kernel_config=ckc)
    g_a, g_b, p_a, p_b = ttnn.chunk(gp, chunks=4, dim=-1); ttnn.deallocate(gp)
    a = ttnn.multiply_(p_a, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    b = ttnn.multiply_(p_b, g_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(g_a); ttnn.deallocate(g_b)
    a = _tchunk(a, (0, 3, 1, 2)); b = _tchunk(b, (0, 3, 2, 1))
    xc = ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=mem, program_config=pc, dtype=ttnn.bfloat16)
    ttnn.deallocate(a); ttnn.deallocate(b)
    xc = ttnn.permute(xc, (0, 2, 3, 1), memory_config=mem)
    xg = ttnn.all_gather(xc, dim=-1)
    ttnn.deallocate(xc)
    xg = ttnn.layer_norm(xg, weight=tm_non_w, bias=tm_non_b, epsilon=1e-5, compute_kernel_config=ckc)
    p_out = ttnn.linear(xg, tm_p_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(), compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    g_out = ttnn.linear(xn, tm_g_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(), compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(xg); ttnn.deallocate(xn)
    return ttnn.multiply_(p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])


# ---------- mesh triattn-start (head-shard: 1 head/device) ----------
# reorder qkv weights to [CZ, n_heads*(3*head_dim)] then shard dim=1 -> per dev [CZ, 96]
qh = ta_sd["linear_q.weight"].reshape(N_HEADS, HEAD_DIM, CZ)
kh = ta_sd["linear_k.weight"].reshape(N_HEADS, HEAD_DIM, CZ)
vh = ta_sd["linear_v.weight"].reshape(N_HEADS, HEAD_DIM, CZ)
qkv_cols = torch.cat([torch.cat([qh[i].t(), kh[i].t(), vh[i].t()], dim=1) for i in range(N_HEADS)], dim=1)
ta_qkv = mk(qkv_cols, dim=1)                                   # per dev [CZ, 96]
ta_g = mk(ta_sd["linear_g.weight"].t(), dim=1)                 # [CZ, n_heads*hd] -> per dev [CZ,32]
ta_bias_w = mk((ta_sd["linear.weight"] * (HEAD_DIM ** 0.5)).t(), dim=1)  # [CZ,n_heads]->per dev[CZ,1]
ta_ln_w, ta_ln_b = mk(ta_sd["layer_norm.weight"]), mk(ta_sd["layer_norm.bias"])
ta_o = mk(ta_sd["linear_o.weight"].t())                        # replicated [n_heads*hd, CZ]
scale = HEAD_DIM ** 0.5


def mesh_triattn(x):
    xr = ttnn.reshape(x, tuple(x.shape)[1:])                   # [L,L,CZ] replicated
    xn = ttnn.layer_norm(xr, weight=ta_ln_w, bias=ta_ln_b, epsilon=1e-5, compute_kernel_config=ckc)
    bias = ttnn.linear(xn, ta_bias_w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
    bias = ttnn.permute(ttnn.unsqueeze(bias, 0), (0, 3, 1, 2))  # [1,1,L,L] per dev
    qkv = ttnn.experimental.minimal_matmul(xn, ta_qkv, compute_kernel_config=ckc, dtype=T._dtype())
    g = ttnn.experimental.minimal_matmul(xn, ta_g, compute_kernel_config=ckc, dtype=T._dtype())
    ttnn.deallocate(xn)
    qkv = ttnn.unsqueeze(qkv, 1)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=1, num_kv_heads=1,
                                                     transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(qkv)
    o = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False,
                                                      scale=scale ** -1,
                                                      program_config=T._sdpa_program_config_for_lengths(q.shape[2], k.shape[2]))
    ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v); ttnn.deallocate(bias)
    o = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)  # [L,L,32] per dev
    o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(g)
    og = ttnn.all_gather(o, dim=-1)                            # [L,L,128] replicated
    ttnn.deallocate(o)
    xo = ttnn.linear(og, ta_o, compute_kernel_config=ckc, dtype=T._dtype(), core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(og)
    return ttnn.reshape(xo, (1, *xo.shape))


# ---------- replicated transition (mesh tax measurement) ----------
trans_m = T.Transition(state_dict=trans_sd, compute_kernel_config=ckc)

xm = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=mesh, dtype=ttnn.bfloat8_b, mesh_mapper=rep)


def d0(o):
    return ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]


o_tm = mesh_trimul(xm); got_tm = d0(o_tm); ttnn.deallocate(o_tm)
o_ta = mesh_triattn(xm); got_ta = d0(o_ta); ttnn.deallocate(o_ta)
md_tm = (got_tm.float() - ref_tm.float()).abs().max().item()
md_ta = (got_ta.float() - ref_ta.float()).abs().max().item()
print(f"[pcc] trimul mesh-vs-single pcc={pcc(got_tm, ref_tm):.6f} maxdiff={md_tm:.3e} "
      f"shapes {tuple(got_tm.shape)}/{tuple(ref_tm.shape)}  | "
      f"triattn pcc={pcc(got_ta, ref_ta):.6f} maxdiff={md_ta:.3e} "
      f"shapes {tuple(got_ta.shape)}/{tuple(ref_ta.shape)}", flush=True)

results["mesh_trimul"] = bench(lambda: mesh_trimul(xm), mesh, "trimul")
results["mesh_triattn"] = bench(lambda: mesh_triattn(xm), mesh, "triattn")
results["mesh_transition"] = bench(lambda: trans_m(xm), mesh, "transition")

# clean all_gather micro-bench: gather a sharded hidden of the trimul/triattn shape
sh_tm = ttnn.from_torch(torch.randn(1, L, L, C), layout=ttnn.TILE_LAYOUT, device=mesh,
                        dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
sh_ta = ttnn.from_torch(torch.randn(1, L, L, HEAD_DIM), layout=ttnn.TILE_LAYOUT, device=mesh,
                        dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
results["ag_tm"] = bench(lambda: ttnn.all_gather(sh_tm, dim=-1), mesh, "ag_tm")
results["ag_ta"] = bench(lambda: ttnn.all_gather(sh_ta, dim=-1), mesh, "ag_ta")

print(f"[mesh]   L={L}  trimul={results['mesh_trimul']*1000:.2f}ms  "
      f"triattn={results['mesh_triattn']*1000:.2f}ms  transition={results['mesh_transition']*1000:.2f}ms", flush=True)

st_tm, st_ta, st_tr = results["single_trimul"], results["single_triattn"], results["single_transition"]
me_tm, me_ta, me_tr = results["mesh_trimul"], results["mesh_triattn"], results["mesh_transition"]
print(f"[SPEEDUP] L={L}  trimul={st_tm/me_tm:.2f}x  triattn={st_ta/me_ta:.2f}x  "
      f"transition={st_tr/me_tr:.2f}x (>1 good, <1 = mesh tax)", flush=True)
print(f"[all_gather] trimul-hidden={results['ag_tm']*1000:.2f}ms  triattn-head={results['ag_ta']*1000:.2f}ms", flush=True)

# ---- project full pairformer layer (per recycle iter, x4) ----
# layer = trimul_start + trimul_end + triattn_start + triattn_end + transition_z
single_layer = 2 * st_tm + 2 * st_ta + st_tr
mesh_layer = 2 * me_tm + 2 * me_ta + me_tr
print(f"[PROJECT] L={L} per-layer(z-path): single={single_layer*1000:.1f}ms "
      f"mesh={mesh_layer*1000:.1f}ms  -> {single_layer/mesh_layer:.2f}x", flush=True)

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
