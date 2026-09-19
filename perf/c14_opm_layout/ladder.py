#!/usr/bin/env python3
"""OPM layout round trip at the executed 512 aa shape: composed arms + per-op micro arms.

Arms interleaved rep by rep with the interior order reversed on odd reps, plus an A/A twin.
Clock forced to 1350 MHz and sampled DURING by a thread that holds no device fd.
Instrument controls: (1) byte counters computed two independent ways and asserted equal,
(2) an in-session DRAM roof against c10-fold-census's 442.9 GB/s and c14-unpriced-block's 442.3,
(3) every micro arm is checked against the in-situ ms/call from c12-profiled-fold's MSALayer
    profile -- a standalone arm that does not reproduce the fold is not evidence.
"""
import json, statistics, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk, torch, ttnn  # noqa: E402

MHZ, REPS = 1350, int(sys.argv[1]) if len(sys.argv) > 1 else 15
S, I, J, C, D, CZ = 64, 512, 512, 32, 32, 128
# in-situ ms/call, c12-profiled-fold runs/msal_prof, 11 MSALayer calls, 512 aa, qb2 p300c @1350
INSITU = {"untilize_z": 2.8320, "reshape_z": 3.0605, "tilize_z": 2.4257, "transpose_z": 2.8829,
          "mul_scale_z": 3.0227, "projo_batched": 10.2965, "z_matmul": 2.3235}

dev = ttnn.open_device(device_id=0)
held = clk.force(MHZ, clk.nodes_open_by_this_process())
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ - 5 for n in held): break
    time.sleep(0.05)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
g = dev.compute_with_storage_grid_size(); CG = ttnn.CoreGrid(y=g.y, x=g.x)
DRAM = ttnn.DRAM_MEMORY_CONFIG
scale = 1.0 / S
torch.manual_seed(0)

# ---- control 1: byte counters, two independent routes, asserted equal ----
Z_EL = I * C * D * J
by_shape = Z_EL * 2
by_tiles = ((I * C) // 32) * ((D * J) // 32) * 32 * 32 * 2
assert by_shape == by_tiles == 536870912, (by_shape, by_tiles)
CTRL = {"z_bytes_by_shape": by_shape, "z_bytes_by_tilegrid": by_tiles, "equal": True}

def tt(x, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, layout=layout, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)

a0 = tt((torch.randn(S, I, C) / 8).to(torch.bfloat16))
b0 = tt((torch.randn(S, J, D) / 8).to(torch.bfloat16))
W = tt((torch.randn(C * D, CZ) / 32).to(torch.bfloat16))
bias = tt((torch.randn(1, CZ) / 8).to(torch.bfloat16))

# ---- control 2: in-session DRAM roof off a known byte count ----
def dram_roof():
    x = tt(torch.zeros(1, 1, 16384, 16384, dtype=torch.bfloat16))
    y = tt(torch.zeros(1, 1, 16384, 16384, dtype=torch.bfloat16))
    ttnn.synchronize_device(dev)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); o = ttnn.add(x, y); ttnn.synchronize_device(dev)
        best = min(best, time.perf_counter() - t0); ttnn.deallocate(o)
    ttnn.deallocate(x); ttnn.deallocate(y)
    return 3 * by_shape / best / 1e9, best * 1e3

# ---- prebuilt operands for the micro arms, at the exact executed shapes ----
def mk_ab():
    a = ttnn.permute(a0, (1, 2, 0))                       # (I,C,S)
    b = ttnn.permute(b0, (2, 1, 0))                       # (D,J,S)
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT); b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
    return a, b
A_P, B_P = mk_ab()
A_FLAT = ttnn.reshape(A_P, (I * C, S))
Z_TILE = ttnn.matmul(A_FLAT, B_P, transpose_b=True, compute_kernel_config=ckc)   # (I*C,D*J)
Z_RM = ttnn.to_layout(Z_TILE, ttnn.ROW_MAJOR_LAYOUT)
Z_RM3 = ttnn.reshape(Z_RM, (I, C * D, J))
Z_T3 = ttnn.to_layout(Z_RM3, ttnn.TILE_LAYOUT)
Z_FIN = ttnn.permute(Z_T3, (0, 2, 1))                                            # (I,J,C*D)
Z_2D = ttnn.reshape(Z_FIN, (I * J, C * D))
ttnn.synchronize_device(dev)

def m_zmm():   ttnn.deallocate(ttnn.matmul(A_FLAT, B_P, transpose_b=True, compute_kernel_config=ckc))
def m_unt():   ttnn.deallocate(ttnn.to_layout(Z_TILE, ttnn.ROW_MAJOR_LAYOUT))
def m_rsh():   ttnn.deallocate(ttnn.reshape(Z_RM, (I, C * D, J)))
def m_til():   ttnn.deallocate(ttnn.to_layout(Z_RM3, ttnn.TILE_LAYOUT))
def m_trn():   ttnn.deallocate(ttnn.permute(Z_T3, (0, 2, 1)))
def m_mul():   ttnn.deallocate(ttnn.multiply(Z_FIN, scale))
def m_resh2d():ttnn.deallocate(ttnn.reshape(Z_FIN, (I * J, C * D)))
def m_pjb():   ttnn.deallocate(ttnn.linear(Z_FIN, W, bias=bias, compute_kernel_config=ckc, core_grid=CG))
def m_pjf():   ttnn.deallocate(ttnn.linear(Z_2D, W, bias=bias, compute_kernel_config=ckc, core_grid=CG))
def m_pjb_ng():ttnn.deallocate(ttnn.linear(Z_FIN, W, bias=bias, compute_kernel_config=ckc))
def m_pjf_ng():ttnn.deallocate(ttnn.linear(Z_2D, W, bias=bias, compute_kernel_config=ckc))

# ---- composed arms: the whole op, a -> out ----
def _layout(z):
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.reshape(z, (I, C * D, J))
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    return ttnn.permute(z, (0, 2, 1))

def shipped(fold_scale=False, flat=False):
    a = ttnn.permute(a0, (1, 2, 0))
    if fold_scale:
        a2 = ttnn.multiply(a, scale); ttnn.deallocate(a); a = a2
    b = ttnn.permute(b0, (2, 1, 0))
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT); b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
    af = ttnn.reshape(a, (I * C, S))
    z = ttnn.matmul(af, b, transpose_b=True, compute_kernel_config=ckc)
    ttnn.deallocate(a); ttnn.deallocate(b)
    z = _layout(z)
    if not fold_scale:
        z = ttnn.multiply_(z, scale)
    if flat:
        z2 = ttnn.reshape(z, (I * J, C * D)); ttnn.deallocate(z); z = z2
    out = ttnn.linear(z, W, bias=bias, compute_kernel_config=ckc, core_grid=CG)
    ttnn.deallocate(z); ttnn.deallocate(out)

ARMS = [
    ("full_shipped",   lambda: shipped()),
    ("full_shipped_AA",lambda: shipped()),
    ("full_foldscale", lambda: shipped(fold_scale=True)),
    ("full_fold_flat", lambda: shipped(fold_scale=True, flat=True)),
    ("z_matmul",       m_zmm), ("untilize_z", m_unt), ("reshape_z", m_rsh),
    ("tilize_z", m_til), ("transpose_z", m_trn), ("mul_scale_z", m_mul),
    ("reshape_2d", m_resh2d), ("projo_batched", m_pjb), ("projo_flat", m_pjf),
    ("projo_batched_nogrid", m_pjb_ng), ("projo_flat_nogrid", m_pjf_ng),
]
for _, fn in ARMS: fn()
ttnn.synchronize_device(dev)
roof_gbs, roof_ms = dram_roof()
CTRL["dram_roof_gbs"] = roof_gbs
CTRL["dram_roof_ms"] = roof_ms
ttnn.synchronize_device(dev)

s = clk.Sampler(held[0]); time.sleep(0.3)
acc = {n: [] for n, _ in ARMS}
for r in range(REPS):
    order = ARMS if r % 2 == 0 else list(reversed(ARMS))
    for n, fn in order:
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); fn()
        ttnn.synchronize_device(dev); acc[n].append((time.perf_counter() - t0) * 1e3)
res = {n: {"ms_min": min(v), "ms_med": statistics.median(v), "n": len(v)} for n, v in acc.items()}
res["_AA_pct"] = abs(res["full_shipped_AA"]["ms_min"] - res["full_shipped"]["ms_min"]) \
                 / res["full_shipped"]["ms_min"] * 100
repro = {k: {"insitu_ms": v, "arm_ms": res[k]["ms_min"], "ratio": res[k]["ms_min"] / v}
         for k, v in INSITU.items() if k in res}
out = {"arms": res, "insitu_repro": repro, "control": CTRL, "grid": [g.y, g.x],
       "nodes": held, "clock": s.stop(), "reps": REPS, "shape": [S, I, J, C, D, CZ]}
ttnn.close_device(dev)
(HERE / "ladder_qb1c2.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
