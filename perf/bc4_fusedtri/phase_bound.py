"""Step 1a of bc4-bet-fusedtri: the in-L1 rate of every compute phase of one AF2 trimul forward at N=224, c=128.

A fused trimul (in-projection by row band kept in L1, gates + mask, per-channel contraction, LN out, out-projection
and gate) runs the same arithmetic on the same cores as these phases; fusing removes DRAM round trips, which every
L1 arm here already excludes. So the sum of the L1 phase times (less one dispatch floor per call) prices the fused
kernel's compute. Rule (NOTES.md, written before the run): sum >= 0.425 ms (<= 30 TFLOP/s whole-kernel) kills the
bet without building the kernel; sum <= 0.283 ms (>= 45) means build it; between, build and name what binds.

Row-local phases run on one half of the rows (one band per core of 256 rows, 98 cores) and count twice.
"""
import argparse, json, os, statistics, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=21)
ap.add_argument("--burst", type=int, default=20)
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio import bmm_res

dev = T.get_device(trace="diffusion")  # a trace region, so each phase is also timed without host dispatch
g = dev.compute_with_storage_grid_size()
GX, GY = g.x, g.y
DRAM = ttnn.DRAM_MEMORY_CONFIG
CARD = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
out = open(A.out, "w")
CLK = []


def clk_loop():
    while True:
        try:
            CLK.append(int(open(f"/sys/class/tenstorrent/tenstorrent!{CARD}/tt_aiclk").read()))
        except Exception:
            pass
        time.sleep(1)


threading.Thread(target=clk_loop, daemon=True).start()


def ckc(fid=ttnn.MathFidelity.HiFi4):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(f, trace=False):
    """Per-call ms over bursts. Eager timing includes Python dispatch (~0.03-0.045 ms/op on a loaded qb2), which
    hides any phase faster than that; trace=True replays a captured burst, so it reads device time plus launch."""
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    if trace:
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(A.burst):
            ttnn.deallocate(f())
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        go = lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)  # noqa: E731
    else:
        def go():
            for _ in range(A.burst):
                ttnn.deallocate(f())
    try:
        go(); ttnn.synchronize_device(dev)
        ms = []
        for _ in range(A.reps):
            t0 = time.perf_counter()
            go(); ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
    finally:
        if trace:
            ttnn.release_trace(dev, tid)
    return statistics.median(ms), min(ms), max(ms)


def emit(r):
    r["aiclk_now"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CARD}/tt_aiclk").read())
    r["load"] = open("/proc/loadavg").read().split()[:3]
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n"); out.flush()


def run(phase, f, mult=1, flop=0, ref=None, conv=None):
    r = dict(phase=phase, mult=mult)
    try:
        if ref is not None:
            o = f(); got = ttnn.to_torch(o).double(); ttnn.deallocate(o)
            got = (conv(got) if conv else got).reshape(ref.shape)
            r["rel_l2_vs_f64"] = round(float((got - ref).norm() / ref.norm()), 5)
        md, lo, hi = timed(f)
        r.update(eager_ms=round(md, 4), eager_min_ms=round(lo, 4), eager_max_ms=round(hi, 4))
        md, lo, hi = timed(f, trace=True)
        r.update(ms=round(md, 4), min_ms=round(lo, 4), max_ms=round(hi, 4))
        if flop:
            r["tflops"] = round(flop / md / 1e9, 2)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    emit(r)
    return r


N, C = A.n, 128
M = N * N // 2                       # one band's rows
Mt = M // 32
cores = max(c for c in range(1, GX * GY + 1) if Mt % c == 0)
pm = Mt // cores
crs = ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(GX, GY), row_wise=True)
hs = lambda cols: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                    ttnn.ShardSpec(crs, [pm * 32, cols], ttnn.ShardOrientation.ROW_MAJOR))
gen = torch.Generator().manual_seed(0)
emit(dict(grid=[GX, GY], arch=str(dev.arch()), n=N, band_rows=M, cores=cores, per_core_M=pm))


def put(t, mem):
    return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)


def mm_prog(nt, kt):
    # fp32 dest holds 4 tiles; ttnn wants out_subblock_w == per_core_N or out_subblock_h == 1
    sw = max(w for w in (1, 2, 4) if nt % w == 0)
    sh = max(h for h in (1, 2, 4) if pm % h == 0 and h * sw <= 4) if sw == nt else 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY), in0_block_w=kt, out_subblock_h=sh,
        out_subblock_w=sw, per_core_M=pm, per_core_N=nt, fuse_batch=True, fused_activation=None, mcast_in0=False)


hz = torch.randn(1, 1, M, C, generator=gen)
z = put(hz, hs(C))
ck = ckc()
res = {}
# dispatch floor: the cheapest sharded op on the same grid
res["floor"] = run("launch_floor", lambda: ttnn.add(z, 0.0, memory_config=hs(C)))

# LN in (sharded LN, falls back to L1 interleaved)
try:
    lncfg = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
                                                         subblock_w=4, block_h=pm, block_w=C // 32, inplace=False)
    lnref = torch.nn.functional.layer_norm(hz.double(), (C,))
    res["ln_in"] = run("ln_in", lambda: ttnn.layer_norm(z, program_config=lncfg, memory_config=hs(C),
                                                        compute_kernel_config=ck), 2, ref=lnref)
except Exception as e:
    emit(dict(phase="ln_in", error=f"{type(e).__name__}: {str(e)[:300]}"))
if "error" in res.get("ln_in", {"error": 1}):
    zi = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
    res["ln_in"] = run("ln_in_l1i", lambda: ttnn.layer_norm(zi, memory_config=ttnn.L1_MEMORY_CONFIG,
                                                            compute_kernel_config=ck), 2)

# in-projection 128 -> 640, HiFi4 and HiFi2
hw = torch.randn(C, 5 * C, generator=gen) / C ** 0.5
w = put(hw, DRAM)
ref = hz.double() @ hw.bfloat16().double()
ref_in = hz.bfloat16().double() @ hw.bfloat16().double()
res["in_proj"] = run("in_proj_hifi4", lambda: ttnn.matmul(z, w, program_config=mm_prog(20, 4), memory_config=hs(640),
                                                          compute_kernel_config=ck), 2, 2 * M * C * 640, ref_in)
run("in_proj_hifi2", lambda: ttnn.matmul(z, w, program_config=mm_prog(20, 4), memory_config=hs(640),
                                         compute_kernel_config=ckc(ttnn.MathFidelity.HiFi2)), 2, 2 * M * C * 640, ref_in)

# gates: sigmoid over 384 gate channels, value*gate over 256, *mask over 256, g_out applied later (sigmoid counted here)
g384 = put(torch.randn(1, 1, M, 384, generator=gen), hs(384))
v256 = put(torch.randn(1, 1, M, 256, generator=gen), hs(256))
m256 = put(torch.ones(1, 1, M, 256), hs(256))
res["sigmoid"] = run("sigmoid_384", lambda: ttnn.sigmoid(g384, memory_config=hs(384)), 2)
# controls: what fusion cannot remove from the sigmoid. relu is a near-free SFPU op, so relu_384 is the L1
# unpack/pack/launch share; sigmoid minus relu is SFPU math every fused kernel still pays. Fast mode is a precision
# change and only a lever if graded on designs.
ctl_relu = run("ctl_relu_384", lambda: ttnn.relu(g384, memory_config=hs(384)), 2)
ctl_fast = run("ctl_sigmoid_fast_384", lambda: ttnn.sigmoid(g384, mode=ttnn.SigmoidMode.FastApproximate,
                                                             memory_config=hs(384)), 2)
res["gate_mul"] = run("mul_256", lambda: ttnn.multiply(v256, m256, memory_config=hs(256)), 2)
res["mask_mul"] = run("mask_mul_256", lambda: ttnn.multiply(v256, m256, memory_config=hs(256)), 2)

# contraction: out[c] = a[c] @ b[c]^T, whole channels per core (bmm_res); abl 6 = no a reads, no out writes
ha = torch.randn(1, C, N, N, generator=gen); hb = torch.randn(1, C, N, N, generator=gen)
a = put(ha, DRAM); b = put(hb, DRAM)
cref = ha.bfloat16().double() @ hb.bfloat16().double().transpose(-1, -2)
cflop = 2 * C * N ** 3
res["contract_dram"] = run("contract_bmm_res", lambda: bmm_res.matmul(a, b, transpose_b=True), 1, cflop, cref)
res["contract"] = run("contract_bmm_res_abl6", lambda: bmm_res.matmul(a, b, transpose_b=True, abl=6), 1, cflop)
run("contract_bmm_res_abl7", lambda: bmm_res.matmul(a, b, transpose_b=True, abl=7), 1)
run("contract_bmm_res_hifi2_abl6", lambda: bmm_res.matmul(a, b, transpose_b=True, abl=6,
                                                           compute_kernel_config=ckc(ttnn.MathFidelity.HiFi2)), 1, cflop)

# LN out, out-projection 128 -> 128, final gate (sigmoid already counted) multiply + residual add
if "ln_in" in res and "error" not in res["ln_in"]:
    res["ln_out"] = dict(res["ln_in"], phase="ln_out")
hp = torch.randn(C, C, generator=gen) / C ** 0.5
wp = put(hp, DRAM)
res["out_proj"] = run("out_proj_hifi4", lambda: ttnn.matmul(z, wp, program_config=mm_prog(4, 4), memory_config=hs(C),
                                                            compute_kernel_config=ck), 2, 2 * M * C * C,
                      hz.bfloat16().double() @ hp.bfloat16().double())
res["out_gate"] = run("out_gate_mul_128", lambda: ttnn.multiply(z, z, memory_config=hs(C)), 2)
res["residual"] = run("residual_add_128", lambda: ttnn.add(z, z, memory_config=hs(C)), 2)

# SUM over traced per-call ms x mult. total_ms keeps every launch (the upper side); net_ms takes the traced floor
# off each phase (the NOTES rule's "one dispatch floor subtracted") and is the fused kernel's lower bound.
fl = res["floor"].get("ms", 0)
tot, net, parts, missing = 0.0, 0.0, {}, []
for k, r in res.items():
    if k in ("floor", "contract_dram"):
        continue
    if "ms" not in r:
        missing.append(k); continue
    parts[k] = round(r["ms"] * r["mult"], 4); tot += r["ms"] * r["mult"]; net += max(r["ms"] - fl, 0) * r["mult"]
flop = 2 * N * N * C * 640 + 2 * C * N ** 3 + 2 * N * N * C * C
emit(dict(phase="SUM", parts_ms=parts, missing=missing, floor_ms=fl, total_ms=round(tot, 4), net_ms=round(net, 4),
          whole_kernel_flop=flop, tflops_total=round(flop / tot / 1e9, 2) if tot else None,
          tflops_net=round(flop / net / 1e9, 2) if net else None,
          sigmoid_math_ms=round((res["sigmoid"]["ms"] - ctl_relu.get("ms", 0)) * 2, 4) if "ms" in ctl_relu else None,
          aiclk_median=statistics.median(CLK) if CLK else None, aiclk_min=min(CLK) if CLK else None,
          aiclk_n=len(CLK)))
