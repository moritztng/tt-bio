"""spd-msa: OuterProductMean's output projection [I*J, C*D] x [C*D, c_z] under different program configs.

At 730 tokens on Wormhole it is [541696, 1024] x [1024, 256] at 67 ms (4.2 TF/s; every bf16 config lpx-matmul tried
lands at 68 ms, bfp8 operands 36 ms). The suspicion: the auto config splits N over cores and re-reads the 1.1 GB
in0 once per N block. Arms keep all of N on each core (1D, in1 multicast) and split M into calls small enough for
the output block to fit L1. Each arm's output is compared with the auto arm (max abs, rel rms); time is the median of
REPS synced calls.

usage: TT_VISIBLE_DEVICES=<chip> python sweep_opm_out.py OUT [ROWS=541696] [REPS=7]
"""
import glob, json, statistics, sys, threading, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 541696
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
LOG = open(OUT / "sweep.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
SAMPLES = []


def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                row[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            except Exception:
                row[n] = -1
        SAMPLES.append(row); time.sleep(0.25)


threading.Thread(target=_sampler, daemon=True).start()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
GX, GY = grid.x, grid.y
NC = GX * GY
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
K, N = 1024, 256
torch.manual_seed(0)
x_h = torch.randn(ROWS, K).bfloat16(); w_h = (torch.randn(K, N) / 32).bfloat16()
x = ttnn.from_torch(x_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
w = ttnn.from_torch(w_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
log(ev="start", rows=ROWS, grid=[GX, GY], nodes=NODES)


def one_d(pm, ibw, sbh, sbw):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(GX, GY), in0_block_w=ibw, out_subblock_h=sbh, out_subblock_w=sbw,
        per_core_M=pm, per_core_N=N // 32, fuse_batch=True, fused_activation=None, mcast_in0=False)


def run(split, pc):
    """`split` calls over row blocks of at most `split` rows (None: one call), each with config `pc`."""
    if split is None or split >= ROWS:
        return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc)
    parts = [ttnn.linear(x[r:min(r + split, ROWS)], w, compute_kernel_config=ckc, program_config=pc)
             for r in range(0, ROWS, split)]
    out = ttnn.concat(parts, dim=0)
    for p in parts:
        ttnn.deallocate(p)
    return out


arms = [("auto", None, None)]
for pm in (8, 16, 24, 32):
    rows = pm * 32 * NC
    for ibw, sb in ((4, (1, 4)), (8, (1, 4)), (4, (2, 2)), (2, (1, 8))):
        arms.append((f"1d_pm{pm}_ibw{ibw}_sb{sb[0]}x{sb[1]}", rows, (pm, ibw, sb)))
for k in (2, 4, 8):
    arms.append((f"auto_split{k}", -(-ROWS // k // 32) * 32, None))

ref = None
for name, split, cfg in arms:
    try:
        pc = one_d(cfg[0], cfg[1], *cfg[2]) if cfg else None
        out = run(split, pc); ttnn.synchronize_device(dev)
        SAMPLES.clear()
        ts = []
        for _ in range(REPS):
            ttnn.deallocate(out)
            t0 = time.perf_counter(); out = run(split, pc); ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        clk = {n: (min(v), int(statistics.median(v)), max(v)) for n in NODES
               if (v := [r[n] for r in SAMPLES if r.get(n, -1) > 0])}
        o = ttnn.to_torch(out).float(); ttnn.deallocate(out)
        if ref is None:
            ref = o
        d = (o - ref)
        log(ev="arm", arm=name, split=split, ms=ts, ms_med=statistics.median(ts), aiclk=clk,
            tflops=2 * ROWS * K * N / statistics.median(ts) / 1e9,
            max_abs_vs_auto=float(d.abs().max()), rel_rms_vs_auto=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()))
    except Exception as e:
        log(ev="arm_fail", arm=name, err=str(e)[:300])
log(ev="end")
