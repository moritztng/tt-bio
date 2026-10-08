"""spd-msa: OuterProductMean's output projection [I*J, C*D] x [C*D, c_z] under different program configs.

At 730 tokens on Wormhole it is [541696, 1024] x [1024, 256] at 67 ms (4.2 TF/s; every bf16 config lpx-matmul tried
lands at 68 ms, bfp8 operands 36 ms). The suspicion: the auto config splits N over cores and re-reads the 1.1 GB
in0 once per N block. Arms keep all of N on each core (1D, in1 multicast) and split M into calls small enough for
the output block to fit L1. Each arm's output is graded against a float64 host product (max abs, rel rms, count of
elements off by more than 5 %); time is the median of
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


def run(split, cfg):
    """`split` calls over row blocks of at most `split` rows (None: one call). With a 1D config `cfg`
    = (pm, ibw, sb) each block gets per_core_M = its own tile rows over the cores, capped at pm, so a
    ragged last block is not handed a config sized for a full one."""
    def one(xs):
        if cfg is None:
            return ttnn.linear(xs, w, compute_kernel_config=ckc)
        pm = min(cfg[0], -(-int(xs.shape[0]) // 32 // NC))
        sbh = cfg[2][0] if pm % cfg[2][0] == 0 else 1
        return ttnn.linear(xs, w, compute_kernel_config=ckc, program_config=one_d(pm, cfg[1], sbh, cfg[2][1]))
    if isinstance(split, str):                     # "bK": K row blocks as a batch dim, a free view
        k = int(split[1:])
        o = ttnn.linear(ttnn.reshape(x, (k, ROWS // k, K)), w, compute_kernel_config=ckc)
        return ttnn.reshape(o, (ROWS, N))
    if split is None or split >= ROWS:
        return one(x)
    parts = [one(x[r:min(r + split, ROWS)]) for r in range(0, ROWS, split)]
    out = ttnn.concat(parts, dim=0)
    for p in parts:
        ttnn.deallocate(p)
    return out


arms = [("auto", None, None)]
for pm in (4, 8, 12, 16):
    rows = pm * 32 * NC
    for ibw, sb in ((2, (1, 4)), (4, (1, 4))):
        arms.append((f"1d_pm{pm}_ibw{ibw}_sb{sb[0]}x{sb[1]}", rows, (pm, ibw, sb)))
for k in (2, 4, 8, 16):
    arms.append((f"auto_split{k}", -(-ROWS // k // 32) * 32, None))
for k in (4, 8, 16, 32, 46):                       # ROWS // k must stay a whole number of tiles
    if (ROWS // 32) % k == 0:
        arms.append((f"auto_batch{k}", f"b{k}", None))

# float64 reference: every arm is graded against it, not against another bf16 arm
ref = (x_h.double() @ w_h.double()).float()
ref_rms = float(ref.pow(2).mean().sqrt())
auto = None
for name, split, cfg in arms:
    try:
        out = run(split, cfg); ttnn.synchronize_device(dev)
        SAMPLES.clear()
        ts = []
        for _ in range(REPS):
            ttnn.deallocate(out)
            t0 = time.perf_counter(); out = run(split, cfg); ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        clk = {n: (min(v), int(statistics.median(v)), max(v)) for n in NODES
               if (v := [r[n] for r in SAMPLES if r.get(n, -1) > 0])}
        o = ttnn.to_torch(out).float(); ttnn.deallocate(out)
        if auto is None:
            auto = o
        d = (o - ref).abs()
        bad = (d > 0.05 * (1 + ref.abs())).nonzero()
        log(ev="arm", arm=name, split=split, ms=ts, ms_med=statistics.median(ts), aiclk=clk,
            tflops=2 * ROWS * K * N / statistics.median(ts) / 1e9, finite=bool(torch.isfinite(o).all()),
            max_abs_vs_f64=float(d.max()), rel_rms_vs_f64=float(d.pow(2).mean().sqrt() / ref_rms),
            bad_vals=[(int(i), int(j), float(ref[i, j]), float(o[i, j])) for i, j in bad[:6].tolist()],
            n_bad=int(bad.shape[0]), bitident_auto=bool(torch.equal(o, auto)), bad_rows=sorted(set(bad[:, 0].tolist()))[:8])
    except Exception as e:
        log(ev="arm_fail", arm=name, err=str(e)[:300])
log(ev="end")
