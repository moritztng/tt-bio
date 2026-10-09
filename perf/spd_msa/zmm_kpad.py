"""spd-msa: the OPM contraction z = a b^T against the depth it contracts over, padded with zero rows.

At depth 9947 K is 311 tiles, a prime, so a program config can only step K one tile at a time. Zero rows
added to both operands contribute exact zeros to every dot product, so padding K to a multiple of 8 or 32
tiles changes only how the matmul may block it. Per (M, K) arm: median ms over REPS synced calls, the
error of the output against the unpadded call and, on a 256-row probe, against float64.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_kpad.py OUT [DEPTH=9947] [REPS=5] [FID=hifi4]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 9947
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
FID = sys.argv[4] if len(sys.argv) > 4 else "hifi4"
LOG = open(OUT / "zmm_kpad.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().split()[0])
        except Exception:
            pass
    return out


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3}[FID],
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
I, C = 736, 32
N = I * C
torch.manual_seed(0)
a_h = (torch.randn(N, DEPTH) / DEPTH ** 0.5).bfloat16()
b_h = torch.randn(N, DEPTH).bfloat16()
PADS = [DEPTH] + [p for p in (-(-DEPTH // 256) * 256, -(-DEPTH // 1024) * 1024) if p != DEPTH]


def up(t, k):
    t = torch.nn.functional.pad(t, (0, k - t.shape[1]))
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


log(ev="start", depth=DEPTH, pads=PADS, fid=FID, arch=str(dev.arch()), aiclk=aiclk())
ref64 = a_h[:256].double() @ b_h.double().t()
base = {}
for k in PADS:
    a, b = up(a_h, k), up(b_h, k)
    for m in (N, 11264, 1024):
        am = a if m == N else a[:m, :]
        try:
            z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
            ts = []
            for _ in range(REPS):
                ttnn.deallocate(z)
                t0 = time.perf_counter()
                z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
            zh = ttnn.to_torch(z); ttnn.deallocate(z)
            rec = dict(ev="arm", k=k, kt=k // 32 + (k % 32 > 0), m=m, ms=ts, ms_med=statistics.median(ts),
                       tflops=2 * m * N * DEPTH / statistics.median(ts) / 1e9, aiclk=aiclk())
            if k == DEPTH:
                base[m] = zh
            else:
                rec.update(bitident=bool(torch.equal(zh, base[m])),
                           max_abs_vs_unpadded=float((zh.double() - base[m].double()).abs().max()))
            d = zh[:256].double() - ref64
            rec.update(rel_rms_f64=float(d.pow(2).mean().sqrt() / ref64.pow(2).mean().sqrt()),
                       finite=bool(torch.isfinite(zh).all()))
            log(**rec)
            del zh
        except Exception as e:
            log(ev="arm_fail", k=k, m=m, err=str(e)[:300])
        if am is not a:
            ttnn.deallocate(am)
    ttnn.deallocate(a); ttnn.deallocate(b)
log(ev="end")
