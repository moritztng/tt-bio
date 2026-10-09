"""spd-msa: which zero-padding of the OPM contraction depth K to ship, across MSA depths.

zmm_kpad.py found K = 311 tiles (prime, depth 9947) 1.2-1.5x slower than 312 and 320 tiles slower again at
M = 11264. This sweeps the rounding rule: K tiles rounded up to a multiple of 1 (none), 2, 4 and 8, at the
depths the 11-set and the size ladder carry, at the whole-row M (23552) and the row-blocked M (11264).

usage: TT_VISIBLE_DEVICES=<chip> python zmm_kpad2.py OUT [FID=hifi3] [REPS=5] [DEPTHS=9947,12830,13602,5889,4097]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FID = sys.argv[2] if len(sys.argv) > 2 else "hifi3"
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
DEPTHS = [int(d) for d in (sys.argv[4] if len(sys.argv) > 4 else "9947,12830,13602,5889,4097").split(",")]
LOG = open(OUT / "zmm_kpad2.jsonl", "a")


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
N = 736 * 32
log(ev="start", depths=DEPTHS, fid=FID, arch=str(dev.arch()), aiclk=aiclk())
for depth in DEPTHS:
    kt = -(-depth // 32)
    torch.manual_seed(0)
    a_h = (torch.randn(N, depth) / depth ** 0.5).bfloat16()
    b_h = torch.randn(N, depth).bfloat16()
    seen = {}
    for mult in (1, 2, 4, 8):
        ktp = -(-kt // mult) * mult
        if ktp in seen.values():
            log(ev="same", depth=depth, mult=mult, kt=ktp); seen[mult] = ktp; continue
        seen[mult] = ktp
        k = ktp * 32 if mult > 1 else depth
        a = ttnn.from_torch(torch.nn.functional.pad(a_h, (0, k - depth)), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.nn.functional.pad(b_h, (0, k - depth)), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        for m in (N, 11264):
            am = a if m == N else a[:m, :]
            try:
                z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
                ts = []
                for _ in range(REPS):
                    ttnn.deallocate(z)
                    t0 = time.perf_counter()
                    z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                fin = bool(torch.isfinite(ttnn.to_torch(z[:64, :])).all())
                ttnn.deallocate(z)
                log(ev="arm", depth=depth, mult=mult, kt=ktp, m=m, ms=ts, ms_med=statistics.median(ts),
                    spread=max(ts) - min(ts), tflops=2 * m * N * depth / statistics.median(ts) / 1e9,
                    finite=fin, aiclk=aiclk())
            except Exception as e:
                log(ev="arm_fail", depth=depth, mult=mult, m=m, err=str(e)[:300])
            if am is not a:
                ttnn.deallocate(am)
        ttnn.deallocate(a); ttnn.deallocate(b)
log(ev="end")
