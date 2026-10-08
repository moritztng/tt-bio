"""spd-msa: Protenix-v2's MSA-module ops, old path against new, on one chip.

Times (synced wall per call, the call is a chain of 20-40 multi-ms device ops so dispatch is noise)
and grades (error against a float64 host reference computed from the same bf16 inputs and weights):

* pwa: PairWeightedAveraging over one 512-row depth chunk, per-head loop vs fused heads
  (`TT_BIO_PWA_FUSED_HEADS`) vs fused without the head padding (`TT_BIO_PWA_UNPADDED`).
* opm: OuterProductMean over the trunk's depth-chunk list, chunked running sum vs one full-depth
  contraction (`TT_BIO_OPM_JOIN_PARTS`), then with the output projection as a row-block batch
  (`TT_BIO_OPM_PROJ_BATCH`), with the residual, scale_bias=True as Protenix builds it.

Shapes default to the campaign cell: 730 tokens padded to 736, MSA depth 9947, c_m 128, c_z 256.
AICLK of every Tenstorrent node is sampled every 0.25 s while the timed loops run.

usage: TT_VISIBLE_DEVICES=<chip> python bench_msa_ops.py OUT [pwa,opm] [TOKENS] [DEPTH] [REPS]
"""
import glob, hashlib, json, os, statistics, sys, threading, time
from pathlib import Path

OUT = Path(sys.argv[1])
WHAT = (sys.argv[2] if len(sys.argv) > 2 else "pwa,opm").split(",")
T_ = int(sys.argv[3]) if len(sys.argv) > 3 else 736
DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else 9947
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
SAMPLES, WINDOW = [], {"on": False}


def _sampler():
    while True:
        if WINDOW["on"]:
            row = {}
            for n in NODES:
                try:
                    row[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk")
                                 .read_text().split()[0])
                except Exception:
                    row[n] = -1
            SAMPLES.append(row)
        time.sleep(0.25)


threading.Thread(target=_sampler, daemon=True).start()


def clocks():
    """min/median/max AICLK per node over the samples since the last call."""
    out = {}
    for n in NODES:
        v = [r[n] for r in SAMPLES if r.get(n, -1) > 0]
        if v:
            out[n] = (min(v), int(statistics.median(v)), max(v), len(v))
    SAMPLES.clear()
    return out


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T

torch.manual_seed(0)
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
log(ev="start", what=WHAT, tokens=T_, depth=DEPTH, reps=REPS, nodes=NODES,
    visible=os.environ.get("TT_VISIBLE_DEVICES"), arch=str(dev.arch()) if hasattr(dev, "arch") else None)

C_M, C_Z, H, HD, C_OPM = 128, 256, 8, 8, 32
bf = lambda t: t.to(torch.bfloat16)


def lin_w(o, i):
    return bf(torch.randn(o, i) / i ** 0.5)


def up(t):
    return ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def err(x, ref):
    x = x.double(); d = x - ref
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                max_abs=float(d.abs().max()), finite=bool(torch.isfinite(x).all()))


def ln64(x, w, b):
    x = x.double()
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * w.double() + b.double()


def timed(fn, reps):
    fn()                                   # warm: program cache
    ttnn.synchronize_device(dev)
    WINDOW["on"] = True
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if _ < reps - 1:
            ttnn.deallocate(out)
    WINDOW["on"] = False
    return out, ts, clocks()


def bench_pwa():
    rows = min(512, DEPTH)
    sd = {"norm_m.weight": bf(1 + 0.1 * torch.randn(C_M)), "norm_m.bias": bf(0.1 * torch.randn(C_M)),
          "norm_z.weight": bf(1 + 0.1 * torch.randn(C_Z)), "norm_z.bias": bf(0.1 * torch.randn(C_Z)),
          "proj_m.weight": lin_w(H * HD, C_M), "proj_g.weight": lin_w(H * HD, C_M),
          "proj_z.weight": lin_w(H, C_Z), "proj_o.weight": lin_w(C_M, H * HD)}
    pwa = T.PairWeightedAveraging(HD, H, sd, ckc)
    m = bf(torch.randn(1, rows, T_, C_M)); z = bf(torch.randn(1, T_, T_, C_Z))
    m_tt, z_tt = up(m), up(z)
    ws = pwa.head_weights(z_tt, None)
    w_host = torch.cat([ttnn.to_torch(w).double().reshape(1, T_, T_) for w in ws])   # [H, T, T]
    # reference from the device's own token weights: this grades the head loop, not the softmax
    mn = ln64(m[0], sd["norm_m.weight"], sd["norm_m.bias"])
    v = mn @ sd["proj_m.weight"].double().t(); g = torch.sigmoid(mn @ sd["proj_g.weight"].double().t())
    v = v.reshape(rows, T_, H, HD); g = g.reshape(rows, T_, H, HD)
    o = torch.einsum("hij,rjhc->rihc", w_host, v) * g
    ref = o.reshape(rows, T_, H * HD) @ sd["proj_o.weight"].double().t()
    res = {}
    for arm, fused, unpad in (("loop", False, False), ("fused", True, False), ("unpadded", True, True)):
        T._PWA_FUSED_HEADS, T._PWA_UNPADDED = fused, unpad
        out, ts, clk = timed(lambda: pwa(m_tt, None, weights=ws), REPS)
        res[arm] = ttnn.to_torch(out).reshape(rows, T_, C_M)
        log(ev="pwa", arm=arm, rows=rows, tokens=T_, ms=ts, ms_med=statistics.median(ts),
            aiclk=clk, err=err(res[arm], ref), stats=list(T.PWA_FUSED_STATS) + list(T.PWA_UNPADDED_STATS))
        ttnn.deallocate(out)
    log(ev="pwa_ab", fused_vs_loop=err(res["fused"], res["loop"].double()),
        unpadded_vs_fused=err(res["unpadded"], res["fused"].double()),
        unpadded_bitident=bool(torch.equal(res["unpadded"], res["fused"])))


def bench_opm():
    sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C_M)), "norm.bias": bf(0.1 * torch.randn(C_M)),
          "proj_a.weight": lin_w(C_OPM, C_M), "proj_b.weight": lin_w(C_OPM, C_M),
          "proj_o.weight": lin_w(C_Z, C_OPM * C_OPM), "proj_o.bias": bf(0.1 * torch.randn(C_Z))}
    opm = T.OuterProductMean(sd, ckc, scale_bias=True)
    m = bf(torch.randn(1, DEPTH, T_, C_M)); z = bf(torch.randn(1, T_, T_, C_Z))
    chunks = [up(m[:, s:s + T.MSA_CHUNK_SIZE].contiguous()) for s in range(0, DEPTH, T.MSA_CHUNK_SIZE)]
    z_tt = up(z)
    # float64 reference on the first R token rows (the op is per row i, so a row subset is exact)
    R = 64
    a, b = [], []
    for s in range(0, DEPTH, 512):                                           # fp64 LN in slices
        mn = ln64(m[0, s:s + 512], sd["norm.weight"], sd["norm.bias"])       # [s, T, C_M]
        a.append(mn[:, :R] @ sd["proj_a.weight"].double().t())
        b.append(mn @ sd["proj_b.weight"].double().t())
    a, b = torch.cat(a), torch.cat(b)
    del mn
    outer = (a.permute(1, 2, 0).reshape(R * C_OPM, DEPTH) @ b.reshape(DEPTH, T_ * C_OPM))
    outer = outer.reshape(R, C_OPM, T_, C_OPM).permute(0, 2, 1, 3).reshape(R, T_, C_OPM * C_OPM) / DEPTH
    ref = outer @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double() / DEPTH + z[0, :R].double()
    del outer, a, b
    res = {}
    for arm, join, pb in (("chunked", False, False), ("joined", True, False), ("joined_pb", True, True)):
        T._OPM_JOIN_PARTS, T._OPM_PROJ_BATCH = join, pb
        out, ts, clk = timed(lambda: opm(chunks, None, None, residual=z_tt), REPS)
        res[arm] = ttnn.to_torch(out).reshape(T_, T_, C_Z)[:R]
        log(ev="opm", arm=arm, depth=DEPTH, tokens=T_, ms=ts, ms_med=statistics.median(ts), aiclk=clk,
            err=err(res[arm], ref), err_update=err(res[arm] - z[0, :R].double(), ref - z[0, :R].double()),
            stats=list(T.OPM_JOIN_PARTS_STATS),
            sha=hashlib.sha1(res[arm].contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:12])
        ttnn.deallocate(out)
    log(ev="opm_ab", joined_vs_chunked=err(res["joined"], res["chunked"].double()),
        proj_batch_bitident=bool(torch.equal(res["joined_pb"], res["joined"])))


for w in WHAT:
    {"pwa": bench_pwa, "opm": bench_opm}[w]()
log(ev="end")
