"""spd-msa: OPM's a/b projections per depth chunk, two c=32 linears over the same normed chunk vs one c=64
linear whose output is cut into a and b by two tile-aligned slices.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_msa/opm_abjoint.py OUT [ROWS=512] [TOKENS=736] [REPS=20]

Both arms compute every output element as the same 128-long dot product, so a and b must be torch.equal to the
two-linear arm. AICLK is read before and after each arm.
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
T_ = int(sys.argv[3]) if len(sys.argv) > 3 else 736
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
LOG = open(OUT / "abjoint.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            v = int((p / "tt_aiclk").read_text().split()[0])
            if v < 5000:
                out[p.name.split("!")[1]] = v
        except Exception:
            pass
    return out


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.manual_seed(0)
dev = T.get_device()
ckc = T.TenstorrentConfig().compute_kernel_config if hasattr(T, "TenstorrentConfig") else None
if ckc is None:
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
C_M, C = 128, 32
bf = lambda t: t.to(torch.bfloat16)
up = lambda t: ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
lin = dict(compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
log(ev="start", rows=ROWS, tokens=T_, reps=REPS, arch=str(dev.arch()), aiclk=aiclk(), ckc=str(ckc))

x = up(torch.randn(ROWS, T_, C_M))
nw, nb = up(torch.ones(C_M) + 0.1 * torch.randn(C_M)), up(0.1 * torch.randn(C_M))
wa, wb = torch.randn(C_M, C) / C_M ** 0.5, torch.randn(C_M, C) / C_M ** 0.5
ba, bb = 0.1 * torch.randn(C), 0.1 * torch.randn(C)
Wa, Wb, Ba, Bb = up(wa), up(wb), up(ba), up(bb)
Wab, Bab = up(torch.cat([wa, wb], 1)), up(torch.cat([ba, bb]))


def norm():
    return ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc)


def two():
    mc = norm()
    a = ttnn.linear(mc, Wa, bias=Ba, **lin)
    b = ttnn.linear(mc, Wb, bias=Bb, **lin)
    ttnn.deallocate(mc)
    return a, b


def joint():
    mc = norm()
    ab = ttnn.linear(mc, Wab, bias=Bab, **lin)
    ttnn.deallocate(mc)
    a = ttnn.slice(ab, (0, 0, 0), (ROWS, T_, C))
    b = ttnn.slice(ab, (0, 0, C), (ROWS, T_, 2 * C))
    ttnn.deallocate(ab)
    return a, b


def joint_only():
    """The c=64 linear alone, to split the arm's time between the matmul and the slices."""
    mc = norm()
    ab = ttnn.linear(mc, Wab, bias=Bab, **lin)
    ttnn.deallocate(mc)
    return ab, None


res = {}
for name, fn in (("two", two), ("joint", joint), ("joint_only", joint_only), ("two", two), ("joint", joint)):
    ts = []
    c0 = aiclk()
    for rep in range(REPS + 2):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        a, b = fn()
        ttnn.synchronize_device(dev)
        if rep >= 2:
            ts.append((time.perf_counter() - t0) * 1e3)
        if rep < REPS + 1:
            ttnn.deallocate(a)
            if b is not None:
                ttnn.deallocate(b)
    res.setdefault(name, (ttnn.to_torch(a), None if b is None else ttnn.to_torch(b)))
    log(ev="arm", arm=name, ms_median=statistics.median(ts), ms_min=min(ts), ms_max=max(ts), n=len(ts),
        aiclk_before=c0, aiclk_after=aiclk())
for name in ("joint",):
    log(ev="ab", arm=name, a_equal=bool(torch.equal(res[name][0], res["two"][0])),
        b_equal=bool(torch.equal(res[name][1], res["two"][1])),
        a_max_abs=float((res[name][0].double() - res["two"][0].double()).abs().max()),
        b_max_abs=float((res[name][1].double() - res["two"][1].double()).abs().max()))
log(ev="end")
