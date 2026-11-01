"""Boltz-2 PWA (8 heads x 32, c_m 64) at served MSA depth: the shipped fused head path step by step,
against a channel-major formulation with no in-tile transpose of the 256-wide v, g or o.

* fused: `PairWeightedAveraging._heads_fused` as it runs for Boltz-2 (head_dim 32 takes it, see
  `_PWA_FULL_HEADS_FUSED`). Two WH transposes and two outer permutes of a [rows, T, 256] tensor per block.
* cmajor: m^T [rows, 64, T] by one WH transpose of the 64-wide input. v^T and g^T come out of a batched
  W^T @ m^T (ttnn cannot broadcast in0 over the batch, so the [256, 64] weight is repeated over rows:
  21 MB at 320 rows). [rows, 256, T] splits into [rows, 8, 32, T] on tile rows, so heads reach the batch
  axis by one outer tile permute and back by another. The gate multiplies in that layout, Wo^T @ (o*g)^T
  sums the heads in K, and one WH transpose of the 64-wide result returns [rows, T, 64].
  Same products and sums; graded against the same float64 reference.

usage: TT_VISIBLE_DEVICES=<chip> python pwa32_split.py OUT [ROWS=320] [TOKENS=768] [REPS=5] [ARMS=fused,cmajor]
       (more arms: cmajor_hifi3, cmajor_hifi2 = cmajor with the head contraction at that fidelity;
        cmajor_pret = token weights pre-transposed, no transpose_b; cmajor_nof32 = head contraction without
        fp32 dest accumulation (twice the dest tiles per subblock); cmajor_hifi2_nof32 = both;
        cmajor_w8 = token weights in bfloat8_b;
        phead = per-head channel-major: each head's v^T [rows, 32, T] is already [rows*32, T] on tile rows, so the
        head contraction is 8 batch-1 matmuls on ttnn's planner with no outer permute either side; o*g per
        head, one concat back to [rows, 256, T]; phead_cg = the same with core_grid on the head matmuls)
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 320
T_ = int(sys.argv[3]) if len(sys.argv) > 3 else 768
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
ARMS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["fused", "cmajor"]
LOG = open(OUT / "split.jsonl", "a")


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

torch.manual_seed(0)
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
C_M, H, HD = 64, 8, 32
TL = ttnn.TILE_LAYOUT
bf = lambda t: t.to(torch.bfloat16)
up = lambda t: ttnn.from_torch(bf(t), layout=TL, device=dev, dtype=ttnn.bfloat16)
lin = dict(compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
log(ev="start", rows=ROWS, tokens=T_, reps=REPS, arch=str(dev.arch()), aiclk=aiclk())

mn = bf(torch.randn(ROWS, T_, C_M))
wv, wg = bf(torch.randn(C_M, H * HD) / C_M ** 0.5), bf(torch.randn(C_M, H * HD) / C_M ** 0.5)
wo = bf(torch.randn(H * HD, C_M) / (H * HD) ** 0.5)
w = bf(torch.softmax(torch.randn(H, T_, T_), -1))
v = mn.double() @ wv.double(); g = torch.sigmoid(mn.double() @ wg.double())
o = torch.einsum("hij,rjhc->rihc", w.double(), v.reshape(ROWS, T_, H, HD)) * g.reshape(ROWS, T_, H, HD)
ref = o.reshape(ROWS, T_, H * HD) @ wo.double()
del v, g, o

mc = up(mn); W = up(w); Wv, Wg, Wo = up(wv), up(wg), up(wo)
rep = lambda x: up(x.t().contiguous().unsqueeze(0).expand(ROWS, -1, -1).contiguous())
WvT, WgT, WoT = rep(wv), rep(wg), rep(wo)                     # [rows, 256, 64] x2, [rows, 64, 256]


def proj(x, wt):
    o = T._narrow_proj_linear(x, wt, ckc, x.dtype) if T._PWA_PROJ else None
    return ttnn.linear(x, wt, **lin) if o is None else o


def err(x):
    x = x.double(); d = x - ref
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(d.abs().max()),
                finite=bool(torch.isfinite(x).all()))


def fused(s):
    rows, T = ROWS, T_
    s("v", lambda: proj(mc, Wv))                                                   # [rows, T, 256]
    s("v_tWH", lambda x: ttnn.permute(x, (0, 2, 1)))                              # [rows, 256, T]
    s("v_outer", lambda x: ttnn.reshape(ttnn.permute(ttnn.reshape(x, (rows, H, HD, T)), (1, 0, 2, 3)),
                                        (H, rows * HD, T)))
    s("head_mm", lambda x: ttnn.matmul(x, W, transpose_b=True, **lin))
    s("o_outer", lambda x: ttnn.reshape(ttnn.permute(ttnn.reshape(x, (H, rows, HD, T)), (1, 0, 2, 3)),
                                        (rows, H * HD, T)))
    s("o_tWH", lambda x: ttnn.permute(x, (0, 2, 1)))                              # [rows, T, 256]
    s("g", lambda x: (x, proj(mc, Wg)), keep=True)
    s("gate_mul", lambda xg: ttnn.multiply_(xg[0], xg[1], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
    s("o_proj", lambda x: proj(x, Wo))


def cmajor(s, head_lin=lin, Wh=None):
    rows, T = ROWS, T_
    s("mT", lambda: ttnn.transpose(mc, -2, -1))                                    # [rows, 64, T]
    s("vT", lambda m: (m, ttnn.matmul(WvT, m, **lin)), keep=True)                  # [rows, 256, T]
    s("v_outer", lambda mv: (mv[0], ttnn.reshape(ttnn.permute(ttnn.reshape(mv[1], (rows, H, HD, T)),
                                                              (1, 0, 2, 3)), (H, rows * HD, T))), keep=True)
    s("head_mm", lambda mv: (mv[0], ttnn.matmul(mv[1], W, transpose_b=True, **head_lin) if Wh is None
                             else ttnn.matmul(mv[1], Wh, **head_lin)), keep=True)
    s("o_outer", lambda mo: (mo[0], ttnn.reshape(ttnn.permute(ttnn.reshape(mo[1], (H, rows, HD, T)),
                                                              (1, 0, 2, 3)), (rows, H * HD, T))), keep=True)
    s("gT", lambda mo: (mo[1], ttnn.matmul(WgT, mo[0], **lin)), keep=True)
    s("gate_mul", lambda og: ttnn.multiply_(og[0], og[1], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
    s("outT", lambda x: ttnn.matmul(WoT, x, **lin))                                # [rows, 64, T]
    s("out_tWH", lambda x: ttnn.transpose(x, -2, -1))                              # [rows, T, 64]


WvTh = [up(wv[:, h * HD:(h + 1) * HD].t().contiguous().unsqueeze(0).expand(ROWS, -1, -1).contiguous())
        for h in range(H)]
WgTh = [up(wg[:, h * HD:(h + 1) * HD].t().contiguous().unsqueeze(0).expand(ROWS, -1, -1).contiguous())
        for h in range(H)]
Wh1 = [up(w[h].unsqueeze(0)) for h in range(H)]                 # [1, T, T] per head, as head_weights returns


def phead(s, head_lin=dict(compute_kernel_config=ckc)):
    rows, T = ROWS, T_
    s("mT", lambda: ttnn.transpose(mc, -2, -1))
    s("vT", lambda m: (m, [ttnn.reshape(ttnn.matmul(WvTh[h], m, **lin), (1, rows * HD, T)) for h in range(H)]),
      keep=True)
    s("head_mm", lambda mv: (mv[0], [ttnn.matmul(v, Wh1[h], transpose_b=True, **head_lin)
                                     for h, v in enumerate(mv[1])]), keep=True)
    s("gT", lambda mo: (mo[1], [ttnn.matmul(WgTh[h], mo[0], **lin) for h in range(H)]), keep=True)
    s("gate_mul", lambda og: [ttnn.multiply_(ttnn.reshape(o, (rows, HD, T)), g,
                                             input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                              for o, g in zip(*og)])
    s("concat", lambda os: ttnn.concat(os, dim=1))                                 # [rows, 256, T]
    s("outT", lambda x: ttnn.matmul(WoT, x, **lin))
    s("out_tWH", lambda x: ttnn.transpose(x, -2, -1))


phead_cg = lambda s: phead(s, lin)


def run(name, body):
    """Each rep runs the chain once with a sync after every step; per-step ms is the median over reps."""
    steps, total, out = {}, [], None
    for r in range(REPS + 1):
        state = {"x": None}
        t_all = 0.0

        def s(step, fn, keep=False):
            nonlocal t_all
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            x = fn() if state["x"] is None else fn(state["x"])
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) * 1e3
            t_all += dt
            if r:
                steps.setdefault(step, []).append(dt)
            state["x"] = x

        body(s)
        if r:
            total.append(t_all)
        out = state["x"]
    o = ttnn.to_torch(out).reshape(ROWS, T_, C_M)
    log(ev="arm", arm=name, total_ms=statistics.median(total), steps={k: round(statistics.median(v), 3)
        for k, v in steps.items()}, err=err(o), aiclk=aiclk())
    return o


def _fid(f, f32=True):
    k = ttnn.WormholeComputeKernelConfig(math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=f32,
                                         packer_l1_acc=True)
    return lambda s: cmajor(s, dict(lin, compute_kernel_config=k))


# cmajor with only the head contraction changed: its time and its error against the same float64 reference.
# pwa3 read HiFi2 at the HiFi4 time (WH 4.81 vs 4.93 ms per 320 block), so math is not what bounds it.
cmajor_hifi3, cmajor_hifi2 = _fid(ttnn.MathFidelity.HiFi3), _fid(ttnn.MathFidelity.HiFi2)
cmajor_nof32 = _fid(ttnn.MathFidelity.HiFi4, f32=False)
cmajor_hifi2_nof32 = _fid(ttnn.MathFidelity.HiFi2, f32=False)
Wt = up(w.transpose(-2, -1).contiguous())
cmajor_pret = lambda s: cmajor(s, Wh=Wt)
W8 = ttnn.from_torch(bf(w.transpose(-2, -1).contiguous()), layout=TL, device=dev, dtype=ttnn.bfloat8_b)
cmajor_w8 = lambda s: cmajor(s, Wh=W8)

res = {}
for name in ARMS:
    try:
        res[name] = run(name, globals()[name])
    except Exception as e:
        log(ev="fail", arm=name, err=str(e)[:600])
if "fused" in res and "cmajor" in res:
    d = (res["cmajor"].double() - res["fused"].double()).abs()
    log(ev="ab", max_abs=float(d.max()), bitident=bool(torch.equal(res["cmajor"], res["fused"])))
log(ev="end")
