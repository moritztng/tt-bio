"""spd-msa: where the unpadded PWA head path spends its 18.6 ms (Wormhole, 512 rows x 736 tokens), and a
transposed formulation that skips most of the head regrouping.

* unpadded: `PairWeightedAveraging._heads_unpadded` step by step, each step synced and timed on its own.
* tposed: v and g are produced channel-major, [H*hd, rows*T], whose row-major bytes ARE [H, hd*rows, T]
  (flat index (h*hd+d)*rows*T + r*T + j either way), so the head regrouping is one reshape instead of two
  permutes and four layout passes. The head matmul runs against the token weights transposed, the gate
  multiplies in that layout, and the output projection contracts the channel axis with transpose_a.
  Same products and sums, so graded against the same float64 reference as bench_msa_ops.py.

* permfirst: one tiled permute puts the channel axis outermost, [H*hd, rows, T]; (head, dim) regroups by views.

usage: TT_VISIBLE_DEVICES=<chip> python pwa_opsplit.py OUT [ROWS=512] [TOKENS=736] [REPS=5] [ARMS=a,b,..]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
T_ = int(sys.argv[3]) if len(sys.argv) > 3 else 736
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
LOG = open(OUT / "opsplit.jsonl", "a")


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
C_M, H, HD = 128, 8, 8
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
bf = lambda t: t.to(torch.bfloat16)
up = lambda t, **k: ttnn.from_torch(bf(t), layout=TL, device=dev, dtype=ttnn.bfloat16, **k)
lin = dict(compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
log(ev="start", rows=ROWS, tokens=T_, reps=REPS, arch=str(dev.arch()), aiclk=aiclk())

mn = bf(torch.randn(ROWS, T_, C_M))                         # the normed chunk
wv, wg, wo = (bf(torch.randn(C_M, H * HD) / C_M ** 0.5), bf(torch.randn(C_M, H * HD) / C_M ** 0.5),
              bf(torch.randn(H * HD, C_M) / (H * HD) ** 0.5))
w = torch.softmax(torch.randn(H, T_, T_), -1)
w = bf(w)
v = mn.double() @ wv.double(); g = torch.sigmoid(mn.double() @ wg.double())
o = torch.einsum("hij,rjhc->rihc", w.double(), v.reshape(ROWS, T_, H, HD)) * g.reshape(ROWS, T_, H, HD)
ref = o.reshape(ROWS, T_, H * HD) @ wo.double()
del v, g, o

mc = up(mn); W = up(w); Wv, Wg, Wo = up(wv), up(wg), up(wo)
WvgT = up(torch.cat([wv, wg], 1).t().contiguous())          # [2*H*HD, C_M]


def err(x):
    x = x.double(); d = x - ref
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(d.abs().max()),
                finite=bool(torch.isfinite(x).all()))


def unpadded(s):
    rows, T = ROWS, T_
    s("v", lambda: ttnn.linear(mc, Wv, **lin))
    s("v_perm120", lambda x: ttnn.permute(x, (1, 2, 0)))
    s("v_to_rm", lambda x: ttnn.to_layout(x, RM))
    s("v_regroup_rm", lambda x: ttnn.permute(ttnn.reshape(x, (T, H, HD * rows)), (1, 0, 2)))
    s("v_to_tile", lambda x: ttnn.to_layout(x, TL))
    s("head_mm", lambda x: ttnn.matmul(W, x, **lin))
    s("o_to_rm", lambda x: ttnn.to_layout(x, RM))
    s("o_regroup_rm", lambda x: ttnn.reshape(ttnn.permute(x, (1, 0, 2)), (T, H * HD, rows)))
    s("o_to_tile", lambda x: ttnn.to_layout(x, TL))
    s("o_perm201", lambda x: ttnn.permute(x, (2, 0, 1)))
    s("g", lambda x: (x, ttnn.linear(mc, Wg, **lin)), keep=True)
    s("gate_mul", lambda xg: ttnn.multiply_(xg[0], xg[1], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
    s("o_proj", lambda x: ttnn.linear(x, Wo, **lin))


def tposed(s):
    rows, T = ROWS, T_
    s("vg_T", lambda: ttnn.matmul(WvgT, ttnn.reshape(mc, (rows * T, C_M)), transpose_b=True, **lin))
    s("split", lambda x: (ttnn.slice(x, (0, 0), (H * HD, rows * T)), ttnn.slice(x, (H * HD, 0), (2 * H * HD, rows * T))),
      keep=True)
    s("v_regroup", lambda vg: (ttnn.reshape(vg[0], (H, HD * rows, T)), vg[1]), keep=True)
    s("head_mm_T", lambda vg: (ttnn.matmul(vg[0], W, transpose_b=True, **lin), vg[1]), keep=True)
    s("o_regroup", lambda og: (ttnn.reshape(og[0], (H * HD, rows * T)), og[1]), keep=True)
    s("gate_mul", lambda og: ttnn.multiply_(og[0], og[1], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
    s("o_proj_Ta", lambda x: ttnn.matmul(x, Wo, transpose_a=True, **lin))
    s("view", lambda x: ttnn.reshape(x, (rows, T, C_M)))


def permfirst(s):
    """Channel axis moved outermost by one tiled permute; splitting it into (head, dim) and merging dim into the
    row axis leaves the last two dims' tiles untouched (rows % 32 == 0), so the regrouping is a view."""
    rows, T = ROWS, T_
    s("v", lambda: ttnn.linear(mc, Wv, **lin))
    s("v_perm201", lambda x: ttnn.permute(x, (2, 0, 1)))                 # [H*HD, rows, T]
    s("v_view", lambda x: ttnn.reshape(x, (H, HD * rows, T)))
    s("head_mm_T", lambda x: ttnn.matmul(x, W, transpose_b=True, **lin))  # [H, HD*rows, T]
    s("o_view", lambda x: ttnn.reshape(x, (H * HD, rows, T)))
    s("o_perm120", lambda x: ttnn.permute(x, (1, 2, 0)))                 # [rows, T, H*HD]
    s("g", lambda x: (x, ttnn.linear(mc, Wg, **lin)), keep=True)
    s("gate_mul", lambda xg: ttnn.multiply_(xg[0], xg[1], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
    s("o_proj", lambda x: ttnn.linear(x, Wo, **lin))


def run(name, body):
    """Each rep runs the chain once with a sync after every step; per-step ms is the median over reps."""
    steps, total, out = {}, [], None
    for rep in range(REPS + 1):
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
            if rep:
                steps.setdefault(step, []).append(dt)
            state["x"] = x

        body(s)
        if rep:
            total.append(t_all)
        out = state["x"]
    o = ttnn.to_torch(out).reshape(ROWS, T_, C_M)
    log(ev="arm", arm=name, total_ms=statistics.median(total), steps={k: round(statistics.median(v), 3)
        for k, v in steps.items()}, err=err(o), aiclk=aiclk())
    return o


res = {}
ARMS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["unpadded", "tposed", "permfirst"]
for name, body in ((a, globals()[a]) for a in ARMS):
    try:
        res[name] = run(name, body)
    except Exception as e:
        log(ev="fail", arm=name, err=str(e)[:400])
for a in (a for a in res if a != "unpadded" and "unpadded" in res):
    d = (res[a].double() - res["unpadded"].double()).abs()
    log(ev="ab", arm=a, max_abs=float(d.max()), bitident=bool(torch.equal(res[a], res["unpadded"])))
log(ev="end")
