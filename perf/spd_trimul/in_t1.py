"""T1 (one kernel: layer_norm -> [a | b] = (xn @ Wp) * sigmoid(xn @ Wg) [* mask on a], written in the
einsum's [1, C, S, S] layout, plus g_out = xn @ Wg_out) against the incumbent c_z = 128 trimul input
side: rel rms and max abs against a float64 torch reference for both, for each of a, b and g_out, T1
finite, and median ms of 10 warm synchronized calls each. One JSON line per N.

The incumbent is the call site's own ops at 512 aa in normal mode: ttnn.layer_norm
(ln_compute_kernel_config), `_in_proj_matmul` with g_out's columns appended and split off (the fused
gate), two `reblock_permute_gated` moves, and the moved mask multiply on a.

T1 has NEVER RUN on a device before this script: run it under `timeout`, smallest N first, so a hang
costs one bounded run (synchronize_device does not return to a SIGALRM).

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_trimul/in_t1.py [--ns 64,128,512]
"""
import argparse, json, statistics, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="64,128,512")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--c", type=int, default=128, help="c_z")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--no-mask", action="store_true")
ap.add_argument("--ln1-ab", action="store_true", help="also run T1 with T1_LN1 off: bit-equality of a, b, g_out and its time")
ap.add_argument("--abl", default="", help="comma list of T1_ABL stage-skip masks to time after the real T1"
                " (kernels/trimul_in_t1: 1 LN, 2 g_out mm, 4 [a|b] mm, 8 gate, 16 z reads, 32 gather, 64 writes)")
ap.add_argument("--set", default="", help="comma list NAME=INT of trimul_tail knobs for T1 (T1_RB, T1_SBW, GIN_MOVE_NB, T1_ROWS)")
ap.add_argument("--grid", default="", help="gx,gy: run T1 on this core grid instead of COMPUTE_GRID_MAIN")
ap.add_argument("--w8", action="store_true", help="also run T1 on bfp8 weights (lever trimul_t1w8): its error and time")
ap.add_argument("--fid", default="", help="HiFi4/HiFi3/HiFi2/LoFi: run T1 at this fidelity instead of the trunk's")
ap.add_argument("--out", help="also append each JSON line here (the qb runner passes it)")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG, reblock_permute as RB
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
ckc = MG.ckc_args(kcfg)
if A.fid:
    ckc = (getattr(ttnn.MathFidelity, A.fid),) + tuple(ckc[1:])
grid = tuple(int(v) for v in A.grid.split(",")) if A.grid else tuple(T.COMPUTE_GRID_MAIN)
TT.set_t1(True)
TT.T1_C256 = 1
for kv in [v for v in A.set.split(",") if v]:
    k, v = kv.split("=")
    assert k in ("T1_RB", "T1_SBW", "GIN_MOVE_NB", "T1_B8", "T1_ROWS"), k
    setattr(TT, k, int(v))
C, h = A.c, A.hidden
gen = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print(json.dumps({"set": A.set, "grid": grid, "ckc": [str(c) for c in ckc], "c": C, "hidden": h,
                  "mask": not A.no_mask}), flush=True)


def rel(a, r):
    a = a.double()
    return (float((a - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()),
            float((a - r).abs().max()))


def bad_units(a, r, il):
    """The units (xt, h, yt) of `a` [1, h, n, n] whose rel rms vs `r` exceeds 0.02, each with its
    core and its position among that core's units (the order the kernels run them in)."""
    n = r.shape[-1]
    St, H = n // 32, 32 // il
    e = (a.double() - r).pow(2).reshape(-1, St, H, il, St, 32).sum((0, 3, 5))
    s = r.pow(2).reshape(-1, St, H, il, St, 32).sum((0, 3, 5)).clamp_min(1e-30)
    cores = TT.t1_split(St * St * H, grid[0] * grid[1])
    out = []
    for xt, hh, yt in (e / s).sqrt().gt(0.02).nonzero().tolist():
        u = (xt * H + hh) * St + yt
        c = next(i for i, (u0, k) in enumerate(cores) if u0 <= u < u0 + k)
        out.append([u, xt, hh, yt, c, u - cores[c][0], cores[c][1]])
    return out


def unit_map(a, r, il, units):
    """For each (xt, h, yt) unit: rel rms of `a` by channel tile and by row of the unit (where in the
    unit the error sits: the last channel tile only would mean the unit's tail races the next unit)."""
    out = []
    for _, xt, hh, yt, *_ in units:
        rows = slice(xt * 32 + hh * il, xt * 32 + hh * il + il)
        e = (a[0, :, rows, yt * 32:yt * 32 + 32].double() - r[0, :, rows, yt * 32:yt * 32 + 32])
        rr = r[0, :, rows, yt * 32:yt * 32 + 32]
        ct = e.shape[0] // 32
        q = (e.reshape(ct, -1).pow(2).sum(1) / rr.reshape(ct, -1).pow(2).sum(1).clamp_min(1e-30)).sqrt()
        rw = (e.pow(2).sum((0, 2)) / rr.pow(2).sum((0, 2)).clamp_min(1e-30)).sqrt()
        out.append({"unit": [xt, hh, yt], "by_q": [round(float(v), 4) for v in q],
                    "by_row": [round(float(v), 4) for v in rw]})
    return out


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


def free(ts):
    for t in ts:
        if t is not None and t.is_allocated():
            ttnn.deallocate(t)


# Checkpoint-shaped weights (torch Linear [out, in]), as TriangleMultiplication holds them.
w = {"g_in": torch.randn(2 * h, C, generator=gen) / C ** 0.5,
     "p_in": torch.randn(2 * h, C, generator=gen) / C ** 0.5,
     "g_out": torch.randn(C, C, generator=gen) / C ** 0.5,
     "gamma": 1 + 0.1 * torch.randn(C, generator=gen), "beta": 0.1 * torch.randn(C, generator=gen)}
w = {k: v.to(torch.bfloat16) for k, v in w.items()}
g_in_t, p_in_t = w["g_in"].t(), w["p_in"].t()
(fused,) = T.TriangleMultiplication._gp_fused_order(None, (g_in_t, p_in_t), h, 1)
wpT, wgT, wgo = TT.t1_host_weights(g_in_t, p_in_t, w["g_out"].t())
dw = {"gamma": up(w["gamma"]), "beta": up(w["beta"]), "wpT": up(wpT), "wgT": up(wgT),
      "wgo": up(wgo), "w_fused": up(torch.cat([fused, w["g_out"].t()], -1)), "w_gp": up(fused)}
off = lambda r: T.gp_off(r, h)  # noqa: E731
up8 = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
dw8 = {"wpT": up8(wpT), "wgT": up8(wgT), "wgo": up8(wgo)} if A.w8 else {}

bad = 0
for n in [int(s) for s in A.ns.split(",") if s]:
    rec = {"n": n}
    # z O(1) with a per-row offset the norm must remove.
    z = (torch.randn(1, n, n, C, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen)).to(torch.bfloat16)
    m = None if A.no_mask else (torch.rand(1, n, n, generator=gen) > 0.1).to(torch.bfloat16)
    dz = up(z)
    dm = None if m is None else up(m)
    d = torch.float64
    zn = torch.nn.functional.layer_norm(z.to(d), (C,), w["gamma"].to(d), w["beta"].to(d), 1e-5)
    gi, pi = zn @ w["g_in"].to(d).t(), zn @ w["p_in"].to(d).t()
    a_ref = pi[..., :h] * torch.sigmoid(gi[..., :h])
    if m is not None:
        a_ref = a_ref * m.to(d)[..., None]
    ref = {"a": a_ref.permute(0, 3, 1, 2), "b": (pi[..., h:] * torch.sigmoid(gi[..., h:])).permute(0, 3, 1, 2),
           "g": zn @ w["g_out"].to(d).t()}
    P = n * n * C * 2

    def incumbent():
        xn = ttnn.layer_norm(dz, weight=dw["gamma"], bias=dw["beta"], epsilon=1e-5,
                             compute_kernel_config=T.ln_compute_kernel_config(kcfg))
        pair = T._in_proj_matmul(xn, dw["w_fused"], kcfg, ttnn.DRAM_MEMORY_CONFIG, None,
                                 split=(4 * h, C))
        if pair is None:      # the split refused: the gate's own projection, as the tail would run it
            gp = T._in_proj_matmul(xn, dw["w_gp"], kcfg, ttnn.DRAM_MEMORY_CONFIG, None)
            g = T._trimul_out_proj(xn, dw["wgo"], kcfg, None)
        else:
            gp, g = pair
        ttnn.deallocate(xn)
        a = RB.reblock_permute_gated(gp, off("p_a"), off("g_a"), h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = RB.reblock_permute_gated(gp, off("p_b"), off("g_b"), h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gp)
        if dm is not None:
            a = ttnn.multiply_(a, ttnn.unsqueeze(dm, 1))
        return [a, b, g]

    def t1():
        return TT.fused_in_ln(dz, dw["gamma"], dw["beta"], dw["wpT"], dw["wgT"], dw["wgo"], ckc, grid,
                              mask=dm)

    try:
        outs = incumbent()
        for k, o in zip("abg", outs):
            rec[f"inc_{k}_rel_rms_f64"], rec[f"inc_{k}_max_abs"] = rel(ttnn.to_torch(o).reshape(ref[k].shape), ref[k])
        free(outs)
        rec["inc_ms"] = timed(lambda: free(incumbent()), A.reps)
    except Exception as e:                                                    # noqa: BLE001
        rec["inc_err"] = str(e).splitlines()[0][:300]

    try:
        rec["t1_why"] = TT.t1_eligible(dz, dw["gamma"], dw["beta"], dw["wpT"], dw["wgT"], dw["wgo"], dm)
        outs = t1()
        if outs is None:
            rec["t1_err"] = f"declined: {rec['t1_why']}"
        else:
            ttnn.synchronize_device(dev)
            host = [ttnn.to_torch(o) for o in outs]
            rec["t1_finite"] = all(bool(torch.isfinite(t).all()) for t in host)
            for k, t in zip("abg", host):
                rec[f"t1_{k}_rel_rms_f64"], rec[f"t1_{k}_max_abs"] = rel(t.reshape(ref[k].shape), ref[k])
            if m is not None:
                il = TT.T1_PLANS.get((C // 32, 2 * h // 32, C // 32), (1, 32))[1]
                bu = bad_units(host[0].reshape(ref["a"].shape), ref["a"], il)
                rec["a_bad_units"], rec["a_bad_first"] = len(bu), bu[:24]   # [u, xt, h, yt, core, pos, units on core]
                rec["a_bad_map"] = unit_map(host[0].reshape(ref["a"].shape), ref["a"], il, bu[:3])
            free(outs)
            rec["t1_ms"] = timed(lambda: free(t1()), A.reps)
            rec["t1_GBps"] = round(4 * P / rec["t1_ms"] / 1e6, 1)
            if "inc_ms" in rec:
                rec["speedup"] = round(rec["inc_ms"] / rec["t1_ms"], 3)
            for m_ in [int(v) for v in A.abl.split(",") if v]:
                TT.T1_ABL = m_
                rec[f"abl{m_}_ms"] = timed(lambda: free(t1()), A.reps)
            TT.T1_ABL = 0
            if A.ln1_ab:
                prev, TT.T1_LN1 = TT.T1_LN1, not TT.T1_LN1
                outs = t1()
                ttnn.synchronize_device(dev)
                rec["ln1_flip_equal"] = [bool(torch.equal(ttnn.to_torch(o), t)) for o, t in zip(outs, host)]
                free(outs)
                rec["ln1_flip_ms"] = timed(lambda: free(t1()), A.reps)
                TT.T1_LN1 = prev
            if A.w8:
                t1w = lambda: TT.fused_in_ln(dz, dw["gamma"], dw["beta"], dw8["wpT"], dw8["wgT"],  # noqa: E731
                                             dw8["wgo"], ckc, grid, mask=dm)
                rec["w8_why"] = TT.t1_eligible(dz, dw["gamma"], dw["beta"], dw8["wpT"], dw8["wgT"], dw8["wgo"], dm)
                outs = t1w()
                if outs is None:
                    rec["w8_err"] = f"declined: {rec['w8_why']}"
                else:
                    ttnn.synchronize_device(dev)
                    for k, o in zip("abg", outs):
                        t = ttnn.to_torch(o)
                        rec[f"w8_{k}_finite"] = bool(torch.isfinite(t).all())
                        rec[f"w8_{k}_rel_rms_f64"], rec[f"w8_{k}_max_abs"] = rel(t.reshape(ref[k].shape), ref[k])
                    free(outs)
                    rec["w8_ms"] = timed(lambda: free(t1w()), A.reps)
                    for m_ in [int(v) for v in A.abl.split(",") if v]:
                        TT.T1_ABL = m_
                        rec[f"w8_abl{m_}_ms"] = timed(lambda: free(t1w()), A.reps)
                    TT.T1_ABL = 0
    except Exception as e:                                                    # noqa: BLE001
        rec["t1_err"] = str(e).splitlines()[0][:300]
        traceback.print_exc()
    rec["t1_stats"] = list(TT.T1_STATS)
    bad += "t1_err" in rec or "inc_err" in rec
    print(json.dumps(rec), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    free([dz, dm])
free(dw.values())
free(dw8.values())
sys.exit(1 if bad else 0)
