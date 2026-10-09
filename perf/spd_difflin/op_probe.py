"""Device time and float64 error of Protenix's diffusion linears under operand formats and K blockings.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_difflin/op_probe.py --out probe.jsonl [--shapes qkv,a1] [--fam normal,fast]

Each shape is a ttnn.linear the fold runs every denoise step (census r10 rows, c730, 5 samples). The input is
fp32 N(0, 1) (a layer-norm output), the weight fp32 N(0, 1/k), so the output is ~N(0, 1); the reference is float64.

Families:
* normal: fp32 dest acc, HiFi4, fp32 output, as normal mode runs the diffusion. `f32` is today's call (fp32
  activation and weight); `a16` casts the activation to bf16 on device first (the cast is timed apart, `cast_us`);
  `a16w16` also stores the weight in bf16. `auto` is ttnn's own program, `k1` an explicit 2D multicast program with
  in0_block_w 1: one K tile per dest pass, the only blocking that is clean of Wormhole's fp32-accumulation erratum
  (state/spd-wherr.md). Subblocks are swept and every one is reported.
* fast: fp32 dest acc off, bf16 output, bf16 activation, as fast mode runs the diffusion (`w16 HiFi4 auto` is today's).

Device time is a captured trace replayed back to back (host dispatch excluded), median of 7 batches. A pixel is
wrong under spd-wherr's rule: error above 8 bf16 ulps of the float64 value and above 16x the call's rms error.
"""
import argparse, json, math, os, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

# name: (M rows, K, N, bias)  -- normal mode pads the 730 tokens to 768 (dit_sdpa32), fast keeps 730.
SHAPES = {
    "qkv": (768, 768, 3072, True),     # tenstorrent.py AttentionPairBias qkv
    "g": (768, 768, 768, False),       # AttentionPairBias gate
    "o": (768, 768, 768, False),       # AttentionPairBias out
    "a1": (768, 768, 1536, False),     # conditioned transition a1 (silu) / a2
    "b": (768, 1536, 768, False),      # conditioned transition b
    "s": (768, 384, 768, True),        # linear_s / linear_a_last on s_t
    "at128": (5919, 128, 128, False),  # atom transformer
    "at256": (5919, 128, 256, False),
    "at256b": (5919, 256, 128, False),
}
SAMPLES = 5


def opened_nodes():
    out = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            out.add(int(t.rsplit("/", 1)[1]))
    return sorted(out)


def aiclk(nodes):
    """AICLK of the opened node(s), read right after a timed arm; a dead ARC's 0xFFFFFFFF is dropped."""
    vals = []
    for n in nodes:
        try:
            v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            if 100 <= v <= 3000:
                vals.append(v)
        except Exception:
            pass
    return min(vals) if vals else None


def wrong(R, Y):
    err = (Y - R).abs()
    rms = err.pow(2).mean().sqrt().item()
    ulp = torch.exp2(torch.floor(torch.log2(R.abs().clamp_min(1e-30))) - 7)
    bad = (err > 8 * ulp) & (err > 16 * rms)
    return int(bad.sum()), float(err.max()), rms / R.pow(2).mean().sqrt().item()


def dev_time(dev, fn, reps=7, target_s=0.01, nodes=()):
    out = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(dev, cq_id=0); o = fn(); ttnn.end_trace_capture(dev, tid, cq_id=0)
    run = lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev); t0 = time.perf_counter(); run(); ttnn.synchronize_device(dev)
    k = max(1, min(100, math.ceil(target_s / max(time.perf_counter() - t0, 1e-6))))
    per, clk = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(k):
            run()
        c = aiclk(nodes)   # read while the queued replays are still running
        ttnn.synchronize_device(dev); per.append((time.perf_counter() - t0) / k)
        if c:
            clk.append(c)
    ttnn.release_trace(dev, tid); ttnn.deallocate(o)
    per.sort()
    return per[len(per) // 2] * 1e6, (per[-1] - per[0]) / per[len(per) // 2], (min(clk) if clk else None)


def programs(mt, kt, nt, grid, acc, ibws):
    """2D multicast programs, every legal subblock (h*w <= 4 under fp32 acc, 8 without), for each in0_block_w."""
    gx, gy = grid
    pm, pn = -(-mt // gy), -(-nt // gx)
    cap = 4 if acc else 8
    out = []
    for ibw in ibws:
        if kt % ibw:
            continue
        subs = sorted(((h, w) for h in range(1, pm + 1) for w in range(1, pn + 1)
                       if pm % h == 0 and pn % w == 0 and h * w <= cap), key=lambda s: -s[0] * s[1])[:3]
        for h, w in subs:
            out.append((f"2d_k{ibw}_pm{pm}_pn{pn}_sb{h}x{w}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
                out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                fused_activation=None, fuse_batch=True)))
    # Narrow N (atom linears): split rows over every core, weight whole on each.
    ncores = gx * gy
    pm1 = -(-mt // ncores)
    if nt <= 8:
        for ibw in ibws:
            if kt % ibw:
                continue
            w = max(x for x in range(1, min(cap, nt) + 1) if nt % x == 0)
            out.append((f"1d_k{ibw}_pm{pm1}_sb1x{w}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=ibw, out_subblock_h=1, out_subblock_w=w,
                out_block_h=pm1, out_block_w=nt, per_core_M=pm1, per_core_N=nt, fuse_batch=True,
                fused_activation=None, mcast_in0=False)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--fam", default="normal,fast")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = T.get_device(trace="protenix")
    g = dev.compute_with_storage_grid_size(); grid = (g.x, g.y)
    arch = T.arch_name()
    nodes = opened_nodes()
    log = a.out.open("a")

    def emit(**kw):
        kw.setdefault("aiclk", aiclk(nodes)); kw.update(arch=arch, grid=grid, t=time.time())
        log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw), flush=True)

    def ckc(fid, acc):
        return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                      math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=True)

    emit(ev="start", shapes=a.shapes, fam=a.fam)
    for name in a.shapes.split(","):
        rows, k, n, has_bias = SHAPES[name]
        for fam in a.fam.split(","):
            tok = rows if fam == "normal" else min(rows, 730) if rows == 768 else rows
            m = SAMPLES * tok
            mt, kt, nt = -(-m // 32), k // 32, n // 32
            acc = fam == "normal"
            arms = []   # (label, in dtype, w dtype, out dtype, fid, program or None)
            if fam == "normal":
                for ind, wd in (("f32", "f32"), ("bf16", "f32"), ("bf16", "bf16")):
                    lab = {"f32": "f32", "bf16": "a16"}[ind] + ("w16" if wd == "bf16" else "")
                    arms.append((lab + "_auto", ind, wd, "f32", "HiFi4", None))
                    for pl, pc in programs(mt, kt, nt, grid, True, (1,)):
                        arms.append((f"{lab}_{pl}", ind, wd, "f32", "HiFi4", pc))
                for fid in ("HiFi3", "HiFi2"):
                    for pl, pc in programs(mt, kt, nt, grid, True, (1,))[:1]:
                        arms.append((f"a16w16_{pl}_{fid}", "bf16", "bf16", "f32", fid, pc))
            else:
                for wd in ("bf16", "b8"):
                    for fid in ("HiFi4", "HiFi2"):
                        arms.append((f"w{wd}_{fid}_auto", "bf16", wd, "bf16", fid, None))
                arms.append(("a8wb8_HiFi2_auto", "b8", "b8", "bf16", "HiFi2", None))
                for pl, pc in programs(mt, kt, nt, grid, False, (2, 4, 8)):
                    arms.append((f"wbf16_HiFi4_{pl}", "bf16", "bf16", "bf16", "HiFi4", pc))
                    arms.append((f"wb8_HiFi2_{pl}", "bf16", "b8", "bf16", "HiFi2", pc))
            DT = dict(f32=ttnn.float32, bf16=ttnn.bfloat16, b8=ttnn.bfloat8_b)
            res = {}
            for d in range(a.draws):
                torch.manual_seed(1000 + d)
                A = torch.randn(SAMPLES, tok, k); W = torch.randn(k, n) / k ** 0.5
                Bv = torch.randn(n) * 0.1 if has_bias else None
                R = A.double() @ W.double() + (Bv.double() if has_bias else 0)
                ta = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                tw = {wd: ttnn.from_torch(W, dtype=DT[wd], layout=ttnn.TILE_LAYOUT, device=dev)
                      for wd in ("f32", "bf16", "b8")}
                tb = {wd: ttnn.from_torch(Bv.reshape(1, n), dtype=DT[wd] if wd != "b8" else ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=dev) for wd in ("f32", "bf16", "b8")} \
                    if has_bias else None
                cast = {}
                for ind in ("bf16", "b8"):
                    cast[ind] = ttnn.typecast(ta, DT[ind])
                if d == 0:
                    for ind in ("bf16", "b8"):
                        us, sp, ck = dev_time(dev, lambda ind=ind: ttnn.typecast(ta, DT[ind]), nodes=nodes)
                        emit(ev="cast", shape=name, fam=fam, to=ind, us=round(us, 1), spread=round(sp, 3), aiclk=ck)
                for lab, ind, wd, od, fid, pc in arms:
                    x = ta if ind == "f32" else cast[ind]
                    kw = dict(bias=tb[wd] if has_bias else None, compute_kernel_config=ckc(fid, acc), dtype=DT[od])
                    if pc is None:
                        kw["core_grid"] = ttnn.CoreGrid(y=grid[1], x=grid[0])
                    else:
                        kw["program_config"] = pc
                    fn = lambda x=x, wd=wd, kw=kw: ttnn.linear(x, tw[wd], **kw)
                    try:
                        y = fn()
                    except Exception as e:
                        if d == 0:
                            emit(ev="arm", shape=name, fam=fam, arm=lab, error=str(e).splitlines()[0][:240])
                        continue
                    Y = ttnn.to_torch(y).double().reshape(R.shape); ttnn.deallocate(y)
                    nb, mx, rel = wrong(R, Y)
                    r = res.setdefault(lab, dict(wrong=0, max_err=0.0, rel=[], us=None, spread=None, clk=None))
                    r["wrong"] += nb; r["max_err"] = max(r["max_err"], mx); r["rel"].append(rel)
                    if d == 0:
                        r["us"], r["spread"], r["clk"] = dev_time(dev, fn, nodes=nodes)
                for t in [ta, *tw.values(), *cast.values(), *((tb or {}).values())]:
                    ttnn.deallocate(t)
            for lab, r in res.items():
                emit(ev="arm", shape=name, fam=fam, m=m, k=k, n=n, arm=lab, us=r["us"] and round(r["us"], 1),
                     spread=r["spread"] and round(r["spread"], 3), wrong=r["wrong"], max_err=round(r["max_err"], 5),
                     rel_rms=float(f"{sum(r['rel']) / len(r['rel']):.3e}"), draws=a.draws, aiclk=r["clk"])
    emit(ev="end")


if __name__ == "__main__":
    main()
