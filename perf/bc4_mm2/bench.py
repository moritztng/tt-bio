#!/usr/bin/env python3
"""bc4-bet-mm2 step 2: every signature of the two matmul sites, on its own operands, as shipped and under each lever.

Runs the PD-L1 campaign to gradient iteration --at. During that iteration the first call of every distinct
signature at `autograd._matmul` and `pair_mm.or_linear` keeps a DRAM clone of its operands and the exact
kwargs it was issued with (program config, compute config, dtype), and every call is counted. When the
iteration returns, each signature is timed:

* base     the call as the model issues it
* hifi2    same, MathFidelity.HiFi2 (the backward ships HiFi4)
* bfp8     same, bfloat8_b output
* hifi2_bfp8
* mmm*     minimal_matmul over the flattened rows, a small block-grid sweep, bf16 out at the shipped fidelity
           (and the best of them at HiFi2 / bfp8 out); pair-sized 2-D weights only, transposed weight made
           once outside the timing (a parameter, constant across the campaign)

Device time is a traced burst: --burst calls captured once, replayed --reps times, one synchronize per replay,
so host enqueue is not in it. `sync_ms` is the same burst eager, which is what the eager path pays when the
host is the bottleneck. Accuracy: rel L2 of each arm against float64 of the same operands.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_TRACE_REGION_MB=2048 python perf/bc4_mm2/bench.py --out OUT.json
"""
import argparse
import collections
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_e2e/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--at", type=int, default=4)
ap.add_argument("--burst", type=int, default=20)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--min-calls", type=float, default=1.0)
ap.add_argument("--min-m", type=int, default=0, help="time only signatures with at least this many rows")
ap.add_argument("--fast", action="store_true", help="base, pair_linear and minimal_matmul arms only")
a = ap.parse_args()
sys.path.insert(0, str(a.bc2))

import torch  # noqa: E402
import ttnn  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from tt_bio import bindcraft2 as bc2, tenstorrent as T  # noqa: E402
from tt_bio import autograd as AG, pair_mm as PM  # noqa: E402

CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
DRAM = ttnn.DRAM_MEMORY_CONFIG
st = dict(n=0)
count = collections.Counter()
keep = {}
clk = []


def _sampler():
    p = f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk"
    while True:
        try:
            clk.append((time.time(), int(open(p).read())))
        except Exception:                                                # noqa: BLE001
            pass
        time.sleep(1)


threading.Thread(target=_sampler, daemon=True).start()


def canon(v):
    if hasattr(v, "math_fidelity"):
        return "cfg:" + ",".join(str(getattr(v, k, "")) for k in ("math_fidelity", "fp32_dest_acc_en",
                                                                   "packer_l1_acc", "math_approx_mode"))
    return re.sub(r"0x[0-9a-fA-F]+", "", repr(v))


def sig(site, x, y, bias, kw):
    def t(v):
        return None if v is None else (tuple(int(d) for d in v.shape), str(v.dtype))
    return json.dumps([site, t(x), t(y), t(bias), sorted((k, canon(v)) for k, v in kw.items())])


def clone(v):
    return None if v is None else ttnn.clone(v, memory_config=DRAM)


real_mm = AG._matmul


def _matmul(x, y, **kw):
    if st["n"] == a.at:
        k = sig("_matmul", x, y, None, kw)
        count[k] += 1
        if k not in keep:
            keep[k] = dict(site="_matmul", x=clone(x), y=clone(y), bias=None, kw=dict(kw))
    return real_mm(x, y, **kw)


AG._matmul = _matmul
real_or = PM.or_linear


def or_linear(linear, x, w, bias=None, **kw):
    if st["n"] == a.at:
        xv, wv, bv = (getattr(v, "value", v) for v in (x, w, bias))
        k = sig("or_linear", xv, wv, bv, kw)
        count[k] += 1
        if k not in keep:
            keep[k] = dict(site="or_linear", x=clone(xv), y=clone(wv), bias=clone(bv), kw=dict(kw), linear=linear)
    return real_or(linear, x, w, bias, **kw)


PM.or_linear = or_linear
TTS = sys.modules.get("tt_bio.taped_ttnn")
if TTS is not None:
    TTS.forget_shim_bindings()


def with_cfg(kw, fid=None):
    c = kw.get("compute_kernel_config")
    if fid is None:
        return c
    base = dict(math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    if c is not None:
        base = {k: getattr(c, k) for k in base}
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, **base)


def call_for(r, fid=None, out_dtype=None):
    kw = dict(r["kw"])
    if fid is not None:
        kw["compute_kernel_config"] = with_cfg(kw, fid)
    if out_dtype is not None:
        kw["dtype"] = out_dtype
    if r["site"] == "_matmul":
        return lambda: real_mm(r["x"], r["y"], **kw)
    return lambda: real_or(r["linear"], r["x"], r["y"], r["bias"], **kw)


def reference(r):
    x = ttnn.to_torch(r["x"]).double()
    y = ttnn.to_torch(r["y"]).double()
    kw = r["kw"]
    if r["site"] == "_matmul":
        if kw.get("transpose_a"):
            x = x.transpose(-1, -2)
        if kw.get("transpose_b"):
            y = y.transpose(-1, -2)
        return torch.matmul(x, y)
    out = torch.matmul(x, y)
    if r["bias"] is not None:
        out = out + ttnn.to_torch(r["bias"]).double().reshape(-1)[: out.shape[-1]]
    act = kw.get("activation")
    if act == "relu":
        out = out.clamp_min(0)
    elif act is not None:
        return None
    return out


def rel(o, ref):
    if ref is None:
        return None
    got = ttnn.to_torch(o).double()
    try:
        got = got.reshape(ref.shape)
    except RuntimeError:
        got = got[tuple(slice(0, s) for s in ref.shape)] if got.dim() == ref.dim() else got.reshape(-1)[: ref.numel()].reshape(ref.shape)
    return float((got - ref).norm() / ref.norm().clamp_min(1e-30))


def free(o):
    if isinstance(o, ttnn.Tensor):
        ttnn.deallocate(o)


def timed(f):
    # warm (compile) + eager burst + traced burst
    free(f())
    ttnn.synchronize_device(dev)
    sync = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(a.burst):
            free(f())
        ttnn.synchronize_device(dev)
        sync.append((time.perf_counter() - t0) * 1e3 / a.burst)
    outs = []
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    try:
        for _ in range(a.burst):
            outs.append(f())
    finally:
        ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / a.burst)
    ttnn.release_trace(dev, tid)
    for o in outs:
        free(o)
    return dict(dev_ms=round(statistics.median(ms), 5), dev_min=round(min(ms), 5), dev_spread=round(max(ms) - min(ms), 5),
                sync_ms=round(statistics.median(sync), 5))


def arm(row, name, f, ref, flop):
    r = dict(arm=name)
    t0 = time.time()
    try:
        o = f()
        r["rel_f64"] = rel(o, ref)
        free(o)
        r.update(timed(f))
        if flop:
            r["tflops"] = round(flop / r["dev_ms"] / 1e9, 2)
    except Exception as e:                                               # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        try:
            ttnn.synchronize_device(dev)
        except Exception:                                                # noqa: BLE001
            pass
    c = [v for t, v in clk if t >= t0 - 1]
    r["aiclk"] = [min(c), int(statistics.median(c))] if c else None
    row["arms"].append(r)
    print(json.dumps(dict(sig=row["i"], **r)), flush=True)
    return r


def mmm_configs(Mt, Kt, Nt):
    gx, gy = T.COMPUTE_GRID_MAIN
    seen = []
    for Mb in (4, 8, 16):
        for Nb in (1, 2, 4, 8):
            for Kb in (4, 8):
                kb = min(Kb, Kt)
                if Mt % Mb or Nt % Nb or Kt % kb:
                    continue
                sh = next(h for h in (4, 2, 1) if Mb % h == 0)
                sw = next(w for w in (4, 2, 1) if Nb % w == 0 and sh * w <= 4)
                c = (Mb, kb, Nb, sh, sw)
                if c not in seen:
                    seen.append(c)
    return [(c, ttnn.MinimalMatmulConfig(M_block_size=c[0], K_block_size=c[1], N_block_size=c[2], subblock_h=c[3],
                                         subblock_w=c[4], compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy)))
            for c in seen]


def bench():
    rows = []
    for i, (k, n) in enumerate(count.most_common()):
        r = keep[k]
        s = json.loads(k)
        row = dict(i=i, site=r["site"], calls_per_it=n, x=s[1], y=s[2], bias=s[3],
                   kw={kk: vv[:300] for kk, vv in s[4]}, arms=[])
        rows.append(row)
        kw = r["kw"]
        x, y = r["x"], r["y"]
        sx, sy = [int(d) for d in x.shape], [int(d) for d in y.shape]
        ta, tb = bool(kw.get("transpose_a")), bool(kw.get("transpose_b"))
        M = sx[-1] if ta else sx[-2]
        K = sx[-2] if ta else sx[-1]
        N = sy[-2] if tb else sy[-1]
        if r["site"] == "or_linear" or len(sy) == 2:
            M = 1
            for d in (sx[:-1] if not ta else sx[:-2] + [sx[-1]]):
                M *= d
            batch = 1
        else:
            batch = 1
            for d in sx[:-2]:
                batch *= d
        flop = 2 * batch * M * K * N
        row["mkn"] = [batch, M, K, N, ta, tb]
        if M < a.min_m:
            continue
        ref = reference(r) if x.dtype != ttnn.float32 or flop < 5e9 else None
        arm(row, "base", call_for(r), ref, flop)
        if n < a.min_calls or x.dtype != ttnn.bfloat16 or y.dtype != ttnn.bfloat16:
            continue
        if not a.fast:
            arm(row, "hifi2", call_for(r, fid=ttnn.MathFidelity.HiFi2), ref, flop)
            arm(row, "bfp8", call_for(r, out_dtype=ttnn.bfloat8_b), ref, flop)
            arm(row, "hifi2_bfp8", call_for(r, fid=ttnn.MathFidelity.HiFi2, out_dtype=ttnn.bfloat8_b), ref, flop)
        # pair_linear (weight resident in L1, Wormhole-only in the model) on a narrow bias-free call
        if (not ta and len(sy) == 2 and M >= 4096 and min(K, N) < 32 and r["bias"] is None
                and r["kw"].get("activation") in (None,)):
            from tt_bio import pair_linear as PL
            wl = ttnn.clone(ttnn.transpose(y, -2, -1) if tb else y, memory_config=DRAM)
            arm(row, "pl", lambda: PL.matmul(x, wl, with_cfg(kw)), ref, flop)
            free(wl)
        # minimal_matmul on the flattened rows, pair-sized calls against a 2-D weight only
        if ta or len(sy) != 2 or M < 4096 or K % 32 or N % 32 or r["kw"].get("activation") not in (None,):
            continue
        x2 = ttnn.reshape(x, [1, 1, M, K])
        w2 = ttnn.transpose(y, -2, -1) if tb else y
        w2 = ttnn.clone(w2, memory_config=DRAM)
        b2 = r["bias"]
        if b2 is not None:
            b2 = ttnn.reshape(b2, [1, int(b2.shape[-1])])
        cfg = with_cfg(kw)
        best = None
        for c, mc in mmm_configs(M // 32, K // 32, N // 32):
            f = (lambda mc=mc, cf=cfg, dt=None: ttnn.experimental.minimal_matmul(
                input_tensor=x2, weight_tensor=w2, bias_tensor=b2, compute_kernel_config=cf, config=mc, dtype=dt))
            res = arm(row, "mmm" + "_".join(map(str, c)), f, ref, flop)
            if "dev_ms" in res and (best is None or res["dev_ms"] < best[0]):
                best = (res["dev_ms"], mc, c)
        if best:
            mc = best[1]
            c2 = with_cfg(kw, ttnn.MathFidelity.HiFi2)
            arm(row, "best_mmm_hifi2", lambda: ttnn.experimental.minimal_matmul(
                input_tensor=x2, weight_tensor=w2, bias_tensor=b2, compute_kernel_config=c2, config=mc), ref, flop)
            arm(row, "best_mmm_bfp8", lambda: ttnn.experimental.minimal_matmul(
                input_tensor=x2, weight_tensor=w2, bias_tensor=b2, compute_kernel_config=cfg, config=mc,
                dtype=ttnn.bfloat8_b), ref, flop)
            arm(row, "best_mmm_hifi2_bfp8", lambda: ttnn.experimental.minimal_matmul(
                input_tensor=x2, weight_tensor=w2, bias_tensor=b2, compute_kernel_config=c2, config=mc,
                dtype=ttnn.bfloat8_b), ref, flop)
        free(w2)
        write(rows, partial=True)
    write(rows, partial=False)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def write(rows, partial):
    c = [v for _, v in clk]
    a.out.write_text(json.dumps(dict(
        sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        host=socket.gethostname(), chip=CHIP, partial=partial, at=a.at, burst=a.burst, reps=a.reps,
        grid=list(T.COMPUTE_GRID_MAIN), loadavg=open("/proc/loadavg").read().split()[:3],
        aiclk=dict(median=int(statistics.median(c)), min=min(c), n=len(c)) if c else None,
        env={k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_BIO_")}, rows=rows), indent=1))


cls = bc2.design_model_class()
real_grads = cls.sequence_gradients
dev = None


def sequence_gradients(self, *args, **kwargs):
    global dev
    st["n"] += 1
    try:
        return real_grads(self, *args, **kwargs)
    finally:
        print(json.dumps(dict(ev="iter", n=st["n"], sigs=len(keep), calls=sum(count.values()))), flush=True)
        if st["n"] == a.at:
            dev = T.get_device()
            ttnn.synchronize_device(dev)
            bench()
            os._exit(0)


cls.sequence_gradients = sequence_gradients
project = a.out.parent / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", "max_trajectories=1", f"project_folder={project}", f"binder_lengths=[{a.binder}]"])))
weights = {k: os.environ[v] for k, v in (("af2_weights", "JAPANFOLD_BC2_AF2_WEIGHTS"),) if os.environ.get(v)}
with bc2.campaign_predictor(**({"checkpoints": weights["af2_weights"]} if weights else {})):
    bc2.run_campaign(settings, str(project), trajectories_per_card=1, **weights)
print("campaign ended before the bench iteration", flush=True)
os._exit(1)
