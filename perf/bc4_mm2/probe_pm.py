#!/usr/bin/env python3
"""bc4-bet-mm2: why `pair_mm.matmul` serves none of BindCraft 2's pair projections on Blackhole.

Calls the real `pair_mm.matmul` at the call forms the shape census found falling through (forward with and
without a 1-D bias, the 4-wide pair-bias projection, the dX forms), armed, and records what it did: served,
declined (and why), or refused (and the exception). Then the same with `pair_linear` allowed off Wormhole.
Each served form is timed (traced burst) and graded against float64, beside the plain ttnn call.
"""
import argparse, json, os, time
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); A = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, autograd as AG, pair_mm as PM
dev = T.get_device(); cfg = AG.precise_config(); DRAM = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
PM.PAIR_MM_FUSED = True
errs = []
_mmm = ttnn.experimental.minimal_matmul


def mmm(*a, **k):
    try:
        return _mmm(*a, **k)
    except Exception as e:
        errs.append(str(e).splitlines()[0][:300])
        raise


ttnn.experimental.minimal_matmul = mmm


def timed(f, burst=20, reps=9):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    outs = []; tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(burst):
        outs.append(f())
    ttnn.end_trace_capture(dev, tid, cq_id=0); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter(); ttnn.execute_trace(dev, tid, cq_id=0, blocking=False); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / burst)
    ttnn.release_trace(dev, tid); [ttnn.deallocate(o) for o in outs]
    return round(min(ms), 5)


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)


g = torch.Generator().manual_seed(1)
FORMS = (("fwd_bias", [1, 224, 224, 128], [128, 128], True, False),
         ("fwd", [1, 224, 224, 128], [128, 128], False, False),
         ("fwd_n4", [224, 224, 128], [128, 4], False, False),
         ("dx_384", [50176, 384], [128, 384], False, True),
         ("dx_n4", [50176, 4], [128, 4], False, True),
         ("dx_n8", [50176, 8], [128, 8], False, True))
rows = []
for name, xs, ws, has_b, tb in FORMS:
    hx = torch.randn(*xs, generator=g).bfloat16(); hw = (torch.randn(*ws, generator=g) / ws[0] ** 0.5).bfloat16()
    hb = torch.randn(ws[0] if tb else ws[1], generator=g).bfloat16() if has_b else None
    ref = hx.double() @ (hw.double().T if tb else hw.double())
    if hb is not None:
        ref = ref + hb.double()
    x, w = up(hx), up(hw); b = up(hb) if hb is not None else None
    base = lambda: (ttnn.matmul(x, w, transpose_b=True, compute_kernel_config=cfg) if tb else
                    ttnn.linear(x, w, bias=b, compute_kernel_config=cfg, dtype=ttnn.bfloat16, memory_config=DRAM))
    for pl_bh in (False, True):
        PM._wormhole = (lambda: True) if pl_bh else T.is_wormhole
        PM.DECLINED.clear(); errs.clear(); s0 = PM.STATS[0]
        try:
            o = PM.matmul(x, w, b, cfg, ttnn.bfloat16, transpose_b=tb)
        except Exception as e:
            o = None; errs.append("raised: " + str(e).splitlines()[0][:300])
        r = dict(form=name, x=xs, w=ws, bias=has_b, tb=tb, pl_bh=pl_bh, served=PM.STATS[0] - s0,
                 declined=[list(map(str, k)) for k in PM.DECLINED], errs=list(errs))
        if o is not None:
            got = ttnn.to_torch(o).double().reshape(ref.shape); ttnn.deallocate(o)
            r["rel_f64"] = round(float((got - ref).norm() / ref.norm()), 6)
            r["dev_ms"] = timed(lambda: PM.matmul(x, w, b, cfg, ttnn.bfloat16, transpose_b=tb))
        ob = base(); gb = ttnn.to_torch(ob).double().reshape(ref.shape); ttnn.deallocate(ob)
        r["base_rel_f64"] = round(float((gb - ref).norm() / ref.norm()), 6)
        r["base_ms"] = timed(base)
        r["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read())
        print(json.dumps(r), flush=True); rows.append(r)
json.dump(dict(rows=rows, loadavg=open("/proc/loadavg").read().split()[:3]), open(A.out, "w"), indent=1)
