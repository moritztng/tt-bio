"""lpx-sdpa microbench: one attention call class at Protenix-v2's exact shapes, many configs, one chip.

A PLAN is a list of GROUPS. A group shares one set of input tensors per operand dtype set, and its arms
run interleaved round-robin, rep by rep, with the group's baseline arm run twice (`A` and its A/A twin
`A'`) so every group carries its own noise floor. Time per call is the back-to-back slope:
N calls enqueued, one synchronize, wall / N, on a warm program cache. The pip ttnn wheel has no Tracy
device profiler, so this is the device-time instrument; for ops under ~1 ms `trace: true` replays a
captured trace instead, which removes host dispatch from the number.

Every arm is checked once against a float32 torch reference on a slice of the batch. Accuracy is out
of scope for the campaign; the check only proves the kernel ran (finite, shaped right, not garbage).

usage: bench.py OUT PLAN.json [CHIP]
Arm fields (all optional except impl):
  impl   stock | fused | generic | explicit
  dt     {"q","k","v","mask","out"} -> bf16 | bfp8 | bfp4 | fp32
  fid    HiFi4 | HiFi3 | HiFi2 | LoFi      acc  fp32 dest acc     approx  math_approx_mode
  exp    exp_approx_mode (stock/generic/fused)      qc, kc  chunk sizes
  im     intermediate CB format for generic/fused (bf16 | bfp8 | bfp4) -- kernel change, see sdpa_generic
  kvbf   kv buffer factor (generic/fused)
  bchunk explicit only: batch rows per matmul-softmax-matmul pass (0 = whole batch)
  sdt    explicit only: score dtype      pdt  explicit only: probability dtype
Group fields: shape {"q":[B,H,S,D], "k":[...], "v":[...], "mask":[MB,MH,S,S]}, scale, reps, n, trace, arms.
"""
import json, math, os, statistics, sys, threading, time, glob, traceback
from pathlib import Path

OUT = Path(sys.argv[1]); PLAN = json.loads(Path(sys.argv[2]).read_text())
CHIP = int(sys.argv[3]) if len(sys.argv) > 3 else None
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time()
    s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s[:400], flush=True)

NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
CLK = []
def _sampler():
    while True:
        row = {}
        for n in NODES:
            try: row[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            except Exception: row[n] = -1
        CLK.append((time.monotonic(), row)); time.sleep(0.25)
threading.Thread(target=_sampler, daemon=True).start()

if CHIP is not None:
    from tt_bio import runtime, worker as W
    from tt_bio.host_controller import worker_payload
    slot = runtime.build_local_workers("tenstorrent", [object()], [CHIP])[0]
    W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.sdpa_generic as SG
import tt_bio.triatt_sdpa as TS

dev = T.get_device(trace="diffusion")
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}")
                 and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
NODE = opened[0]
GRID = tuple(T.COMPUTE_GRID_MAIN)
log(ev="nodes_open", nodes=opened, grid=GRID, arch=str(dev.arch()), tt_bio=T.__file__)

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi3": ttnn.MathFidelity.HiFi3,
       "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}

def ckc_obj(a):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=FID[a.get("fid", "HiFi2")],
                                            math_approx_mode=bool(a.get("approx", True)),
                                            fp32_dest_acc_en=bool(a.get("acc", False)), packer_l1_acc=False)

def ckc_tuple(a):
    return (FID[a.get("fid", "HiFi2")], bool(a.get("approx", True)), bool(a.get("acc", False)), False)

def dts(a):
    d = {"q": "bf16", "k": "bf16", "v": "bf16", "mask": "bf16", "out": None}
    d.update(a.get("dt", {}))
    if d["out"] is None:
        d["out"] = d["q"]
    return d

# ---------------- inputs ----------------
HOST = {}   # operand -> fp32 host tensor (one set per group)
DEVT = {}   # (operand, dtype) -> device tensor

def host_inputs(g):
    torch.manual_seed(0)
    sh = g["shape"]
    HOST.clear()
    for o in ("q", "k", "v"):
        HOST[o] = torch.randn(sh[o]) * 0.5
    HOST["mask"] = torch.randn(sh["mask"])

def dev_tensor(o, dt):
    key = (o, dt)
    if key not in DEVT:
        DEVT[key] = ttnn.from_torch(HOST[o], dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return DEVT[key]

def free_inputs():
    for t in DEVT.values():
        ttnn.deallocate(t)
    DEVT.clear()

# ---------------- implementations ----------------
def make_call(a, g):
    d = dts(a); scale = g["scale"]
    q, k, v, m = (dev_tensor(o, d[o]) for o in ("q", "k", "v", "mask"))
    impl = a["impl"]
    if impl == "stock":
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=GRID, exp_approx_mode=bool(a.get("exp", False)),
                                    q_chunk_size=a["qc"], k_chunk_size=a["kc"])
        ck = ckc_obj(a)
        def call():
            return ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=m, is_causal=False, scale=scale, program_config=pc,
                compute_kernel_config=ck)
        return call
    if impl in ("generic", "fused"):
        B, H, S, D = (int(x) for x in q.shape)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, H, S, D]), DT[d["out"]], ttnn.TILE_LAYOUT, dev,
                                             ttnn.DRAM_MEMORY_CONFIG)
        kw = dict(exp_approx_mode=bool(a.get("exp", False)), kv_buffer_factor=a.get("kvbf", 2))
        if a.get("im"):
            kw["im_dtype"] = DT[a["im"]]
        if a.get("oim"):
            kw["out_im_dtype"] = DT[a["oim"]]
        if impl == "fused":
            cores = GRID[0] * GRID[1]
            q_pf = TS.q_parallel_factor(S, H, a["qc"], cores, cap=0)
            split = (cores // (H * q_pf), H, q_pf)
            p = SG.plan(q, k, v, m, out, a["qc"], a["kc"], GRID, ckc_tuple(a), scale, split)
            if not (p["nh_per_core"] == 1 and p["q_per_core"] == 1 and p["bcast_batch"] and not p["use_padded_mask"]):
                raise RuntimeError(f"fused fill_preconditions: q_per_core={p['q_per_core']} "
                                   f"nh_per_core={p['nh_per_core']} padded_mask={p['use_padded_mask']}")
            kw.update(split=split, kernel_dir=TS.KERNEL_DIR,
                      mask_cb_tiles=p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"],
                      defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]})
        def call():
            SG.sdpa(dev, q, k, v, m, out, a["qc"], a["kc"], GRID, ckc_tuple(a), scale, **kw)
            return None
        call.out = out
        return call
    if impl == "explicit":
        B = int(q.shape[0]); bc = a.get("bchunk") or B
        ck = ckc_obj(a); sdt = DT[a.get("sdt", "bf16")]; pdt = DT[a.get("pdt", a.get("sdt", "bf16"))]
        odt = DT[d["out"]]
        def call():
            outs = []
            for b0 in range(0, B, bc):
                qs = q if bc == B else ttnn.slice(q, [b0, 0, 0, 0], [min(b0 + bc, B)] + list(q.shape)[1:])
                ks = k if bc == B else ttnn.slice(k, [b0, 0, 0, 0], [min(b0 + bc, B)] + list(k.shape)[1:])
                vs = v if bc == B else ttnn.slice(v, [b0, 0, 0, 0], [min(b0 + bc, B)] + list(v.shape)[1:])
                s = ttnn.matmul(qs, ks, transpose_b=True, compute_kernel_config=ck, dtype=sdt)
                s2 = ttnn.multiply(s, scale); ttnn.deallocate(s)
                s3 = ttnn.add(s2, m, dtype=sdt); ttnn.deallocate(s2)
                p = ttnn.softmax(s3, dim=-1, compute_kernel_config=ck); ttnn.deallocate(s3)
                if p.dtype != pdt:
                    p2 = ttnn.typecast(p, pdt); ttnn.deallocate(p); p = p2
                o = ttnn.matmul(p, vs, compute_kernel_config=ck, dtype=odt); ttnn.deallocate(p)
                if bc != B:
                    for t in (qs, ks, vs): ttnn.deallocate(t)
                outs.append(o)
            if len(outs) == 1:
                return outs[0]
            o = ttnn.concat(outs, dim=0)
            for t in outs: ttnn.deallocate(t)
            return o
        return call
    raise ValueError(impl)

def run_once(call):
    o = call()
    if o is None:
        o = getattr(call, "out", None); own = False
    else:
        own = True
    return o, own

def reference_check(a, g, call):
    """rel_rms of the device output against fp32 torch over a batch slice of the SAME rounded inputs."""
    d = dts(a)
    o, own = run_once(call)
    ttnn.synchronize_device(dev)
    ob = ttnn.to_torch(o).float()
    if own:
        ttnn.deallocate(o)
    nb = min(2, ob.shape[0])
    qr, kr, vr = (ttnn.to_torch(dev_tensor(x, d[x])).float()[:nb] for x in ("q", "k", "v"))
    mr = ttnn.to_torch(dev_tensor("mask", d["mask"])).float()
    mr = mr[:nb] if mr.shape[0] > 1 else mr
    S_q, S_k = qr.shape[2], kr.shape[2]
    sc = torch.einsum("bhqd,bhkd->bhqk", qr, kr) * g["scale"] + mr[..., :S_q, :S_k]
    ref = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(sc, -1), vr)
    got = ob[:nb, :, :S_q, :ref.shape[-1]]
    finite = bool(torch.isfinite(ob).all())
    rel = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
    return dict(finite=finite, rel_rms=rel, out_shape=list(ob.shape))

def timed_rep(call, n, trace_id=None):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    if trace_id is not None:
        for _ in range(n):
            ttnn.execute_trace(dev, trace_id, cq_id=0, blocking=False)
    else:
        for _ in range(n):
            o = call()
            if o is not None:
                ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n

def clk_window(t0, t1):
    c = sorted(r[NODE] for s, r in CLK if t0 <= s <= t1)
    return dict(n=len(c), med=c[len(c) // 2] if c else None, min=c[0] if c else None, max=c[-1] if c else None)

# ---------------- groups ----------------
for gi, g in enumerate(PLAN["groups"]):
    gname = g.get("name", f"g{gi}")
    host_inputs(g)
    arms = [dict(g["base"], name="A")] + [dict(g["base"], name="A'")] + g["arms"]
    live = []
    for a in arms:
        rec = dict(group=gname, arm=a["name"], cfg={k: v for k, v in a.items() if k != "name"})
        try:
            call = make_call(a, g)
            t = time.perf_counter(); o, own = run_once(call); ttnn.synchronize_device(dev)
            rec["first_s"] = time.perf_counter() - t
            if own: ttnn.deallocate(o)
            if not g.get("nocheck"):
                rec["check"] = reference_check(a, g, call)
            # pick N so one rep is >= ~60 ms of device time
            per = timed_rep(call, 2)
            rec["n"] = max(2, min(200, int(math.ceil(0.06 / max(per, 1e-5)))))
            rec["trace"] = None
            if g.get("trace"):
                tid = ttnn.begin_trace_capture(dev, cq_id=0); o = call(); ttnn.end_trace_capture(dev, tid, cq_id=0)
                rec["trace"] = tid; call.keep = o
            live.append((a, call, rec))
        except Exception as e:
            msg = str(e).split("\n")[0][:300]
            log(ev="refused", **rec, error=msg)
            try: ttnn.synchronize_device(dev)
            except Exception: pass
    times = {r["arm"]: [] for _, _, r in live}
    t0 = time.monotonic()
    reps = g.get("reps", 10)
    for rep in range(reps):
        order = live if rep % 2 == 0 else live[::-1]     # ABBA-style, cancels slow drift
        for a, call, rec in order:
            times[rec["arm"]].append(timed_rep(call, rec["n"], rec["trace"]))
    t1 = time.monotonic()
    clk = clk_window(t0, t1)
    base = statistics.median(times["A"])
    aa = abs(statistics.median(times["A'"]) / base - 1)
    for a, call, rec in live:
        v = times[rec["arm"]]
        med = statistics.median(v)
        log(ev="arm", **{k: rec[k] for k in ("group", "arm", "cfg", "n", "check") if k in rec},
            ms=1e3 * med, ms_min=1e3 * min(v), ms_max=1e3 * max(v),
            spread_pct=100 * (max(v) - min(v)) / med, speedup=base / med, aa_floor_pct=100 * aa,
            reps=len(v), aiclk=clk, trace=rec["trace"] is not None)
        if rec["trace"] is not None:
            ttnn.release_trace(dev, rec["trace"])
        o = getattr(call, "out", None)
        if o is not None: ttnn.deallocate(o)
    free_inputs()
    log(ev="group_done", group=gname, base_ms=1e3 * base, aa_floor_pct=100 * aa, aiclk=clk, s=t1 - t0)
log(ev="end")
os._exit(0)
