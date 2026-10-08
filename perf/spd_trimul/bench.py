"""spd-trimul module bench: Protenix-v2's triangle multiplication at a fold's shape, lever arms interleaved.

One process, one chip. Both variants (starting = outgoing, ending = incoming) run every arm, round-robin
rep by rep, with the baseline twice (`base` and its A/A twin `base'`) so each run carries its own floor.
Time per call is the back-to-back slope: `n` calls enqueued, one synchronize, wall / n, warm program
cache. At ~67 ms per call on Wormhole the host dispatch of ~30 programs is under 1 %.

Accuracy per arm, on the same inputs: the float64 torch reference of the trimul math (`reference`), and
the baseline arm's output. rel_rms is over valid rows only (the bucket's padded tokens are masked). The
reference is checked against the baseline first; if the BASELINE misses it, the reference convention is
wrong and the run stops, so an arm can never be graded against a wrong reference.

AICLK is sampled from a separate process (ttnn holds the GIL through a sync), from the node this
process opened.

usage: bench.py OUT [--n 736] [--valid 730] [--arms base,ibw,into,...] [--reps 5] [--calls 4] [--chip C]
       [--weights layer.pt]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--valid", type=int, default=730)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--arms", default="base,ibw,into,ibw+into,ibw+hifi2,ibw+lofi")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=4)
ap.add_argument("--chip", type=int, default=None)
ap.add_argument("--weights", default=None, help="torch .pt with one trimul's state dict (real weights)")
ap.add_argument("--variants", default="start,end")
ap.add_argument("--opsplit", action="store_true", help="also log a synced per-op wall split per arm")
ap.add_argument("--resid", action="store_true",
                help="call the way the Pairformer does, add_to_input=True: z += update in place, so every arm\n"
                     "times the residual add too (the epi2 lever folds it into the tail)")
A = ap.parse_args()

OUT = Path(A.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    s = json.dumps(kw, default=str)
    LOG.write(s + "\n"); LOG.flush(); print(s[:600], flush=True)


if A.chip is not None:
    from tt_bio import runtime, worker as W
    from tt_bio.host_controller import worker_payload
    slot = runtime.build_local_workers("tenstorrent", [object()], [A.chip])[0]
    W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RB
import tt_bio.trimul_tail as TTL

dev = T.get_device()
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}")
                 and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")
                 and os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1].isdigit()})
NODE = opened[0] if opened else -1
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
_SAMPLER = subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:      # dies with the bench, never orphaned
    try: v = open("/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(f"{{time.monotonic()}}\\t{{v}}\\n"); f.flush(); time.sleep(0.25)
"""], stderr=open(OUT / "sampler.err", "w"))
log(ev="nodes_open", nodes=opened, grid=tuple(T.COMPUTE_GRID_MAIN), arch=str(dev.arch()), tt_bio=T.__file__,
    args=vars(A))


def aiclk(t0, t1, pad=0.3):
    """(min, max, n) MHz over [t0 - pad, t1 + pad]: the sampler ticks every 250 ms, longer than a rep."""
    v = []
    for line in CLKF.read_text().splitlines():
        ts, mhz = line.split("\t")
        if t0 - pad <= float(ts) <= t1 + pad and int(mhz) > 0:
            v.append(int(mhz))
    return (min(v), max(v), len(v)) if v else None


# ---------------- arms: name -> list of (setter, value) ----------------
LEVERS = {
    "ibw": [(T.set_trimul_ibw_full, True)],
    "into": [(T.set_trimul_back_into, True)],
    "nointo": [(T.set_trimul_back_into, False)],
    "hifi2": [(T.set_trimul_einsum_fid, "hifi2")],
    "lofi": [(T.set_trimul_einsum_fid, "lofi")],
    "b8in": [(T.set_trimul_inproj_b8, True)],
    "epi1": [(TTL.set_epi, 1)],
    "epi2": [(TTL.set_epi, 2)],
}


def arm_setters(name):
    if name in ("base", "base'"):
        return []
    out = []
    for part in name.split("+"):
        if part not in LEVERS:
            raise SystemExit(f"unknown lever {part!r}; known: {sorted(LEVERS)}")
        out += LEVERS[part]
    return out


def apply(setters):
    prev = [(f, f(v)) for f, v in setters]
    return prev


def restore(prev):
    for f, v in reversed(prev):
        f(v)


# ---------------- per-op split: every device-op entry point synced and wall-timed ----------------
import contextlib
OPS = []
_WRAP = ["matmul", "layer_norm", "concat", "multiply_", "multiply", "add_", "add", "typecast", "generic_op",
         "permute", "transpose", "reallocate", "clone", "to_memory_config", "unsqueeze"]


@contextlib.contextmanager
def op_timer():
    saved, depth = {}, [0]

    def wrap(name, fn):
        def w(*a, **k):
            if depth[0]:
                return fn(*a, **k)
            depth[0] += 1
            try:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = fn(*a, **k)
                ttnn.synchronize_device(dev)
                OPS.append((f"{name}@{sys._getframe(1).f_code.co_name}", 1e3 * (time.perf_counter() - t0)))
                return r
            finally:
                depth[0] -= 1
        return w
    for n in _WRAP:
        saved[n] = getattr(ttnn, n)
        setattr(ttnn, n, wrap(n, saved[n]))
    saved["mm"] = ttnn.experimental.minimal_matmul
    ttnn.experimental.minimal_matmul = wrap("minimal_matmul", saved["mm"])
    try:
        yield
    finally:
        ttnn.experimental.minimal_matmul = saved.pop("mm")
        for n, f in saved.items():
            setattr(ttnn, n, f)


# ---------------- weights, inputs, reference ----------------
cz, h, N, NV = A.cz, A.hidden, A.n, A.valid
if A.weights:
    W0 = {k: v.float() for k, v in torch.load(A.weights, map_location="cpu").items()}
else:
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g) / s[-1] ** 0.5
    W0 = {"norm_in.weight": 1 + 0.1 * torch.randn(cz, generator=g), "norm_in.bias": 0.1 * torch.randn(cz, generator=g),
          "norm_out.weight": 1 + 0.1 * torch.randn(h, generator=g), "norm_out.bias": 0.1 * torch.randn(h, generator=g),
          "g_in.weight": r(2 * h, cz), "p_in.weight": r(2 * h, cz), "g_out.weight": r(cz, cz), "p_out.weight": r(cz, h)}

from tt_bio.af2 import compute_kernel_config
CKC = T.trunk_compute_kernel_config(compute_kernel_config())
gz = torch.Generator().manual_seed(1)
z_host = torch.randn(1, N, N, cz, generator=gz)
m_host = torch.zeros(1, N, N)
m_host[:, :NV, :NV] = 1
z_host = z_host.to(torch.bfloat16).float()            # the reference sees exactly what the chip sees
z_dev = ttnn.from_torch(z_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
m_dev = ttnn.from_torch(m_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)


def reference(ending):
    """float64 trimul: OpenFold/Protenix semantics in tt-bio's weight layout (a = first half of g_in/p_in)."""
    d = torch.float64
    z = z_host.to(d)
    Wd = {k: v.to(d) for k, v in W0.items()}
    ln = torch.nn.functional.layer_norm
    x = ln(z, (cz,), Wd["norm_in.weight"], Wd["norm_in.bias"], 1e-5)
    gp = torch.sigmoid(x @ Wd["g_in.weight"].T) * (x @ Wd["p_in.weight"].T)
    m = m_host.to(d)[..., None]
    a, b = gp[..., :h] * m, gp[..., h:] * m
    if ending:
        o = torch.einsum("bkic,bkjc->bijc", a, b)
    else:
        o = torch.einsum("bikc,bjkc->bijc", a, b)
    o = ln(o, (h,), Wd["norm_out.weight"], Wd["norm_out.bias"], 1e-5)
    return (o @ Wd["p_out.weight"].T) * torch.sigmoid(x @ Wd["g_out.weight"].T)


def call(mod):
    """One timed call. With --resid the Pairformer's form: z_dev += update in place, nothing freed."""
    if A.resid:
        mod(z_dev, m_dev, add_to_input=True)
    else:
        ttnn.deallocate(mod(z_dev, m_dev))


def accuracy_call(mod):
    """The update as a host tensor. With --resid, on a fresh upload of z_host, minus z_host."""
    if not A.resid:
        y = mod(z_dev, m_dev)
        ttnn.synchronize_device(dev)
        yt = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
        return yt
    zc = ttnn.from_torch(z_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = mod(zc, m_dev, add_to_input=True)
    ttnn.synchronize_device(dev)
    yt = ttnn.to_torch(y).float() - z_host
    ttnn.deallocate(y)
    if zc.is_allocated():
        ttnn.deallocate(zc)
    return yt


def err(y, ref):
    y, ref = y[0, :NV, :NV].double(), ref[0, :NV, :NV].double()
    d = y - ref
    return {"rel_rms": (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
            "max_abs": d.abs().max().item(), "finite": bool(torch.isfinite(y).all())}


# ---------------- run ----------------
arms = A.arms.split(",")
if "base" in arms and "base'" not in arms:
    arms.insert(arms.index("base") + 1, "base'")
res = {"args": vars(A), "nodes": opened, "arch": str(dev.arch()), "variants": {}}
for var in A.variants.split(","):
    ending = var == "end"
    mod = T.TriangleMultiplication(ending, W0, CKC)
    t0 = time.time()
    ref = reference(ending)
    log(ev="reference", variant=var, s=round(time.time() - t0, 1))
    outs = {}
    for arm in arms:                                   # warm + accuracy, one arm at a time
        prev = apply(arm_setters(arm))
        try:
            back0 = list(RB.STATS_BACK)
            tail0, resid0 = list(TTL.STATS), list(TTL.RESID_STATS)
            fired = {"in0_block_w": T._triangle_mul_program_config(-(-N // 32)).in0_block_w,
                     "ibw_refused": sorted(T._TRIMUL_IBW_FULL_REFUSED)}
            for _ in range(2):
                yt = accuracy_call(mod)
            fired["back_kernel_calls"] = RB.STATS_BACK[0] - back0[0]
            fired["mm_transpose"] = dict((f"{k[0]}/{k[1]}", v) for k, v in T.TRIMUL_MM_TRANSPOSE_STATS.items())
            fired["einsum_fid"] = T._TRIMUL_EINSUM_FID or "trunk"
            fired["back_into"] = T._TRIMUL_BACK_INTO
            fired["inproj_b8"] = T._TRIMUL_INPROJ_B8
            fired["tail_f1"] = [a - b for a, b in zip(TTL.STATS, tail0)]
            fired["tail_epi"] = TTL.EPI
            fired["tail_resid"] = [a - b for a, b in zip(TTL.RESID_STATS, resid0)]
        finally:
            restore(prev)
        outs[arm] = yt
        e = err(yt, ref)
        e["fired"] = fired
        e["vs_base_max_abs"] = (yt - outs["base"]).abs().max().item() if "base" in outs else None
        e["equal_base"] = bool(torch.equal(yt, outs["base"])) if "base" in outs else None
        log(ev="accuracy", variant=var, arm=arm, **e)
        if arm == "base" and not (e["finite"] and e["rel_rms"] < 0.05):
            log(ev="abort", why="baseline misses the float64 reference: reference convention wrong", **e)
            raise SystemExit(2)
    if A.opsplit:                                      # one synced call per arm, per-op wall
        for arm in arms:
            if arm == "base'":
                continue
            prev = apply(arm_setters(arm))
            try:
                for rep in range(3):
                    OPS.clear()
                    with op_timer():
                        call(mod)
                    if rep == 2:
                        log(ev="opsplit", variant=var, arm=arm, total_ms=round(sum(t for _, t in OPS), 3),
                            ops=[[n, round(t, 3)] for n, t in OPS])
            finally:
                restore(prev)
    times = {a: [] for a in arms}
    for rep in range(A.reps):                          # timing, interleaved
        for arm in arms:
            prev = apply(arm_setters(arm))
            try:
                ttnn.synchronize_device(dev)
                t0 = time.monotonic()
                for _ in range(A.calls):
                    call(mod)
                ttnn.synchronize_device(dev)
                t1 = time.monotonic()
            finally:
                restore(prev)
            times[arm].append((t1 - t0) / A.calls)
            log(ev="rep", variant=var, arm=arm, rep=rep, ms=round(1e3 * times[arm][-1], 3), aiclk=aiclk(t0, t1))
    row = {}
    base = statistics.median(times["base"]) if "base" in times else None
    for arm, v in times.items():
        med = statistics.median(v)
        row[arm] = {"ms": round(1e3 * med, 3), "spread_ms": round(1e3 * (max(v) - min(v)), 3),
                    "x_vs_base": round(base / med, 4) if base else None, "reps": len(v)}
    clk = aiclk(0, time.monotonic())
    log(ev="variant_done", variant=var, arms=row, aiclk=clk)
    res["variants"][var] = {"arms": row, "aiclk": clk}
    del mod
(OUT / "result.json").write_text(json.dumps(res, indent=1))
log(ev="done")
