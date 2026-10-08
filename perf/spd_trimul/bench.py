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
ap.add_argument("--arms", default="base,ibw,into,ibw+into")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=4)
ap.add_argument("--chip", type=int, default=None)
ap.add_argument("--weights", default=None, help="torch .pt with one trimul's state dict (real weights)")
ap.add_argument("--variants", default="start,end")
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


def aiclk(t0, t1):
    v = []
    for line in CLKF.read_text().splitlines():
        ts, mhz = line.split("\t")
        if t0 <= float(ts) <= t1 and int(mhz) > 0:
            v.append(int(mhz))
    return (min(v), max(v), len(v)) if v else None


# ---------------- arms: name -> list of (setter, value) ----------------
LEVERS = {
    "ibw": [(T.set_trimul_ibw_full, True)],
    "into": [(T.set_trimul_back_into, True)],
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
            for _ in range(2):
                y = mod(z_dev, m_dev)
                ttnn.synchronize_device(dev)
                yt = ttnn.to_torch(y).float()
                ttnn.deallocate(y)
        finally:
            restore(prev)
        outs[arm] = yt
        e = err(yt, ref)
        e["vs_base_max_abs"] = (yt - outs["base"]).abs().max().item() if "base" in outs else None
        e["equal_base"] = bool(torch.equal(yt, outs["base"])) if "base" in outs else None
        log(ev="accuracy", variant=var, arm=arm, **e)
        if arm == "base" and not (e["finite"] and e["rel_rms"] < 0.05):
            log(ev="abort", why="baseline misses the float64 reference: reference convention wrong", **e)
            raise SystemExit(2)
    times = {a: [] for a in arms}
    for rep in range(A.reps):                          # timing, interleaved
        for arm in arms:
            prev = apply(arm_setters(arm))
            try:
                ttnn.synchronize_device(dev)
                t0 = time.monotonic()
                for _ in range(A.calls):
                    ttnn.deallocate(mod(z_dev, m_dev))
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
