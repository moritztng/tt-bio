"""Where the diffusion atom attention module's time goes: each step of `_attention_superset` timed on its own.

Same operands as opbench.py's atom arms (5 samples, 5,919 atoms, 4 heads, head_dim 32, superset window 160 keys),
one dtype per run. Each step runs NCALL times between two syncs, so a step's number is its own device time plus its
host dispatch, which is what the fold pays when the step is not overlapped. Variants of a step sit beside it.
usage: atom_steps.py OUT CHIP [fp32|bf16]
"""
import argparse, json, os, statistics, subprocess, sys, time, types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("chip", type=int); ap.add_argument("dtype", nargs="?", default="fp32")
ap.add_argument("--reps", type=int, default=10); ap.add_argument("--ncall", type=int, default=10)
a = ap.parse_args()
OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "steps.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s[:300], flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [a.chip])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio.protenix import AtomTransformer as AT

dev = T.get_device()
NODE = sorted(int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
              if os.path.exists(f"/proc/self/fd/{fd}")
              and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/"))[0]
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:
    try: v = open("/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(str(time.monotonic()) + chr(9) + v + chr(10)); f.flush(); time.sleep(0.25)
"""])
DT = ttnn.float32 if a.dtype == "fp32" else ttnn.bfloat16
up = lambda t: ttnn.from_torch(t, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
log(ev="open", arch=str(dev.arch()), dtype=a.dtype, node=NODE)

M, N, H, dh, nq, nk = 5, 5919, 4, 32, 32, 128
NP = -(-N // nq) * nq; nb = NP // nq
torch.manual_seed(1)
X = {o: up(torch.randn(M, N, H * dh) * 0.5) for o in "qkv"}
hz = torch.randn(nb, H, nq, nk)
key = torch.arange(nb)[:, None, None] * nq - 48 + torch.arange(nk)[None, None, :]
mask = ((key >= 0) & (key < N)).expand(nb, nq, nk).float()
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
n = types.SimpleNamespace(N_HEADS=H, HEAD_DIM=dh, N_QUERIES=nq, N_KEYS=nk, PAD_LEFT=48, device=dev, dtype=DT,
                          _sdpa=False, compute_kernel_config=CKC)
for f in ("_superset", "_superset_bias"):
    setattr(n, f, types.MethodType(getattr(AT, f), n))
lead, Wd = n._superset(); S = Wd // nq; nbk = nb + S - 1
zs = n._superset_bias(up(hz), mask, nb)
smx = T.softmax_ckc("protenix.atom_transformer")


# The steps of _attention_superset, in order, each a function of the previous results.
def heads(x, front, rows):
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.pad(x, [[0, 0], [front, rows - front - N], [0, 0]], 0.0)
    x = ttnn.permute(ttnn.reshape(x, (M, rows, H, dh)), (0, 2, 1, 3))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def windows(h):
    x = ttnn.reshape(h, (M * H, nbk, nq, dh))
    x = ttnn.concat([ttnn.slice(x, [0, j, 0, 0], [M * H, j + nb, nq, dh]) for j in range(S)], dim=2)
    return ttnn.reshape(x, (M, H * nb, Wd, dh))


R = {}
STEPS = [
    ("q heads (RM pad, permute)", lambda: R.__setitem__("Qs", ttnn.reshape(heads(X["q"], 0, NP), (M, H * nb, nq, dh)))),
    ("k heads (RM pad, permute)", lambda: R.__setitem__("Kh", heads(X["k"], lead, nbk * nq))),
    ("k windows (5 slices, concat)", lambda: R.__setitem__("Ks", windows(R["Kh"]))),
    ("v heads + windows", lambda: R.__setitem__("Vs", windows(heads(X["v"], lead, nbk * nq)))),
    ("k permute", lambda: R.__setitem__("KsT", ttnn.permute(R["Ks"], (0, 1, 3, 2)))),
    ("qk matmul", lambda: R.__setitem__("sc", T.batched_matmul(R["Qs"], R["KsT"], compute_kernel_config=CKC))),
    ("scale_add bias", lambda: R.__setitem__("sc2", T.scale_add(R["sc"], dh ** -0.5, zs))),
    ("softmax", lambda: R.__setitem__("p", ttnn.softmax(R["sc2"], dim=-1, compute_kernel_config=smx))),
    ("pv matmul", lambda: R.__setitem__("o", T.batched_matmul(R["p"], R["Vs"], compute_kernel_config=CKC))),
    ("out merge (permute, RM slice, tile)", lambda: R.__setitem__("out", ttnn.to_layout(ttnn.slice(ttnn.to_layout(
        ttnn.reshape(ttnn.permute(ttnn.reshape(R["o"], (M, H, NP, dh)), (0, 2, 1, 3)), (M, NP, H * dh)),
        ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0], [M, N, H * dh]), ttnn.TILE_LAYOUT))),
]
zrow = lambda r: ttnn.zeros((M, r, H * dh), dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev)
Z_FRONT, Z_TAIL = zrow(lead), zrow(nbk * nq - lead - NP)


def rows_np(x):
    """(M, N, C) -> (M, NP, C) zero-padded rows, TILE throughout (NP is the tile-padded N)."""
    return ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)


def heads_tile():
    """Q (M, H, NP, dh) and K, V (M, H, nbk*nq, dh) by one nlp_create_qkv_heads, no ROW_MAJOR step."""
    q = ttnn.reshape(rows_np(X["q"]), (M, 1, NP, H * dh))
    kv = ttnn.concat([rows_np(X["k"]), rows_np(X["v"])], dim=-1)                  # (M, NP, 2C)
    kv = ttnn.concat([ttnn.concat([Z_FRONT] * 2, dim=-1), kv, ttnn.concat([Z_TAIL] * 2, dim=-1)], dim=1)
    kv = ttnn.reshape(kv, (M, 1, nbk * nq, 2 * H * dh))
    return ttnn.experimental.nlp_create_qkv_heads(q, kv, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)


def merge_tile():
    o = ttnn.experimental.nlp_concat_heads(ttnn.reshape(R["o"], (M, H, NP, dh)), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.slice(ttnn.reshape(o, (M, NP, H * dh)), [0, 0, 0], [M, N, H * dh])


# (name, fn, reference key or None, how the result maps onto the reference)
VARIANTS = [
    ("qk matmul transpose_b (no k permute)", lambda: ttnn.matmul(R["Qs"], R["Ks"], transpose_b=True,
                                                                 compute_kernel_config=CKC), "sc", None),
    ("softmax in place", lambda: ttnn.softmax_in_place(ttnn.clone(R["sc2"]), compute_kernel_config=smx), "p", None),
    ("clone of scores (in-place baseline)", lambda: ttnn.clone(R["sc2"]), None, None),
    ("q+k+v heads TILE (nlp_create_qkv_heads)", heads_tile, "Kh", lambda o: o[1]),
    ("out merge TILE (nlp_concat_heads, tile slice)", merge_tile, "out", None),
]
for name, f in STEPS:
    f()
ref = ttnn.to_torch(R["out"]).double()
for name, f, rk, pick in VARIANTS:
    try:
        o = f()
        o = pick(o) if pick else o
        eq = None
        if rk is not None:
            x, y = ttnn.to_torch(o).double(), ttnn.to_torch(R[rk]).double()
            eq = dict(shape=list(x.shape), ref_shape=list(y.shape),
                      max_abs=float((x.reshape(y.shape) - y).abs().max()) if x.numel() == y.numel() else None)
        log(ev="variant_ok", name=name, check=eq)
    except Exception as e:
        log(ev="variant_refused", name=name, error=str(e).splitlines()[0][:300])
VARIANTS = [(nm, f) for nm, f, _r, _p in VARIANTS]


def clk_window(t0, t1):
    return [int(v) for t, v in (l.split() for l in open(CLKF)) if t0 <= float(t) <= t1 and v.isdigit()]


times = {nm: [] for nm, _ in STEPS + VARIANTS}; clk = []
for rep in range(a.reps):
    for nm, f in STEPS + VARIANTS:
        ttnn.synchronize_device(dev); t0 = time.monotonic(); p0 = time.perf_counter()
        try:
            for _ in range(a.ncall):
                f()
        except Exception:
            continue
        ttnn.synchronize_device(dev)
        times[nm].append((time.perf_counter() - p0) / a.ncall * 1e3); clk += clk_window(t0, time.monotonic())
tot = 0.0
for nm, _ in STEPS + VARIANTS:
    v = times[nm]
    if not v:
        continue
    med = statistics.median(v)
    tot += med if any(nm == s for s, _ in STEPS) else 0
    log(ev="step", name=nm, ms=med, ms_min=min(v), ms_max=max(v), variant=not any(nm == s for s, _ in STEPS))
c = sorted(clk)
log(ev="end", steps_total_ms=tot, aiclk_med=c[len(c) // 2] if c else None, aiclk_min=c[0] if c else None,
    out_finite=bool(torch.isfinite(ref).all()))
