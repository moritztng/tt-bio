"""spd-trikern-mm op bench: the trimul einsum (+ its mask multiply) at Protenix-v2's c730 shape, arms interleaved.

One process, one chip. a, b = [1, C, S, S] bf16 DRAM (C = 128 channels per chunk, S = 736), the einsum
`ttnn.matmul(a, b, transpose_{a,b})` with the production program config (`_triangle_mul_program_config(Kt,
full=True)`), normal ckc (HiFi3, fp32 DST, packer L1 acc) or fast ckc (HiFi4, bf16 DST). Every arm is timed as a
back-to-back slope (`--calls` calls, one sync) per rep, arms round-robin, with an A/A twin of the first arm.

Accuracy per arm: rel_rms over `--refch` channels against a float64 torch einsum of the same bf16 operands, and
`torch.equal` against the arm named in SAME (the production arm of the same variant), where an arm should be
bit-exact by construction.

AICLK sampled out of process from the node this process opened (250 ms).

usage: mmbench.py OUT --chip C [--arms a,b,...] [--reps 7] [--calls 8] [--c 128] [--s 736]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--chip", type=int, required=True)
ap.add_argument("--arms", default="")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--calls", type=int, default=8)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--s", type=int, default=736)
ap.add_argument("--refch", type=int, default=3)
ap.add_argument("--devprof", action="store_true", help="per-RISC device durations of one call per arm (Tracy build)")
A = ap.parse_args()
OUT = Path(A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s[:700], flush=True)


from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [A.chip])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
NODES = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                if os.path.exists(f"/proc/self/fd/{fd}")
                and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")
                and os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1].isdigit()})
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:
    try: v = open("/sys/class/tenstorrent/tenstorrent!{NODES[0]}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(str(time.monotonic()) + chr(9) + v + chr(10)); f.flush(); time.sleep(0.25)
"""])
REV = os.popen(f"git -C {ROOT} rev-parse --short HEAD 2>/dev/null").read().strip() or (
    (ROOT / "REVISION").read_text().strip() if (ROOT / "REVISION").exists() else "unknown")
log(ev="nodes_open", nodes=NODES, arch=str(dev.arch()), grid=list(T.COMPUTE_GRID_MAIN), git=REV, args=vars(A),
    mm_pipe=T.MM_PIPE, overlay=os.environ.get("TT_BIO_METAL_OVERLAY"), runtime_root=os.environ.get("TT_METAL_RUNTIME_ROOT"))


def aiclk(t0, t1, pad=0.3):
    v = []
    for line in CLKF.read_text().splitlines():
        ts, mhz = line.split("\t")
        if t0 - pad <= float(ts) <= t1 + pad and int(mhz) > 0:
            v.append(int(mhz))
    return (min(v), max(v), len(v)) if v else None


C, S = A.c, A.s
St = S // 32
torch.manual_seed(0)
ha = torch.randn(1, C, S, S).to(torch.bfloat16)
hb = torch.randn(1, C, S, S).to(torch.bfloat16)
up = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                                           memory_config=mc)
a_d, b_d = up(ha), up(hb)
hm = torch.zeros(1, 1, S, S); hm[..., :730, :730] = 1
m_d = up(hm.to(torch.bfloat16))
REFCH = list(range(0, C, max(1, C // A.refch)))[:A.refch]


def ref(tr_a, tr_b):
    a64, b64 = ha[0, REFCH].double(), hb[0, REFCH].double()
    x = a64.transpose(-1, -2) if tr_a else a64
    y = b64.transpose(-1, -2) if tr_b else b64
    return x @ y


def ckc(fid, fp32, l1acc=True, full=False):
    c = ttnn.types.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                               fp32_dest_acc_en=fp32, packer_l1_acc=l1acc) \
        if dev.arch() == ttnn.Arch.WORMHOLE_B0 else \
        ttnn.types.BlackholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                                fp32_dest_acc_en=fp32, packer_l1_acc=l1acc)
    c.dst_full_sync_en = full
    return c


NORMAL, FAST = ckc("HiFi3", True), ckc("HiFi4", False)


def pc(sub=(1, 1), ibw_full=True):
    T._triangle_mul_program_config.cache_clear()
    return T._triangle_mul_program_config(St, ibw_full, sub)


ARMS, SAME, REFS = {}, {}, {}


def mm_arm(name, tr_a, tr_b, cfg, k, dtype=ttnn.bfloat16, same=None, mc=ttnn.DRAM_MEMORY_CONFIG):
    def run():
        return ttnn.matmul(a_d, b_d, compute_kernel_config=k, memory_config=mc, program_config=cfg,
                           dtype=dtype, transpose_a=tr_a, transpose_b=tr_b)
    ARMS[name] = run
    REFS[name] = (tr_a, tr_b)
    if same:
        SAME[name] = same


# --- einsum arms. s = starting variant (transpose_b), e = ending (transpose_a), n = no transpose (timing probe)
P11 = pc()
P13, P31 = pc((1, 3)), pc((3, 1))
log(ev="program_config", p11=str(P11))
mm_arm("s", False, True, P11, NORMAL)
mm_arm("s'", False, True, P11, NORMAL, same="s")
mm_arm("e", True, False, P11, NORMAL)
mm_arm("n", False, False, P11, NORMAL)
mm_arm("s_sb13", False, True, P13, NORMAL, same="s")
mm_arm("s_sb31", False, True, P31, NORMAL, same="s")
mm_arm("e_sb13", True, False, P13, NORMAL, same="e")
mm_arm("e_sb31", True, False, P31, NORMAL, same="e")
mm_arm("s_fast", False, True, P11, FAST)
mm_arm("e_fast", True, False, P11, FAST)
mm_arm("s_fast_sb13", False, True, P13, FAST, same="s_fast")
mm_arm("s_lofi", False, True, P11, ckc("LoFi", True))             # diagnostic: is it math-bound
mm_arm("s_b8out", False, True, P11, NORMAL, dtype=ttnn.bfloat8_b)    # diagnostic: is it output-bytes-bound
mm_arm("s_fullsync", False, True, P11, ckc("HiFi3", True, full=True), same="s")


def mask_arm():
    # the production mask multiply: in place on a [1, C, S, S] chunk by a [1, 1, S, S] mask (a copy, so a_d survives)
    t = ttnn.clone(a_d)
    return ttnn.multiply_(t, m_d)


ARMS["mask"] = mask_arm


def clone_arm():
    return ttnn.clone(a_d)                                           # roof probe: 1 read + 1 write of a chunk


ARMS["clone"] = clone_arm

names = [n for n in (A.arms.split(",") if A.arms else ARMS)]
res = {"args": vars(A), "nodes": NODES, "git": REV, "acc": {}, "ms": {}}
outs = {}
for n in names:                                                      # warm + accuracy
    try:
        y = ARMS[n]()
        ttnn.synchronize_device(dev)
        y = ARMS[n]()
        ttnn.synchronize_device(dev)
    except Exception as e:                                           # noqa: BLE001
        log(ev="arm_failed", arm=n, err=str(e)[:600])
        names = [x for x in names if x != n]
        continue
    if n in REFS:
        yt = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
        r = ref(*REFS[n])
        sel = yt[0, REFCH].double()
        e = {"rel_rms": float(((sel - r).pow(2).mean() / r.pow(2).mean()).sqrt()), "finite": bool(torch.isfinite(yt).all()),
             "sha": __import__("hashlib").sha256(yt.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[:16]}
        if n in SAME and SAME[n] in outs:
            e["equal_" + SAME[n]] = bool(torch.equal(yt, outs[SAME[n]]))
            e["max_abs_vs_" + SAME[n]] = float((yt - outs[SAME[n]]).abs().max())
        if n in ("s", "e", "s_fast", "e_fast"):
            outs[n] = yt
        res["acc"][n] = e
        log(ev="accuracy", arm=n, **e)
    else:
        ttnn.deallocate(y)
        if n == "clone":
            ttnn.deallocate(y) if y.is_allocated() else None

if A.devprof:
    ttnn.ReadDeviceProfiler(dev)
    ttnn.get_latest_programs_perf_data()
    keys = (("k", "DEVICE KERNEL DURATION [ns]"), ("br", "DEVICE BRISC KERNEL DURATION [ns]"),
            ("nc", "DEVICE NCRISC KERNEL DURATION [ns]"), ("t0", "DEVICE TRISC0 KERNEL DURATION [ns]"),
            ("t1", "DEVICE TRISC1 KERNEL DURATION [ns]"), ("t2", "DEVICE TRISC2 KERNEL DURATION [ns]"))
    for n in names:
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev); ttnn.get_latest_programs_perf_data()
        y = ARMS[n]()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(y)
        ttnn.ReadDeviceProfiler(dev)
        progs = []
        for pl in ttnn.get_latest_programs_perf_data().values():
            for p in pl:
                r = p.program_analyses_results
                progs.append(dict(cores=p.core_count, **{s: round(r[k].duration / 1e3, 2) for s, k in keys if k in r}))
        log(ev="devprof", arm=n, progs=progs)

times = {n: [] for n in names}
for rep in range(A.reps):
    for n in (names if rep % 2 == 0 else names[::-1]):
        ttnn.synchronize_device(dev)
        t0 = time.monotonic()
        for _ in range(A.calls):
            ttnn.deallocate(ARMS[n]())
        ttnn.synchronize_device(dev)
        t1 = time.monotonic()
        times[n].append((t1 - t0) / A.calls)
        log(ev="rep", arm=n, rep=rep, ms=round(1e3 * times[n][-1], 4), aiclk=aiclk(t0, t1))
for n, v in times.items():
    res["ms"][n] = {"med": round(1e3 * statistics.median(v), 4), "min": round(1e3 * min(v), 4),
                    "max": round(1e3 * max(v), 4)}
res["aiclk"] = aiclk(0, time.monotonic())
log(ev="summary", ms=res["ms"], aiclk=res["aiclk"])
(OUT / "result.json").write_text(json.dumps(res, indent=1))
log(ev="done")
