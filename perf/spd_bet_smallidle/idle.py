"""How long the card idles in a small fold, and where the host is meanwhile.

Idle = unprofiled warm fold wall minus the device kernel time of the same input. The two come from two
processes, because the device profiler slows the fold it measures:

    python perf/spd_bet_smallidle/idle.py --out RUN/ef2-256 --model esmfold2 --yaml F.yaml --a3m F.a3m
        warm fold, then --reps timed folds (AICLK sampled, harness phase brackets), then one fold under
        py-spy (--spy BIN, python stacks of every thread) and one under py-spy --native.
    python perf/spd_bet_smallidle/idle.py --out RUN/ef2-256 --profile ...
        kernel time of one warm fold from the device profiler (needs the Tracy build and the profiler
        environment of perf/spd_census/run.sh).

RFD4 takes --spec DESIGN.yaml instead of --yaml/--a3m and times one design.
"""
import argparse, collections, ctypes, json, os, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--model", required=True)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--yaml", type=Path)
ap.add_argument("--a3m", type=Path)
ap.add_argument("--spec", type=Path, help="rfd4: design spec yaml")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--profile", action="store_true")
ap.add_argument("--spy", help="py-spy binary; adds the sampled folds")
ap.add_argument("--flush", type=int, default=1000, help="--profile: read the profiler every this many device ops")
a = ap.parse_args()
if a.profile:
    os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")
a.out.mkdir(parents=True, exist_ok=True)

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE)); sys.path.insert(0, str(TREE / "scripts" / "gpu_vs_tt"))
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
torch.set_grad_enabled(False)

inst = None
if a.model == "rfd4":
    from tt_bio import rfd4 as R
    from tt_bio.rfd4.engine import get_engine
    T.set_fast_mode(a.fast)
    engine = get_engine()
    def run(i):
        tok = []
        t = time.perf_counter()
        d = a.out / f"r{'p' if a.profile else ''}{i}"; shutil.rmtree(d, ignore_errors=True)   # the writer refuses an existing cif
        R.design(a.spec, d, num_designs=1, seed=42, engine=engine,
                 on_design=lambda _id, n, _cif: tok.append(n))
        return time.perf_counter() - t, {"n_token": tok[-1] if tok else None}
else:
    import tt_baseline as B
    one_fold, _meta, state = B.build_fold(a.model, a.out / "msa", a.yaml, a.a3m, fast=a.fast)
    inst = B.Instrument(a.model, state) if a.model in B.PHASES else None
    def run(i):
        t = time.perf_counter()
        _s, m = one_fold()
        return time.perf_counter() - t, {"plddt": m.get("plddt"), "n_tokens": m.get("n_tokens")}

dev = T.get_device()
nodes = sorted({int(l.rsplit("/", 1)[1]) for l in (os.readlink(f"/proc/self/fd/{f}") for f in os.listdir("/proc/self/fd")
                                                   if os.path.islink(f"/proc/self/fd/{f}")) if l.startswith("/dev/tenstorrent/")})
clk = []
def sampler():
    while True:
        try: clk.append(int(Path(f"/sys/class/tenstorrent/tenstorrent!{nodes[0]}/tt_aiclk").read_text().split()[0]))
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=sampler, daemon=True).start()
def clock():
    r = dict(min=min(clk), max=max(clk), n=len(clk)) if clk else None
    clk.clear(); return r
def load(): return float(open("/proc/loadavg").read().split()[0])

res = dict(model=a.model, fast=a.fast, input=str(a.spec or a.yaml), profile=a.profile,
           tree=(TREE / "TREE_SHA").read_text().strip() if (TREE / "TREE_SHA").exists() else None,
           ttnn=ttnn.__file__, grid=str(dev.compute_with_storage_grid_size()), nodes=nodes)
t = time.monotonic(); run(0); res["warm_wall_s"] = round(time.monotonic() - t, 2); clock()

if a.profile:
    import ttnn.decorators as D
    OPID = ttnn._ttnn.get_device_operation_id
    ST = {"on": False, "depth": 0, "flushed": 0, "missing": 0, "ids": 0}
    K = collections.Counter(); NP = collections.Counter()   # duration bucket (us) -> kernel ns, programs
    FLUSH = [a.flush]
    BUCKETS = (50, 100, 200, 500, 2000, 10000)

    def drain():
        ttnn.synchronize_device(dev)
        for attempt in range(4):  # the read asks ARC for the clock; on a loaded box ARC can miss its 1 s deadline
            try:
                ttnn.ReadDeviceProfiler(dev); break
            except RuntimeError as e:
                if not any(m in str(e) for m in ("ARC to respond", "Failed to get AICLK")) or attempt == 3: raise
                time.sleep(2)
        hi = OPID(); seen = set()
        for progs in ttnn.get_latest_programs_perf_data().values():
            for p in progs:
                seen.add(p.program_execution_uid.runtime_id >> 10)
                k = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                if k is None or not ST["on"]: continue
                b = next((b for b in BUCKETS if k.duration < b * 1000), 10**9)
                K[b] += k.duration; NP[b] += 1
        if ST["on"]:
            miss = sum(1 for i in range(ST["flushed"], hi) if i not in seen)
            ST["missing"] += miss; ST["ids"] += hi - ST["flushed"]
            if miss > 0.01 * (hi - ST["flushed"]) and FLUSH[0] > 50:   # an overflowed profiler buffer drops programs
                FLUSH[0] //= 2
        ST["flushed"] = hi

    def hook(orig):
        def call(self, *args, **kw):
            if ST["depth"]:
                return orig(self, *args, **kw)
            ST["depth"] += 1
            try:
                out = orig(self, *args, **kw)
                if OPID() - ST["flushed"] >= FLUSH[0]:
                    drain()
                return out
            finally:
                ST["depth"] -= 1
        return call
    D.FastOperation.__call__ = hook(D.FastOperation.__call__)
    D.Operation.__call__ = hook(D.Operation.__call__)

    run(1); drain()                 # a second warm fold, so the hooked path is warm too
    clock(); ST["on"] = True
    s, meta = run(2); drain()
    tot = sum(K.values()) or 1
    res.update(meta, profiled_fold_s=round(s, 3), aiclk_mhz=clock(), kernel_s=round(tot / 1e9, 3),
               programs=sum(NP.values()), missing_ids=ST["missing"], ids=ST["ids"],
               by_duration_us={("<%d" % b if b < 10**9 else ">=%d" % BUCKETS[-1]):
                               dict(kernel_s=round(K[b] / 1e9, 3), programs=NP[b]) for b in sorted(K)})
    (a.out / "profile.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)); sys.exit(0)

reps = []
for i in range(a.reps):
    if inst: inst.on()
    l0 = load(); s, meta = run(1 + i)
    row = dict(fold_s=round(s, 3), aiclk_mhz=clock(), load=l0, **meta)
    if inst: row["phases"] = inst.row(s); inst.off()
    reps.append(row)
res["reps"] = reps
res["fold_s_median"] = sorted(r["fold_s"] for r in reps)[len(reps) // 2]
(a.out / "timed.json").write_text(json.dumps(res, indent=1))     # the timed folds survive a failed sampler

if a.spy:
    ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1).value, 0, 0, 0)   # PR_SET_PTRACER_ANY: let our child attach
    res["spy"] = {}
    for name, extra in (("py", []), ("native", ["--native"])):
        out = a.out / f"stacks-{name}.txt"
        p = subprocess.Popen([a.spy, "record", "--pid", str(os.getpid()), "-r", "100", "--idle", "--threads",
                              "-f", "raw", "-o", str(out), *extra], stdout=subprocess.DEVNULL,
                             stderr=open(a.out / f"spy-{name}.log", "w"))
        time.sleep(3 if not extra else 8)          # attach (and, for --native, symbol load) before the fold starts
        alive = p.poll() is None
        s, _ = run(10)
        if p.poll() is None: p.send_signal(signal.SIGINT)
        try: p.wait(60)
        except subprocess.TimeoutExpired: p.terminate()
        res["spy"][name] = dict(fold_s=round(s, 3), aiclk_mhz=clock(), attached=alive,
                                samples_file=out.name if out.exists() else None)

(a.out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
