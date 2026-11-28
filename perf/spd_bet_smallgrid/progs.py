"""Device kernel time of one fold or design, split by the number of cores each program ran on.

The question is whether a small fold leaves half the core grid unused, so that two folds could share one chip.
Every device program reports its core count and kernel duration to the device profiler; this sums kernel time
per core count over one warm run. No attribution to call sites: perf/spd_census does that.

Needs the Tracy tt-metal build and the profiler environment of perf/spd_census/run.sh.

    python perf/spd_bet_smallgrid/progs.py --out RUN/ef2-256 --model esmfold2 --size 256 [--fast]
    python perf/spd_bet_smallgrid/progs.py --out RUN/rfd4-96 --model rfd4 --spec uncond96.yaml
"""
import argparse, collections, json, os, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--model", required=True, choices=("esmfold2", "esmfold2-fast", "rfd4"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--size", type=int, default=256, help="esmfold2: cdk2x2_<size> fixture")
ap.add_argument("--spec", type=Path, help="rfd4: design spec yaml")
ap.add_argument("--flush", type=int, default=1000, help="read the profiler every this many device ops")
a = ap.parse_args()
os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")
a.out.mkdir(parents=True, exist_ok=True)

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE)); sys.path.insert(0, str(TREE / "scripts" / "gpu_vs_tt"))
import torch, ttnn
import ttnn.decorators as D
import tt_bio.tenstorrent as T
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
torch.set_grad_enabled(False)

OPID = ttnn._ttnn.get_device_operation_id
ST = {"on": False, "depth": 0, "flushed": 0, "missing": 0, "ids": 0}
K = collections.Counter(); NP = collections.Counter()      # cores -> kernel ns, programs
SMALL = collections.Counter()                               # kernel-duration bucket (us) -> kernel ns, all cores
SPAN = [None, None]                                         # first kernel start, last kernel end (device ns)
FLUSH = [a.flush]


def drain():
    dev = T.get_device()
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
            K[p.core_count] += k.duration; NP[p.core_count] += 1
            SMALL[next((b for b in (50, 100, 200, 500) if k.duration < b * 1000), 10**9)] += k.duration
            SPAN[0] = k.start_timestamp if SPAN[0] is None else min(SPAN[0], k.start_timestamp)
            SPAN[1] = k.end_timestamp if SPAN[1] is None else max(SPAN[1], k.end_timestamp)
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

if a.model == "rfd4":
    from tt_bio import rfd4 as R
    from tt_bio.rfd4.engine import get_engine
    T.set_fast_mode(a.fast)
    engine = get_engine()
    tok = []
    def run(i):
        R.design(a.spec, a.out / f"r{i}", num_designs=1, seed=42, engine=engine,
                 on_design=lambda _id, n, _cif: tok.append(n))
        return {"n_token": tok[-1] if tok else None}
else:
    import tt_baseline as B
    fx = TREE / "perf" / "size512" / "fixtures"
    one_fold = B.build_fold(a.model, a.out / "msa", fx / f"cdk2x2_{a.size}.yaml", fx / f"cdk2x2_{a.size}.a3m",
                            fast=a.fast)[0]
    def run(i):
        s, m = one_fold()
        return {"fold_s": s, "plddt": m.get("plddt")}

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

t = time.monotonic(); warm = run(0); warm_s = time.monotonic() - t
drain()
clk.clear(); ST["on"] = True
t = time.monotonic(); meta = run(1); drain(); wall = time.monotonic() - t
ST["on"] = False

tot = sum(K.values()) or 1
res = dict(model=a.model, fast=a.fast, size=a.size if a.model != "rfd4" else None, spec=str(a.spec) if a.spec else None,
           tree=(TREE / "TREE_SHA").read_text().strip() if (TREE / "TREE_SHA").exists() else None,
           grid=str(dev.compute_with_storage_grid_size()), nodes=nodes, **meta,
           aiclk_mhz=dict(min=min(clk), max=max(clk), n=len(clk)) if clk else None,
           warm_wall_s=round(warm_s, 2), wall_s=round(wall, 2), kernel_s=round(tot / 1e9, 3),
           device_span_s=round((SPAN[1] - SPAN[0]) / 1e9, 3) if SPAN[0] is not None else None,
           programs=sum(NP.values()), missing_ids=ST["missing"], ids=ST["ids"],
           share_le36=round(sum(v for c, v in K.items() if c <= 36) / tot, 4),
           by_cores={str(c): dict(kernel_s=round(v / 1e9, 3), share=round(v / tot, 4), programs=NP[c])
                     for c, v in sorted(K.items())},
           kernel_share_by_program_duration_us={("<%d" % b if b < 10**9 else ">=500"): round(v / tot, 4)
                                                for b, v in sorted(SMALL.items())})
(a.out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
