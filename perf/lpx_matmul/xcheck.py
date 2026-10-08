"""Cross-check the sweep's trace-replay device time against the device profiler's kernel time.

For the heaviest N signatures of an r1 run, three variants each (as called; fp32 acc off; bfp8 operands + LoFi + no
acc + bfp8 out, same program config): 8 eager calls, profiler drained after each, median DEVICE KERNEL DURATION summed
over the call's programs, then devtime.Bench's trace replay of the same call. Needs a Tracy build of tt-metal
(TT_METAL_DEVICE_PROFILER=1, e.g. lpx-census's ~/lpx/census/env.sh with this row's chip and lease dir).

usage (on .107):  xcheck.py R1_DIR OUT.jsonl N_TOP [rank,rank,...]
"""
import json, os, sys, time
from pathlib import Path

R1 = Path(sys.argv[1]); OUT = open(sys.argv[2], "a"); NTOP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
RANKS = {int(r) for r in sys.argv[4].split(",")} if len(sys.argv) > 4 else None

import ttnn
import tt_bio.tenstorrent as T
from devtime import Bench
from replay import parse

dev = T.get_device(trace="protenix")
B = Bench(dev)
NODE = int(os.environ.get("LPX_NODE", "-1"))
def aiclk():
    try: return int(Path(f"/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read_text().split()[0])
    except Exception: return None

def drain():
    ttnn.synchronize_device(dev); ttnn.ReadDeviceProfiler(dev)
    return [p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration
            for ps in ttnn.get_latest_programs_perf_data().values() for p in ps]

OPS = dict(linear=ttnn.linear, matmul=ttnn.matmul)
calls = json.loads((R1 / "calls.json").read_text())
ck = {e["key"]: e["ckc"] for e in map(json.loads, open(R1 / "bench.jsonl")) if e.get("ev") == "sweep"}
done = 0
for rank, c in enumerate(calls):
    if done >= NTOP: break
    if c["op"] not in OPS or c["key"] not in ck or (RANKS and rank not in RANKS): continue
    op, a, kw = parse(c["key"])
    for var, i0, fid, acc, o in (("base", None, None, None, None), ("noacc", None, None, False, None),
                                 ("b8b8_lofi_noacc_ob8", "b8", "LoFi", False, "b8")):
        args, kws = list(a), dict(kw)
        args[0] = B.mk(a[0], B.DT[i0] if i0 else None)
        args[1] = B.mk(a[1], B.DT[i0] if i0 else None)
        for k2, v in kw.items():                 # bias: a tensor kwarg (rank 14 died on it unbuilt)
            if isinstance(v, tuple) and v[:1] == ("T",): kws[k2] = B.mk(v, B.DT[i0] if i0 else None)
        kws["compute_kernel_config"] = B.ckc(ck[c["key"]], fid, acc)
        if o: kws["dtype"] = B.DT[o]
        fn = lambda: OPS[op](*args, **kws)
        rec = dict(rank=rank, op=op, var=var, n_total=c["n_total"])
        try:
            drain()
            ks = []
            for _ in range(8):
                B.free(fn()); ks.append(sum(drain()) / 1e3)
            ks = sorted(ks[2:]); rec.update(prof_us=ks[len(ks) // 2], prof_spread=(ks[-1] - ks[0]) / ks[len(ks) // 2])
            r = B.time(fn); drain()
            rec.update(trace_us=r["us"], trace_spread=r["spread"], mode=r["mode"], ratio=r["us"] / rec["prof_us"])
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        finally:
            for t in (*args[:2], *(v for v in kws.values() if isinstance(v, ttnn.Tensor))): B.free(t)
        rec.update(aiclk=aiclk(), t_unix=time.time())
        OUT.write(json.dumps(rec) + "\n"); OUT.flush(); print(json.dumps(rec), flush=True)
    done += 1
os._exit(0)
