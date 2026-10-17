"""Measured Wormhole roofs on one chip: device kernel time from the profiler, warm, 5 reps, median.

matmul N^3 (DRAM interleaved, default program config) for bf16 / bfp8_b / bfp4_b x LoFi / HiFi2 / HiFi3 / HiFi4,
fp32 dest acc off and on; DRAM bandwidth from ttnn.add (3 streams) and ttnn.clone (2) on large bf16 tensors.
usage: roofline_wh.py OUT.json
"""
import json, sys, time, glob
from pathlib import Path
import torch, ttnn
import tt_bio.tenstorrent as T

d = T.get_device()
NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
def aiclk():
    out = {}
    for n in NODES:
        try: out[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
        except (OSError, ValueError): out[n] = None
    return out
def drain():
    ttnn.synchronize_device(d); ttnn.ReadDeviceProfiler(d)
    return sorted([(p.program_execution_uid.runtime_id, p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration,
                    p.core_count) for ps in ttnn.get_latest_programs_perf_data().values() for p in ps])
def med(v): v = sorted(v); return v[len(v) // 2]
res = {"grid": str(d.compute_with_storage_grid_size()), "aiclk_before": aiclk(), "matmul": [], "bw": []}
drain()
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3,
       "HiFi4": ttnn.MathFidelity.HiFi4}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
for N in (2048, 4096, 6144):
    x = torch.randn(N, N)
    for dn, dt in DT.items():
        a = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d)
        for fn, fid in FID.items():
            for acc in (False, True):
                cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=acc, packer_l1_acc=True)
                ks = []
                try:
                    for i in range(6):
                        c = ttnn.matmul(a, a, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
                        pr = drain(); ks.append(sum(k for _, k, _ in pr)); ttnn.deallocate(c)
                except RuntimeError as e:  # default program config can overflow L1 at some (N, format, acc)
                    print(json.dumps(dict(N=N, dtype=dn, fid=fn, fp32_acc=acc, err=str(e).splitlines()[0][:200])), flush=True)
                    continue
                k = med(ks[1:]); fl = 2.0 * N ** 3
                byt = 2 * N * N * {"bf16": 2, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}[dn] + N * N * 2
                r = dict(N=N, dtype=dn, fid=fn, fp32_acc=acc, kernel_us=k / 1e3, tflops=fl / k / 1e3,
                         gbs=byt / k, cores=pr[-1][2])
                res["matmul"].append(r); print(json.dumps(r), flush=True)
        ttnn.deallocate(a)
for R, C in ((4096, 4096), (8192, 4096), (8192, 8192)):
    x = ttnn.from_torch(torch.randn(R, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
    for op, streams in (("add", 3), ("clone", 2)):
        ks = []
        for i in range(6):
            y = ttnn.add(x, x) if op == "add" else ttnn.clone(x)
            pr = drain(); ks.append(sum(k for _, k, _ in pr)); ttnn.deallocate(y)
        k = med(ks[1:]); byt = streams * R * C * 2
        r = dict(op=op, shape=[R, C], kernel_us=k / 1e3, gbs=byt / k, cores=pr[-1][2])
        res["bw"].append(r); print(json.dumps(r), flush=True)
    ttnn.deallocate(x)
res["aiclk_after"] = aiclk()
json.dump(res, open(sys.argv[1], "w"), indent=1)
