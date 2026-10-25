"""Triangle attention's head-major bias under triatt_ln_b8 (WH fast): narrow to bfp8 then add the key mask (shipped,
cs10 640 fast: add 556 us on bfp8 + typecast 65 us per call, x 1048) vs add in bf16 then narrow. torch.equal, ms/call.

The mask is OpenDDE's additive -1e9 key mask, [1, 1, 1, S], real keys first.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/tab_bench.py OUT.json [--seq 640] [--real 611] [--heads 12]
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=640); ap.add_argument("--real", type=int, default=611)
ap.add_argument("--heads", type=int, default=12); ap.add_argument("--ncall", type=int, default=20); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = {"seq": a.seq, "real": a.real, "heads": a.heads, "arch": str(dev.arch()), "seeds": {}}
for seed, sd in ((0, 1.0), (1, 4.0), (2, 16.0)):
    torch.manual_seed(seed)
    b = up(torch.randn(1, a.heads, a.seq, a.seq) * sd)
    m = up(torch.where(torch.arange(a.seq) < a.real, 0.0, -1e9).reshape(1, 1, 1, a.seq))

    def ship():
        n = ttnn.typecast(b, ttnn.bfloat8_b)
        o = ttnn.add(n, m); n.deallocate(); return o

    def early():
        s = ttnn.add(b, m)
        o = ttnn.typecast(s, ttnn.bfloat8_b); s.deallocate(); return o

    r = {}
    outs = {}
    for name, f in (("ship", ship), ("early", early)):
        o = f(); outs[name] = ttnn.to_torch(o).float(); o.deallocate()
        ms = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(a.ncall):
                f().deallocate()
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) / a.ncall * 1e3)
        r[name] = {"ms": ms, "ms_min": min(ms)}
    ref = (ttnn.to_torch(b).double() + ttnn.to_torch(m).double())[..., :a.real]
    for name in outs:
        got = outs[name][..., :a.real].double()
        r[name]["rel_rms_real_keys"] = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
    d = outs["early"] != outs["ship"]
    r["equal"] = bool(torch.equal(outs["early"], outs["ship"]))
    r["n_diff"] = int(d.sum()); r["n_diff_real_keys"] = int(d[..., :a.real].sum())
    res["seeds"][f"sd{sd}"] = r
    print(json.dumps({f"sd{sd}": r}), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
