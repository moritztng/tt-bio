"""OpenDDE token-DiT linears at the partner short bucket, HiFi4 (shipped, both modes) vs HiFi2, bf16 operands.

The cs7 census at 288 (.107 cs7/c288) puts diffusion at ~49 % of device time, and its linears run HiFi4 with fp32
dest in both modes at ~0.5 of the roof: qkv [5,576,768]@[768,3072] 475 us, three s-term linb [.,768]@[768,1536]
~220 us, the transition out [.,1536]@[1536,768] ~205 us, gate/out [.,768]@[768,768] 129 us. dit_b8 (bfp8 + HiFi2)
failed OpenDDE's fast grade on its bfp8 weights; this asks what HiFi2 alone buys on bf16 operands, and at what
float64 error, before building a lever.

Arms per shape: hifi4 (fp32 dest, the shipped ckc) and hifi2 (fp32 dest). Timing: NCALL back-to-back calls after
one sync, REPS times, arms interleaved, median us per call. Accuracy: rel_rms vs float64 x @ w (+ b) on every 7th row.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/dit_fid_bench.py OUT.json [--rows 2880]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--rows", type=int, default=5 * 576)
ap.add_argument("--ncall", type=int, default=20); ap.add_argument("--reps", type=int, default=5)
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
SHAPES = {"qkv": (768, 3072), "linb": (768, 1536), "trans_out": (1536, 768), "gate_out": (768, 768)}
ckc = lambda f: ttnn.WormholeComputeKernelConfig(math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True,
                                                 packer_l1_acc=True)
ARMS = {"hifi4": ckc(ttnn.MathFidelity.HiFi4), "hifi2": ckc(ttnn.MathFidelity.HiFi2)}
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = {"rows": a.rows, "aiclk_note": "sample AICLK beside this run (run_cell aiclk.tsv)", "shapes": {}}
for name, (K, N) in SHAPES.items():
    torch.manual_seed(K + N)
    hx, hw, hb = torch.randn(1, a.rows, K), torch.randn(K, N) * K ** -0.5, torch.randn(N) * 0.1
    x, w, b = up(hx), up(hw), up(hb.reshape(1, N))
    sel = slice(0, a.rows, 7)
    ref = ttnn.to_torch(x)[0, sel].double() @ ttnn.to_torch(w).double() + ttnn.to_torch(b).double()[0]
    call = lambda arm: ttnn.linear(x, w, bias=b, compute_kernel_config=ARMS[arm], core_grid=T.CORE_GRID_MAIN)
    err = {}
    for arm in ARMS:
        o = ttnn.to_torch(call(arm))[0, sel].double()
        err[arm] = float(((o - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
    t = {arm: [] for arm in ARMS}
    for _ in range(a.reps):
        for arm in ARMS:
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(a.ncall):
                o = call(arm); ttnn.deallocate(o)
            ttnn.synchronize_device(dev); t[arm].append((time.perf_counter() - t0) / a.ncall * 1e6)
    res["shapes"][name] = {"K": K, "N": N, "us": {k: statistics.median(v) for k, v in t.items()}, "rel_rms": err}
    print(name, res["shapes"][name], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
