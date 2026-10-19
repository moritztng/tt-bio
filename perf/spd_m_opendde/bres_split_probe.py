"""Does a split bmm_res call (BRES_SPLIT, N where b[c] does not fit L1) finish, and match ttnn.matmul?

One arm per process (a hang must not take the others with it): python bres_split_probe.py SEQ C MODE [split]
MODE: start (transpose_b, bf16), end (transpose_a, bf16), fast (bfloat8_b, no transposes). split: 1 (default) or 0.
Prints the plan, the wall of one synced call, torch.equal and rel_rms vs ttnn.matmul.
"""
import os, sys, time

seq, C, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
split = (sys.argv[4] if len(sys.argv) > 4 else "1") == "1"

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import bmm_res

dev = T.get_device()
torch.manual_seed(0)
fast = mode == "fast"
dt = ttnn.bfloat8_b if fast else ttnn.bfloat16
up = lambda t: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
A = up(torch.randn(1, C, seq, seq))
B = up(torch.randn(1, C, seq, seq) * seq ** -0.5)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3 if not fast else ttnn.MathFidelity.LoFi,
                                       math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
ta, tb = {"start": (False, True), "end": (True, False), "fast": (False, False)}[mode]
kt = nt = seq // 32
plan = bmm_res._plan(kt, nt, bmm_res.MG.tile_bytes(dt), bmm_res.MG.tile_bytes(dt), transpose_a=ta, split=split)
print(f"seq {seq} C {C} {mode} split {split} plan (ns, depth) {plan}", flush=True)
ref = ttnn.to_torch(ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                dtype=ttnn.bfloat16, transpose_a=ta, transpose_b=tb))
bmm_res.BRES_SPLIT = split
t0 = time.perf_counter()
o = bmm_res.matmul(A, B, compute_kernel_config=ckc, transpose_a=ta, transpose_b=tb)
if o is None:
    print("declined", flush=True)
    sys.exit(0)
ttnn.synchronize_device(dev)
print(f"returned {time.perf_counter() - t0:.3f} s", flush=True)
got = ttnn.to_torch(o)
d = got.double() - ref.double()
print(f"equal {torch.equal(got, ref)} rel_rms {float(d.pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt()):.3e}",
      flush=True)
CH = slice(0, C, 16)
At, Bt = ttnn.to_torch(A)[:, CH].double(), ttnn.to_torch(B)[:, CH].double()
r64 = (At.transpose(-1, -2) if ta else At) @ (Bt.transpose(-1, -2) if tb else Bt)
rel = lambda x: float((x[:, CH].double() - r64).pow(2).mean().sqrt() / r64.pow(2).mean().sqrt())
print(f"rel64 bres {rel(got):.3e} ttnn {rel(ref):.3e}", flush=True)
