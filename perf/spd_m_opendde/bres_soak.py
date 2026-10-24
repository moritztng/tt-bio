"""Soak `bmm_res.matmul` at the trimul einsum shape the WH fast + trunk_hifi2 hang stopped in.

tt-triage on two caught hangs (.107 ht/H5, ht/H6) found the same program both times: the einsum through bmm_res,
[1, 384, 608, 608] bf16 a and b, one core of the grid still running, its reader blocked in `cb_reserve_back` on the raw
a CB, the writer starved, and the compute threads inside the per-row transpose_wh -> mm_block_init_short switch of the
transposed-a path. This calls that one op in a loop, nothing else, so a hang here is the kernel's own and a fix can be
tested in minutes instead of 25 fold processes.

A synchronize every --sync calls bounds how far the host runs ahead; with TT_METAL_OPERATION_TIMEOUT_SECONDS set, a
hang raises instead of blocking, and the last heartbeat line says how many calls ran clean.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/bres_soak.py [--fid HiFi2] [--ta] [--tb] [--drain] [--full-tinit] [--full-minit] [--seconds 600]
"""
import argparse, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--fid", default="HiFi2")
ap.add_argument("--ta", action="store_true")
ap.add_argument("--tb", action="store_true")
ap.add_argument("--drain", action="store_true", help="BRES_DRAIN: drain every thread at each TA phase switch")
ap.add_argument("--full-tinit", action="store_true", help="BRES_FULL_TINIT: full transpose_wh_init at the TA switch")
ap.add_argument("--full-minit", action="store_true", help="BRES_FULL_MINIT: full mm_block_init at the TA switch")
ap.add_argument("--seq", type=int, default=608)
ap.add_argument("--channels", type=int, default=384)
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--sync", type=int, default=200)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import bmm_res
bmm_res.SOAK = tuple(d for d, on in (("BRES_DRAIN", a.drain), ("BRES_FULL_TINIT", a.full_tinit),
                                          ("BRES_FULL_MINIT", a.full_minit)) if on)

dev = T.get_device()
torch.manual_seed(0)
shape = (1, a.channels, a.seq, a.seq)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
x, y = up(torch.randn(shape)), up(torch.randn(shape))
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, a.fid), fp32_dest_acc_en=True)
tag = f"bres_soak fid={a.fid} ta={int(a.ta)} tb={int(a.tb)} soak={",".join(bmm_res.SOAK) or "-"} S={a.seq} C={a.channels}"


def call():
    out = bmm_res.matmul(x, y, compute_kernel_config=ckc, transpose_a=a.ta, transpose_b=a.tb)
    assert out is not None, "bmm_res declined"
    ttnn.deallocate(out)


first = bmm_res.matmul(x, y, compute_kernel_config=ckc, transpose_a=a.ta, transpose_b=a.tb)
import hashlib
print(f"{tag} digest {hashlib.sha256(ttnn.to_torch(first).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]}",
      flush=True)
ttnn.deallocate(first)
t0, n = time.time(), 1
print(f"{tag} start", flush=True)
while time.time() - t0 < a.seconds:
    for _ in range(a.sync):
        call()
    ttnn.synchronize_device(dev)
    n += a.sync
    print(f"{time.strftime('%H:%M:%S')} calls {n} {1e3 * (time.time() - t0) / n:.3f} ms/call", flush=True)
print(f"{tag} CLEAN calls {n} in {time.time() - t0:.0f} s", flush=True)
