"""spd-msa: does a chip hand back what it was given? Upload a large bf16 tensor, read it back, compare
bit for bit; then the same tensor through an identity-like device op (multiply by 1) and a matmul
against the identity, both read back. Bad positions are reported by row, so a fixed DRAM fault shows
up at the same rows across repeats.

usage: TT_VISIBLE_DEVICES=<chip> python dram_roundtrip.py OUT [ROWS=541696] [COLS=1024] [REPEATS=3]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 541696
COLS = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
REP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
LOG = open(OUT / "roundtrip.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
torch.manual_seed(0)
x_h = torch.randn(ROWS, COLS).bfloat16()
eye = ttnn.from_torch(torch.eye(COLS).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def diff(name, o, r):
    bad = (o.view(torch.int16) != r.view(torch.int16)).nonzero()
    rows = sorted(set(bad[:, 0].tolist()))
    log(ev=name, n_bad=int(bad.shape[0]), rows=rows[:16], n_rows=len(rows),
        max_abs=float((o.float() - r.float()).abs().max()))


for k in range(REP):
    x = ttnn.from_torch(x_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    diff(f"readback{k}", ttnn.to_torch(x), x_h)
    y = ttnn.multiply(x, 1.0)
    diff(f"mul1_{k}", ttnn.to_torch(y), x_h); ttnn.deallocate(y)
    y = ttnn.matmul(x, eye, compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True))
    diff(f"eye_{k}", ttnn.to_torch(y), x_h); ttnn.deallocate(y)
    ttnn.deallocate(x)
log(ev="end")
