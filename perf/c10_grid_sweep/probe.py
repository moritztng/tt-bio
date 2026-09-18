#!/usr/bin/env python3
"""Bounded preflight: device grid, FORCE_AICLK response, core_grid acceptance per shape."""
import fcntl, json, os, struct, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as T

IOC = (0xFA << 8) | 17
POST, POLL = 1 << 0, 1 << 1
FORCE_AICLK = 0x33

def smc(fd, mt, *args):
    msg = [mt] + list(args) + [0] * (7 - len(args))
    fcntl.ioctl(fd, IOC, struct.pack("=IIII8I", 48, POST, 0, 0, *msg))
    dl = time.time() + 2.0
    while time.time() < dl:
        buf = bytearray(struct.pack("=IIII8I", 48, POLL, 0, 0, *([0] * 8)))
        try:
            fcntl.ioctl(fd, IOC, buf, True)
        except OSError as e:
            if e.errno == 11:
                time.sleep(0.005); continue
            raise
        r = struct.unpack("=IIII8I", bytes(buf))[4:]
        return r[0] & 0xFF, r[0] >> 16
    raise TimeoutError("no ARC response")

dev = T.get_device()
a = dev.compute_with_storage_grid_size()
out = {"arch": str(dev.arch()), "device_grid": [int(a.x), int(a.y)],
       "COMPUTE_GRID_MAIN": list(T.COMPUTE_GRID_MAIN),
       "measured": T.COMPUTE_GRID_MEASURED, "host": os.uname().nodename}
node = 0
root = Path(f"/sys/class/tenstorrent/tenstorrent!{node}")
fd = os.open(f"/dev/tenstorrent/{node}", os.O_RDWR | os.O_APPEND)
out["aiclk_before"] = int((root / "tt_aiclk").read_text())
out["force_response"] = list(smc(fd, FORCE_AICLK, 1350))
time.sleep(0.3)
out["aiclk_after"] = int((root / "tt_aiclk").read_text())

ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
def dram(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
S = 512
shapes = {
  "mm_768_1536":  (dram(torch.randn(1,S,768)*.1),  dram(torch.randn(768,1536)*.1)),
  "mm_768_768":   (dram(torch.randn(1,S,768)*.1),  dram(torch.randn(768,768)*.1)),
  "mm_1536_768":  (dram(torch.randn(1,S,1536)*.1), dram(torch.randn(1536,768)*.1)),
  "mm_768_3072":  (dram(torch.randn(1,S,768)*.1),  dram(torch.randn(768,3072)*.1)),
  "mm_pair_qk":   (dram(torch.randn(1,16,S,128)*.1), dram(torch.randn(128,512)*.1)),
  "mm_pair_av":   (dram(torch.randn(1,16,S,512)*.1), dram(torch.randn(512,128)*.1)),
  "cube":         (dram(torch.randn(4096,4096)*.05), dram(torch.randn(4096,4096)*.05)),
}
LADDER = [(11,10),(9,8),(8,8),(8,6),(8,4),(6,4),(4,4)]
acc = {}
for nm,(A,B) in shapes.items():
    acc[nm] = {}
    for (gx,gy) in LADDER:
        try:
            o = ttnn.linear(A, B, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            core_grid=ttnn.CoreGrid(y=gy, x=gx))
            ttnn.synchronize_device(dev); ttnn.deallocate(o)
            acc[nm][f"{gx}x{gy}"] = "ok"
        except Exception as e:
            acc[nm][f"{gx}x{gy}"] = f"FAIL {type(e).__name__}: {str(e)[:120]}"
out["core_grid_acceptance"] = acc
out["aiclk_end"] = int((root / "tt_aiclk").read_text())
out["release_response"] = list(smc(fd, FORCE_AICLK, 0))
os.close(fd)
print(json.dumps(out, indent=1))
