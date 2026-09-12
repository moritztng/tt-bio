"""Does a row slab cost LESS THAN HALF the whole op? One chip, warm, real modules.

The shard is bit-exact (slab_bitexact.py). What is unproven is speed, and there is a named reason to
doubt it: the trimul slab declines the E6 fused gated channel move and the custom reblock kernels,
because both want a square destination and one four-role projection, so it falls back to stock
ttnn.permute. E6 is worth 1.2981x on the starting trimul and 1.3329x on the ending one at 512 aa.

The shard only pays if one slab costs less than half the whole op, because on two chips the two
slabs run concurrently and the block's wall clock is ONE slab. So that is the ratio measured here,
on a single chip, which is all this question needs:

    slab_speedup = whole / slab      >2.00 is superlinear, 2.00 is a clean halving,
                                     <2.00 is the E6 decline eating into the win,
                                     <1.00 means a slab is slower than the whole op.

Warm and median of N: slab_bitexact.py's per-op times are cold and include kernel compilation (it
shows a 1339 ms trimul against a 36.70 ms whole block), so they are not evidence.
"""

import json
import os
import pathlib
import statistics as st
import sys
import time

import torch

# Time the checkout this script LIVES IN, not the `tt_bio` the env has installed, exactly as
# slab_bitexact.py does and for the same reason: `python perf/.../slab_perf.py` puts the SCRIPT's
# directory on sys.path, not the cwd, and the editable install in /home/ttuser/tt-bio-dev/env
# resolves to a different tree. Without this the run either times code nobody edited or, if that
# tree predates `row_slab`, dies on an unexpected keyword.
_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

OUT = os.environ.get("SLABPERF_OUT", "/tmp/b2z2_slabperf.json")
REPS = int(os.environ.get("SLABPERF_REPS", "7"))
S, C = 512, 128

import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import reference as ref  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


dev = tt.get_device()
KC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

torch.manual_seed(0)
rl = ref.PairformerLayer(384, C, 16, 0.25, 32, 4, v2=True)
# Boltz-2 zeroes both trimul output projections at init (final_init_), so a freshly built reference
# layer returns exactly zero. Timing is not data-dependent on Tensix, but randomize anyway so this
# script cannot be mistaken for a correctness check.
w = {k: torch.randn_like(v) * 0.05 if v.dtype.is_floating_point else v
     for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, w, KC)
log(f"device={dev} layer built ({len(w)} tensors)")

f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
z = f(torch.randn(1, S, S, C))
m1 = torch.ones(1, S)
mask = f(m1[:, :, None] * m1[:, None, :])
attn = f((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

OPS = {
    "trimul_start": (layer.triangle_multiplication_start, mask),
    "trimul_end": (layer.triangle_multiplication_end, mask),
    "triatt_start": (layer.triangle_attention_start, attn),
    "triatt_end": (layer.triangle_attention_end, attn),
    "transition_z": (layer.transition_z, None),
}
res = {"S": S, "reps": REPS, "ops": {}}


def bench(call):
    ttnn.deallocate(call())            # warm / compile
    ttnn.synchronize_device(dev)
    v = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        o = call()
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return st.median(v)


for name, (op, extra) in OPS.items():
    if extra is None:
        whole = bench(lambda: op(z))
        slab = bench(lambda: op(z, row_slab=(0, S // 2)))
    else:
        whole = bench(lambda: op(z, extra))
        slab = bench(lambda: op(z, extra, row_slab=(0, S // 2)))
    r = {"whole_ms": whole * 1e3, "slab_ms": slab * 1e3, "slab_speedup": whole / slab}
    res["ops"][name] = r
    verdict = ("superlinear" if r["slab_speedup"] > 2.05 else
               "clean halving" if r["slab_speedup"] > 1.90 else
               "partial" if r["slab_speedup"] > 1.0 else "SLOWER THAN WHOLE")
    log(f"{name:14s} whole {whole*1e3:8.3f} ms | slab {slab*1e3:8.3f} ms | "
        f"speedup {r['slab_speedup']:5.3f}x  {verdict}")

tot_w = sum(v["whole_ms"] for v in res["ops"].values())
tot_s = sum(v["slab_ms"] for v in res["ops"].values())
res["pair_track_total"] = {"whole_ms": tot_w, "slab_ms": tot_s, "speedup": tot_w / tot_s}
log(f"{'PAIR-TRACK SUM':14s} whole {tot_w:8.3f} ms | slab {tot_s:8.3f} ms | "
    f"speedup {tot_w/tot_s:5.3f}x")
json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
