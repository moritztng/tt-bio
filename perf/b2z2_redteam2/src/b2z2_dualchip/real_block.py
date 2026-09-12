"""A REAL PairformerLayer, on my own build, replicated across the p300c pair.

Every number this row has quoted for the block (36.4994 ms) is borrowed from another row's build.
The orchestrator asked each row to price its lever as a fraction of a block it measured itself and
to name the commit. This measures it.

Two things it settles:

  1. My own block time, at 512 aa and at 256 aa. The ratio bounds W, the shardable fraction: the
     pair track is quadratic in sequence length and the single track is linear, so a block that
     scales ~4x from 256 to 512 is almost entirely pair-track work, which is the part a shard
     halves. A block that scales ~2x is not.
  2. The mesh tax on a REAL block of ~272 programs, not on one op. 1.03-1.06x was measured per op;
     over a whole block it either holds or it does not, and the shard is charged it 280 times.

mode=mesh   1x2 mesh, everything replicated -> both chips do the identical whole block in parallel,
            so the wall clock is what one chip takes.
mode=single one chip, no mesh, in its own process (a submesh cannot be timed -- its parent's
            synchronize does not drain it, which is what produced this campaign's phantom mesh tax).
"""

import json
import os
import statistics
import sys
import time

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "mesh"
OUT = os.environ.get("BLOCK_OUT", f"/tmp/b2z2_block_{MODE}.json")
REPS = int(os.environ.get("BLOCK_REPS", "7"))

from tt_bio.device_lease import CardSetLease  # noqa: E402

# Only the mesh arm takes the lease here. The single arm goes through tt_bio.get_device(), which
# acquires the same flock itself -- pre-acquiring makes the process collide with its OWN lease and
# die after the 120 s timeout claiming the card is held by this very pid.
if (sys.argv[1] if len(sys.argv) > 1 else "mesh") == "mesh":
    CardSetLease().acquire()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import reference as ref  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


if MODE == "mesh":
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
    tt._device = dev          # so Module.__init__ -> get_device() lands on the pair
else:
    dev = tt.get_device()
log(f"mode={MODE} device={dev}")

KC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

# Boltz-2 trunk geometry: PairformerModule(8, 32, 4, 24, 16, True) with v2=True.
torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)
log(f"layer built, {len(weights)} weight tensors")

res = {"mode": MODE, "reps": REPS, "sizes": {}}
try:
    for S in (256, 512):
        s = torch.randn(1, S, 384, dtype=torch.float32)
        z = torch.randn(1, S, S, 128, dtype=torch.float32)
        m1 = torch.ones(1, S)
        pair_mask = m1[:, :, None] * m1[:, None, :]
        attn = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9

        f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        mask_tt, attn_tt = f(pair_mask), f(attn)

        # The block's residual updates are in-place (`z = ttnn.add_(z, z_update)`), so the tensor
        # it returns IS the tensor it was given. Freeing the output therefore frees the input, and
        # the next call dies with "Buffer is not allocated". So: fresh s and z per rep, uploaded
        # OUTSIDE the timed region, and free those rather than the returned handles.
        s_tt, z_tt = f(s), f(z)
        layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)   # warm: compiles the block
        ttnn.synchronize_device(dev)
        ttnn.deallocate(z_tt)
        ttnn.deallocate(s_tt)

        v = []
        for _ in range(REPS):
            s_tt, z_tt = f(s), f(z)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)
            ttnn.synchronize_device(dev)
            v.append(time.perf_counter() - t0)
            ttnn.deallocate(z_tt)
            ttnn.deallocate(s_tt)
        v.sort()
        med = statistics.median(v)
        res["sizes"][S] = {"median_s": med, "min_s": v[0], "max_s": v[-1],
                           "spread_pct": (v[-1] - v[0]) / med * 100, "samples": v}
        log(f"S={S:4d}  block {med*1e3:8.3f} ms   (min {v[0]*1e3:.3f} max {v[-1]*1e3:.3f}, "
            f"spread {(v[-1]-v[0])/med*100:.1f} %)")
        for t in (mask_tt, attn_tt):
            ttnn.deallocate(t)

    if 256 in res["sizes"] and 512 in res["sizes"]:
        r = res["sizes"][512]["median_s"] / res["sizes"][256]["median_s"]
        res["scaling_256_to_512"] = r
        log(f"scaling 256->512 = {r:.3f}x   (4.00 = all pair-track/quadratic, 2.00 = all linear)")
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    log(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    if MODE == "mesh":
        try:
            ttnn.close_mesh_device(dev)
            log("closed cleanly")
        except Exception as e:
            log(f"close failed: {e}")
json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
