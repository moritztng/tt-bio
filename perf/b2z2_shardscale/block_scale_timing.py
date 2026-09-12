"""The Pairformer block wall at 1, 2, 4 and 8 chips, and the collective cost that pays for it.

`b2z2-dual-chip-fold` measured a 1.191x block on a two-chip p300c. Nobody has run the trunk shard
past two, so nobody knows whether the curve is linear, concave or already flat -- and the shape is
the only part of a Wormhole measurement that transfers to a Blackhole box.

Three arms in one process, because a device open costs more than the measurement:

  whole   `PairformerLayer(row_shard=False)` on the 1xN mesh. Every chip computes the identical
          whole block, so the wall is one chip's block plus the mesh's per-program dispatch tax.
          At N=1 it is the plain single-chip block and it is the denominator the campaign cares
          about: what does a shard buy over ONE processor.
  shard   `PairformerLayer(row_shard=True)`: the five-op chain, four all_gathers, i axis split N
          ways. This is the deliverable.
  gather  a size sweep of `ttnn.all_gather` at THIS width, priced by the bytes each device ends
          up holding, so the block's link cost is read off a curve measured on the same mesh in
          the same process rather than off either published fit. CONTEXT 4-D: the p300c fit is
          10x wrong below ~2 MB, and sharding N ways divides the per-device bytes by N, which
          walks the block's own transfers toward exactly that regime as N grows.

    MESH_N=4 TT_VISIBLE_DEVICES=16,17,18,19 SCALE_S=512 \
    PYTHONPATH=$PWD python3 perf/b2z2_shardscale/block_scale_timing.py
"""

import json
import os
import pathlib
import statistics
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = str(_HERE.parents[1])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, str(_HERE))

import torch  # noqa: E402
import meshdesc  # noqa: E402

N = int(os.environ.get("MESH_N", "2"))
SIZES = [int(x) for x in os.environ.get("SCALE_S", "512").split(",")]
REPS = int(os.environ.get("SCALE_REPS", "15"))
WARM = int(os.environ.get("SCALE_WARM", "3"))
OUT_PATH = os.environ.get("SCALE_OUT", f"/tmp/b2z2_scale_{N}.json")
ARMS_BSHARD = os.environ.get("SCALE_BSHARD", "0") == "1"
C_Z, C_S = 128, 384

if N > 1:
    meshdesc.install(N)

import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


OPENS = [0]


def _mesh_open(device_id, kwargs):
    OPENS[0] += 1
    with tt._device_init_lock():
        if N > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = (ttnn.open_mesh_device(ttnn.MeshShape(1, N), **kwargs) if N > 1
               else ttnn.open_device(device_id=device_id, **kwargs))
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
dev = tt.get_device()
assert OPENS[0] == 1, "the mesh was opened more than once"
log(f"device={dev} arch={dev.arch()} grid={tt.CORE_GRID_MAIN} N={N} sizes={SIZES}")

RES = {"mesh_n": N, "sizes": SIZES, "reps": REPS, "arch": str(dev.arch()), "bshard": ARMS_BSHARD,
       "visible": os.environ.get("TT_VISIBLE_DEVICES"), "by_size": {}, "gather": []}

kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)

torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
layer = tt.PairformerLayer(32, 4, 24, 16, True,
                           {k: v.float() for k, v in rl.state_dict().items()}, KC)
log(f"layer built, {len(rl.state_dict())} weight tensors")

REPL = ttnn.replicate_tensor_to_mesh_mapper(dev) if N > 1 else None


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


def time_block(row_shard, s_host, z_host, PAIR_MASK, ATTN, b_shard=False):
    """Median block wall. The residual updates are in place, so the block returns the tensor it
    was given: s and z are rebuilt every rep and uploaded OUTSIDE the timed region, and it is
    those handles that get freed, not the returned ones."""
    for _ in range(WARM):
        s_t, z_t = up(s_host), up(z_host)
        layer(s_t, z_t, PAIR_MASK, ATTN, ATTN, row_shard=row_shard, b_shard=b_shard)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(z_t)
        ttnn.deallocate(s_t)
    v = []
    for _ in range(REPS):
        s_t, z_t = up(s_host), up(z_host)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        layer(s_t, z_t, PAIR_MASK, ATTN, ATTN, row_shard=row_shard, b_shard=b_shard)
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(z_t)
        ttnn.deallocate(s_t)
    v.sort()
    return {"median_ms": statistics.median(v) * 1e3, "min_ms": v[0] * 1e3, "max_ms": v[-1] * 1e3,
            "spread_pct": (v[-1] - v[0]) / statistics.median(v) * 100,
            "samples_ms": [x * 1e3 for x in v]}


for S in SIZES:
    # mesh_partition splits evenly and a slab has to stay tile-aligned, so the token axis must be
    # a multiple of 32*N at width N. 640 aa is on the bit-exact ladder at 2 chips and off it at 8.
    if S % (32 * N):
        log(f"S={S} skipped at N={N}: {S}/{N} rows is not a whole number of tiles")
        continue
    m1 = torch.ones(1, S)
    z_host = torch.randn(1, S, S, C_Z, dtype=torch.float32)
    s_host = torch.randn(1, S, C_S, dtype=torch.float32)
    PAIR_MASK = up(m1[:, :, None] * m1[:, None, :])
    ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)
    arms = {}
    # `bshard` is the third arm: the row shard with the `b` role split too. It is the deliverable
    # of `b2z2-shard-replication-attack` and it is timed in the SAME process, against the same
    # `whole` and `shard` arms, so the three numbers share a device open and a program cache.
    for arm, rs, bs in (("whole", False, False), ("shard", True, False), ("bshard", True, True)):
        if rs and N == 1:
            continue
        if bs and not ARMS_BSHARD:
            continue
        arms[arm] = time_block(rs, s_host, z_host, PAIR_MASK, ATTN, b_shard=bs)
        a = arms[arm]
        log(f"S={S:4d} {arm:6s} block {a['median_ms']:8.3f} ms  (min {a['min_ms']:.3f} "
            f"max {a['max_ms']:.3f}, spread {a['spread_pct']:.1f} %)")
    if "shard" in arms:
        arms["shard_ratio_same_mesh"] = arms["whole"]["median_ms"] / arms["shard"]["median_ms"]
        log(f"S={S:4d} shard vs whole ON THIS MESH: {arms['shard_ratio_same_mesh']:.4f}x")
    if "bshard" in arms:
        arms["bshard_ratio_same_mesh"] = arms["whole"]["median_ms"] / arms["bshard"]["median_ms"]
        arms["bshard_vs_shard"] = arms["shard"]["median_ms"] / arms["bshard"]["median_ms"]
        log(f"S={S:4d} bshard vs whole ON THIS MESH: {arms['bshard_ratio_same_mesh']:.4f}x  "
            f"| bshard vs shard: {arms['bshard_vs_shard']:.4f}x")
    RES["by_size"][str(S)] = arms
    for t_ in (PAIR_MASK, ATTN):
        ttnn.deallocate(t_)

# --- the collective, on this mesh, priced by the bytes it actually moves -------------------------
# Sizes are the FULL tensor each device ends up holding. The block's own gather is the pair track
# at [1, S, S, 128] bf16; the sweep brackets it so the latency knee at this width is visible and
# not assumed.
if N > 1:
    full_bytes = max(SIZES) ** 2 * C_Z * 2
    sizes = [b for b in (0.25e6, 1e6, 2e6, 8e6, 33e6, full_bytes, 2 * full_bytes) if b >= N * 2048]
    shard_map = ttnn.shard_tensor_to_mesh_mapper(dev, dim=1)
    for b in sizes:
        rows = max(N, int(round(b / (C_Z * 2 * 32))) * 32)
        rows = (rows // (32 * N)) * (32 * N) or 32 * N
        host = torch.zeros(1, rows, C_Z)
        t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            mesh_mapper=shard_map)
        g = ttnn.all_gather(t, dim=1)        # warm: compiles
        ttnn.synchronize_device(dev)
        ttnn.deallocate(g)
        v = []
        for _ in range(REPS):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            g = ttnn.all_gather(t, dim=1)
            ttnn.synchronize_device(dev)
            v.append(time.perf_counter() - t0)
            ttnn.deallocate(g)
        v.sort()
        med = statistics.median(v)
        out_b = rows * C_Z * 2
        # A ring all_gather moves (N-1)/N of the output into each device, on N-1 serial hops.
        moved = out_b * (N - 1) / N
        RES["gather"].append({"out_bytes": out_b, "per_device_in_bytes": out_b // N,
                              "moved_bytes": moved, "median_us": med * 1e6,
                              "gbps_per_dir": moved / med / 1e9,
                              "min_us": v[0] * 1e6, "max_us": v[-1] * 1e6})
        log(f"gather out {out_b/1e6:7.2f} MB (in {out_b/N/1e6:6.2f} MB/dev)  "
            f"{med*1e6:9.1f} us  {moved/med/1e9:6.2f} GB/s/dir")
        ttnn.deallocate(t)

tt.cleanup()
if N > 1:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
