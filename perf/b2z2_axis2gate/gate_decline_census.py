"""Why the fused channel move is declined on a row slab, and what the decline costs the op.

`b2z2-shard-replication-attack` found `trimul_end` fitting `A + B/f` at R2 0.70 with two flat
points, and named the cause without pricing it: `eligible_gated` reads the destination width off
axis 2, the ending variant's slab lands there, so the DRAM window `N >= 256` declines the fused
kernel from four chips up. This measures it on ONE chip -- `row_slab=(0, S/f)` runs exactly the
shapes a device at width f runs, with no link and no mesh in the measurement.

Three things come out, and the first is the one the gate never gave anybody:

  DECLINE   `reblock_permute.REJECTS` delta per op per fraction, so "declined" is a line in the
            log with a reason and a shape on it instead of a lever that reads as doing nothing.
  COST      the same op timed with the window shipped and with it open, in one process against one
            weight set, plus the engagement counter (`STATS_GATED[0]`) for each -- a timing
            without an engagement delta under it is not evidence the gate moved.
  PARITY    `torch.equal` of the two arms' outputs at every fraction. The fused path and the
            four-way split are the same arithmetic; if they ever differ the window is not a
            performance knob and the whole change is off.

    SLABC_S=512 GATE_OUT=... TT_VISIBLE_DEVICES=20 PYTHONPATH=$PWD python3 \
        perf/b2z2_axis2gate/gate_decline_census.py
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

import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import reblock_permute as _reblock  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

S = int(os.environ.get("SLABC_S", "512"))
FRACTIONS = [int(x) for x in os.environ.get("SLABC_F", "2,4,8,16").split(",")]
REPS = int(os.environ.get("SLABC_REPS", "9"))
WARM = int(os.environ.get("SLABC_WARM", "2"))
OUT_PATH = os.environ.get("GATE_OUT", f"/tmp/b2z2_gatecensus_{S}.json")
C_Z, C_S = 128, 384

# The two arms. "shipped" is the window as it stands: a slab is declined the moment its
# destination row falls under GATED_DRAM_N_MIN. "open" drops the group-count clause to 1, which
# admits every slab, so the per-fraction ratio between the arms IS the decline's price and the
# fraction where it stops paying is where the shipped threshold belongs.
ARMS = {"shipped": 10 ** 9, "open": 1}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


dev = tt.get_device()
log(f"device={dev} arch={dev.arch()} grid={tt.CORE_GRID_MAIN} S={S} fractions={FRACTIONS}")

kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)

torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
layer = tt.PairformerLayer(32, 4, 24, 16, True,
                           {k: v.float() for k, v in rl.state_dict().items()}, KC)

m1 = torch.ones(1, S)
z_host = torch.randn(1, S, S, C_Z, dtype=torch.float32)
s_host = torch.randn(1, S, C_S, dtype=torch.float32)


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


Z = up(z_host)
MASK = up(m1[:, :, None] * m1[:, None, :])
ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

OPS = {
    "trimul_start": lambda r: layer.triangle_multiplication_start(Z, MASK, row_slab=(0, r)),
    "trimul_end": lambda r: layer.triangle_multiplication_end(Z, MASK, row_slab=(0, r)),
}
WHOLE = {
    "trimul_start": lambda: layer.triangle_multiplication_start(Z, MASK),
    "trimul_end": lambda: layer.triangle_multiplication_end(Z, MASK),
}


def rejects_snapshot():
    return dict(_reblock.REJECTS)


def rejects_delta(before):
    out = {}
    for k, v in _reblock.REJECTS.items():
        d = v - before.get(k, 0)
        if d:
            out[f"{k[0]} {list(k[1])}"] = d
    return out


def timed(fn, label):
    """Median wall, plus the engagement and decline the call itself produced.

    The counters are read around ONE cold call before the warmups, so they count one op's worth of
    moves and not REPS of them."""
    rej0, eng0 = rejects_snapshot(), _reblock.STATS_GATED[0]
    r = fn()
    ttnn.synchronize_device(dev)
    served = _reblock.STATS_GATED[0] - eng0
    declined = rejects_delta(rej0)
    out_host = ttnn.to_torch(r)
    ttnn.deallocate(r)
    for _ in range(WARM):
        r = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(r)
    v = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    v.sort()
    med = statistics.median(v) * 1e3
    log(f"  {label:38s} {med:8.3f} ms  served={served:3d}  "
        f"declined={declined if declined else '{}'}")
    return {"median_ms": med, "min_ms": v[0] * 1e3, "max_ms": v[-1] * 1e3,
            "spread_pct": (v[-1] - v[0]) / statistics.median(v) * 100,
            "gated_moves_served": served, "declined": declined,
            "samples_ms": [x * 1e3 for x in v]}, out_host


RES = {"S": S, "fractions": FRACTIONS, "reps": REPS, "arch": str(dev.arch()),
       "grid": str(tt.CORE_GRID_MAIN), "visible": os.environ.get("TT_VISIBLE_DEVICES"),
       "arms": ARMS, "by_arm": {a: {} for a in ARMS}, "parity": {}}

HOST = {}
for arm, gmin in ARMS.items():
    _reblock.GATED_DRAM_GROUPS_MIN = gmin
    log(f"=== arm {arm}: GATED_DRAM_N_MIN={_reblock.GATED_DRAM_N_MIN} "
        f"GATED_DRAM_GROUPS_MIN={gmin} ===")
    for f in FRACTIONS:
        if S % (32 * f):
            log(f"f={f} skipped: {S}/{f} rows is not a whole number of tiles")
            continue
        r = S // f
        log(f"--- 1/{f}: {r} of {S} rows ---")
        for name, fn in OPS.items():
            m, host = timed(lambda fn=fn, r=r: fn(r), f"{name} 1/{f}")
            RES["by_arm"][arm][f"{name}/{f}"] = m
            HOST[(arm, name, f)] = host
    log(f"--- whole tensor (arm {arm}) ---")
    for name, fn in WHOLE.items():
        m, host = timed(fn, f"{name} whole")
        RES["by_arm"][arm][f"{name}/1"] = m
        HOST[(arm, name, 1)] = host

# --- parity: the two arms must be the same bytes -------------------------------------------------
for name in OPS:
    for f in FRACTIONS + [1]:
        a, b = HOST.get(("shipped", name, f)), HOST.get(("open", name, f))
        if a is None or b is None:
            continue
        eq = bool(torch.equal(a, b))
        mx = float((a.float() - b.float()).abs().max())
        RES["parity"][f"{name}/{f}"] = {"torch_equal": eq, "max_abs": mx}
        log(f"  parity {name} 1/{f}: torch.equal={eq} max_abs={mx}")

# --- the ratio table -----------------------------------------------------------------------------
RES["ratio"] = {}
for k in RES["by_arm"]["shipped"]:
    s_, o_ = RES["by_arm"]["shipped"][k], RES["by_arm"]["open"].get(k)
    if not o_:
        continue
    RES["ratio"][k] = {"shipped_ms": s_["median_ms"], "open_ms": o_["median_ms"],
                       "speedup": s_["median_ms"] / o_["median_ms"],
                       "served_shipped": s_["gated_moves_served"],
                       "served_open": o_["gated_moves_served"]}
    log(f"  {k:22s} shipped {s_['median_ms']:8.3f}  open {o_['median_ms']:8.3f}  "
        f"{s_['median_ms']/o_['median_ms']:.4f}x   served {s_['gated_moves_served']} -> "
        f"{o_['gated_moves_served']}")

tt.cleanup()
pathlib.Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
