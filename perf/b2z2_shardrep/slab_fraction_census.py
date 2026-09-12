"""What the 28.679 ms constant is made of, measured op by op on ONE chip.

`b2z2-trunk-shard-scale-wh` fit the row-sharded PairformerLayer to `28.679 ms + 58.916 ms/N` and
named the constant's cause without measuring its size: the `b` role of both triangle products reads
the whole pair tensor on every device. The campaign has been wrong about a constant term's
composition twice, so this measures it instead.

The instrument needs no mesh. `row_slab=(0, S/f)` is what a device at width f computes: those
output rows and nothing else, reading all of `x`. It is the same code path `row_input` takes -- the
two spellings meet at `slab = shard or row_slab is not None` -- so the per-op curve

    t_op(f) = A_op + B_op / f

is the per-device compute of that op at width f, with no link, no fabric and no mesh tax in it.
Sum A_op over the chain and the total must land on the mesh fit's constant; that agreement is the
check on both instruments, and they share no measurement path.

    SLABC_S=512 SLABC_OUT=... TT_VISIBLE_DEVICES=28 PYTHONPATH=$PWD python3 \
        perf/b2z2_shardrep/slab_fraction_census.py
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
from tt_bio import tenstorrent as tt  # noqa: E402

S = int(os.environ.get("SLABC_S", "512"))
FRACTIONS = [int(x) for x in os.environ.get("SLABC_F", "1,2,4,8").split(",")]
REPS = int(os.environ.get("SLABC_REPS", "9"))
WARM = int(os.environ.get("SLABC_WARM", "2"))
OUT_PATH = os.environ.get("SLABC_OUT", f"/tmp/b2z2_slabcensus_{S}.json")
C_Z, C_S = 128, 384


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
SS = up(s_host)
MASK = up(m1[:, :, None] * m1[:, None, :])
ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)


def timed(fn, label):
    """Median wall of `fn`, whose return value is freed outside the timed region."""
    for _ in range(WARM):
        r = fn()
        ttnn.synchronize_device(dev)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    v = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    v.sort()
    med = statistics.median(v) * 1e3
    log(f"  {label:34s} {med:8.3f} ms  (min {v[0]*1e3:7.3f} max {v[-1]*1e3:7.3f}, "
        f"spread {(v[-1]-v[0])/statistics.median(v)*100:4.1f} %)")
    return {"median_ms": med, "min_ms": v[0] * 1e3, "max_ms": v[-1] * 1e3,
            "samples_ms": [x * 1e3 for x in v]}


def s_track():
    """The single track, which the shard replicates on purpose. Not a function of f."""
    s_norm = ttnn.layer_norm(SS, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                             epsilon=1e-5, compute_kernel_config=layer.compute_kernel_config)
    u = layer.attention_pair_bias(s_norm, Z, seq_mask=ATTN)
    ttnn.deallocate(s_norm)
    ttnn.deallocate(u)
    u2 = layer.transition_s(SS)
    return u2


OPS = {
    "trimul_start": lambda r: layer.triangle_multiplication_start(Z, MASK, row_slab=(0, r)),
    "trimul_end": lambda r: layer.triangle_multiplication_end(Z, MASK, row_slab=(0, r)),
    "triatt_start": lambda r: layer.triangle_attention_start(Z, ATTN, row_slab=(0, r)),
    "triatt_end": lambda r: layer.triangle_attention_end(Z, ATTN, row_slab=(0, r)),
    "transition_z": lambda r: layer.transition_z(Z, row_slab=(0, r)),
}

RES = {"S": S, "fractions": FRACTIONS, "reps": REPS, "arch": str(dev.arch()),
       "visible": os.environ.get("TT_VISIBLE_DEVICES"), "by_fraction": {}, "fit": {}}

for f in FRACTIONS:
    if S % (32 * f):
        log(f"f={f} skipped: {S}/{f} rows is not a whole number of tiles")
        continue
    r = S // f
    log(f"--- fraction 1/{f}: {r} of {S} output rows per device ---")
    RES["by_fraction"][str(f)] = {name: timed(lambda fn=fn, r=r: fn(r), name)
                                  for name, fn in OPS.items()}

log("--- the s track, which no width changes ---")
RES["s_track"] = timed(s_track, "s_track")

log("--- the whole unsharded block, for the denominator ---")


def whole_block():
    s_t, z_t = up(s_host), up(z_host)
    layer(s_t, z_t, MASK, ATTN, ATTN, row_shard=False)
    return None


for _ in range(WARM):
    whole_block()
    ttnn.synchronize_device(dev)
v = []
for _ in range(REPS):
    s_t, z_t = up(s_host), up(z_host)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    layer(s_t, z_t, MASK, ATTN, ATTN, row_shard=False)
    ttnn.synchronize_device(dev)
    v.append(time.perf_counter() - t0)
    ttnn.deallocate(z_t)
    ttnn.deallocate(s_t)
v.sort()
RES["whole_block"] = {"median_ms": statistics.median(v) * 1e3, "min_ms": v[0] * 1e3,
                      "max_ms": v[-1] * 1e3, "samples_ms": [x * 1e3 for x in v]}
log(f"  whole block {statistics.median(v)*1e3:8.3f} ms")


def fit(xs, ys):
    """Least squares of y = A + B*x with x = 1/f. Two unknowns, len(xs) points."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    B = (n * sxy - sx * sy) / den
    A = (sy - B * sx) / n
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    ss_res = sum((y - (A + B * x)) ** 2 for x, y in zip(xs, ys))
    return A, B, (1 - ss_res / ss_tot if ss_tot else 1.0)


got = sorted(int(k) for k in RES["by_fraction"])
A_sum = 0.0
for name in OPS:
    xs = [1.0 / f for f in got]
    ys = [RES["by_fraction"][str(f)][name]["median_ms"] for f in got]
    A, B, r2 = fit(xs, ys)
    RES["fit"][name] = {"A_ms": A, "B_ms": B, "r2": r2,
                        "replicated_pct_of_op": 100 * A / (A + B) if A + B else 0.0}
    A_sum += A
    log(f"fit {name:14s} A {A:8.3f} ms  B {B:8.3f} ms  R2 {r2:7.4f}  "
        f"replicated {100*A/(A+B):5.1f} % of the op")
A_sum += RES["s_track"]["median_ms"]
RES["fit"]["_A_total_ms"] = A_sum
RES["fit"]["_A_total_incl_s_track"] = True
log(f"A_total (five ops + s track) = {A_sum:.3f} ms  "
    f"against the mesh fit's 28.679 ms constant")

tt.cleanup()
pathlib.Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
