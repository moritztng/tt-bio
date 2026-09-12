"""compute(N) with the `b` role split, measured so the constant term can be compared to 28.679.

`b2z2-trunk-shard-scale-wh` fitted **compute(N) = block_wall(N) - 4 x G_pair(N)** and got
`28.679 ms + 58.916 ms/N`. `b2z2-shard-replication-attack` fitted the RAW BLOCK WALL of the
`b_shard` arm and got 34.907 ms. Those two constants are not comparable: one has the link
subtracted and the other does not, and `b_shard` adds two more collectives on top of the four.
This file measures everything the comparable fit needs, in ONE process on ONE mesh:

  whole / shard / bshard   the three block arms, timed ROUND-ROBIN rather than in blocks. The box
                           is shared and loadavg moves during a run; arm-blocked timing books that
                           drift as an arm difference, interleaving does not. loadavg is recorded
                           per rep so the claim can be checked rather than asserted.
  G_pair                   [1, S, S, 128] gathered on dim 1 -- the four the row shard takes.
  G_b                      [1, 128, S, S] gathered on dim 3 -- the two `b_shard` adds.
                           Both are value-checked against a full ramp on every device before they
                           are timed, so a gather that returned the local slab N times cannot be
                           priced as a gather.

Three things are read out of the process rather than assumed:

  * `ROW_SHARD_CALLS` -- the engagement counter. An absent shard produces a perfect result and a
    plausible wall; the parent row lost a whole fold run to exactly that. Asserted, not printed.
  * `reblock_permute.REJECTS` -- `eligible_gated` declines silently at N>=4 because it reads the
    destination width off axis 2, where the ending trimul's slab lands. It bends both sharded arms,
    so it inflates both constants and cancels in their difference. `b2z2-reblock-axis2-gate` owns
    the fix; this row owes the count.
  * `_L1_OUT_RUNG` -- a shard's real failure mode is L1 pressure stepping a config down a rung.

Parity is checked in the same process against the `whole` arm on EVERY device, with a perturbed
control that must be rejected. It is not checked against `tt_bio.reference`, which zero-initialises
23 of a PairformerLayer's weights including both trimuls' `p_out`.

    MESH_N=2 TT_VISIBLE_DEVICES=16,17 TT_BIO_LEASE_CARDS=16,17 \
    TT_BIO_LEASE_HOLDER=worker:b2z2-bshard-timing CURVE_OUT=... \
    PYTHONPATH=$PWD python3 perf/b2z2_bshardtime/bshard_compute_curve.py
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
sys.path.insert(0, str(_HERE.parents[0] / "b2z2_shardscale"))

import torch  # noqa: E402
import meshdesc  # noqa: E402

N = int(os.environ.get("MESH_N", "2"))
S = int(os.environ.get("CURVE_S", "512"))
REPS = int(os.environ.get("CURVE_REPS", "15"))
WARM = int(os.environ.get("CURVE_WARM", "3"))
OUT_PATH = os.environ.get("CURVE_OUT", f"/tmp/b2z2_bshardcurve_{N}.json")
C_Z, C_S = 128, 384

assert N > 1, "this fit is about what a mesh removes; N=1 is the denominator, not a point on it"
assert S % (32 * N) == 0, f"{S} rows do not split {N} ways in whole tiles"
meshdesc.install(N)

import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import reblock_permute as rbp  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


OPENS = [0]


def _mesh_open(device_id, kwargs):
    OPENS[0] += 1
    with tt._device_init_lock():
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, N), **kwargs)
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
dev = tt.get_device()
assert OPENS[0] == 1, "the mesh was opened more than once"
log(f"device={dev} arch={dev.arch()} grid={tt.CORE_GRID_MAIN} N={N} S={S} reps={REPS}")

RES = {"mesh_n": N, "S": S, "reps": REPS, "warm": WARM, "arch": str(dev.arch()),
       "visible": os.environ.get("TT_VISIBLE_DEVICES"), "loadavg_start": os.getloadavg(),
       "arms": {}, "gather": {}, "parity": {}, "counters": {}, "reblock": {}}

kernel_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)

torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
layer = tt.PairformerLayer(32, 4, 24, 16, True,
                           {k: v.float() for k, v in rl.state_dict().items()}, KC)
REPL = ttnn.replicate_tensor_to_mesh_mapper(dev)
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0)
log(f"layer built, {len(rl.state_dict())} weight tensors")


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


m1 = torch.ones(1, S)
z_host = torch.randn(1, S, S, C_Z, dtype=torch.float32)
s_host = torch.randn(1, S, C_S, dtype=torch.float32)
PAIR_MASK = up(m1[:, :, None] * m1[:, None, :])
ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

ARMS = (("whole", False, False), ("shard", True, False), ("bshard", True, True))


def run_block(rs, bs, keep=False):
    """One block. s and z are rebuilt and uploaded OUTSIDE the caller's timed region."""
    s_t, z_t = up(s_host), up(z_host)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    s_o, z_o = layer(s_t, z_t, PAIR_MASK, ATTN, ATTN, row_shard=rs, b_shard=bs)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    out = None
    if keep:
        out = (ttnn.to_torch(s_o, mesh_composer=COMP).float(),
               ttnn.to_torch(z_o, mesh_composer=COMP).float())
    ttnn.deallocate(z_t)
    ttnn.deallocate(s_t)
    return dt, out


# --- parity first, so a wrong arm is never timed -------------------------------------------------
# Every arm against the `whole` arm's bytes, on EVERY device of the mesh. The `whole` arm is also
# its own control: every chip computes the identical block, so its N copies must already agree.
ref_out = None
for arm, rs, bs in ARMS:
    tt_counts_before = dict(tt.ROW_SHARD_CALLS)
    _, out = run_block(rs, bs, keep=True)
    got = {"chain": tt.ROW_SHARD_CALLS["chain"] - tt_counts_before["chain"],
           "b_shard": tt.ROW_SHARD_CALLS["b_shard"] - tt_counts_before["b_shard"]}
    want = {"chain": 1 if rs else 0, "b_shard": 1 if bs else 0}
    assert got == want, f"{arm}: engagement counter {got}, expected {want}"
    if ref_out is None:
        ref_out = out
    dev_s, dev_z = out[0].shape[0] // N, out[1].shape[0] // N
    md = {}
    for name, g, r, per in (("s", out[0], ref_out[0], dev_s), ("z", out[1], ref_out[1], dev_z)):
        md[name] = max((g[i * per:(i + 1) * per] - r[:per]).abs().max().item() for i in range(N))
    RES["parity"][arm] = {"max_abs_diff": md, "counters": got,
                          "devices_checked": N, "shapes": [list(out[0].shape), list(out[1].shape)]}
    log(f"parity {arm:6s} max|diff| s={md['s']:.6g} z={md['z']:.6g} over {N} devices  {got}")
    assert md["s"] == 0.0 and md["z"] == 0.0, f"{arm} is not bit-exact against the whole arm"

# the control that must break what the check reads: perturb z on the last device's rows
z_pert = z_host.clone()
z_pert[:, -32:, :, 0] += 5.0
_z_saved, z_host = z_host, z_pert
_, out = run_block(True, True, keep=True)
z_host = _z_saved
per = out[1].shape[0] // N
ctl = max((out[1][i * per:(i + 1) * per] - ref_out[1][:per]).abs().max().item() for i in range(N))
RES["parity"]["perturbed_control"] = {"max_abs_diff_z": ctl, "rejected": ctl > 0.0}
log(f"perturbed control: max|diff| z={ctl:.6g}  rejected={ctl > 0.0}")
assert ctl > 0.0, "the perturbed control passed: the parity check does not read the values"
del ref_out, out

# --- the three arms, round-robin ----------------------------------------------------------------
for _ in range(WARM):
    for arm, rs, bs in ARMS:
        run_block(rs, bs)
rbp.REJECTS.clear()
samples = {a: [] for a, _, _ in ARMS}
loads = []
for rep in range(REPS):
    order = ARMS if rep % 2 == 0 else tuple(reversed(ARMS))
    for arm, rs, bs in order:
        dt, _ = run_block(rs, bs)
        samples[arm].append(dt * 1e3)
    loads.append(os.getloadavg()[0])
RES["loadavg_per_rep"] = loads
for arm, _, _ in ARMS:
    v = sorted(samples[arm])
    RES["arms"][arm] = {"median_ms": statistics.median(v), "min_ms": v[0], "max_ms": v[-1],
                        "spread_pct": (v[-1] - v[0]) / statistics.median(v) * 100,
                        "samples_ms": samples[arm]}
    log(f"{arm:6s} block {statistics.median(v):8.3f} ms  (min {v[0]:.3f} max {v[-1]:.3f}, "
        f"spread {(v[-1]-v[0])/statistics.median(v)*100:.1f} %)")
RES["reblock"]["rejects_during_timing"] = {f"{k[0]}|{list(k[1])}": v for k, v in rbp.REJECTS.items()}
RES["counters"] = dict(tt.ROW_SHARD_CALLS)
RES["l1_out_rung"] = {str(k): v for k, v in tt._L1_OUT_RUNG.items()}
log(f"reblock rejects during timing: {RES['reblock']['rejects_during_timing']}")
log(f"_L1_OUT_RUNG: {RES['l1_out_rung']}")
for t_ in (PAIR_MASK, ATTN):
    ttnn.deallocate(t_)

# --- the two collectives, value-checked, on this mesh in this process -----------------------------
def gather(label, full_shape, dim):
    slab = list(full_shape)
    assert slab[dim] % N == 0
    ramp = torch.arange(full_shape[dim], dtype=torch.float32)
    view = [1] * len(full_shape)
    view[dim] = full_shape[dim]
    host = ramp.view(view).expand(full_shape).contiguous()
    t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(dev, dim=dim))
    g = ttnn.all_gather(t, dim=dim)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(g, mesh_composer=COMP)
    per = got.shape[0] // N
    want = host.to(torch.bfloat16).float()
    exact = all(torch.equal(got[i * per:(i + 1) * per].float(), want) for i in range(N))
    ttnn.deallocate(g)
    v = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        g = ttnn.all_gather(t, dim=dim)
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    v.sort()
    med = statistics.median(v)
    out_b = 2
    for d in full_shape:
        out_b *= d
    RES["gather"][label] = {"shape": list(full_shape), "dim": dim, "exact": bool(exact),
                            "out_bytes": out_b, "median_us": med * 1e6, "min_us": v[0] * 1e6,
                            "max_us": v[-1] * 1e6,
                            "gbps_per_dir": out_b * (N - 1) / N / med / 1e9}
    log(f"gather {label:8s} {str(list(full_shape)):22s} dim={dim} exact={exact}  "
        f"{med*1e6:9.1f} us  {out_b*(N-1)/N/med/1e9:6.2f} GB/s/dir")
    ttnn.deallocate(t)
    assert exact, f"{label} gather did not reproduce the ramp; its time is not a gather's time"


gather("pair", (1, S, S, C_Z), 1)     # x4, what the row shard takes
gather("b", (1, C_Z, S, S), 3)        # x2, what b_shard adds

RES["loadavg_end"] = os.getloadavg()
tt.cleanup()
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
