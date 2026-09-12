"""Row slabs of the three shardable Pairformer ops, against the whole-tensor result, bit-exact.

The dual-chip design gives each Blackhole of the p300c pair the WHOLE pair track and half of the
output ROWS. That is only a shard if a slab of rows is exactly the same bytes as those rows of the
unsharded op: the residual chain adds `z_update` into `z` in place, so a slab that is merely close
would put the two chips on diverging copies of z within one block and there is nothing downstream
to pull them back together.

It is achievable because the i axis is not a reduction axis. Both triangle products contract over
k, the triangle attentions reduce over their key axis, and the transition is elementwise, so
splitting i regroups no sum. This script is what turns that argument into a measurement, at the
production shape (pair [1, 512, 512, 128], 4 triangle-attention heads of width 32) and on real
Boltz-2 layer geometry rather than a toy fixture.

For each op: compute it whole, compute it as two 256-row slabs, concatenate the slabs, and require
`torch.equal`. Then the negative control -- perturb one channel row of z INSIDE the second slab's
own row range, recompute that slab, and require the SAME check to reject it. A check that cannot
fail has not checked anything.

Run (the device is whatever `tt_bio.tenstorrent.get_device()` hands back, one chip, no mesh):

    python3 perf/b2z2_dualchip/slab_bitexact.py

Exit status is 0 only if every op is bit-exact AND every negative control was rejected.
"""

import os
import sys
import time

import torch

import ttnn
from tt_bio import tenstorrent as tt
from tt_bio import reference as ref

S = int(os.environ.get("SLAB_S", "512"))
SPLIT = int(os.environ.get("SLAB_SPLIT", str(S // 2)))
C_Z = 128
SLABS = ((0, SPLIT), (SPLIT, S))


def randomize_(module):
    """Give every parameter a nonzero value.

    Boltz-2's reference init is `final_init_` on both output projections of the triangle
    multiplication and `gating_init_`/zero elsewhere, so a freshly constructed layer returns
    EXACTLY ZERO from trimul and transition. Every comparison in this file would then hold for
    reasons that have nothing to do with slabs -- the first run of the negative control below
    reported max abs diff 0.0 against the WRONG rows, which is what caught it. A parity fixture
    has to carry weights that make the op produce structure.
    """
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


dev = tt.get_device()
log(f"device={dev}  S={S}  slabs={SLABS}")

KC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

# Boltz-2 trunk geometry, the same construction perf/b2z2_dualchip/real_block.py times:
# PairformerModule(8, 32, 4, 24, 16, True) with v2=True. Real weight shapes matter here -- the
# fused input projection's role split is a column-order argument, and a fixture with a square
# g_in/p_in would not distinguish a correct split from a transposed one.
torch.manual_seed(0)
rl = ref.PairformerLayer(384, C_Z, 16, 0.25, 32, 4, v2=True)
randomize_(rl)
weights = {k: v.float() for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)
log(f"layer built, {len(weights)} weight tensors")

z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
# A RAGGED mask, not all ones: the trimul multiplies it into `a` only, so it has to follow `a`
# onto the slab axis -- rows for the starting variant and COLUMNS for the ending one. An all-ones
# mask cannot tell a correct slice from a missing one, or from the wrong axis.
m1 = torch.zeros(1, S)
m1[:, :S - 32] = 1.0
pair_mask_t = m1[:, :, None] * m1[:, None, :]
attn_t = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


pair_mask = up(pair_mask_t)
attn_mask = up(attn_t)


def _call(op, z_host, slab):
    """One op call on a freshly uploaded z, as a torch tensor. Frees everything it allocated."""
    z = up(z_host)
    out = op(z, slab)
    host = ttnn.to_torch(out)
    ttnn.deallocate(out)
    ttnn.deallocate(z)
    return host


# Each entry takes (z, row_slab) and returns the op's output. `row_slab=None` is the unsharded
# call, which must reach exactly the ops it reaches today -- that is half of what is being checked.
OPS = {
    "trimul_start": lambda z, sl: layer.triangle_multiplication_start(z, pair_mask, row_slab=sl),
    "trimul_end": lambda z, sl: layer.triangle_multiplication_end(z, pair_mask, row_slab=sl),
    "triatt_start": lambda z, sl: layer.triangle_attention_start(z, attn_mask, row_slab=sl),
    "triatt_end": lambda z, sl: layer.triangle_attention_end(z, attn_mask, row_slab=sl),
    "transition_z": lambda z, sl: layer.transition_z(z, row_slab=sl),
}


def report_mismatch(name, want, got):
    d = (want.float() - got.float()).abs()
    bad = int((want != got).sum())
    print(f"    {name}: {bad} of {want.numel()} elements differ, max abs {float(d.max()):.6g}, "
          f"first at {torch.nonzero(want != got)[0].tolist()}", flush=True)


def routes():
    """Which kernel each shape-routed site served, so a failure names the leg that moved."""
    out = []
    for mod, attr, label in (
        ("_triatt_qkv", "STATS", "qkv_heads served/declined"),
        ("_triatt_sdpa", "STATS", "fused sdpa served/declined"),
        ("_reblock", "STATS_GATED", "gated move served/declined"),
    ):
        m = getattr(tt, mod, None)
        if m is not None and hasattr(m, attr):
            out.append(f"{label}={getattr(m, attr)}")
    out.append(f"sdpa_picks={dict(tt.SDPA_CHUNK_PICKS)}")
    out.append(f"sdpa_routes={dict(tt.SDPA_ROUTE_COUNTS)}")
    return "  ".join(out)


failures = []
for name, op in OPS.items():
    t0 = time.perf_counter()
    whole = _call(op, z_t, None)
    t_whole = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = [_call(op, z_t, sl) for sl in SLABS]
    t_slab = time.perf_counter() - t0
    joined = torch.cat(parts, dim=1)

    ok = joined.shape == whole.shape and torch.equal(joined, whole)
    log(f"{name:13s} whole {tuple(whole.shape)} {t_whole*1e3:7.1f} ms | slabs "
        f"{[tuple(p.shape) for p in parts]} {t_slab*1e3:7.1f} ms | "
        f"torch.equal {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{name}: slab concat is not bit-exact against the whole-tensor result")
        if joined.shape == whole.shape:
            report_mismatch(name, whole, joined)
        print(f"    routes: {routes()}", flush=True)

    # Negative control. One channel row of z, inside the SECOND slab's own row range, moved far
    # enough that no rounding can absorb it. Both variants reach it: the starting trimul reads
    # row r0 into `a`, the ending one reads it into `b` (which every output row contracts
    # against), the attentions carry it through q or the shared bias, and the transition is
    # elementwise on it. If the rebuilt concat still compares equal, the comparison above is not
    # reading what it claims to read.
    z_bad = z_t.clone()
    z_bad[0, SLABS[1][0], 0, :] += 5.0
    bad = torch.cat([parts[0], _call(op, z_bad, SLABS[1])], dim=1)
    rejected = not torch.equal(bad, whole)
    log(f"{name:13s} negative control: perturbed z[0,{SLABS[1][0]},0,:] -> "
        f"{'REJECTED (good)' if rejected else 'ACCEPTED (the check is blind)'}")
    if not rejected:
        failures.append(f"{name}: the negative control passed, so the check cannot fail")

for t in (pair_mask, attn_mask):
    ttnn.deallocate(t)

print()
if failures:
    log(f"FAIL ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
log(f"PASS: {len(OPS)} ops bit-exact as two row slabs at S={S}, every negative control rejected")
sys.exit(0)
