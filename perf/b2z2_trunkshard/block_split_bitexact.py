"""A whole Pairformer block, split two ways on the i axis, against the unsplit block, bit-exact.

`b2z2-dual-chip-fold` proved the five pair-track ops bit-exact one at a time. A block is not five
ops, it is a residual chain: `z = z + op(z)` five times, each op reading the z the previous one
wrote. A slab that is right on its own can still be wrong in the chain, because an op that writes
only its own rows leaves every other chip's copy of z stale and the next op reads that copy.

This runs the chain both ways on ONE device and requires `torch.equal`. Splitting on the host and
running the halves sequentially answers the numerics question completely -- does the model's
arithmetic survive an i-axis partition -- with no mesh, no fabric and no link. Whether it is fast
on two chips is a different question and a different row owns the hardware for it.

It also checks the op inventory, which is the part that decides how many gathers a block costs.
Ops marked ROWS in `tt_bio.row_shard.PAIR_CHAIN` are fed ONLY their own rows: if such an op really
does read a row it does not own, it computes different bytes and the same `torch.equal` catches it.
And the starve control does the converse -- it feeds a FULL op only its own rows and requires the
check to REJECT it, which is what proves the gather at that site is load-bearing rather than
assumed.

    python3 perf/b2z2_trunkshard/block_split_bitexact.py

Exit status is 0 only if the split block is bit-exact on both tracks AND every negative control
was rejected. Env: TRUNKSHARD_S (512), TRUNKSHARD_SHARDS (2), TRUNKSHARD_OUT.
"""

import pathlib
import sys

# Score the checkout this script lives in, not whatever `tt_bio` the environment installed.
_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import json
import os
import time

import torch

import ttnn
from tt_bio import reference as ref
from tt_bio import row_shard
from tt_bio import tenstorrent as tt

S = int(os.environ.get("TRUNKSHARD_S", "512"))
N_SHARDS = int(os.environ.get("TRUNKSHARD_SHARDS", "2"))
OUT = os.environ.get("TRUNKSHARD_OUT", "/tmp/b2z2_block_split.json")
C_Z, C_S = 128, 384


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def randomize_(module):
    """Give every parameter a nonzero value.

    Boltz-2's reference init zeroes both trimul output projections (`final_init_`), so a freshly
    built layer returns exactly zero and every comparison below would pass for reasons that have
    nothing to do with the split. Same fixture rule `slab_bitexact.py` records.
    """
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


dev = tt.get_device()
BOUNDS = row_shard.row_shard_bounds(S, N_SHARDS)
log(f"device={dev} arch={dev.arch()} S={S} shards={BOUNDS} "
    f"balance-ceiling={row_shard.row_shard_ceiling(S, N_SHARDS):.4f}x")
log(f"tt_bio from {tt.__file__}")

kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)

# Boltz-2 trunk geometry: PairformerModule(8, 32, 4, 24, 16, True), v2.
torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
randomize_(rl)
weights = {k: v.float() for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)
log(f"layer built, {len(weights)} weight tensors")

z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
s_t = torch.randn(1, S, C_S, dtype=torch.float32)
# A ragged mask, not all ones: the trimul multiplies it into `a` only, so it has to follow `a`
# onto the slab axis. An all-ones mask cannot tell a correct slice from a missing one.
m1 = torch.zeros(1, S)
m1[:, :S - 32] = 1.0
pair_mask_t = m1[:, :, None] * m1[:, None, :]
attn_t = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def run(sharded, z_host=None, bounds=None, chain=None):
    """One block, sharded or not, on a freshly uploaded (s, z). Returns host (s, z)."""
    z, s = up(z_t if z_host is None else z_host), up(s_t)
    mask, am = up(pair_mask_t), up(attn_t)
    if sharded:
        s_o, z_o = row_shard.pairformer_block_sharded(
            layer, s, z, bounds or BOUNDS, mask=mask, attn_mask_start=am, attn_mask_end=am,
            chain=chain or row_shard.PAIR_CHAIN)
    else:
        s_o, z_o = layer(s, z, mask, am, am)
    out = (ttnn.to_torch(s_o), ttnn.to_torch(z_o))
    # The residual chain updates in place, so z_o IS z and s_o IS s -- free once, not twice.
    ttnn.deallocate(z_o)
    ttnn.deallocate(s_o)
    ttnn.deallocate(mask)
    ttnn.deallocate(am)
    return out


def cmp(a, b):
    d = (a.float() - b.float()).abs().max().item()
    return bool(a.shape == b.shape and torch.equal(a, b)), d


log("reference: the unsplit block")
s_ref, z_ref = run(False)
log(f"  s{list(s_ref.shape)} z{list(z_ref.shape)}")

log(f"split into {len(BOUNDS)} row slabs")
s_sp, z_sp = run(True)
z_ok, z_d = cmp(z_sp, z_ref)
s_ok, s_d = cmp(s_sp, s_ref)
log(f"  pair track z: equal={z_ok} max abs diff {z_d}")
log(f"  single track s: equal={s_ok} max abs diff {s_d}")

failures = []
if not z_ok:
    failures.append(f"pair track differs, max abs diff {z_d}")
if not s_ok:
    failures.append(f"single track differs, max abs diff {s_d}")

# --- negative control 1: a value control that breaks exactly the bytes torch.equal reads --------
# Perturb one channel row of z inside the LAST slab's own row range, rerun the split block, and
# require the same check to reject it. If this passes, the check is reading something else.
log("negative control 1: perturbed input inside the last slab")
z_bad = z_t.clone()
z_bad[0, BOUNDS[-1][0], 0, :] += 5.0
s_b, z_b = run(True, z_host=z_bad)
nc1_ok, nc1_d = cmp(z_b, z_ref)
log(f"  rejected={not nc1_ok} (max abs diff {nc1_d})")
if nc1_ok:
    failures.append("negative control 1 was NOT rejected: the check cannot fail")

# --- negative control 2: a starve control that tests the INVENTORY, not the values --------------
# Every op the inventory calls FULL forces one gather, and a gather is 67.1 MB of link traffic. So
# each FULL claim gets demoted to ROWS in turn -- the op is handed only its own rows, exactly as if
# it were row-local -- and the run MUST fail. If starving an op changes nothing, that op never
# needed its gather, the count is too high and the shard is being over-priced. This is the part of
# the inventory a measurement can settle; the rest is a code reading.
log("negative control 2: starve each FULL op of the rows it does not own, one at a time")
nc2 = {}
for idx, (op_name, need) in enumerate(row_shard.PAIR_CHAIN):
    if need != "FULL":
        continue
    starved = (row_shard.PAIR_CHAIN[:idx] + ((op_name, "ROWS"),)
               + row_shard.PAIR_CHAIN[idx + 1:])
    try:
        _, z_st = run(True, chain=starved)
    except Exception as e:                                              # noqa: BLE001
        nc2[op_name] = {"rejected": True, "how": "raised", "detail": type(e).__name__}
        log(f"  {op_name}: rejected, the starved op does not even produce a valid chain "
            f"({type(e).__name__})")
        continue
    if tuple(z_st.shape) != tuple(z_ref.shape):
        nc2[op_name] = {"rejected": True, "how": "shape", "detail": list(z_st.shape)}
        log(f"  {op_name}: rejected on shape {list(z_st.shape)} vs {list(z_ref.shape)}")
        continue
    ok, d = cmp(z_st, z_ref)
    nc2[op_name] = {"rejected": not ok, "how": "values", "detail": d}
    log(f"  {op_name}: rejected={not ok} (max abs diff {d})")
    if ok:
        failures.append(f"negative control 2 was NOT rejected for {op_name}: it does not need a "
                        "gather and the inventory over-counts them")

# --- localisation: when the block is not bit-exact, name the op -----------------------------------
# The split regroups no reduction, so a divergence is never the arithmetic: it is a kernel PICK.
# Several projections here are routed by shape, and a slab changes M. `tenstorrent.py:6820` says so
# in as many words and names this script's job -- "a shape where a pick DID flip would take a
# different kernel and would not be bit-exact, and the parity script is what says which of the two
# a shape is". So when the chain differs, run each op on its own, whole against slabs, and report
# the ones that moved. An op that is bit-exact alone but not in the chain would be a different and
# much worse finding than one that is wrong on its own.
per_op = {}
if not (z_ok and s_ok):
    log("localising: each op on its own, whole against slabs")
    ops = {
        "triangle_multiplication_start": lambda z, sl: layer.triangle_multiplication_start(
            z, MASK, row_slab=sl),
        "triangle_multiplication_end": lambda z, sl: layer.triangle_multiplication_end(
            z, MASK, row_slab=sl),
        "triangle_attention_start": lambda z, sl: layer.triangle_attention_start(
            z, AM, row_slab=sl),
        "triangle_attention_end": lambda z, sl: layer.triangle_attention_end(z, AM, row_slab=sl),
        "transition_z": lambda z, sl: layer.transition_z(z, row_slab=sl),
    }
    MASK, AM = up(pair_mask_t), up(attn_t)

    def routes():
        """Which kernel each shape-routed site served, so a divergence names the leg that moved."""
        out = {}
        for mod, attr in (("_triatt_qkv", "STATS"), ("_triatt_sdpa", "STATS"),
                          ("_reblock", "STATS_GATED")):
            m = getattr(tt, mod, None)
            if m is not None and hasattr(m, attr):
                v = getattr(m, attr)
                # served/declined counters are plain lists in some modules, dicts in others
                out[mod] = dict(v) if isinstance(v, dict) else list(v)
        out["sdpa_picks"] = dict(tt.SDPA_CHUNK_PICKS)
        out["sdpa_routes"] = dict(tt.SDPA_ROUTE_COUNTS)
        return out

    def _delta(a, b):
        """Only the counters that moved between two snapshots."""
        out = {}
        for k in set(a) | set(b):
            x, y = a.get(k), b.get(k)
            if x == y:
                continue
            if isinstance(y, dict):
                out[k] = {kk: y.get(kk, 0) - (x or {}).get(kk, 0)
                          for kk in set(y) | set(x or {}) if y.get(kk, 0) != (x or {}).get(kk, 0)}
            else:
                out[k] = [b - a_ for a_, b in zip(x or [0] * len(y), y)]
        return out

    for name, fn in ops.items():
        zt = up(z_t)
        r_before = routes()
        whole = ttnn.to_torch(fn(zt, None))
        r_whole = routes()
        parts = []
        for r0, r1 in BOUNDS:
            o = fn(zt, (r0, r1))
            parts.append(ttnn.to_torch(o))
            ttnn.deallocate(o)
        r_slab = routes()
        ttnn.deallocate(zt)
        joined = torch.cat(parts, dim=1)
        ok, d = cmp(joined, whole)
        per_op[name] = {"bit_exact": ok, "max_abs_diff": d,
                        "routes_whole": _delta(r_before, r_whole),
                        "routes_slab": _delta(r_whole, r_slab)}
        log(f"  {name}: equal={ok} max abs diff {d}")
        if not ok:
            # Rounding regrouped by a different block config, or a wrong answer? A regroup moves
            # a last bit or two on a few elements; a bug moves a large fraction of them by a large
            # relative amount. The numbers below are what tells those apart, and the difference
            # decides whether the shape is merely unsupported or actively broken.
            a, b = joined.float(), whole.float()
            diff = (a - b).abs()
            n = int((diff > 0).sum())
            rel = (diff / b.abs().clamp_min(1e-6)).max().item()
            per_op[name].update({"n_differing": n, "n_elements": int(diff.numel()),
                                 "frac_differing": n / diff.numel(),
                                 "mean_abs_diff": diff.mean().item(),
                                 "max_rel_diff": rel,
                                 "ref_abs_max": b.abs().max().item()})
            log(f"    {n}/{diff.numel()} elements differ ({100 * n / diff.numel():.2f} %), "
                f"mean abs {diff.mean().item():.3e}, max rel {rel:.3e}, "
                f"|ref| max {b.abs().max().item():.3f}")
            log(f"    whole took {per_op[name]['routes_whole']}")
            log(f"    slabs took {per_op[name]['routes_slab']}")
    ttnn.deallocate(MASK)
    ttnn.deallocate(AM)

# --- negative control 3: the slabs, concatenated in the wrong order ------------------------------
log("negative control 3: slabs concatenated in the wrong order")
if len(BOUNDS) > 1 and len({r1 - r0 for r0, r1 in BOUNDS}) == 1:
    k = BOUNDS[0][1]
    swapped = torch.cat([z_sp[:, k:], z_sp[:, :k]], dim=1)
    nc3_ok, nc3_d = cmp(swapped, z_ref)
    log(f"  rejected={not nc3_ok} (max abs diff {nc3_d})")
    if nc3_ok:
        failures.append("negative control 3 was NOT rejected")
else:
    nc3_ok, nc3_d = None, None
    log("  skipped: slabs are not equal-sized, a swap is not a shape-valid perturbation")

res = {
    "arch": "WORMHOLE_B0" if dev.arch() == ttnn.Arch.WORMHOLE_B0 else str(dev.arch()),
    "S": S, "shards": [list(b) for b in BOUNDS],
    "balance_ceiling": row_shard.row_shard_ceiling(S, N_SHARDS),
    "gathers_per_block": row_shard.gathers_per_block(),
    "gathers_per_block_with_injectable_bias": row_shard.gathers_per_block(injectable_bias=True),
    "chain": [list(c) for c in row_shard.PAIR_CHAIN],
    "per_op": per_op,
    "z_bit_exact": z_ok, "z_max_abs_diff": z_d,
    "s_bit_exact": s_ok, "s_max_abs_diff": s_d,
    "nc1_rejected": not nc1_ok, "nc1_max_abs_diff": nc1_d,
    "nc2_starve": nc2,
    "nc3_rejected": None if nc3_ok is None else not nc3_ok, "nc3_max_abs_diff": nc3_d,
    "pass": not failures,
}
pathlib.Path(OUT).write_text(json.dumps(res, indent=2))
log(f"wrote {OUT}")

if failures:
    print("FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASS: a {len(BOUNDS)}-way i-axis split of the whole Pairformer block is bit-exact at "
      f"S={S} on both tracks, and all negative controls were rejected")
sys.exit(0)
