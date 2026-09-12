"""The row slab, addressed by `ttnn.mesh_partition` instead of by row number, bit-exact.

`b2z2-dual-chip-fold` proved the slab itself: each pair-track op computes its own output rows and
those rows are bit-identical to those rows of the whole-tensor result. It addressed the slab as
`row_slab=(r0, r1)`, a pair of Python ints, and that is the one thing an SPMD mesh cannot run --
every device executes the same program, so `r0` cannot be a different constant on each of them.
This proves the same ops under the addressing that CAN: `RowSlab.mesh(n)`, which reaches its slab
through `ttnn.mesh_partition` and needs no per-device constant at all.

Two arms, because they answer different questions.

* **1x1** is the arm the directive requires before any of this touches the pair. A one-device
  partition is the identity, so this arm proves the rewritten addressing is neutral -- every op
  still returns exactly what it returned before -- and that every slab site accepts a partitioned
  tensor. It CANNOT distinguish `mesh_partition` being applied from it being ignored, which is why
  the call is counted rather than assumed, and why the 1x2 arm exists.
* **1x2** is the real one. The partition genuinely halves, each device holds a different slab, and
  the two slabs concatenated on the host must equal the unsharded result to the bit. No fabric and
  no all_gather: the gather is done by the host composer, so this runs on a Wormhole pair as well
  as a Blackhole one, and it isolates the addressing from the link.

    MESH_N=1 python3 perf/b2z2_trunkshard/mesh_slab_bitexact.py
    MESH_N=2 TT_VISIBLE_DEVICES=4,5 python3 perf/b2z2_trunkshard/mesh_slab_bitexact.py

Exit status is 0 only if every op is bit-exact under mesh addressing AND every control was
rejected. Env: MESH_N (1), MESH_S (512), MESH_OUT.
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import json
import os
import time

import torch

N = int(os.environ.get("MESH_N", "1"))
S = int(os.environ.get("MESH_S", "512"))
OUT = os.environ.get("MESH_OUT", f"/tmp/b2z2_mesh_slab_{N}.json")
C_Z, C_S = 128, 384

from tt_bio.device_lease import CardSetLease  # noqa: E402

# A multi-device open takes the lease here; the single-device path goes through tt.get_device(),
# which acquires the same flock itself and would collide with a pre-acquired one.
if N > 1:
    CardSetLease().acquire()

import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import row_shard  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio.row_shard import RowSlab  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Count the partitions. On a 1x1 mesh the op is the identity, so "it ran" and "it was skipped"
# produce the same bytes; the only way to tell them apart is to watch the call.
PARTITIONS = [0]
_mesh_partition = ttnn.mesh_partition


def _counted(*a, **k):
    PARTITIONS[0] += 1
    return _mesh_partition(*a, **k)


ttnn.mesh_partition = _counted

if N > 1:
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, N), l1_small_size=32768)
    tt._device = dev          # so Module.__init__ -> get_device() lands on the mesh
else:
    dev = tt.get_device()
log(f"mesh 1x{N} device={dev} S={S} arch={dev.arch()}")

kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)


def randomize_(module):
    """Boltz-2's reference init zeroes both trimul output projections, so a freshly built layer
    returns exactly zero and every comparison would hold for reasons unrelated to the slab."""
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
randomize_(rl)
layer = tt.PairformerLayer(32, 4, 24, 16, True, {k: v.float() for k, v in rl.state_dict().items()},
                           KC)
log(f"layer built, mesh has {N} device(s)")

z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
s_t = torch.randn(1, S, C_S, dtype=torch.float32)
# Ragged, not all ones: the trimul multiplies the mask into `a` only, so it has to follow `a` onto
# the slab axis. An all-ones mask cannot tell a correct partition from a missing one.
m1 = torch.zeros(1, S)
m1[:, :S - 32] = 1.0
pair_mask_t = m1[:, :, None] * m1[:, None, :]
attn_t = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9

REPL = ttnn.replicate_tensor_to_mesh_mapper(dev) if N > 1 else None
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0) if N > 1 else None


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


def down(t, sharded):
    """Bring a mesh tensor back. Replicated: take one device's copy, they are identical by
    construction. Sharded: every device holds a different row slab, so the composer stacks them on
    a new leading axis and they concatenate back onto the row axis here. This is the gather, done
    on the host, so no fabric is involved and the addressing is isolated from the link."""
    if N == 1:
        return ttnn.to_torch(t)
    c = ttnn.to_torch(t, mesh_composer=COMP)
    if not sharded:
        return c[0:1]
    per = c.shape[0] // N
    return torch.cat([c[i * per:(i + 1) * per] for i in range(N)], dim=1)


MASK, AM = up(pair_mask_t), up(attn_t)
OPS = {
    "trimul_start": lambda z, sl: layer.triangle_multiplication_start(z, MASK, row_slab=sl),
    "trimul_end": lambda z, sl: layer.triangle_multiplication_end(z, MASK, row_slab=sl),
    "triatt_start": lambda z, sl: layer.triangle_attention_start(z, AM, row_slab=sl),
    "triatt_end": lambda z, sl: layer.triangle_attention_end(z, AM, row_slab=sl),
    "transition_z": lambda z, sl: layer.transition_z(z, row_slab=sl),
}


def run_op(fn, slab, host=None, sharded=False):
    z = up(z_t if host is None else host)
    out = fn(z, slab)
    got = down(out, sharded)
    ttnn.deallocate(out)
    ttnn.deallocate(z)
    return got


def eq(a, b):
    return bool(a.shape == b.shape and torch.equal(a, b)), (a.float() - b.float()).abs().max().item()


failures, res = [], {}

# --- the partition itself: does it actually split, and by how much -------------------------------
probe = up(z_t)
part = ttnn.mesh_partition(probe, 1)
part_shape = [int(d) for d in part.shape]
log(f"mesh_partition([1,{S},{S},{C_Z}], dim=1) -> per-device {part_shape}")
if part_shape[1] != S // N:
    failures.append(f"mesh_partition gave {part_shape[1]} rows per device, expected {S // N}")
ttnn.deallocate(part)
ttnn.deallocate(probe)

# --- every op, mesh-addressed against the unsharded result ---------------------------------------
mesh_slab = RowSlab.mesh(N)
log(f"each op: RowSlab.mesh({N}) against row_slab=None")
for name, fn in OPS.items():
    whole = run_op(fn, None)
    got = run_op(fn, mesh_slab, sharded=N > 1)
    ok, d = eq(got, whole)
    res[name] = {"bit_exact": ok, "max_abs_diff": d}
    log(f"  {name}: equal={ok} max abs diff {d}")
    if not ok:
        failures.append(f"{name} is not bit-exact under mesh addressing, max abs diff {d}")

# --- the whole block, mesh-addressed -------------------------------------------------------------
# On one device the gather is a no-op, so the whole residual chain runs; on a mesh each FULL op
# would need an all_gather and that is the pair's question, not this one.
block = {}
if N == 1:
    log("whole block: RowSlab.mesh(1) against the unsharded block")
    z, s = up(z_t), up(s_t)
    # The residual chain updates in place, so the returned z IS the input z: download, then free
    # once. Freeing both is the "Buffer is not allocated" landmine real_block.py records.
    s_o, z_o = layer(s, z, MASK, AM, AM)
    s_ref, z_ref = ttnn.to_torch(s_o), ttnn.to_torch(z_o)
    ttnn.deallocate(z_o)
    ttnn.deallocate(s_o)
    z2, s2 = up(z_t), up(s_t)
    s_o, z_o = row_shard.pairformer_block_sharded(
        layer, s2, z2, (mesh_slab,), mask=MASK, attn_mask_start=AM, attn_mask_end=AM,
        gather=lambda parts: list(parts)[0])
    s_got, z_got = ttnn.to_torch(s_o), ttnn.to_torch(z_o)
    ttnn.deallocate(z_o)
    ttnn.deallocate(s_o)
    zok, zd = eq(z_got, z_ref)
    sok, sd = eq(s_got, s_ref)
    block = {"z_bit_exact": zok, "z_max_abs_diff": zd, "s_bit_exact": sok, "s_max_abs_diff": sd}
    log(f"  pair track z: equal={zok} max abs diff {zd}")
    log(f"  single track s: equal={sok} max abs diff {sd}")
    if not (zok and sok):
        failures.append(f"the mesh-addressed block is not bit-exact (z {zd}, s {sd})")

# --- control 1: the check must reject a perturbed input ------------------------------------------
log("control 1: perturb the input and require the same check to reject it")
z_bad = z_t.clone()
z_bad[0, S // 2, 0, :] += 5.0
whole = run_op(OPS["trimul_start"], None)
bad = run_op(OPS["trimul_start"], mesh_slab, host=z_bad, sharded=N > 1)
nc1_ok, nc1_d = eq(bad, whole)
log(f"  rejected={not nc1_ok} (max abs diff {nc1_d})")
if nc1_ok:
    failures.append("control 1 was not rejected: the check cannot fail")

# --- control 2: the partition has to be load-bearing ---------------------------------------------
# On a mesh, device 0's slab must be the FIRST rows and not the whole tensor or somebody else's.
# On 1x1 this is not testable at all -- a one-device partition IS the identity -- and saying so is
# the honest version of the 1x1 arm.
log("control 2: the partition is load-bearing")
if N > 1:
    got = run_op(OPS["transition_z"], mesh_slab, sharded=False)   # device 0's slab only
    whole = run_op(OPS["transition_z"], None)
    per = S // N
    first_ok, _ = eq(got[:, :per] if got.shape[1] >= per else got, whole[:, :per])
    wrong_ok, _ = eq(got[:, :per] if got.shape[1] >= per else got, whole[:, per:2 * per])
    log(f"  device 0 holds the first {per} rows: {first_ok}; holds the second slab: {wrong_ok}")
    if not first_ok:
        failures.append("device 0's slab is not the first rows of the result")
    if wrong_ok:
        failures.append("device 0's slab matches the SECOND slab, so the partition is not per-device")
    nc2 = {"device0_is_first_slab": first_ok, "device0_is_second_slab": wrong_ok}
else:
    nc2 = {"not_testable": "a 1x1 mesh_partition is the identity; the 1x2 arm is what tests this"}
    log("  not testable on 1x1: a one-device partition is the identity. See the 1x2 arm.")

log(f"ttnn.mesh_partition was called {PARTITIONS[0]} times")
if PARTITIONS[0] == 0:
    failures.append("ttnn.mesh_partition was never called: the mesh addressing is not being used")

out = {"arch": str(dev.arch()), "mesh": f"1x{N}", "S": S, "ops": res, "block": block,
       "partition_shape": part_shape, "partition_calls": PARTITIONS[0],
       "control_1_rejected": not nc1_ok, "control_2": nc2, "pass": not failures}
pathlib.Path(OUT).write_text(json.dumps(out, indent=2))
log(f"wrote {OUT}")

if failures:
    print("FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASS: every pair-track op is bit-exact addressed by ttnn.mesh_partition on a 1x{N} mesh, "
      f"and every control was rejected")
sys.exit(0)
