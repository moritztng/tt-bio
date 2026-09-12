"""Is a ROW-SHARDED Pairformer block bit-identical to the replicated one, on a real 1x2 mesh?

`slab_bitexact.py` proves the algebra of one op on one device with a Python row range. It cannot
prove the thing the mesh needs, for two reasons:

  * a ttnn mesh is SPMD, so `row_slab=(0, 256)` makes BOTH chips compute rows 0-256. The shard has
    to arrive as a TENSOR, from `ttnn.mesh_partition`, whose result depends on the device's
    position in the mesh. That is what `row_input` is.
  * it checks ops, not the BLOCK. The residual chain feeds each op the previous op's output, and
    which ops need a full, FRESH `z` rather than their own rows is the whole design. Get that
    wrong and every op is still individually bit-exact while the block is silently wrong.

So: build one PairformerLayer on a 1x2 mesh with replicated weights, run the block replicated and
row-sharded on the same inputs, and demand `torch.equal` on both outputs, on BOTH chips.

The negative control is the gather set itself. `TT_BIO_MESH_PARITY_DROP=<n>` drops the n-th gather
(0-based, in the order the block issues them), which leaves one op reading a `z` whose far rows are
one op stale. If the check still passes with a gather dropped, it is not reading the shard.

Run:
    TT_VISIBLE_DEVICES=2,3 TT_BIO_LEASE_CARDS=2,3 perf/b2z2_dualchip/mesh_block_parity.py
"""

import os
import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import torch

import ttnn
from tt_bio import reference as ref
from tt_bio import tenstorrent as tt

S, C_Z, C_S = int(os.environ.get("MESH_S", "512")), 128, 384
DROP = os.environ.get("TT_BIO_MESH_PARITY_DROP")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def randomize_(module):
    """Nonzero everywhere: Boltz-2's init zeroes both trimul output projections, and a block that
    returns z unchanged compares equal for reasons that have nothing to do with the shard."""
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
failures = []
try:
    tt._configure_active_compute_grid(mesh)
    mesh.enable_program_cache()
    # Every Module builds its weights through `get_device()`, so point it at the mesh. A plain
    # `from_torch(device=mesh)` with no mapper replicates, which is what the fold harness relies on.
    tt._DEVICE = mesh
    log(f"mesh={mesh}  S={S}  drop_gather={DROP}")

    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
    randomize_(rl)
    layer = tt.PairformerLayer(32, 4, 24, 16, True,
                               {k: v.float() for k, v in rl.state_dict().items()}, KC)
    log(f"layer built, {len(rl.state_dict())} weight tensors")

    if DROP is not None:
        # Negative control: silence one gather. The sharded block then runs an op on a z whose
        # far rows are one residual step behind, which is exactly the failure a wrong gather set
        # produces, and the check has to reject it.
        n = int(DROP)
        orig = tt.PairformerLayer._pair_track_row_sharded
        state = {"i": 0}

        def patched(self, z, mask, ams, ame):
            real = ttnn.all_gather

            def counting(t, dim, *a, **kw):
                i, state["i"] = state["i"], state["i"] + 1
                if i == n:
                    return ttnn.clone(t)      # same shape as the shard, NOT the gathered tensor
                return real(t, dim, *a, **kw)
            ttnn.all_gather = counting
            try:
                return orig(self, z, mask, ams, ame)
            finally:
                ttnn.all_gather = real
        tt.PairformerLayer._pair_track_row_sharded = patched

    m1 = torch.zeros(1, S)
    m1[:, :S - 32] = 1.0          # ragged, so a dropped mask slice cannot pass
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
    s_t = torch.randn(1, S, C_S, dtype=torch.float32)
    pair_mask = up(m1[:, :, None] * m1[:, None, :])
    attn = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

    cat = ttnn.concat_mesh_to_tensor_composer(mesh, 0)

    def run(row_shard):
        s, z = up(s_t), up(z_t)
        s_o, z_o = layer(s, z, pair_mask, attn, attn, row_shard=row_shard)
        hs = ttnn.to_torch(s_o, mesh_composer=cat)
        hz = ttnn.to_torch(z_o, mesh_composer=cat)
        ttnn.deallocate(s_o)
        ttnn.deallocate(z_o)
        return hs, hz

    t0 = time.perf_counter()
    s_rep, z_rep = run(False)
    t_rep = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_shd, z_shd = run(True)
    t_shd = time.perf_counter() - t0
    log(f"replicated {t_rep * 1e3:7.1f} ms | row-sharded {t_shd * 1e3:7.1f} ms  (cold, not a "
        f"perf number)")

    # `cat` stacks chip 0's copy then chip 1's along dim 0. Both must equal the replicated result:
    # a shard that forgot a gather is wrong on the chip that does not own the rows it read.
    for name, rep, shd in (("z", z_rep, z_shd), ("s", s_rep, s_shd)):
        per_chip = [torch.equal(shd[i:i + 1], rep[i:i + 1]) for i in range(rep.shape[0])]
        ok = all(per_chip)
        d = (rep.float() - shd.float()).abs()
        log(f"{name}: row-sharded == replicated on each chip {per_chip}  "
            f"max abs diff {float(d.max()):.6g}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            n_bad = int((rep != shd).sum())
            log(f"    {n_bad} of {rep.numel()} elements differ, first at "
                f"{torch.nonzero(rep != shd)[0].tolist()}")
            failures.append(f"{name}: the row-sharded block is not bit-exact")
        # The replicated run is itself a control on the mesh: both chips must agree.
        if not torch.equal(rep[0:1], rep[1:2]):
            failures.append(f"{name}: the REPLICATED run differs between chips, so the mesh "
                            f"itself is not lockstep and nothing above means anything")

    for t in (pair_mask, attn):
        ttnn.deallocate(t)
finally:
    tt._DEVICE = None
    ttnn.close_mesh_device(mesh)

print()
if DROP is not None:
    ok = bool(failures)
    log(f"NEGATIVE CONTROL (gather {DROP} dropped): "
        f"{'REJECTED (good)' if ok else 'ACCEPTED -- the check is blind'}")
    sys.exit(0 if ok else 1)
if failures:
    log(f"FAIL ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
log(f"PASS: a row-sharded Pairformer block at S={S} is bit-identical to the replicated one on "
    f"both chips of the 1x2 mesh")
sys.exit(0)
