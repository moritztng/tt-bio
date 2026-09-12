"""Is the i-axis shard BIT-EXACT against the unsharded computation?

Claimed in FINDINGS as an inference: the trimul contraction sums over k, the shard splits i, so no
reduction order changes and bit-exactness should hold. That is an argument, not a measurement, and
it is the claim a reader will lean on hardest. So measure it.

For each op: compute it REPLICATED (each chip does the whole thing) and SHARDED+GATHERED (each chip
does half the i rows, then all_gather restores the full tensor), pull both back, and demand
torch.equal -- not allclose, not PCC.

The negative control matters as much as the check (memory: a control must break what the check
reads). Here it is a deliberately perturbed gather: one element of the far half flipped. If the
comparison cannot reject that, it is not testing what it claims to test.
"""

import json
import os
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("PARITY_OUT", "/tmp/b2z2_parity.json")
CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
repl = ttnn.replicate_tensor_to_mesh_mapper(mesh)
res = {}

def host(t, shard_dim=None):
    """Bring a mesh tensor back. Replicated -> take chip 0's copy; sharded -> concat the shards."""
    c = ttnn.to_torch(t, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, 0))
    return c[0:1] if shard_dim is None else c

try:
    torch.manual_seed(0)
    # 1. row-local: layer_norm on the pair track, i-axis = dim 1
    z = torch.empty(1, 512, 512, 128, dtype=torch.bfloat16).uniform_(-1, 1)
    w = torch.empty(128, dtype=torch.bfloat16).uniform_(0.5, 1.5)
    z_rep = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    z_shd = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, 1))
    w_rep = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)

    ref = host(ttnn.layer_norm(z_rep, weight=w_rep, epsilon=1e-5, compute_kernel_config=KC))
    got = host(ttnn.all_gather(ttnn.layer_norm(z_shd, weight=w_rep, epsilon=1e-5,
                                               compute_kernel_config=KC), dim=1), shard_dim=1)[0:1]
    ok = torch.equal(ref.float(), got.float())
    bad = got.clone(); bad[0, 400, 7, 3] = bad[0, 400, 7, 3] + 1.0
    ctrl = not torch.equal(ref.float(), bad.float())
    res["layer_norm_pair"] = {"bit_exact": bool(ok), "negative_control_rejects": bool(ctrl),
                              "max_abs_diff": float((ref.float() - got.float()).abs().max())}
    log(f"layer_norm  bit_exact={ok}  negctrl_rejects={ctrl}  maxdiff={res['layer_norm_pair']['max_abs_diff']}")

    # 2. the trimul contraction, i-axis = dim 2. This is the one the argument was about.
    a = torch.empty(1, 32, 512, 512, dtype=torch.bfloat16).uniform_(-1, 1)
    b = torch.empty(1, 32, 512, 512, dtype=torch.bfloat16).uniform_(-1, 1)
    a_rep = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    a_shd = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, 2))
    b_rep = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)

    ref2 = host(ttnn.matmul(a_rep, b_rep, compute_kernel_config=KC))
    got2 = host(ttnn.all_gather(ttnn.matmul(a_shd, b_rep, compute_kernel_config=KC), dim=2),
                shard_dim=2)[0:1]
    ok2 = torch.equal(ref2.float(), got2.float())
    bad2 = got2.clone(); bad2[0, 5, 400, 11] = bad2[0, 5, 400, 11] + 1.0
    ctrl2 = not torch.equal(ref2.float(), bad2.float())
    res["trimul_contraction"] = {"bit_exact": bool(ok2), "negative_control_rejects": bool(ctrl2),
                                 "max_abs_diff": float((ref2.float() - got2.float()).abs().max())}
    log(f"trimul mm   bit_exact={ok2}  negctrl_rejects={ctrl2}  maxdiff={res['trimul_contraction']['max_abs_diff']}")
finally:
    try:
        ttnn.close_mesh_device(mesh)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")
json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
