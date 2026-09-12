"""Do `b_shard`'s term and the reblock gate's term touch the same readers? Yes, and a short circuit hides it.

The orchestrator's amendment asks this because two rows are re-fitting one `compute(N)` curve on
different terms, and a term counted twice is how this campaign already produced a 7.10 % double
count. The answer is in `TriangleMultiplication`'s gate:

    gated = ... and _reblock.eligible_gated(gp_in_fused, slice_c, memory_config)
                and (not slab or _reblock.eligible_gated(gp_b_fused, slice_c, memory_config))

`gp_b_fused` is built ONLY under a slab, and under `b_shard` its source is the slab `a` took rather
than the whole pair tensor. So `b_shard` is what decides the SHAPE the second check reads, and the
second check is the reblock gate `b2z2-reblock-axis2-gate` is fixing. The terms are not disjoint.

Today that is invisible, because Python's `and` short-circuits: at N>=4 the FIRST check already
fails on the ending trimul, so the second is never evaluated and both sharded arms decline exactly
once per block. That is why the defect cancels in the difference of the two constants this row
published. **It stops cancelling the moment the first check is fixed.**

This file measures both halves rather than arguing them. `eligible_gated` is wrapped so every call
is recorded with its shape and verdict and the verdict is passed through unchanged, so the
short-circuit is visible as an ABSENCE of calls. Then the `b` role's own slab shape is put through
the same gate directly, which is what the first check is hiding.

    MESH_N=4 TT_VISIBLE_DEVICES=24,25,26,27 GATE_OUT=... \
    PYTHONPATH=$PWD python3 perf/b2z2_bshardtime/gate_reader_overlap.py
"""

import json
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = str(_HERE.parents[1])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, str(_HERE.parents[0] / "b2z2_shardscale"))

import torch  # noqa: E402
import meshdesc  # noqa: E402

N = int(os.environ.get("MESH_N", "4"))
S = int(os.environ.get("GATE_S", "512"))
OUT_PATH = os.environ.get("GATE_OUT", f"/tmp/b2z2_gate_overlap_{N}.json")
C_Z, C_S = 128, 384
meshdesc.install(N)

import ttnn  # noqa: E402
from tt_bio import reblock_permute as rbp  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _mesh_open(device_id, kwargs):
    with tt._device_init_lock():
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, N), **kwargs)
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
dev = tt.get_device()
log(f"device={dev} arch={dev.arch()} N={N} S={S}")

# Wrap the gate. The verdict is passed through untouched -- this observes, it does not steer.
CALLS = []
_real_gate = rbp.eligible_gated


def _watched(xw, slice_c, memory_config):
    v = _real_gate(xw, slice_c, memory_config)
    CALLS.append({"shape": [int(d) for d in xw.shape], "slice_c": int(slice_c),
                  "buffer": str(memory_config.buffer_type), "verdict": bool(v)})
    return v


rbp.eligible_gated = _watched
# The model module imported the symbol into its own namespace, so patch there too or the wrapper
# never runs -- the same class of miss as a gate that reads a different number from the one the
# build uses.
_mod_ref = getattr(tt, "_reblock", None)
assert _mod_ref is rbp, "tt_bio.tenstorrent._reblock is not the module being patched"

kernel_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
layer = tt.PairformerLayer(32, 4, 24, 16, True,
                           {k: v.float() for k, v in rl.state_dict().items()}, KC)
REPL = ttnn.replicate_tensor_to_mesh_mapper(dev)


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


m1 = torch.ones(1, S)
z_host = torch.randn(1, S, S, C_Z, dtype=torch.float32)
s_host = torch.randn(1, S, C_S, dtype=torch.float32)
PAIR_MASK = up(m1[:, :, None] * m1[:, None, :])
ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

RES = {"mesh_n": N, "S": S, "arch": str(dev.arch()),
       "visible": os.environ.get("TT_VISIBLE_DEVICES"), "arms": {}}

for arm, rs, bs in (("whole", False, False), ("shard", True, False), ("bshard", True, True)):
    CALLS.clear()
    rbp.REJECTS.clear()
    s_t, z_t = up(s_host), up(z_host)
    layer(s_t, z_t, PAIR_MASK, ATTN, ATTN, row_shard=rs, b_shard=bs)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(z_t)
    ttnn.deallocate(s_t)
    calls = list(CALLS)
    RES["arms"][arm] = {
        "gate_calls": len(calls),
        "accepts": sum(1 for c in calls if c["verdict"]),
        "declines": sum(1 for c in calls if not c["verdict"]),
        "by_shape": sorted({(str(c["shape"]), c["verdict"]) for c in calls}),
        "calls": calls,
        "rejects": {f"{k[0]}|{list(k[1])}": v for k, v in rbp.REJECTS.items()},
    }
    a = RES["arms"][arm]
    log(f"{arm:6s} gate calls {a['gate_calls']:2d}  accept {a['accepts']}  decline {a['declines']}"
        f"  shapes {a['by_shape']}")

# What the short circuit is hiding: put the `b` role's own slab shape through the same gate.
# The ending trimul's slab lands on axis 2, so `gp_b_fused` under `b_shard` is [1, S, S/N, 2*c].
probe = {}
for label, shape in (("a role, ending, slab (checked today)", (1, S, S // N, 256)),
                     ("b role, ending, slab (short-circuited)", (1, S, S // N, 256)),
                     ("b role, ending, WHOLE z (plain row shard)", (1, S, S, 256)),
                     ("b role, starting, slab", (1, S // N, S, 256))):
    t = ttnn.from_torch(torch.zeros(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, mesh_mapper=REPL)
    mc = ttnn.DRAM_MEMORY_CONFIG
    v = _real_gate(t, 128, mc)
    probe[label] = {"shape": list(shape), "verdict": bool(v)}
    log(f"probe {label:42s} {str(list(shape)):22s} -> {'ACCEPT' if v else 'DECLINE'}")
    ttnn.deallocate(t)
RES["direct_probe"] = probe
RES["short_circuit_hides"] = (
    probe["a role, ending, slab (checked today)"]["verdict"] is False
    and probe["b role, ending, slab (short-circuited)"]["verdict"] is False
    and probe["b role, ending, WHOLE z (plain row shard)"]["verdict"] is True)
log(f"short circuit hides a second decline that only exists under b_shard: "
    f"{RES['short_circuit_hides']}")

rbp.eligible_gated = _real_gate
for t_ in (PAIR_MASK, ATTN):
    ttnn.deallocate(t_)
tt.cleanup()
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
