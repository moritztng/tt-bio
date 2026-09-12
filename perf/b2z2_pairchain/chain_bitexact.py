"""The five-op pair track, row-sharded over a real two-chip mesh: is it bit-exact, and if not, where?

`perf/b2z2_dualchip/slab_bitexact.py` proves one op at a time with a Python row range, which an
SPMD mesh cannot run. `perf/b2z2_trunkshard/mesh_slab_bitexact.py` proves four of the five ops
under `ttnn.mesh_partition`, one op at a time. Neither reaches the thing a sharded fold needs: the
RESIDUAL CHAIN, where each op reads the previous op's output and an op that writes only its own
rows leaves every other device's copy of z stale.

Three arms, and the middle one is why this file exists.

  chain      the engine's own `PairformerLayer.__call__(..., row_shard=True)` against the same
             block replicated. Both tracks, on BOTH chips, `torch.equal`. This is the deliverable.
  localise   the same five ops driven step by step from here, with the sharded z gathered and
             compared to the unsharded z AFTER EVERY OP. A chain that fails tells you only that it
             failed; this says which op, and whether the op or the gather before it is at fault.
  controls   a dropped gather (the chain then reads a z one residual step stale) and a perturbed
             input. Both must be rejected, or the check is not reading the shard.

The device is injected at `_open_device_locked`, the way `perf/b2z2_dualchip/mesh_fold.py` does it,
so `tt_bio.tenstorrent.get_device()` returns the mesh and every `Module` builds its weights on it.
Setting a module global by hand instead is how this harness's predecessor opened a SECOND device
underneath a live fabric mesh; see `state/b2z2-pairtrack-chain-wh.md`.

Run (WH pair, adjacent chips, the n300 descriptor):
    MESH_N=2 TT_VISIBLE_DEVICES=12,13 TT_BIO_LEASE_CARDS=12,13 \
    TT_MESH_GRAPH_DESC_PATH=<ttnn>/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
    python3 perf/b2z2_pairchain/chain_bitexact.py

Env: MESH_N (2), CHAIN_S (512), CHAIN_DROP (drop the n-th gather, 0-based), CHAIN_OUT.
Exit status 0 only if the chain is bit-exact on both tracks and both controls were rejected.
"""

import json
import os
import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

N = int(os.environ.get("MESH_N", "2"))
S = int(os.environ.get("CHAIN_S", "512"))
DROP = os.environ.get("CHAIN_DROP")
# `b_shard` splits the `b` role of both triangle products over the mesh and gathers the projection
# instead of replicating the read. It adds collectives INSIDE the trimul, so under CHAIN_B_SHARD
# the gather index CHAIN_DROP enumerates counts every collective the chain makes, b gathers
# included, and each of them has to be rejected.
B_SHARD = os.environ.get("CHAIN_B_SHARD", "0") == "1"
OUT_PATH = os.environ.get("CHAIN_OUT", f"/tmp/b2z2_chain_{N}.json")
C_Z, C_S = 128, 384

import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


OPENS = [0]


def _mesh_open(device_id, kwargs):
    """What `_open_device_locked` does, on a 1xN mesh. Reimplemented rather than replaced: the
    compute grid retune and the program cache are not optional, and a mesh installed without the
    first one leaves every program config tuned to the 11x10 module default (an 8x9 Wormhole then
    asks for core (10, 0) and dies)."""
    OPENS[0] += 1
    with tt._device_init_lock():
        if N > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        log(f"opening 1x{N} mesh, kwargs={kwargs}")
        dev = (ttnn.open_mesh_device(ttnn.MeshShape(1, N), **kwargs) if N > 1
               else ttnn.open_device(device_id=device_id, **kwargs))
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
dev = tt.get_device()
log(f"device={dev} opens={OPENS[0]} arch={dev.arch()} grid={tt.CORE_GRID_MAIN} "
    f"b_shard={B_SHARD}")
assert OPENS[0] == 1, "the mesh was opened more than once"

RES = {"arch": str(dev.arch()), "mesh": f"1x{N}", "S": S, "visible": os.environ.get("TT_VISIBLE_DEVICES"),
       "drop": DROP, "b_shard": B_SHARD, "chain": {}, "localise": {}, "controls": {}}
failures = []

kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)


def randomize_(module):
    """Boltz-2's init zeroes both trimul output projections, so a freshly built layer returns z
    unchanged and every comparison below would hold for reasons unrelated to the shard."""
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
randomize_(rl)
layer = tt.PairformerLayer(32, 4, 24, 16, True,
                           {k: v.float() for k, v in rl.state_dict().items()}, KC)
log(f"layer built, {len(rl.state_dict())} weight tensors")

REPL = ttnn.replicate_tensor_to_mesh_mapper(dev) if N > 1 else None
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0) if N > 1 else None


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


def down_all(t):
    """Every device's copy, stacked on a new leading axis. For a replicated tensor the copies must
    agree; for a sharded one they are different row slabs."""
    if N == 1:
        return ttnn.to_torch(t).unsqueeze(0)
    return ttnn.to_torch(t, mesh_composer=COMP)


def down_rows(t):
    """A tensor each device holds a different row slab of, concatenated back onto the row axis."""
    c = down_all(t)
    per = c.shape[0] // N
    return torch.cat([c[i * per:(i + 1) * per] for i in range(N)], dim=1)


# Ragged, not all ones: the trimul multiplies the mask into `a` only, so a mask that does not
# follow `a` onto the slab axis has to be visible.
m1 = torch.zeros(1, S)
m1[:, :S - 32] = 1.0
z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
s_t = torch.randn(1, S, C_S, dtype=torch.float32)
PAIR_MASK = up(m1[:, :, None] * m1[:, None, :])
ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)


def eq(a, b):
    return bool(a.shape == b.shape and torch.equal(a, b)), float((a.float() - b.float()).abs().max())


# --- localise: the five ops, driven from here, compared after every one -------------------------
# The chain in `PairformerLayer._pair_track_row_sharded` is the thing under test, so this driver
# repeats its op order rather than calling it: a failure there says only "the block is wrong",
# where this says which op's output first differs, and comparing the gathered z after each step
# separates "the op computed the wrong rows" from "the gather put them back wrong".
STEPS = [
    ("trimul_start",
     lambda z, ri: layer.triangle_multiplication_start(z, PAIR_MASK, row_input=ri,
                                                       b_shard=B_SHARD),
     lambda z: layer.triangle_multiplication_start(z, PAIR_MASK)),
    ("trimul_end",
     lambda z, ri: layer.triangle_multiplication_end(z, PAIR_MASK, row_input=ri,
                                                     b_shard=B_SHARD),
     lambda z: layer.triangle_multiplication_end(z, PAIR_MASK)),
    ("triatt_start", lambda z, ri: layer.triangle_attention_start(z, ATTN, row_input=ri),
     lambda z: layer.triangle_attention_start(z, ATTN)),
    ("triatt_end", lambda z, ri: layer.triangle_attention_end(z, ATTN, row_input=ri),
     lambda z: layer.triangle_attention_end(z, ATTN)),
    ("transition_z", lambda z, ri: layer.transition_z(z, row_input=ri),
     lambda z: layer.transition_z(z)),
]


def localise():
    """Run the pair track twice, unsharded and sharded, and compare the full z after every op."""
    out = {}
    z_ref = up(z_t)
    z = up(z_t)
    z_rows = ttnn.mesh_partition(z, dim=1) if N > 1 else z
    first_bad = None
    for name, sharded_fn, whole_fn in STEPS:
        u = whole_fn(z_ref)
        z_ref = ttnn.add_(z_ref, u)
        ttnn.deallocate(u)
        try:
            u = sharded_fn(z, z_rows)
        except Exception as e:  # noqa: BLE001
            out[name] = {"raised": f"{type(e).__name__}: {str(e)[:200]}"}
            log(f"  {name}: RAISED {type(e).__name__}: {str(e)[:160]}")
            first_bad = first_bad or name
            break
        z_rows = ttnn.add_(z_rows, u)
        ttnn.deallocate(u)
        if N > 1:
            old = z
            z = ttnn.all_gather(z_rows, dim=1)
            ttnn.deallocate(old)
        else:
            z = z_rows
        ok, d = eq(down_all(z)[0:1], down_all(z_ref)[0:1])
        out[name] = {"bit_exact": ok, "max_abs_diff": d}
        log(f"  after {name}: equal={ok} max abs diff {d}")
        if not ok and first_bad is None:
            first_bad = name
    out["first_divergent_op"] = first_bad
    for t in (z, z_ref):
        if t.is_allocated():
            ttnn.deallocate(t)
    return out


log("localise: the five pair-track ops, sharded vs whole, compared after each")
RES["localise"] = localise()
if RES["localise"].get("first_divergent_op"):
    failures.append(f"the chain first diverges at {RES['localise']['first_divergent_op']}")

# --- the deliverable: the engine's own chain, block level ----------------------------------------
if DROP is not None:
    n = int(DROP)
    orig = tt.PairformerLayer._pair_track_row_sharded
    state = {"i": 0}

    def patched(self, z, mask, ams, ame, **kw):
        real = ttnn.all_gather

        def counting(t, dim, *a, **kw):
            i, state["i"] = state["i"], state["i"] + 1
            if i == n:
                # The RIGHT shape and the WRONG rows: every device's own slab, twice, with no
                # traffic. Returning the shard instead (which is what this control used to do)
                # is rejected by a shape check three ops later, which proves the gather is
                # structurally required and says nothing about whether the comparison reads
                # values. This one can only be caught by reading them.
                # N copies, not two: at width 4 a doubled slab is the wrong SHAPE and the chain
                # then dies on a shape error, which rejects the control for a reason that says
                # nothing about whether the comparison reads values.
                return ttnn.concat([t] * N, dim=dim)
            return real(t, dim, *a, **kw)
        ttnn.all_gather = counting
        try:
            return orig(self, z, mask, ams, ame, **kw)
        finally:
            ttnn.all_gather = real
    tt.PairformerLayer._pair_track_row_sharded = patched
    log(f"NEGATIVE CONTROL: gather {n} dropped")


def run_block(row_shard, z_host=None):
    s, z = up(s_t), up(z_t if z_host is None else z_host)
    s_o, z_o = layer(s, z, PAIR_MASK, ATTN, ATTN, row_shard=row_shard,
                     b_shard=B_SHARD and row_shard)
    hs, hz = down_all(s_o), down_all(z_o)
    ttnn.deallocate(s_o)
    ttnn.deallocate(z_o)
    return hs, hz


log("chain: the engine's PairformerLayer(row_shard=True) against the replicated block")
t0 = time.perf_counter()
s_rep, z_rep = run_block(False)
t_rep = time.perf_counter() - t0
t0 = time.perf_counter()
try:
    s_shd, z_shd = run_block(True)
    RES["chain"]["ran"] = True
except Exception as e:  # noqa: BLE001
    RES["chain"] = {"ran": False, "raised": f"{type(e).__name__}: {str(e)[:400]}"}
    log(f"  row_shard=True RAISED {type(e).__name__}: {str(e)[:300]}")
    failures.append(f"the chain raised {type(e).__name__}")
    s_shd = z_shd = None
t_shd = time.perf_counter() - t0

if z_shd is not None:
    log(f"  replicated {t_rep * 1e3:.1f} ms | row-sharded {t_shd * 1e3:.1f} ms (cold, not a perf number)")
    for name, rep, shd in (("z", z_rep, z_shd), ("s", s_rep, s_shd)):
        per_chip = [torch.equal(shd[i:i + 1], rep[i:i + 1]) for i in range(rep.shape[0])]
        ok, d = eq(shd, rep)
        RES["chain"][name] = {"bit_exact_per_chip": per_chip, "max_abs_diff": d}
        log(f"  {name}: sharded == replicated per chip {per_chip}  max abs diff {d}")
        if not all(per_chip):
            failures.append(f"{name}: the row-sharded block is not bit-exact")
        # The replicated run is its own control: a mesh whose devices disagree makes every
        # comparison above meaningless.
        if N > 1 and not torch.equal(rep[0:1], rep[1:2]):
            failures.append(f"{name}: the REPLICATED block differs between chips")

# --- control: perturb the input --------------------------------------------------------------
if DROP is None and z_shd is not None:
    z_bad = z_t.clone()
    z_bad[0, S - 64, 0, 0] += 5.0      # inside the LAST device's rows
    _, z_p = run_block(True, z_host=z_bad)
    ok, d = eq(z_p, z_rep)
    RES["controls"]["perturbed_input"] = {"rejected": not ok, "max_abs_diff": d}
    log(f"control: perturbed input rejected={not ok} (max abs diff {d})")
    if ok:
        failures.append("the perturbed-input control was ACCEPTED; the check is blind")

# Close the mesh and take fabric down IN PROCESS. `perf/b2z2_dualchip/mesh_fold.py` ends on
# `os._exit(0)`, which skips teardown entirely and leaves the mesh open at process exit -- the
# ledger records an ethernet core wedged after every one of those runs. Every run of this file
# closes cleanly, and none of them has wedged a chip.
tt.cleanup()
if N > 1:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")

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
log(f"PASS: the five-op row-sharded pair track at S={S}"
    f"{' with b_shard' if B_SHARD else ''} is bit-identical to the replicated block "
    f"on every chip of the 1x{N} mesh")
sys.exit(0)
