"""Which slab site inside `triangle_attention_end` does `mesh_partition` reach differently from
`slice`, and why.

The op takes a slab of four things it computed itself: the transposed normed pair tensor, the
triangle bias, q (after the head split) and, when the mask has a query extent, the mask. Two
mechanisms reach that slab -- `ttnn.slice(t, r0, r1)` for the row-number spelling and
`ttnn.mesh_partition(t, dim)` for the mesh one -- and on DEVICE 0 of a 1x2 mesh they address the
same rows, so every site must give the same bytes and the op the same answer.

It does not: the mesh spelling differs from the row-number one by ~0.47-1.04 on this op alone,
while the other four pair-track ops are 0.0 (`perf/b2z2_trunkshard/mesh_slab_bitexact.py`).

So bisect it. Run the op on device 0 with every site taking `slice(0, half)`, then flip ONE site
at a time to `mesh_partition` and compare device 0's output against the same reference. Whichever
flip moves the answer is the site where the two mechanisms are not the same slab, and comparing the
two slabs' bytes directly at that site says what changed.

    MESH_N=2 TT_VISIBLE_DEVICES=12,13 python3 perf/b2z2_pairchain/triatt_end_bisect.py

Env: MESH_N (2), BISECT_S (512), BISECT_OUT.
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
S = int(os.environ.get("BISECT_S", "512"))
OUT_PATH = os.environ.get("BISECT_OUT", "/tmp/b2z2_triatt_bisect.json")
C_Z, C_S = 128, 384
HALF = S // N

import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _mesh_open(device_id, kwargs):
    with tt._device_init_lock():
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, N), **kwargs)
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
dev = tt.get_device()
log(f"device={dev} arch={dev.arch()} grid={tt.CORE_GRID_MAIN}")

kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)


def randomize_(module):
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
op = layer.triangle_attention_end

REPL = ttnn.replicate_tensor_to_mesh_mapper(dev) if N > 1 else None
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0) if N > 1 else None


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


def dev0(t):
    """Device 0's copy of a mesh tensor."""
    if N == 1:
        return ttnn.to_torch(t)
    c = ttnn.to_torch(t, mesh_composer=COMP)
    return c[0:1]


m1 = torch.zeros(1, S)
m1[:, :S - 32] = 1.0
z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
ATTN = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

RES = {"arch": str(dev.arch()), "mesh": f"1x{N}", "S": S, "half": HALF, "sites": {}, "arms": {}}

# --- the site table, filled by one instrumented run ---------------------------------------------
_real_slab_take = tt._slab_take
SHARD_SITES = None      # None = every site takes mesh_partition; a set = only those do
CALLS = []


def _patched(t, axis, r0, r1, shard):
    """Record every slab site, and route it by `SHARD_SITES`.

    With `shard=True` the op has already localised its range to `(0, HALF)`, so `slice(0, HALF)` on
    the replicated tensor IS device 0's partition. Every site therefore has two spellings of one
    slab, and on device 0 they must give identical bytes."""
    i = len(CALLS)
    CALLS.append({"i": i, "axis": axis, "r0": r0, "r1": r1, "shape": [int(d) for d in t.shape],
                  "layout": str(t.layout), "buffer": str(t.memory_config().buffer_type),
                  "dtype": str(t.dtype)})
    use_mesh = shard and (SHARD_SITES is None or i in SHARD_SITES)
    CALLS[-1]["mesh"] = use_mesh
    if shard and not use_mesh:
        return _real_slab_take(t, axis, 0, r1, False)
    return _real_slab_take(t, axis, r0, r1, shard)


tt._slab_take = _patched


def run(sites, shard=True):
    """One run of the ending op on device 0. `sites=None` -> every site meshes; a set -> only
    those sites mesh and the rest slice; `shard=False` -> the pure row-number arm."""
    global SHARD_SITES, CALLS
    SHARD_SITES, CALLS = sites, []
    z = up(z_t)
    if shard:
        ri = ttnn.mesh_partition(z, dim=1) if N > 1 else z
        out = op(z, ATTN, row_input=ri)
    else:
        out = op(z, ATTN, row_slab=(0, HALF))
    got = dev0(out)
    ttnn.deallocate(out)
    sites_seen = list(CALLS)
    return got, sites_seen


def eq(a, b):
    return bool(a.shape == b.shape and torch.equal(a, b)), float((a.float() - b.float()).abs().max())


# Reference: the whole-tensor op, rows [0, HALF). The row-number slab is already proven bit-exact
# against this (`perf/b2z2_dualchip/slab_bitexact.py`), so it is the arm every other one is read
# against, and it is measured here rather than assumed.
z = up(z_t)
whole = dev0(op(z, ATTN))
ref_rows = whole[:, :HALF]
log(f"reference: whole-tensor op, rows [0, {HALF}) of {list(whole.shape)}")

idx, sites = run(None, shard=False)
ok, d = eq(idx, ref_rows)
RES["arms"]["row_number_slab"] = {"bit_exact": ok, "max_abs_diff": d}
log(f"row-number slab (0, {HALF}): equal={ok} max abs diff {d}")
RES["sites"]["row_number"] = sites

allmesh, sites = run(None, shard=True)
ok, d = eq(allmesh, ref_rows)
RES["arms"]["mesh_all_sites"] = {"bit_exact": ok, "max_abs_diff": d}
log(f"mesh, every site: equal={ok} max abs diff {d}")
RES["sites"]["mesh"] = sites
n_sites = len(sites)
log(f"{n_sites} slab sites: " + ", ".join(
    f"#{c['i']} axis{c['axis']} {c['shape']} {c['layout']} {c['buffer']}" for c in sites))

none_mesh, _ = run(set(), shard=True)
ok, d = eq(none_mesh, ref_rows)
RES["arms"]["mesh_no_sites"] = {"bit_exact": ok, "max_abs_diff": d}
log(f"mesh-shaped call, every site SLICED: equal={ok} max abs diff {d}  "
    f"(this arm isolates the sites from the rest of the shard path)")

for k in range(n_sites):
    got, _ = run({k}, shard=True)
    ok, d = eq(got, ref_rows)
    c = sites[k]
    RES["arms"][f"mesh_site_{k}"] = {"bit_exact": ok, "max_abs_diff": d, "site": c}
    log(f"mesh at site #{k} only (axis {c['axis']}, {c['shape']}, {c['buffer']}): "
        f"equal={ok} max abs diff {d}")

# --- and the bytes themselves, at every site -----------------------------------------------------
# If a site's two spellings give the same bytes, it cannot be the one that moved the answer.
log("the two spellings, byte for byte, at each recorded site shape")
RES["byte_compare"] = {}
for k, c in enumerate(sites):
    t = up(torch.randn(*([1] * (4 - len(c["shape"])) + c["shape"]), dtype=torch.float32)
           .reshape(c["shape"]))
    a = ttnn.mesh_partition(t, dim=c["axis"])
    b = ttnn.slice(t, [0] * len(c["shape"]),
                   [c["r1"] if i == c["axis"] else int(d) for i, d in enumerate(c["shape"])])
    ok, d = eq(dev0(a), dev0(b))
    RES["byte_compare"][f"site_{k}"] = {"shape": c["shape"], "axis": c["axis"],
                                        "same_bytes": ok, "max_abs_diff": d}
    log(f"  site #{k} {c['shape']} axis {c['axis']}: partition == slice on device 0: {ok} ({d})")
    for x in (t, a, b):
        ttnn.deallocate(x)

# Which (q_chunk, k_chunk) each shape actually ran. The k_chunk sets the online-softmax reduction
# order, so "the slab is bit-exact" and "the slab took the whole tensor's k_chunk" have to be the
# same statement; printing the picks is what makes that checkable instead of inferred.
RES["sdpa_picks"] = {f"q{q}_k{k}": v for (q, k), v in tt.SDPA_CHUNK_PICKS.items()}
log(f"SDPA picks (q_len, k_len) -> [q_chunk, k_chunk, route]: {RES['sdpa_picks']}")

tt.cleanup()
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
