"""What does `ttnn.mesh_partition(t, dim)` actually address on a RANK-3 tensor?

`perf/b2z2_pairchain/triatt_end_bisect.py` localised `triangle_attention_end`'s mesh divergence to
one slab site: the transposed normed pair tensor, which is rank 3 (`[S, S, C]`) because the op
reshapes the leading 1 away. Its two spellings disagree by 6.23 there, while both rank-4 sites
(the bias `[1, 4, S, S]` and q `[S, 4, S, 32]`) agree to the bit.

This says which slab `mesh_partition` hands back, by comparing it against every candidate slice of
the same tensor, at rank 4 and at rank 3, on both devices. No model, no weights: it is a property
of the collective and nothing else.

    MESH_N=2 TT_VISIBLE_DEVICES=12,13 python3 perf/b2z2_pairchain/mesh_partition_rank_probe.py
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
S = int(os.environ.get("PROBE_S", "128"))
C = int(os.environ.get("PROBE_C", "64"))
OUT_PATH = os.environ.get("PROBE_OUT", "/tmp/b2z2_partition_rank.json")

import ttnn  # noqa: E402
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
REPL = ttnn.replicate_tensor_to_mesh_mapper(dev)
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0)
log(f"device={dev} arch={dev.arch()}")

HALF = S // N
RES = {"arch": str(dev.arch()), "mesh": f"1x{N}", "S": S, "C": C, "cases": {}}

# Distinct per index on every axis, so any mis-addressed slab is visible rather than lucky.
base = (torch.arange(S).view(S, 1, 1) * 1000.0
        + torch.arange(S).view(1, S, 1)
        + torch.arange(C).view(1, 1, C) * 0.001)


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                           mesh_mapper=REPL)


def per_device(t):
    c = ttnn.to_torch(t, mesh_composer=COMP)
    per = c.shape[0] // N
    return [c[i * per:(i + 1) * per] for i in range(N)]


def candidates(host, dim):
    """Every slab of `host` along every axis, as {name: tensor}, for the partition to be matched
    against. The answer is whichever one it equals -- named, not inferred."""
    out = {}
    for ax in range(host.dim()):
        n = host.shape[ax]
        if n % N:
            continue
        w = n // N
        for k in range(N):
            sl = [slice(None)] * host.dim()
            sl[ax] = slice(k * w, (k + 1) * w)
            out[f"axis{ax}[{k * w}:{(k + 1) * w}]"] = host[tuple(sl)]
    return out


def case(name, host):
    t = up(host)
    entry = {"host_shape": list(host.shape), "dims": {}}
    for dim in range(host.dim()):
        try:
            p = ttnn.mesh_partition(t, dim=dim)
        except Exception as e:  # noqa: BLE001
            entry["dims"][str(dim)] = {"raised": f"{type(e).__name__}: {str(e)[:120]}"}
            log(f"  {name} dim={dim}: RAISED {type(e).__name__}")
            continue
        shards = per_device(p)
        cands = candidates(host, dim)
        matched = []
        for i, sh in enumerate(shards):
            hit = [k for k, v in cands.items()
                   if v.shape == sh.shape[-v.dim():] and torch.equal(v.float(), sh.reshape(v.shape).float())]
            matched.append(hit)
        entry["dims"][str(dim)] = {"shard_shape": [int(d) for d in shards[0].shape],
                                   "matches_per_device": matched}
        log(f"  {name} dim={dim}: shard {[int(d) for d in shards[0].shape]}  "
            f"device matches {matched}")
        ttnn.deallocate(p)
    ttnn.deallocate(t)
    RES["cases"][name] = entry


log(f"rank-4 [1,{S},{S},{C}] -- the shape every proven site has")
case("rank4", base.unsqueeze(0))
log(f"rank-3 [{S},{S},{C}] -- the shape triangle_attention_end slabs")
case("rank3", base)
log(f"rank-2 [{S},{C}]")
case("rank2", base[0])

tt.cleanup()
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
