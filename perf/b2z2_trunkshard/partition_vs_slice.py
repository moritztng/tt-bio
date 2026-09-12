"""Does `ttnn.mesh_partition` return the same bytes as the `ttnn.slice` it replaces?

`mesh_slab_bitexact.py` found that four of the five pair-track ops are bit-identical under mesh
addressing and one, `triangle_attention_end`, is not -- while its row-number slab IS bit-identical
on the same mesh. So the difference is in the addressing, for that op's operands specifically. It
takes three slabs where the others take one, on three different tensor shapes. This compares the
two primitives directly on each of those shapes, with nothing else in the way.

    TT_VISIBLE_DEVICES=6,7 TT_MESH_GRAPH_DESC_PATH=<n300 mgd> \
        python3 perf/b2z2_trunkshard/partition_vs_slice.py

Read the result with care. The three shapes `triatt_end` slabs come back IDENTICAL under the two
primitives, so the divergence is not one mis-partitioned operand. The fourth case, the pair track
on dim 1, comes back DIFFERENT -- and that contradicts the op-level control in
`mesh_slab_bitexact.py`, which shows `transition_z` (a dim-1 slab of that exact tensor) putting the
first 256 rows on device 0 and matching the unsharded result to the bit. One of the two is reading
the mesh composer wrongly and it has not been settled, so neither the fourth line nor any
conclusion drawn from it should be quoted yet.
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import os

import torch

from tt_bio.device_lease import CardSetLease  # noqa: E402

N = int(os.environ.get("MESH_N", "2"))
if N > 1:
    CardSetLease().acquire()

import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

dev = (ttnn.open_mesh_device(ttnn.MeshShape(1, N), l1_small_size=32768) if N > 1
       else tt.get_device())
if N > 1:
    tt._device = dev
    tt._configure_active_compute_grid(dev)
print(f"device {dev}", flush=True)

REPL = ttnn.replicate_tensor_to_mesh_mapper(dev) if N > 1 else None
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0) if N > 1 else None

# The three shapes triatt_end takes a slab of, and the axis each is taken on.
CASES = [
    ("x_gate  (the normed, transposed pair tensor)", (512, 512, 128), 1),
    ("bias    (the triangle bias, head-major)", (1, 4, 512, 512), 2),
    ("q       (the query, head-major from nlp_create_qkv_heads)", (1, 4, 512, 32), 2),
    ("pair    (the pair track, for contrast -- this one works)", (1, 512, 512, 128), 1),
]

bad = []
for label, shape, dim in CASES:
    torch.manual_seed(0)
    host = torch.randn(*shape, dtype=torch.float32)
    t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        mesh_mapper=REPL)
    per = shape[dim] // N
    lo, hi = [0] * len(shape), list(shape)
    lo[dim], hi[dim] = 0, per
    by_slice = ttnn.slice(t, lo, hi)
    by_part = ttnn.mesh_partition(t, dim)
    def down(x):
        if N == 1:
            return ttnn.to_torch(x)
        c = ttnn.to_torch(x, mesh_composer=COMP)
        return c[0:1] if len(shape) == 4 else c[:shape[0]]
    a, b = down(by_slice), down(by_part)
    same = a.shape == b.shape and torch.equal(a, b)
    d = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
    print(f"{label}\n    dim {dim}, {shape} -> {per}: slice {list(a.shape)} vs partition "
          f"{list(b.shape)}  same={same} max abs diff {d}", flush=True)
    if not same:
        bad.append(label)
    for x in (t, by_slice, by_part):
        ttnn.deallocate(x)

print("\nDIFFER: " + (", ".join(bad) if bad else "none -- the two primitives agree everywhere"))
sys.exit(1 if bad else 0)
