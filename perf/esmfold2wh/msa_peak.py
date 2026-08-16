"""Predict the Wormhole DRAM peak of the MSA encoder from Blackhole.

Runs the encoder standalone at production shapes with the Wormhole small-grid
tile budgets forced on, sampling the allocator around each block. Blackhole has
32 GB so the unblocked arm also runs, which is what makes the comparison a
prediction and not a hope.
"""
import glob, os, sys, time, torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p2")
from safetensors.torch import load_file
from tt_bio import tenstorrent
import tt_bio.esmfold2 as E

L = int(sys.argv[1]); M = int(sys.argv[2]); BLOCK = int(sys.argv[3])
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots/*"))[0]
sd = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    for k, v in load_file(f).items():
        if k.startswith("msa_encoder."):
            sd[k[len("msa_encoder."):]] = v

dev = tenstorrent.get_device()
def used_gib():
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks / 2**30

# Wormhole small-grid budgets, forced. PAIR_TILE_AREA also drives OuterProductMean's
# existing row tiling, so the shape of every allocation matches a Wormhole run.
tenstorrent.SMALL_GRID_PAIR_TILE_AREA = 65536
tenstorrent.SMALL_GRID_SEQ_TILE = 256
tenstorrent.SMALL_GRID_MSA_TILE_AREA = BLOCK
print(f"L={L} M={M} msa_tile_area={BLOCK} -> rows={tenstorrent.msa_row_tile(L, M)}", flush=True)

B = 1
g = torch.Generator().manual_seed(0)
mk = lambda *s: torch.randn(*s, generator=g)
inputs = dict(x_pair=mk(B, L, L, 256), x_inputs=mk(B, L, 451),
    msa_oh=torch.nn.functional.one_hot(torch.randint(0, 33, (B, L, M), generator=g), 33).float(),
    has_deletion=mk(B, L, M).gt(0).float(), deletion_value=mk(B, L, M).abs(),
    msa_mask=torch.ones(B, L, M))
mod = E.MSAEncoder(4, 8, 16); mod.load_state_dict(sd, strict=False)

peak = [0.0]; where = [None]; top = []
# Sample after every allocating ttnn op: the block-level sample misses the
# transient the block is built out of, which is the whole point of the measurement.
for _name in ("matmul", "layer_norm", "concat", "add", "multiply", "sigmoid",
              "silu", "from_torch", "permute", "reshape", "chunk", "unsqueeze"):
    _f = getattr(ttnn, _name)
    def _wrap(f=_f):
        def g(*a, **kw):
            r = f(*a, **kw)
            u = used_gib()
            if u > peak[0]:
                peak[0] = u
                where[0] = (getattr(f, "__name__", "?"), tuple(getattr(r, "shape", ()) or ()))
            top.append((u, getattr(f, "__name__", "?"), tuple(getattr(r, "shape", ()) or ())))
            return r
        return g
    setattr(ttnn, _name, _wrap())

print(f"before: {used_gib():.3f} GiB", flush=True)
t = time.time()
out = mod(**inputs)
print(f"RESULT L={L} M={M} block_area={BLOCK} peak={peak[0]:.3f} GiB at {where[0]}  wall={time.time()-t:.1f}s", flush=True)

top.sort(reverse=True)
seen = set()
for u, n, sh in top:
    k = (n, sh)
    if k in seen: continue
    seen.add(k)
    print(f"  {u:7.3f} GiB after {n}{sh}", flush=True)
    if len(seen) >= 12: break
