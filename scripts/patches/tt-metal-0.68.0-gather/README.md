# ttnn.gather fixes backported onto our pinned tt-metal 0.68.0

Two upstream commits, cherry-picked as source patches. No version bump, nothing else pulled in.

| patch | upstream | what it fixes |
|---|---|---|
| `0001-gather-multirow-correctness.patch` | `da92873a78a` (#40390, 2026-03-30) | multi-core gather returned wrong data for every tile-row after the first |
| `0002-gather-needed-bitmap.patch` | `dcdde8f414d` (#53112) | the reader rescanned all 1024 index values once per input tile |

v0.68.0 was tagged 2026-03-29, one day before the correctness fix landed.

Both commits touch only JIT-compiled dataflow kernels, so **nothing here needs a tt-metal C++
build**. `apply.sh` copies the installed runtime into a private prefix and patches it there; the
shared wheel other workers run on is never written.

```sh
scripts/patches/tt-metal-0.68.0-gather/apply.sh /path/to/prefix
TT_METAL_RUNTIME_ROOT=/path/to/prefix/ttnn \
TT_METAL_CACHE=/path/to/prefix-cache \
PYTHONPATH=/path/to/prefix \
python3 your_script.py
```

`TT_METAL_RUNTIME_ROOT` is required. Without it the runtime derives its root from the package's
parent directory and looks for `tt_metal/` one level too high. `TT_METAL_CACHE` must be private
because the JIT cache key does not include the kernel source, so a shared cache serves the
unpatched binary and the patch silently does nothing.

## 0002 is a port, not a cherry-pick

Upstream writes `dcdde8f414d` against a later kernel that uses the `DataflowBuffer` wrapper,
which 0.68.0 does not have. `/tmp`-free reproduction of the port lives in the commit that added
this directory: the `needed[]` pre-scan is re-expressed against 0.68.0's raw `cb_*` calls, and
every substitution asserts it matched exactly once so a silently-empty patch is impossible.

## What we measured, on card 2 of qb2

`scripts/rfd3_port/p97_gather_backport.py` (correctness), `p98_gathered_vs_dense.py` (production
shape), `p99_gather_bands.py` (atom bands). Raw JSON in `perf/p97`, `perf/p98`, `perf/p99`.

**Correctness: fixed.** Stock 0.68.0 is exact to a key axis of 1920 and 93.75% wrong at 2048.
Patched, `ttnn.gather` on dim 3 is exact at 1920, 2048, 3072, 4096 and 6080, in fp32 and bf16,
with both a random and an RFD3-style banded index, and 0 of 3112960 elements are wrong at the
production `[1,4,6080,6080]`. Exactness here is `!=` against `torch.gather`, not a PCC.

**0002 is worth 23.8x on its own.** At the production shape the gather op alone goes 4079.26 ms
-> 171.14 ms. Upstream reported 12.5x on their shape; ours is bigger because our index is banded,
so only ~2 of 190 input tiles per index tile carry a hit.

**The gathered attention route still loses, in every band.** Interleaved against the dense
softmax it replaces, both arms warmed, A/A floor 0.01-0.60 ms:

| atoms | gather factory | dense ms | gathered ms | ratio |
|---|---|---|---|---|
| 512 | single-core | 0.11 | 0.70 | 6.4x |
| 1024 | single-core | 0.29 | 1.69 | 5.8x |
| 1920 | single-core | 0.72 | 4.47 | 6.2x |
| 2048 | multi-core | 0.79 | 38.40 | 48.9x |
| 3040 | multi-core | 1.85 | 67.04 | 36.2x |
| 6080 | multi-core | 7.38 | 282.02 | 38.2x |

At 6080 the gathered chain is 171 ms of gather plus 111 ms of scatter against a 7.4 ms dense
softmax.

## Why it still loses

`gather_program_factory.cpp` splits work with `split_work_to_cores(core_grid, Wt_index, true)` —
over the **index** width only. RFD3 gathers 128 neighbours, so `Wt_index` is 4 tiles and 4 cores
serve the op however many atoms there are. The writer kernel then loops `w` over all `Wt_input`
tiles for every row on every core, so each of those 4 cores reads the entire 591.5 MB input:
2.37 GB of DRAM traffic to produce 12.4 MB of output. Neither backported commit touches the work
split or the read replication, and the step from 1920 to 2048 atoms in the table above is the
`GATHER_WT_THRESHOLD = 60` factory switch, not a data effect.
