# Multi-card single-input sharding spike

**Verdict: NO-GO.** Do not queue a full implementation follow-up.

## The gap

`tt-bio predict --devices` is data-parallel: one whole target per card. A single
large fold (e.g. N=1024) runs on exactly one card while the others sit idle. This
caps the latency of the single biggest jobs. The spike asked whether sharding one
input's compute across cards can lower that latency.

## What was measured (qb2, real hardware, warm)

Chosen op: the ESMFold2/Protenix trunk **triangle multiplication** at N=1024,
C_Z=256. Its core is a per-channel batched matmul `[1,C,N,N] @ [1,C,N,N]`,
embarrassingly parallel across channels (no reduction inside the matmul), so it is
the best case for sharding.

Single-card anchors (one full call, N=1024):

| op | single-card compute |
|---|---|
| full trimul call | **154.5 ms** |
| pair transition | 109.3 ms |
| trimul core matmul (C=256) | 9.3 ms |

Sharded compute scales almost perfectly (matmul core, mesh SPMD):

| cards | compute scaling |
|---|---|
| 2 | **1.99x** |
| 4 | **3.83x** |

So the arithmetic parallelizes. The problem is everything around it.

## Blocker 1 — there is no card-to-card fabric on this box

Bringing up any fabric collective (`ttnn.all_gather`) fails:

```
Fabric Router Sync: Timeout after 10000 ms. Device 3: Expected 0xa2b2c2d2, got 0xa0b0c0d0
```

UMD reports `Opening local chip ids {0,1,2,3} and remote chip ids {}` — the four
P150s are independent PCIe cards with no working inter-card ethernet. The
`p150_x2`/`p150_x4` mesh descriptors declare 4 eth channels/edge; the hardware
exposes at most 2 (RELAXED-mode warning), and the routers never sync. On-device
single-input sharding is therefore **impossible today** regardless of the math.

## Blocker 2 — the only path (host PCIe) is 6-100x slower than single-card

The sole inter-card path is host-mediated: card → host RAM → card. A sharded
trimul must, per call, replicate the 512 MiB layer-normed input to every card and
gather the 512 MiB output back. Measured on the real PCIe path:

| cards | host replicate (in) | host gather (out) | comms / call | compute saved | net vs single |
|---|---|---|---|---|---|
| 2 | 622 ms | 329 ms | **951 ms** | ~77 ms | **0.15x (6.6x slower)** |
| 4 | 672 ms | 283 ms | **955 ms** | ~114 ms | **0.16x (6.4x slower)** |

Comms (~950 ms) dwarfs the compute it parallelizes (154 ms full op, 9 ms matmul
core). Against the matmul core alone the ratio is ~100x. There is no N or K where
host-routed sharding wins.

## Even a hypothetical working fabric is marginal, and needs a whole-trunk rewrite

To net >1.3x at 4 cards against the 154.5 ms full trimul, comms per call must stay
under ~79 ms while moving ~1 GiB (512 MiB in + 512 MiB out) — i.e. >13.6 GB/s
sustained collective bandwidth. With ~2 eth channels/edge that sits right at the
break-even knee, not a comfortable win.

And trimul is the friendly case. A real sharded trunk keeps the 512 MiB pair state
sharded by channel across all 48 blocks, so every channel-mixing op forces a
cross-card reduction each block: trimul input layer-norm stats, the trimul output
projection (256→256), and the pair transition (a per-position MLP over channels,
109 ms). That is ~3 collectives × 512 MiB × 48 blocks ≈ 70+ GiB of fabric traffic
per fold. The surface area is the entire trunk, not one op, and the best-case
payoff even with ideal fabric is ~2x.

## Recommendation

**No-go.** Data-parallel `--devices` (one target per card, zero inter-card comms)
remains the correct architecture; it already saturates throughput. Single-input
latency sharding is blocked by hardware (no working eth fabric) and, even past
that, is a marginal win requiring a full sharded-trunk rewrite.

Re-evaluate only if a future QuietBox ships with functional card-to-card ethernet.
The one durable positive: the parallelizable compute scales 1.99x/3.83x on 2/4
cards, so the math is not the obstacle — the interconnect is.

## Reproduce

```
# compute scaling + host-comms (per K)
TT_MESH_GRAPH_DESC_PATH=/tmp/p150_x{2,4}_c2.textproto \
  python3 scripts/trimul_shard_spike.py --k {2,4} --n 1024
# single-card full-op anchor
TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
  python3 scripts/trimul_fullop_time.py
```
