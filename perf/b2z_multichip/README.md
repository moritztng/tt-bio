# Multi-chip Boltz-2: pricing the link on the Wormhole Galaxy

Row-sharding the pair tensor `z` across chips needs three collectives per pairformer block: both
triangle multiplications contract over a full axis and need the other operand gathered, and
`triangle_attention_end` is per-column and needs a transpose. `triangle_attention_start` and
`transition_z` are row-local and free. At 512 aa, `z` is 512x512x128 bf16 = 67.108864 MB, so each
collective delivers 33.554 MB to each of two chips.

These scripts measure what that costs. They do not touch `tt_bio/`.

| script | what it answers |
|---|---|
| `probe_api.py` | what the installed ttnn exposes for mesh + collectives, and the system mesh shape |
| `probe_open.py`, `probe_open2.py`, `probe_open3.py` | which (mesh shape, fabric mode, reliability mode) triple opens |
| `probe_eth.py`, `probe_eth2.py` | the cluster's own view of chip-to-chip ethernet |
| `link_bench.py` | all-gather / reduce-scatter latency and burst cost at the real `z` shapes |
| `host_bw.py` | host<->device bandwidth, which prices the relay path that needs no fabric |

Results live in `results/`. State doc: `~/.coworker/state/b2z-multichip-fold.md`.

Run everything with `TT_VISIBLE_DEVICES` set. Unpinned, UMD opens all 32 ASICs and blocks on a
sibling worker's `CHIP_IN_USE_<n>_PCIe` lock.

As of 2026-09-12 `link_bench.py` cannot run on whglx: tt-fabric does not initialise there, in any of
18 combinations tried. The state doc has the signatures and the two ways out.
