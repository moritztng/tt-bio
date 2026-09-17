# c12_profiled_fold — in-fold device time per op, at a pinned during-sampled 1350 MHz

The campaign prices its levers off `c10-fold-census`, which replayed each launch key standalone
with `ttnn.DRAM_MEMORY_CONFIG` on every operand and every output, and published the `@grid110` arm
for `linear` and `matmul`. The fold runs neither: many of those ops are L1-resident, and no live
site passes `core_grid`. Two biases, opposite signs, partly cancelling. This directory measures the
same classes with the fold's own operands instead.

## Why not just profile the fold

465,664 ttnn calls. tt-metal's device profiler costs 48 B of DRAM per dispatched program per RISC
per core and ~350 kB of host CSV per program, so a whole-fold capture is on the order of 10^5 GB of
CSV against 3.1 TB free. `--enable-sum-profiling` does not change this: it buys two extra optional
markers per program (`profiler_state_manager.cpp:39`) and still allocates a per-program DRAM slot.
Above 1000 dispatched programs without `--op-support-count` the failure is loud and total — dropped
marker warnings and no ops report at all.

So the unit of measurement is one *instance* of a repeating unit, grabbed out of a live fold and
replayed fenced under the profiler, weighted by the fold's own calls-per-unit. Grabbing the whole
unit is what kills the replay bias: the unit's own code allocates its own intermediates and picks
its own memory and program configs, so there is no DRAM-operand overprice and no grid arm the fold
does not use. The precursor trick is `b2z-kernel-cycle-census`'s, reused: unwind the fold with a
sentinel the moment the wanted call is in hand, and for diffusion-side units run recycling 1 with
the pairformer and MSA stacks truncated. Shapes and memory configs are untouched; only the values
flowing in differ, and a bf16 matmul's cycle count does not depend on its values.

## Board, not card

A tt-metal **source** build refuses a single P300 chip. `TT_VISIBLE_DEVICES=3` dies in
`generate_cluster_descriptor`: one chip of a two-chip board makes the cluster type `CUSTOM`, which
then demands a mesh graph descriptor (`tt_cluster.cpp:198,273`). Legal subsets on qb2 are `{0,1}`,
`{2,3}` and all four. The shipped `ttnn==0.68.0` wheel predates the assert and takes a single card,
but it has the device profiler compiled out. **So the `counts` phase runs on one card under the
wheel, and any profiled phase needs the whole board.**

## Reproduce

```bash
# spine: unit call census + fold wall of record, one card, shipped wheel
RUN=counts_c3 PHASE=counts METAL= VIS=3 LEASE=3 CLKNODES=3 ./run.sh --plain-n 1

# instrument proof and a profiled unit, whole board, Tracy source build
RUN=probe1 PHASE=probe ./run.sh
RUN=pfl1 PHASE=unit OPSUP=20000 ./run.sh --unit PairformerLayer --reps 3 --ops
RUN=dtl1 PHASE=unit OPSUP=20000 ./run.sh --unit DiffusionTransformerLayer --reps 20 --ops \
    --short-precursor

python3 reduce.py --run runs/counts_c3 --unit-run runs/pfl1 --unit-run runs/dtl1 \
    --node 3 --out runs/table.json
```

`run.sh` holds AICLK at the target for the whole session through tt-kmd's ARC queue and releases it
in a trap; `clk.py` samples `tt_aiclk` at 1 kHz from its own process so the sampler never takes the
folding process's GIL. `reduce.py` qualifies every timed interval against those samples and refuses
an interval that is not min = max = target with no read error and no gap over 10 ms.

## What each number is, and is not

| output | what it is | what it is not |
|---|---|---|
| `fold_s_plain_median` | the session's own fold wall, plain arm, qualified clock | not the cell of record unless the box was quiet |
| `tree[].incl_s` / `excl_s` | unsynced host-side inclusive/exclusive wall per unit path | **not device time.** A ttnn call is an async enqueue; these are weights and coverage |
| `tree[].calls` | the fold's own calls per unit | — |
| `bracket_cost_ratio` | what the counting bracket costs the fold | — |
| `units[].device_ms_per_call` | summed `DEVICE KERNEL DURATION` in the fenced window, per unit call | not a wall: it excludes the gaps between programs |
| `units[].by_class` | device ms split back into the census's python classes | only as good as `align_quality`, which is printed beside it |

The class split is the hard part and it is reported with its own quality number. The ops report
carries device op codes, not python names, and the census's classes do not partition the same way:
`linear` and `matmul` are both `MatmulDeviceOperation`; `multiply_`, `add_`, `multiply` and `add`
are all `BinaryNgDeviceOperation`. `--ops` records the python call sequence in the same fenced
region, and `align` walks the two sequences in order, crediting a row to a name only when the
name's recorded operand shapes contain the row's `INPUT_0` shape. A split taken from a poor
alignment is not quotable.
