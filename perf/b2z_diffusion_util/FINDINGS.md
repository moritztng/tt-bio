# The diffusion step runs on 75.7 % of the grid. The pairformer block runs on 94.4 %.

ARCH: BH. qb2 physical card 0, one Blackhole processor of a p300c. **Measured grid: 110
available worker cores (11x10)**, read from the tracy report's `AVAILABLE WORKER CORE COUNT`
and from the profiler API's `num_available_cores`, both on every one of the 1066 programs.
Not assumed. pc's card is a 130-core part, so the sibling row's 97.28 % and the 94.36 % below
are the same block on two different grids, not two measurements of one number.

tt-metal source build at `/home/ttuser/tt-metal-b2z`, tag v0.68.0, the commit the shipped
`ttnn==0.68.0` wheel is cut from, `ENABLE_TRACY=ON`. tt-bio's fused trimul/SDPA kernels
compile and run unmodified on it (128 `GenericOpDeviceOperation` programs in the precursor),
so unlike the pc census there is no kernel substitution to declare.

## One `Diffusion` call, 1066 dispatched programs

| | PairformerLayer | Diffusion step |
|---|---|---|
| programs per call | 272 | **1066** |
| device kernel time per call | 35.8397 ms | **22.0163 ms** |
| calls per fold | 280 | 200 |
| duration-weighted mean core utilization | **94.36 %** | **75.74 %** |
| programs below half the grid | 8 | **128** |
| idle core-fraction-seconds per fold | 0.566 s | **1.068 s** |

Where the step's kernel time runs: **46.9 %** of it on all 110 cores, 7.5 % at 75-90 %,
33.5 % at 50-75 %, and **12.0 % below half the grid**. The pairformer block, same card, is
79.3 % / 15.6 % / 4.6 % / 0.4 %.

## One cause: the token track is 512 rows, and 512 rows is 16 tiles

**89.2 % of the step's idle core-seconds (0.952 s of 1.068 s per fold) is on a tensor of
shape `[1, 1, 512, D]`.** 512 tokens is 16 tiles of 32, and 16 is what every row-parallel
program there has to divide among 110 cores:

* `LayerNorm` and `NlpCreateHeads` split rows and nothing else, so they get **16 cores**,
  14.5 % of the grid, on 124 programs per call.
* `Matmul` can also split the output's N axis, so it does better, but its grid height must
  divide 16 tile rows and cannot exceed the grid's 10, which leaves 8. The step's matmul
  grids are therefore 8x8 = **64**, 8x10 = **80**, 8x11 = **88** cores and never 110. 2 of
  the 10 core rows are unreachable before any other consideration.

The atom track does not have this problem in the same way: it is `[1, 140, 32, D]`, 140
windows, and lands on 70 or 94 cores.

## Ranked, each group priced against a roof

`us ea` is measured, `ideal` is a perfect linear re-spread to 110 cores, `roof` is
`max(bytes / 429.9 GB/s, flops / 85.96 TFLOP/s)` on the campaign's measured Blackhole roofs.
`recov` is `measured - max(ideal, roof)` summed over the fold: the honest prize, because
re-spreading a program stops paying the moment it reaches the byte roof.

| op | cores | n/call | us ea | ideal | roof | idle s/fold | recov s/fold | shape |
|---|---|---|---|---|---|---|---|---|
| Matmul | 64 | 193 | 20.08 | 11.68 | 7.03 | 0.3241 | 0.3241 | `[1,1,512,768] -> [1,1,512,768]` |
| LayerNorm | 16 | 100 | 15.93 | 2.32 | 3.66 | 0.2723 | 0.2455 | `[1,1,512,768] -> [1,1,512,768]` |
| NlpCreateHeads | 16 | 24 | 42.53 | 6.19 | 9.76 | 0.1745 | 0.1573 | `[1,1,512,3072] -> [1,16,512,64]` |
| Matmul | 80 | 76 | 22.24 | 16.17 | 14.05 | 0.0922 | 0.0922 | `[1,1,512,768] -> [1,1,512,1536]` |
| Matmul | 64 | 26 | 22.91 | 13.33 | 14.05 | 0.0498 | 0.0461 | `[1,1,512,1536] -> [1,1,512,768]` |
| Matmul | 88 | 24 | 39.64 | 31.71 | 28.11 | 0.0381 | 0.0381 | `[1,1,512,768] -> [1,1,512,3072]` |
| Matmul | 70 | 18 | 24.57 | 15.63 | 8.16 | 0.0322 | 0.0322 | `[1,140,32,128] -> [1,140,32,256]` |
| Matmul | 70 | 24 | 16.01 | 10.19 | 5.41 | 0.0279 | 0.0279 | `[1,140,32,128] -> [1,140,32,128]` |

**1.0682 s/fold of idle core-seconds, and 1.0205 s of it survives the roof check.** Nothing
here is bandwidth-bound or compute-bound at the core count it currently gets; the 16-core
LayerNorm group runs at 21.5 % of the byte roof while holding 14.5 % of the grid. Full table
including every group in `out/step_rank_floor.json`.

## The step on the roofline

One diffusion step moves **3133.28 MB** and does **328.48 GFLOP**. Per-op, that is 7.9382 ms
against a measured 22.0163 ms of kernel time: a **2.773x** deficit, the same shape of problem
the campaign found in the pairformer block. Perfect re-spreading to 110 cores would take the
step to 16.6752 ms and **2.101x**.

So grid under-fill is **24.3 % of the diffusion step's own roof deficit**. The other three
quarters is the campaign's standing unexplained in-kernel time and is not a grid problem.

## Two instruments, one answer

| | tracy ops report | profiler API |
|---|---|---|
| available cores | 110 | 110 |
| programs per call | 1066 | 1066 (10663 over 10 calls, incl. 3 fence) |
| duration-weighted mean utilization | **0.75740** | **0.76936** |
| idle core-seconds/fold | 1.0682 | 1.1564 |

The API (`ttnn.ReadDeviceProfiler` + `get_all_programs_perf_data()`) never builds the tracy
host log, which is what OOM-killed the census on pc's 30 GB. It exposes only `core_count`,
`num_available_cores` and durations -- no op code, no shapes -- so it can answer the
utilization question but cannot rank offenders. Its 1.2-point higher mean is expected: the
only duration it exposes on this build is `DEVICE FW DURATION`, which carries per-program
firmware overhead on every assigned core, not `DEVICE KERNEL DURATION`. Their core-count
histograms match program for program: 16 -> 124, 64 -> 220, 70 -> 51, 80 -> 77, 88 -> 24,
90 -> 6, 94 -> 6, 110 -> 554 per call, on both.

Independently, re-running this report over the sibling row's own committed step capture (a
different grab, a different process, 3 reps) gives **0.757388** against this run's
**0.757404**.

## Instrument proof

The same report, run over the sibling's PairformerLayer capture on the same card, gives
**94.36 %**, which is `103.8 / 110` -- the figure `b2z-kernel-cycle-census` reported from its
own separately written code path. The tool reproduces a number it did not produce.

Two defects fixed in the inherited report code, both of which silently return a wrong answer
rather than failing:

1. `shape()` read `INPUT_0_W`, but the column is `INPUT_0_W_PAD[LOGICAL]`. Every shape in the
   sibling's tables is therefore `(0,0,0,0)`, which reads as a program-cache artifact and is
   actually a wrong key. Without the fix the one-cause finding above is invisible.
2. Cells are `padded[logical]`, e.g. `512[500]`, when the report is generated with
   `--no-op-info-cache`; `float()` throws and the helper returns its default of 0.

And one instrument limit worth carrying: on a long precursor the device's marker buffer wraps
and drops the *oldest* rows, so the opening fence can be gone while the run looks clean. 9936
rows survived here where 10 calls alone need 10 660. `find_region` now anchors on the closing
fence and **checks** the split by requiring every repetition to be the same op-code sequence;
only 3 of the 10 captured calls pass that check, and those 3 are what is reported.
