# The per-op launch floor on Blackhole: worth 0.594 s, not the 2.8 s the campaign expected

The floor of record takes `max(traffic, arithmetic)` per op and charges nothing for launching the
op. `roof-difftx-arith-efficiency` measured launch on Wormhole at **20.6 us fixed + 8.3 us per
block** and it was never re-measured on Blackhole, so the campaign carried a registered prediction:
the true floor lands **15.5-17.0 s** rather than 12.706 s, putting the 17.270 s fold at 90-100 % of
roof instead of 73.6 %, because **353,384 of its 465,664 op calls** are small enough for launch to
be the binding term.

Measured per op class on the part of record, that prediction is wrong by a factor of five.

| floor | s | of the 17.270 s fold | prize |
|---|---|---|---|
| committed, `max(traffic, arithmetic)` | 12.706 | 73.6 % | 4.564 s |
| + launch, clean intercepts only | **13.231** | 76.6 % | 4.039 s |
| + launch, every measured floor | **13.300** | 77.0 % | **3.970 s** |
| every op at what it costs run alone | 17.103 | 99.0 % | 0.167 s |

**The third term is +0.594 s, 4.7 % of the floor.** The fold moves from 73.6 % to 77.0 % of roof,
not to 90-100 %. **The campaign is not finished: 3.970 s is still open.** The last row is not a
floor and is read separately below, because it is the more interesting number.

## Why the prediction missed by 5x

**"465,664 op calls" is not 465,664 device programs.** It counts top-level `ttnn` calls in the
capture, and a third of them launch nothing:

    150,160   host-side metadata, or a python wrapper whose device child is already counted.
              `ttnn.Tensor.__getitem__` appears 12,776 times beside exactly 12,776 `ttnn.slice`
              calls, because it IS those calls. Charging it would be double counting.
     24,832   no output tensor allocated and not in place: nothing was launched.
      3,920   `ttnn.generic_op`, the fused kernels. No stock op to sweep, so charged zero.
     84,224   priced by the class ladder
    202,528   priced by its own (class, shape, K) ladder

Of the 287k calls that do run a program, **103,672 have launch as their binding term**, and those
hold **1.002 s** of the floor. The other 183k are already traffic or arithmetic bound, so pricing
their launch changes nothing.

**And the Wormhole constant does not carry.** On Blackhole under trace replay the fixed cost is
**4.13 us to 54.96 us depending on the class**, not one 20.6 us number: `to_memory_config` to L1 is
4.13 us, the eltwise family 6.19-6.22 us, `ttnn.linear` 11.41 us, `layer_norm` 14.38 us,
`nlp_create_qkv_heads` 40.37 us, `ttnn.softmax` 54.96 us. Applying 20.6 us flat would have
overcharged the fold's 83,728 eltwise-family calls by 3.3x and undercharged softmax by 2.7x.

## Two things that had to be got right, and one that is a footgun

**1. A synced reading puts host dispatch into a roofline.** The floor is published on a trace-replay
reading, `reps` enqueues captured once and replayed, so the number is device time with the host
removed. It matters exactly where the op is cheap:

| class | synced | trace replay | ratio |
|---|---|---|---|
| `add` | 9.72 us | 6.22 us | 1.56 |
| `multiply` | 9.54 us | 6.22 us | 1.53 |
| `concat` | 6.67 us | 4.86 us | 1.37 |
| `layer_norm` | 14.25 us | 14.38 us | 0.99 |
| `softmax` | 54.74 us | 54.96 us | 1.00 |
| `nlp_create_qkv_heads` | 40.15 us | 40.37 us | 0.99 |

For the cheap classes the synced reading is `max(host, device)` and the host half is the larger one,
so up to a third of it is dispatch a real fold hides under device time. For the expensive classes
the two agree to 1 %. Publishing the synced column would have inflated the term by roughly half on
the classes that dominate the call count.

**2. The launch floor is shape dependent, not class dependent.** One ladder per class, swept on a
single `[1, R, 768]` aspect ratio, is wrong by up to 1.79x at a **fixed tile count**:

| `ttnn.layer_norm`, same class, same op | tiles | measured | the class ladder predicts | ratio |
|---|---|---|---|---|
| `[1, 512, 384]` | 192 | 10.13 us | 15.05 us | 0.67 |
| `[1, 140, 32, 128]` | 560 | 12.07 us | 17.12 us | 0.71 |
| `[1, 512, 768]` | 384 | 15.93 us | 16.00 us | 1.00 |
| `[1, 512, 1536]` | 768 | 27.22 us | 18.44 us | 1.48 |
| `[1, 1024, 512, 64]` | 32768 | 395.60 us | 221.59 us | 1.79 |

The governing variable is the row **width**, the reduction length, which the class ladder holds
fixed at 768: it prices every narrower shape too high and every wider one too low. `layer_norm`
alone is 33,312 calls and 65 % of the whole term, so this is not a detail. The table is therefore
keyed `(class, output shape, K)` and the class ladder is only the fallback. 202,528 of the 286,752
keyable calls are priced by their own shape's ladder; the remaining 84,224 fall back.

**3. `ttnn.reshape` at `[1, 768, 512]` hangs the card.** Two independent ladder runs died on the
same entry, spinning a dispatch thread with no progress for minutes, and had to be killed by pid.
Everything before it in the same run is unaffected (each row writes out as it completes), which is
why the ladder is resumable and why 19 of 122 keys are measured rather than all of them.

## The number that is worth more than the floor

Price every op at **what it costs run alone on a quiet card** (the ladder read at the op's own tile
count, not extrapolated to zero) and the total is **17.103 s against a 17.270 s fold: 99.0 %.**

That is an over-reading and not a floor. An isolated arm carries its own dispatch and its own cache
state, the class-fallback rows carry the arm's shape rather than the fold's, and the 3,920
`generic_op` calls keep their analytic term. But it bounds where the remaining 3.970 s can come
from. The fold is running its ops at within 1 % of the speed those ops reach alone on an idle card,
so there is no pool of scheduling slack, dispatch bubbles or launch serialisation to recover.
**Closing any part of 3.970 s means making the kernels themselves faster or issuing fewer of them.**
That is consistent with what the campaign already found the hard way: per-class grid sizing, the
DRAM result write, the SDPA mask path, none of them an aggregate byte count.

## What is still unpriced, and the bound on it

The 3,920 `ttnn.generic_op` calls hold **4.084 s** of the floor and are charged **zero** launch,
because a fused kernel has no stock op to sweep. So 13.300 s is a lower bound on the three-term
floor. The miss is bounded: at the most expensive class floor measured anywhere in the fold
(`softmax`, 54.96 us) those 3,920 calls would add **0.215 s**, and at the median class floor
(6.56 us) **0.026 s**. Neither reopens the prediction.

103 of the 122 (class, shape, K) keys are not measured per shape and fall back to the class ladder.
They are the low-call-count tail: the 19 measured keys carry 202,528 of the 286,752 keyable calls.
`perf/roof_launch/fold_shapes.json` is ordered by calls, `shape_ladder.py --resume` picks up where
it stopped, so a quiet box can extend it without redoing anything.

## Session discipline

| table | reading | loadavg before / after | A/A |
|---|---|---|---|
| `launch_sweep_qb2c1.json` | synced, 156 points | 4.46 / 4.46 | cube 0.15 %, add 0.93 % |
| `launch_trace_qb2c1.json` | trace replay, the one the floor uses | 2.94 / 2.94 | cube 1.57 %, add 1.62 % |
| `shape_control_qb2c1.json` | trace replay | 1.30 / 1.36 | cube 0.62 % |
| `shape_ladder_k_qb2c1.json` | trace replay | 1.03 / not taken | not taken |

The shape ladder has **no closing A/A and no closing loadavg**, because the run was killed on the
`reshape` wedge rather than exiting. Its rows are quoted as measured; every one of them is a
four-point ladder whose fit r2 is in the json, and the five rows whose ladder is not monotone (all
five are `ttnn.linear`, r2 from 0.00 to 0.98) fall back to their cheapest measured point rather
than to a meaningless intercept. A third of the way through that run another worker started a capacity gate
on card 0 and the box went from loadavg 1 to 8, which is the other reason the ladder stopped where
it did: extending it under that load would produce rows worse than the fallback.

## Two readings, and why both are published

`launch_floor_us` per row is the ladder's y-intercept where the ladder is monotone and the fit is
clean, and its cheapest measured point where it is not. The intercept is launch and nothing else,
extrapolated to zero rows, so it is additive-independent of traffic and arithmetic and `max()` of
the three is still a floor. The cheapest-measured-point fallback is a valid lower bound on the op's
device time but it is not purely launch: `linear|1x512x3072` reads 38.42 us at an eighth of the rows
against 39.05 us at full size, so calling 38.42 us "launch" over-reads. **Strict** keeps only the
clean intercepts and charges zero for the rest, 13.231 s. **All floors** takes both, 13.300 s. The
two differ by 0.069 s, so nothing in the verdict turns on the choice.

## The control

`--launch-shapes` and `--launch` are off by default, and with them off the chain reproduces byte for
byte. Re-run on the same capture with these edits in the tree:

    python3 perf/roof_quiet/rejoin.py \
      --attrib  <artifacts>/roof-quiet-attrib-refold/attrib_quiet_512_qb2c0.json \
      --captures <artifacts>/roof-quiet-attrib-refold/captures \
      --roofs shape_roofs_qb2c3_shipped.json --time-stat trimmed --tag ctrl_nolaunch

All five outputs are identical to `perf/roof_quiet/out_shipped_trimmed/`, `true_floor.json` at
sha256 `8a20a82c8e1330bef94c62caa0fc2be6f25dd0f32ab905bb6f334e16301100e1` both sides. Adding
`--launch-shapes perf/roof_launch/shape_ladder_k_qb2c1.json --launch
perf/roof_launch/launch_trace_qb2c1.json --tag launch` gives `out_launch_trimmed/`.

## Reused, not rewritten

- `perf/roof_quiet/rejoin.py` runs the committed three-script chain; this row added a two-flag
  passthrough to it, the same shape as `roof-triatt-rate-fix`'s `--roofs` passthrough.
- `perf/roof_true/true_floor.py` is the join. The third term is a `max()` argument inside the
  existing `op_terms`, not a second join.
- `perf/roof_launch/op_census.py` calls `true_floor.setup`, `Join` and `op_terms` verbatim, so the
  op set, the byte column and the two existing terms are the committed ones. It emits
  `fold_shapes.json` from `true_floor.launch_key`, so the sweep and the join key on one definition.
- `perf/roof_launch/launch_sweep.py` is `roof-difftx`'s method (ladder per class, fit
  `fixed + per-block`, three readings) on the part of record; `shape_control.py` and
  `shape_ladder.py` import its `trace_point`, `fit`, `tiles` and `roofs`.
- `perf/roof_shape/weigh.py`'s class table is **unchanged**. The brief expected the term to go in
  there, but `weigh.py` prices a class per fold and the launch floor is per op call, so it belongs
  where the per-op `max()` already is.

## The campaign verdict this settles

The headline "86.7 % of roofline" was already superseded twice, by the quiet re-capture and by the
TriangleAttention rate fix, to 73.6 %. The launch term moves it to **77.0 %**, and the registered
prediction of 90-100 % is refuted. **The remaining 3.970 s is real headroom, not a measurement
artifact**, and the isolated-op reading says it is in the kernels rather than in the dispatch.
