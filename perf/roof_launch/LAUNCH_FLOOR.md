# The per-op launch floor on Blackhole: worth 0.487 s, not the 2.8 s the campaign expected

The floor of record takes `max(traffic, arithmetic)` per op and charges nothing for launching the
op. `roof-difftx-arith-efficiency` measured launch on Wormhole at **20.6 us fixed + 8.3 us per
block** and it was never re-measured on Blackhole, so the campaign carried a registered prediction:
the true floor lands **15.5-17.0 s** rather than 12.706 s, putting the 17.270 s fold at 90-100 % of
roof instead of 73.6 %, because **353,384 of its 465,664 op calls** are small enough for launch to
be the binding term.

Measured per op class on the part of record, that prediction is wrong by a factor of six.

| floor | s | of the 17.270 s fold | prize |
|---|---|---|---|
| committed, `max(traffic, arithmetic)` | 12.706 | 73.6 % | 4.564 s |
| + launch, clean intercepts only | **13.152** | 76.2 % | 4.118 s |
| + launch, every measured floor | **13.193** | 76.4 % | **4.077 s** |
| every op at what it costs run alone | 17.821 | 103.2 % | none |

**The third term is +0.487 s, 3.7 % of the floor.** The fold moves from 73.6 % to 76.4 % of roof,
not to 90-100 %. **The campaign is not finished: 4.077 s is still open.** The last row is not a
floor and is read separately below, because it is the more interesting number.

Both launch rows are the second pass, with 111 of the 122 (class, shape, K) keys measured per
shape rather than the first pass's 19. Wider coverage moved the floor **down** 0.107 s, from
13.300 s, and why it moved down rather than up is the one new finding here.

## Why the prediction missed by 5x

**"465,664 op calls" is not 465,664 device programs.** It counts top-level `ttnn` calls in the
capture, and a third of them launch nothing:

    150,160   host-side metadata, or a python wrapper whose device child is already counted.
              `ttnn.Tensor.__getitem__` appears 12,776 times beside exactly 12,776 `ttnn.slice`
              calls, because it IS those calls. Charging it would be double counting.
     24,832   no output tensor allocated and not in place: nothing was launched.
      3,920   `ttnn.generic_op`, the fused kernels. No stock op to sweep, so charged zero.
      6,664   priced by the class ladder
    280,088   priced by its own (class, shape, K) ladder

Of the 287k calls that do run a program, **105,728 have launch as their binding term**, and those
hold **0.881 s** of the floor. The other 181k are already traffic or arithmetic bound, so pricing
their launch changes nothing.

**And the Wormhole constant does not carry.** On Blackhole under trace replay the fixed cost is
**4.13 us to 54.96 us depending on the class**, not one 20.6 us number: `to_memory_config` to L1 is
4.13 us, the eltwise family 6.19-6.22 us, `ttnn.linear` 11.41 us, `layer_norm` 14.38 us,
`nlp_create_qkv_heads` 40.37 us, `ttnn.softmax` 54.96 us. Applying 20.6 us flat would have
overcharged the fold's 83,728 eltwise-family calls by 3.3x and undercharged softmax by 2.7x.
Per shape the spread is wider again: 1.01 us for `concat|1x800x512x64` up to 586.46 us for
`linear|512x512x128|K1024`, median 5.99 us across the 111 keys, call-weighted median 6.09 us. The
rows above ~100 us are all 16-call giants on quarter-gigabyte tensors whose own traffic term is
milliseconds, so none of them binds.

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
keyed `(class, output shape, K)` and the class ladder is only the fallback. 280,088 of the 286,752
keyable calls are priced by their own shape's ladder. Of the 6,664 that fall back, 5,080 are
`ttnn.reshape`, whose class arm happens to be built at the fold's own output shape anyway (next
section), so only **1,584 calls, 0.6 %, are priced on a shape that is not their own**.

**3. An identity `ttnn.reshape` hangs the card, and the output shape is not what does it.** Two
independent first-pass ladder runs died on the `[1, 768, 512]` entry, spinning a dispatch thread
with no progress for minutes, and had to be killed by pid. Everything before it in the same run is
unaffected (each row writes out as it completes), which is why the ladder is resumable.

The second pass refuses `ttnn.reshape` outright rather than retrying it, and that costs the table
nothing, because the committed class sweep already prices the fold's exact output shape without
hanging. `launch_sweep.py`'s reshape arm is `(1, 16, 48, R) -> (1, 768, R)`, and at R=512 that is
`[1, 768, 512]`: `launch_trace_qb2c1.json` has it at **22.81 us**, with R=1024 beside it at 35.78
us and no refusals in the file. What the shape ladder built instead was `(1, 768, 512) ->
(1, 768, 512)`, an **identity** reshape, because its generic builder derives the input from the
output's own element count. So the wedge is the identity path, not the shape, and the hang is a
builder defect on this side as much as a ttnn defect on the other. Not re-run to confirm: two
wedges is enough, and the class fallback for those 5,080 calls is keyed on the fold's own shape
family, which is what a per-shape row would have measured anyway.

## Why wider shape coverage moved the floor down

A partial cost table does not bias a floor in a random direction. It biases it **upward**, and the
second session is a clean demonstration: 43 of the 92 newly measured keys moved the floor at all,
25 of them down by 0.133 s and 18 up by 0.025 s, net **-0.107 s**.

Taken as a raw launch term the class fallback was in fact *under*-pricing those 92 keys by 0.090 s
net (it overcharged 58 of them by 0.244 s and undercharged 34 by 0.333 s). That is not a
contradiction, because launch only reaches the floor where it beats `max(traffic, arithmetic)`, and
its largest errors were on ops that are nowhere near launch-bound:

| key | class ladder | its own shape | `max(traffic, arithmetic)` | effect on the floor |
|---|---|---|---|---|
| `matmul\|1x128x512x512\|K512` | 11.39 us | 143.34 us | 2366.17 us | none |
| `linear\|1x512x512x128\|K128` | 11.41 us | 89.63 us | 544.13 us | none |
| `layer_norm_w\|1x512x512x128` | 22.83 us | 53.14 us | 316.07 us | none |
| `layer_norm\|1x140x32x128` | 14.38 us | 5.63 us | 0.00 us | -0.0210 s |
| `layer_norm_w\|1x16x512x64` | 22.83 us | 4.62 us | 2.49 us | -0.0186 s |
| `nlp_concat_heads\|140x1x32x128` | 14.57 us | 2.49 us | 5.40 us | -0.0110 s |
| `to_layout\|1x1x4128x128` | 12.89 us | 3.73 us | 4.98 us | -0.0095 s |
| `to_memory_config_l1\|1x140x32x128` | 4.13 us | 6.09 us | 2.70 us | +0.0094 s |

A 12.6x error on `matmul` is worth exactly zero seconds, because that op's arithmetic term is
16x its own launch floor either way. The keys that do move the floor are the cheap narrow ones, and
there a ladder swept at one width, 768, prices narrow shapes too high: it is the overcharges that
leak into the floor, and the undercharges mostly die inside the `max()`. `ttnn.to_layout` is the
clearest single case. At 12.89 us per class it held 0.031 s of the floor and all 2,400 of its calls
read as launch-bound; priced per shape at 3.6-3.7 us, **not one `to_layout` call in the fold is
launch-bound any more.**

The general form, and the reason this pass was worth a card hour: any floor published off a
partially-keyed cost table is an over-estimate, and correcting it does not need the whole tail,
only the cheap end of it.

## The number that is worth more than the floor

Price every op at **what it costs run alone on a quiet card** (the ladder read at the op's own tile
count, not extrapolated to zero) and the total is **17.821 s against a 17.270 s fold: 103.2 %.**
The first pass read 99.0 % here on a quarter of the shape coverage; measuring the tail per shape
pushed it past the fold.

That is an over-reading and not a floor. An isolated arm carries its own dispatch and its own cache
state, `_interp` runs off the top slope for an op above its ladder's last point, and the 3,920
`generic_op` calls keep their analytic term. Over 100 % does not mean the fold beats physics, it
means this reading has run out of headroom as a bound: the fold executes its ops at or faster than
those same ops reach alone on an idle card, so there is no pool of scheduling slack, dispatch
bubbles or launch serialisation to recover, and the conclusion the first pass drew at 99.0 % holds
with more margin rather than less.
**Closing any part of 4.077 s means making the kernels themselves faster or issuing fewer of them.**
That is consistent with what the campaign already found the hard way: per-class grid sizing, the
DRAM result write, the SDPA mask path, none of them an aggregate byte count.

## What is still unpriced, and the bound on it

The 3,920 `ttnn.generic_op` calls hold **4.084 s** of the floor and are charged **zero** launch,
because a fused kernel has no stock op to sweep. So 13.193 s is a lower bound on the three-term
floor. The miss is bounded: these are fused TriangleAttention and TriangleMultiplication kernels,
so the stock ops they stand in for are `linear` and `sdpa`, measured at 4.8-22.0 us per shape in
this table. At the dearest of those (`sdpa|1x16x512x64`, 22.01 us) the 3,920 calls would add
**0.086 s**, and at the table's call-weighted median (6.09 us) **0.024 s**. Neither reopens the
prediction.

11 of the 122 keys are still not measured per shape, 6,664 calls, 2.3 %. Three are `ttnn.reshape`
(5,080 calls, refused, and already class-priced at their own shape). Six have a row axis of one or
two tiles, so scaling it down 8x/4x/2x gives fewer than three distinct ladder points and no fit is
possible at all: `linear|1x256`, `multiply|1x256`, `cos|1x256`, `layer_norm_w|1x256`,
`linear|1x768`, `slice|32x64`. One is `matmul|1x768x512` with no reduction length in the census row
to build the operand from, and one is `slice|64x32` with two points. Together they are 1,584 calls
priced on a foreign shape, and the class ladder's error on 1,584 calls is at most a few
milliseconds. `perf/roof_launch/fold_shapes.json` is ordered by calls and `shape_ladder.py
--resume` picks up where it stopped, so this table can still be extended, but there is nothing left
in it worth a card hour.

## Session discipline

| table | reading | loadavg before / after | A/A |
|---|---|---|---|
| `launch_sweep_qb2c1.json` | synced, 156 points | 4.46 / 4.46 | cube 0.15 %, add 0.93 % |
| `launch_trace_qb2c1.json` | trace replay, the one the floor uses | 2.94 / 2.94 | cube 1.57 %, add 1.62 % |
| `shape_control_qb2c1.json` | trace replay | 1.30 / 1.36 | cube 0.62 % |
| `shape_ladder_k_qb2c1.json`, keys 1-19 | trace replay | 1.03 / not taken | not taken |
| `shape_ladder_k_qb2c1.json`, keys 20-111 | trace replay | 0.61 / 2.23 | cube 1.53 % |

The first 19 keys have **no closing A/A and no closing loadavg**, because that run was killed on
the `reshape` wedge rather than exiting; a third of the way through it another worker started a
capacity gate on card 0 and the box went from loadavg 1 to 8, which is why it stopped where it did.
The second session closed properly: `--budget-s` stops the loop and takes the after-roofs, and the
two sessions' opening cube agrees to 2.9 % (113.39 against 116.63 TFLOP/s) with a 1.53 % A/A inside
the second one. It ran on card 1 against a single sibling job on card 0, loadavg 0.61 at the start
and 2.23 at the end, its own process included.

Every row is a four-point ladder whose fit r2 is in the json. 86 of the 111 take the fit intercept;
the 25 whose ladder is not monotone or whose intercept is negative fall back to their cheapest
measured point, which by construction can never exceed the op's own measured device time.

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
sha256 `8a20a82c8e1330bef94c62caa0fc2be6f25dd0f32ab905bb6f334e16301100e1` both sides, re-verified
after the second ladder session. Adding `--launch-shapes
perf/roof_launch/shape_ladder_k_qb2c1.json --launch perf/roof_launch/launch_trace_qb2c1.json
--tag launch111` gives `out_launch111_trimmed/`, the 13.193 s row.

The 13.300 s row is still reproducible: run the same command with the 19-key table
(`git show 82b161bae:perf/roof_launch/shape_ladder_k_qb2c1.json`) and every number in the
`launch` block comes back identical to `out_launch_trimmed/`, down to the last digit, with only the
recorded table path differing. So the 0.107 s is the new rows and nothing else.

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
  `shape_ladder.py` import its `trace_point`, `fit`, `tiles` and `roofs`. The second pass added
  nine builders to `shape_ladder.py` for the classes it could not construct at an arbitrary shape
  (`slice`, `sdpa`, `pad`, `concat`, `transpose`, `softmax`, `chunk`, `cos`, `to_memory_config` to
  L1), plus `--max-tiles` and `--budget-s`. The join, the census and the class table are untouched.
- `perf/roof_shape/weigh.py`'s class table is **unchanged**. The brief expected the term to go in
  there, but `weigh.py` prices a class per fold and the launch floor is per op call, so it belongs
  where the per-op `max()` already is.

## The campaign verdict this settles

The headline "86.7 % of roofline" was already superseded twice, by the quiet re-capture and by the
TriangleAttention rate fix, to 73.6 %. The launch term moves it to **76.4 %**, and the registered
prediction of 90-100 % is refuted. **The remaining 4.077 s is real headroom, not a measurement
artifact**, and the isolated-op reading says it is in the kernels rather than in the dispatch.
