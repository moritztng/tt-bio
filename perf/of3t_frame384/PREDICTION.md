# of3t-frame384 — pre-registration, written before the first run

Row `of3t-frame384`, branch `wk/of3t-frame384` off `origin/wk/of3t` at `f00130c43`. CPU only, no
card taken and no device opened. Committed before `ref_grad.py` was invoked once.

## What is being built

Two crop-384 local references, from the SAME captured boundary the device arms are driven from:

  * boundary `/home/ttuser/of3t_trunk043ref/boundary_n384.pt`
  * cotangent `/home/ttuser/of3t_gradients/cap/block47_boundary.pt`
  * producer `perf/of3t_trunkg043/ref_grad.py`, the same script that built the c64 pair
    (`perf/of3t_trunkg043/REF_F64_c64.json` -> `/home/ttuser/of3t_trunkg043/ref_f64_c64.pt`,
    `REF_BF16AUTO_c64.json` -> `ref_bf16auto_c64.pt`). Not a second producer.
  * arms `--policy f64` and `--policy bf16auto`, upstream 0.4.3 tree
    `/home/ttuser/of3t_trunk043ref/of3pkg043`, 48 blocks, `--crop 384`.

## Prediction 1 — the frame-matched ratio at 384

At crop 64 the frame-matched reading is ours **0.3833065668** against a floor of
**0.4007237405**, ratio **0.9565x** (`perf/of3t_apbback/REFAUDIT.json`). The cross-frame reading
on the same device tensors is **5.4144x**.

**I expect 384 to land near 0.9565x, inside an A26-style bar, and I am predicting a band rather
than a point:**

  * ours (shipped renorm arm at n384) against `ref_f64_n384`: **0.30 to 0.90**
  * the in-frame floor (`ref_bf16auto_n384` against `ref_f64_n384`): **0.30 to 0.90**
  * the ratio ours/floor: **0.70 to 1.30**, and inside `sqrt(2) * floor / norm_ratio`

The reasoning is one number. At c64 the same device tensors read 0.3833 against the capture's own
float64 and 1.7043 against the model float64 — 4.45x from the reference alone. At 384 our arm
reads **2.159527121735274** against the model float64 (`perf/of3t_ditmodel/TRUNK_D174.json`,
2,736 tensors, 5.828 % of the model's squared gradient norm). If the frame contributes the same
kind of factor at 384 as at 64, the frame-matched numerator lands near 0.5, and upstream's own
bf16 recipe on the same boundary should sit in the same place because both sides then share every
choice except the arithmetic width.

**What it would mean if the ratio is NOT inside the bar.** Then the c64 frame-matched reading does
not generalise in width, and the trunk has a real model-scope backward defect that the c64
comparison was too small to show. That is a result and it is reported as one: the deliverable is
the number, not a particular sign of it. The three candidate mechanisms, named now so the answer
is not retrodicted onto whichever one fits:

  1. an accumulation that grows with token count, which `of3t-trunkg043` already saw in DEPTH
     (0.94x to 3.81x the floor over blocks 44-47, 4.22x to 66.88x over blocks 0-11 at c64);
  2. a width-shaped term — `of3t-padshape` priced removing the width dependence as
     0.532795 (crop 384) -> 0.423374 (crop 64) at model scope, and that prize was already
     refuted as stated because its endpoint was cross-frame;
  3. the softmax backward `of3t-apbback` named, which recovers 51.55 % of block 47's error mass
     at crop 384 in the frame where it is measurable and 0.0 at model scope.

A ratio near 1.0 at 384 does not distinguish these; it says none of them is large enough to
matter against upstream's own bf16 recipe. A ratio well above the bar says at least one is.

## Prediction 2 — it does not fit without activation checkpointing

`perf/of3t_apbback/N384_OOM.json`: the same producer at `--policy f32 --crop 384` was SIGKILLed
by the OOM killer at **228.2 GB** anon-rss on qb2's 249 GB. float64 doubles every saved
activation, so plain float64 at 384 needs on the order of **450-500 GB** and qb1's 503 GB total
(441 GB available at the start of this pass) does not leave room for it. I expect to need
per-block activation checkpointing, and I pre-register the control that makes it admissible:

  * plain c64 float64 on qb1 must reproduce qb2's `ref_f64_c64.pt` — if it does not, the
    interpreter is part of the frame and the cross-host difference is quoted, not hidden;
  * checkpointed c64 float64 must be **bit-identical** to plain c64 float64 in the same process
    family, on all 2,736 tensors, max absolute difference exactly 0.0, and the same for
    `bf16auto`. Recompute under the same autocast state is deterministic, so this is a claim that
    can fail and therefore a control.

Both sides are CPU float64. No device is involved in either reference, so there is no hardware to
attribute a bit-identity claim to (D155's frozen `BLK47_VALIDATION.json` entry).

Peak RSS and wall clock are reported for both n384 arms whatever they come out at. If float64 at
384 does not fit even checkpointed, the row says what it needs and does not silently crop.

## Prediction 3 — the frame itself

`ref_f64_n384` against `grads_f64_043.pt` (sha256 `1d4ea922...`, digest checked before loading)
is the frame, measured for the record. At block scope and crop 384 the capture's own float64 is
**0.865188** from the model float64 at cos 0.5298 (`of3t-apbback` COST). I expect the 48-block
crop-384 version of that distance to be **large — 0.5 or more** — because it is the same
comparison at a wider scope, and that is the quantity that makes every cross-frame ratio in this
campaign unreadable rather than merely pessimistic.

## Bars, fixed here

  * A16 zero baseline, measured in this row's own scorer, not inherited.
  * A/A: the same tensor file scored against itself must read exactly 0, before any number.
  * A26-style reachable bar recomputed for THIS scope from THIS floor, `sqrt(2) * floor / r`,
    the same arithmetic `of3t-modelboundary` uses (0.10592054683439786 -> 0.14735268326440318
    via r = 1.0165697057473722). Not borrowed from another scope.
  * every model-scope figure carries its crop (D180).
  * the charter clause is PROPOSED, never edited. The orchestrator repoints conditions.
