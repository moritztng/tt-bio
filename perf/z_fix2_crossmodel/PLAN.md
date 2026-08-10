# protenix-trunk--z-fix2-crossmodel — does FIX-2 close openfold3 and boltz2 at 512 aa?

TASK TYPE: VERIFY/BENCHMARK (Phase 3, cross-model correctness check on a shared-code fix) |
PLAYBOOKS loaded: VERIFY + ACCELERATE | memories read: protenix-v2-448aa-l1-cb-clash-cc39a867d,
tt-bio-l1-budget-batch-blind-defect-class, perfwar-sprint-context, perfwar-harness-leaked-buffer-and-capture-superset-traps,
release-gate-cumulative-catches-shape-class-escape, donecheck-hostspecific-path-unsatisfiable-on-remote-host,
tt-bio-worktree-run-recipe, verify-against-live-state-before-concluding, model-merge-approval-gate

**STATUS: PLAN ONLY, NO MEASUREMENTS TAKEN.** This pass is the opus5 planning pass. Every number
marked `TBD-<arm>` is a placeholder for the execution pass to overwrite with a measurement. The
predictions in §3 are registered here **before the device is opened**, which is the point of writing
them down. Nothing in this document is merged and nothing here is production: it is an experiment on
throwaway arms built in a private worktree.

Card 2 on qb2 (`ttuser@tt-quietbox2`), ttnn **0.68.0**, 11x10 grid = 110 cores. qb1 runs 0.67.4, so
every wall-clock second below is a ratio and not a campaign absolute. This leg delivers **no gain** in
ms/fold; its product is a verdict (folds / does not fold) and a byte count, both load-independent.
`TT_BIO_REBLOCK_PERMUTE` is left at its **default, which is OFF** (`reblock_permute.py:288`,
`_ENABLED = os.environ.get("TT_BIO_REBLOCK_PERMUTE", "0") == "1"`). Card 1 carries
`z-permute-flip-land`'s detached campaign and is not touched.

---

## 1. Three corrections to the brief, verified this pass, that change the work

### 1.1 Both fixes are already on `origin/main`. The brief's arm 1 is stale.

`origin/main` is `211e7791` today and `c06bd76c` ("merge: protenix-trunk--z-crashband-fix — close
[385,506]aa L1 CB clash crash band", 2026-08-10 16:02:46 +0200) is an ancestor of it. Verified by
`git merge-base --is-ancestor` for both commits and by grep:

```
5a207fee (FIX-2)  ancestor of origin/main: YES   origin/main:tt_bio/tenstorrent.py:425 _BMM_CFG_REFUSED
142e0109 (FIX-D)  ancestor of origin/main: YES   origin/main:tt_bio/tenstorrent.py:2268 "The normed pair tensor is dead…"
```

`z-permute-flip-land` measured the openfold3 512 aa failure against `origin/main` at **`6fa5a701`**,
and says so in its own doc: *"on `origin/main` at `6fa5a701`: no crash-band fix and no esmfold2
CB-clash fix has landed."* `6fa5a701` predates `c06bd76c`. So the brief's premise is sound for the
main that existed when it was written, and stale for the main that exists now.

**Consequence.** "`origin/main` must FAIL" is not a runnable arm. The baseline has to be built by
reverting, and today's `origin/main` is the FIX-2 + FIX-D arm. The four arms are re-defined in §4.

### 1.2 The brief's predicate arithmetic for openfold3 double-counts. The corrected margin is 186 880 B.

The brief adds the reported address to the reported circular-buffer end: *"808 960 + 995 840 =
1 804 800 … over by 343 040"*. The address is measured from the bottom of the bank and `held` is its
complement, so adding them counts the same region twice. The identity is visible in
`z-crashband-fix`'s own table and it is exact, not approximate:

| N | reported "L1 buffer allocated at" | 1 461 760 − address | that leg's separately measured held/bank |
|---:|---:|---:|---:|
| 385 | 862 208 | **599 552** | 599 552 |
| 448 | 681 984 | **779 776** | 779 776 |

So `held_per_bank = bank − reported_address`, and **the predicate can be read straight off the throw
with no allocator instrumentation at all.** For openfold3 and boltz2 at 512 aa, using the qb1 bank
value pending its re-measurement on card 2:

```
held_per_bank = 1 461 760 − 808 960 = 652 800
held + CB     =   652 800 + 995 840 = 1 648 640   >   1 461 760      the predicate fires
excess                                = 186 880 B
```

Two independent framings give the same number, which is the check that the reading is right: the
circular-buffer region occupies `[0, 995 840)` and the lowest L1 buffer starts at 808 960, so the two
regions **overlap by 995 840 − 808 960 = 186 880 B** — identical to the excess above. The brief's
343 040 is not a byte count of anything.

### 1.3 An empty `_BMM_CFG_REFUSED` is the EXPECTED result in the both-fixes arm, and is not the A/A trap.

`z-crashband-fix` measured this directly on protenix: *"`_BMM_CFG_REFUSED` is empty at every size,
i.e. with FIX-D in, FIX-2 never has to fire — it is a backstop, not the mechanism."* FIX-D removes the
co-residency, so the tuned program config still fits and nothing is ever refused. The positive control
the brief asks for therefore lives **only in the FIX-2-only arm**, and a reader who demands a
non-empty `_BMM_CFG_REFUSED` from today's `origin/main` will be chasing a result that should not
exist. Both expectations are written into the arm table in §4 so the execution pass cannot confuse
them.

---

## 2. The throwing op is identified from arithmetic alone, before any run

`z-permute-flip-land`'s artifact `perf/z_flip_land/sweep_openfold3_qb2c1.json` records, for
`perf/size512/fixtures/cdk2x2_512.yaml`, verbatim:

```
TT_THROW @ /project/tt_metal/impl/program/program.cpp:1052: tt::exception
info:
Statically allocated circular buffers in program 2561 clash with L1 buffers on core range
[(x=0,y=0) - (x=2,y=9)]. L1 buffer allocated at 808960 and static circular buffer region
ends at 995840
```

byte-identical with the `reblock_permute` kernel ON and OFF, and boltz2 records the same two
addresses and the same core range in `sweep_boltz2_qb2c1.json`. The program id is a per-session
counter and carries no information across processes; the core range and the two addresses do.

**`995 840` and `[(0,0)-(2,9)]` are reproduced exactly by `_batched_matmul_search`
(`tenstorrent.py:338`) and nothing else in the file produces a
`MatmulMultiCoreReuseProgramConfig`.** Re-deriving the crash-band leg's closed form on this grid,
with `Nt = 16` (pair N=512), `block_w = 1` (head_dim 24 pads to 32, so `k_tiles = 1` is odd and
`_batched_matmul_block_w` returns 1 on its odd branch), `cores = 110`:

| candidate `p` | legal? | blocks `= 16·16/p` | `CB(p) = 2(p+16)·1·2048 + p·16·6144` | `+ 111 104` overhead |
|---:|---|---:|---:|---:|
| 4 | yes | 64 | 475 136 | 586 240 |
| **8** | **yes, chosen** | **32** | **884 736** | **995 840** |
| 16 | yes (`p == m_tiles`) | 16 | 1 703 936 | 1 815 040 |

The search takes `max(saturating)`, the fewest blocks that still reach
`_BATCHED_MATMUL_SATURATION_BLOCKS = 32`, so `p = 8`, and `CB + 111 104 = 995 840` **to the byte**.
`cores = batch·m_tiles/p = 16·16/8 = 32`, which tt-metal's column-major fill prints as the first three
whole columns of an 11x10 grid, `[(x=0,y=0) - (x=2,y=9)]`. Both halves of the fingerprint fall out of
the model.

Two things follow that matter more than the identification itself:

- **The 111 104 B fixed per-core ttnn overhead constant, measured on qb1's 13x10 grid at 0.67.4,
  reproduces to the byte on qb2's 11x10 grid at 0.68.0.** It is neither grid-dependent nor
  version-dependent across those two points. That is a free validation of the crash-band model on a
  grid and a wheel it was not fitted to, and it is worth recording separately from this leg's verdict.
- **`batch = 16`, `m_tiles = n_tiles = 16`, `k_tiles = 1` is the `AttentionPairBias` q@kᵀ class** —
  `[1,16,512,24] @ [1,16,24,512]`, 16 heads of width 24 over 512 tokens. It is the same class, the same
  operand shapes and the same chosen `p` as protenix's `Nt = 16` row. The sibling call four lines down,
  `o = batched_matmul(probs, v)`, has `n_tiles = 1` and models to `348 672 B`, so it is not the thrower.

### The roofline placement, so the reader knows what kind of defect this is

The only roof this leg needs is a **capacity** roof and it is measured on card 2 rather than
inherited: `ttnn.get_max_worker_l1_unreserved_size()` = TBD-bank B per core (qb1 at 0.67.4 reports
1 461 760; the predicate is re-evaluated with whatever card 2 reports and the qb1 figure is used only
as the cross-check). For the throwing call, with k padded 24→32:

```
FLOPs = 2 · 16 · 512 · 512 · 32 = 268 435 456
bytes = 524 288 (q) + 524 288 (kᵀ) + 8 388 608 (out, bf16) = 9 437 184
arithmetic intensity = 28.4 FLOP/byte
```

against the machine balance of 338 FLOP/byte, so this op sits **11.9x on the memory side** and it is
**memory-bound**: the binding roof is the DRAM write roof, because the output is 8 388 608 of the
9 437 184 bytes it moves. That placement is here to say what this defect is *not*. Nothing about the
clash is a bandwidth or an occupancy problem, and the limiter is neither roof: the op never runs. It
is a static circular-buffer placement failure raised by `ProgramImpl::validate_circular_buffer_region`
at program compile, before any NOC transaction is issued, because `_batched_matmul_search` sizes its
CB budget against `get_max_worker_l1_unreserved_size()` on an **idle** device and cannot see what the
live block already holds. Core occupancy is a side-effect of the same choice and not a cause: `p = 8`
engages 32 of the grid's 110 cores, and the `p = 4` config the search declined would have engaged 64
while asking for 409 600 B less of circular buffer. **Recorded, not chased** (charter §1): preferring
the smallest CB among saturating configs is a third possible repair, it is out of this leg's scope, and
the hard limits forbid writing it here.

This is the same defect class as `tt-bio-l1-budget-batch-blind-defect-class`: an L1 budget helper that
prices only what it can see at plan time throws the first time a caller arrives with a live tensor it
did not price.

---

## 3. Predictions, registered before the device is opened

| # | prediction | expected | confidence | why |
|---|---|---|---:|---|
| **P1** | **FIX-2 alone closes openfold3 and boltz2 at 512 aa, with FIX-D reverted** | **HOLDS** | high | §2 puts the throw inside the shared `batched_matmul` that FIX-2 wraps, and protenix's own `Nt = 16` case, which FIX-2 closed on its own, held **1 039 872** B/bank against these models' **652 800** — strictly the harder case of the two |
| **P2** | FIX-D alone ALSO closes them | HOLDS | moderate | 652 800 B/bank is close to `[1,512,512,128]` bf16 spread over 110 banks (610 304 B/bank at 2048 B pages), i.e. the h=1.5 normed pair tensor with openfold3's C=128 in place of protenix's C=256, and openfold3 reaches `tenstorrent.Pairformer` |
| **P3** | today's `origin/main`, carrying both, already folds both models at 512 aa | HOLDS | high | it is P1 or P2 or both, and either suffices |
| **P4** | every arm that folds is **bit-exact** against every other arm, same plDDT to six decimals and same CIF sha256 | HOLDS | high | the throwing class has `k_tiles = 1`, a single K block, so no planner has any K-blocking freedom to regroup the accumulation and `packer_l1_acc` cannot differ. This is a stronger statement than the brief's "recomputes the same `in0_block_w`": there is only one width to compute |
| **P5** | opendde at 512 aa is untouched and its `_BMM_CFG_REFUSED` is empty | HOLDS | high | it folded 512 aa clean before FIX-2 existed, so no shape class was ever refused, and FIX-2 is inert unless a throw occurs |
| **P6** | in the both-fixes arm `_BMM_CFG_REFUSED` is **empty**, and non-empty only in the FIX-2-only arm | HOLDS | high | §1.3 |

**The failure mode that would make this leg more valuable than success**, and the one to look for: P1
falsified with a *different* core range or a *different* CB end than 995 840. That would mean a second
site throws once the first is caught, and the predicate has a term nobody has found. If that happens,
the deliverable is the new signature plus its `held`/CB evaluation, and **not** a repair by any other
route.

Settled after measurement: **TBD-verdict** (each of P1 to P6 gets CONFIRMED or FALSIFIED with the row
that settled it, and P1's verdict is this leg's headline).

---

## 4. The four arms, re-defined so they are runnable on today's main

Arms are working-tree-only reverse patches of the two commits, applied on top of `origin/main`, one
process per arm. **They are not toggled in-process:** doing that means reimplementing
`batched_matmul` and `AttentionPairBias.__call__` inside the harness, and a reimplementation that
drifts from the real function is exactly the control-fidelity failure this org has already paid for
twice. The brief's "four arms in one process" is honoured where it actually buys something, which is
device open/close churn: **every (model, target) pair runs inside one process per arm.**

Verified this pass, on `211e7791` with a clean tree: `git apply --check -R` succeeds for the FIX-D
hunk alone, the FIX-2 hunk alone, and both together. No conflict, no fuzz.

| arm | working tree | folds? | `_BMM_CFG_REFUSED` after 512 aa | role |
|---|---|---|---|---|
| **A** | both reverted | **must FAIL** | empty (FIX-2 absent) | reproduces the throw verbatim on card 2, and yields `held` and CB from the message |
| **B** | FIX-D reverted | **the headline** | **must be NON-EMPTY**, one key | the positive control that FIX-2 fired |
| **C** | `origin/main`, unmodified | expect folds | **expect EMPTY**, and that is correct (§1.3) | what a user gets today |
| **D** | FIX-2 reverted | expect folds if P2 | empty | separates FIX-D from FIX-2; **mandatory, not optional**, because P2 says both can close it and attribution needs it |

**Arm identity is fingerprinted in every JSON row, not trusted from the patch.** The harness records

```python
fix2_present = "_BMM_CFG_REFUSED" in inspect.getsource(T.batched_matmul)
fixd_present = "The normed pair tensor is dead" in inspect.getsource(T.AttentionPairBias.__call__)
```

so a mislabelled arm is impossible rather than merely unlikely. An arm whose fingerprint does not
match its label is discarded and re-run.

---

## 5. Exactly what to run, in order

### 5.0 Preconditions, all verified this pass

```
worktree   /home/ttuser/.coworker/wt/protenix-trunk--z-fix2-crossmodel   HEAD 211e7791 == origin/main, tree clean
PY         /home/ttuser/tt-bio-dev/env/bin/python3
targets    perf/size512/fixtures/cdk2x2_298.yaml, cdk2x2_512.yaml        both already on origin/main
paths      /home/ttuser/ypx_msa  /home/ttuser/w6_gate_msa  /home/ttuser/esm  /home/ttuser/of3-weights/of3-p2-155k.pt
card       /dev/tenstorrent/2 free (fuser empty); card 1 held by 443538/443655, do not touch
```

Environment for every device command, verbatim:

```sh
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-fix2-crossmodel
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT"
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-fix2-crossmodel
export PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export RELEASE_GATE_MSA_DIR=/home/ttuser/w6_gate_msa
```

`PYTHONPATH=$WT` is not optional: the env installs `tt_bio` editable against the **shared** checkout,
so a script run without it silently imports stale code. `census_sweep.py` already asserts
`Path(T.__file__).is_relative_to(REPO)` and that assert must stay.

### 5.1 Build the harness (no device, ~30 min)

Do not author a folding harness. `perf/z_flip_land/census_sweep.py` on
`origin/wk/protenix-trunk--z-permute-flip-land` is 308 lines, already carries the p300 single-chip
mesh-descriptor fix, the boltz2 `conf_kwargs` config injection whose re-derivation is the single
biggest avoidable risk in this family of legs, `seed_msa_cache`, per-target folds off one model load,
CIF hashing and the `_L1_OUT_REFUSED` tally.

```sh
mkdir -p perf/z_fix2_crossmodel
git show origin/wk/protenix-trunk--z-permute-flip-land:perf/z_flip_land/census_sweep.py \
    > perf/z_fix2_crossmodel/arm_sweep.py
git show 142e0109 -- tt_bio/tenstorrent.py > perf/z_fix2_crossmodel/fixd.patch
git show 5a207fee -- tt_bio/tenstorrent.py > perf/z_fix2_crossmodel/fix2.patch
```

Then make exactly these seven edits to `arm_sweep.py`, and no others:

1. **`--arm {A,B,C,D}` argument**, recorded in the header and in every row.
2. **Arm fingerprint** (§4), computed after `import tt_bio.tenstorrent as T` and written into the
   header *and* every row. Assert it matches `--arm`; abort with a non-zero exit if it does not.
3. **Full traceback.** `err = repr(e)[:600]` becomes `err = repr(e)` plus a separate
   `err_tb = traceback.format_exc()` field, untruncated. Nobody has ever seen the Python frames for
   the openfold3 throw; they are what confirms the call site named in §2 rather than derived.
4. **Predicate fields**, parsed out of the throw text with
   `r"core range \[(.*?)\].*?allocated at (\d+).*?ends at (\d+)"`:
   `core_range`, `l1_buffer_addr`, `cb_region_end`, `held_per_bank = bank - l1_buffer_addr`,
   `sum_vs_bank`, `over_by`. `bank` comes from the header's measured
   `ttnn.get_max_worker_l1_unreserved_size()`.
5. **`_BMM_CFG_REFUSED` tally**, cleared before each fold alongside the existing
   `T._L1_OUT_REFUSED.clear()`: `bmm_cfg_refused_n` and the full sorted key list. This is the leg's
   positive control and it is the one field that must never be dropped.
6. **A `ttnn.matmul` wrapper**, installed before the model is built, that records
   `(operand shapes, dtype, str(e))` for every raise and re-raises unchanged. In arm B the throw is
   caught inside FIX-2 and never reaches the fold's own `except`, so **this wrapper is the only way to
   read `held` and the CB end in an arm that passes.** Verify it is behaviour-neutral by checking arm
   C's plDDT against arm A's at 298 aa.
7. **Full sha256, and default-off.** `hashlib.sha256(...).hexdigest()[:16]` becomes the full digest
   (the brief asks for sha256, and 16 hex is a truncation). Call `fold(target, a3m, on=False)` only,
   never `on=True`: `z-permute-flip-land`'s artifacts were taken with the `reblock_permute` kernel ON,
   which is **not** the default, so its plDDTs and shas are not usable as this leg's control and this
   leg takes its own.

Delete the `--control` second-fold path or leave it unused. It re-folds with the kernel flipped, which
is a different question from this one.

### 5.2 The isolated bit-exactness probe (~3 min, card 2, run this FIRST)

`perf/z_fix2_crossmodel/bmm_equal_probe.py`, before any fold, because it settles P4 in seconds and it
is the parity evidence with the fewest confounders:

```python
q  = ttnn.from_torch(tq, ...)   # [1,16,512,24] bf16 DRAM interleaved, torch.manual_seed(0)
kt = ttnn.from_torch(tk, ...)   # [1,16,24,512] bf16 DRAM interleaved
cfg = T._batched_matmul_config(16, 16, 1, 16, 2)                       # must come back p=8
a = ttnn.matmul(q, kt, compute_kernel_config=ckc, program_config=cfg)  # the tuned path
b = ttnn.matmul(q, kt, compute_kernel_config=ckc)                      # FIX-2's fallback path
assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
```

On an idle device both succeed, so this compares the two numerics FIX-2 chooses between with nothing
else changing. Record `cfg.per_core_M`, `cfg.in0_block_w`, the modelled `CB + 111 104`, and the
measured bank. Predicted `torch.equal` **True**, for the §3 P4 reason. A False here is a finding that
outranks everything else in this leg and it changes the merge recommendation, so it gets its own
section rather than a footnote.

### 5.3 The arms (~90 min of device time across the four)

`perf/z_fix2_crossmodel/run_arm.sh <A|B|C|D>`, modelled on `perf/z_flip_land/run_qb2.sh`: `cd $WT`
(this worktree, never a parent or another slug's), per-step `.done` markers so a pass that ends
mid-arm leaves finished steps intact, one log and one JSON per step.

```sh
apply_arm() {                                  # working tree only; never committed
  git checkout -- tt_bio/tenstorrent.py
  case "$1" in
    A) git apply -R perf/z_fix2_crossmodel/fixd.patch perf/z_fix2_crossmodel/fix2.patch ;;
    B) git apply -R perf/z_fix2_crossmodel/fixd.patch ;;
    C) : ;;
    D) git apply -R perf/z_fix2_crossmodel/fix2.patch ;;
  esac
  git diff --stat tt_bio/tenstorrent.py
}
```

Arm A first: it is the cheapest, it establishes the verbatim throw on this card, and it hands the
other three their control values.

| arm | models and targets, one process per model | expected wall |
|---|---|---|
| **A** | `openfold3` and `boltz2` on `cdk2x2_298.yaml,cdk2x2_512.yaml`, then `opendde` on `cdk2x2_512.yaml` | ~15 min |
| **B** | `openfold3` and `boltz2` on `cdk2x2_298.yaml,cdk2x2_512.yaml` | ~25 min |
| **C** | as B, plus `opendde` on `cdk2x2_512.yaml` | ~30 min |
| **D** | as B | ~25 min |

```sh
$PY -u perf/z_fix2_crossmodel/arm_sweep.py --arm A --model openfold3 \
    --targets perf/size512/fixtures/cdk2x2_298.yaml,perf/size512/fixtures/cdk2x2_512.yaml \
    --out perf/z_fix2_crossmodel/armA_openfold3.json
```

298 aa is in every arm on purpose: it is the size that folds on main today, so it is the parity
control the brief asks for, and four arms times one plDDT is a check that costs 80 s per arm. opendde
at 512 aa appears in arm A and arm C only, which is the exact A/B that answers deliverable 6.

On a wedge: `~/.local/bin/tt-smi -r 2`, then kill the stray by explicit pid. Never `pkill` inside a
compound command. Commit and push after every arm; do not carry four arms' results in an uncommitted
tree.

### 5.4 Acceptance checks, each one binary

1. Arm A reproduces `clash with l1 buffers on core range [(x=0,y=0) - (x=2,y=9)]` with **808 960** and
   **995 840** for both openfold3 and boltz2, and `err_tb` names a `batched_matmul` frame in
   `tt_bio/tenstorrent.py`. If the addresses differ from `z-permute-flip-land`'s, say so and use
   yours; if the Python frame is not `batched_matmul`, §2 is wrong and that supersedes P1.
2. Arm B: both models fold, and `bmm_cfg_refused_n == 1` for the 512 aa row with the key
   `(16, (512, 24), (24, 512), 'DataType.BFLOAT16')` or whatever the real logical shapes are. **Zero
   is a failed arm, not a pass.**
3. Arm C: both models fold and `bmm_cfg_refused_n == 0` at 512 aa (§1.3).
4. Arm D: verdict recorded either way; it decides P2.
5. Parity at 298 aa: one plDDT to six decimals and one full CIF sha256 across all four arms, per
   model. boltz2 carries no top-level plDDT through this path, so its evidence is the sha alone and
   the doc says so rather than leaving a blank.
6. Parity at 512 aa: arms B, C and D agree with each other, digest for digest. No pre-fix reference
   exists at 512 aa because the fold never completed, and the doc states that rather than implying a
   comparison it did not make.
7. The predicate table, per model, with the **measured** bank from card 2: `held`, CB region, their
   sum, the comparison against the bank, and predicted-vs-measured. For the passing arms C and D there
   is no throw to read, so the honest statement is the inequality the completed fold implies:
   `held_C <= bank − 995 840` (= 465 920 with the qb1 bank value), because the tuned config is still
   in force there and it was placed successfully.
8. opendde 512 aa: arm A digest == arm C digest, and `bmm_cfg_refused_n == 0` in both.

### 5.5 Gate mechanics, from the brief plus the two traps verified this pass

- The DONE_CHECK script is **not** on qb2. Stage it: `scp
  /home/moritz/.coworker/workstreams/protenix-trunk--z-fix2-crossmodel_donecheck.py
  ttuser@tt-quietbox2:/home/ttuser/.coworker/workstreams/`. `_org_gate.py` and `_perfwar_gate.py` are
  already there, and so are `state/orgs/protenix-trunk/{CHARTER,STATUS}.md` and
  `state/perfwar/FINDINGS.md`.
- Stage this doc at **both** `/home/moritz/.coworker/state/<slug>.md` and
  `/home/ttuser/.coworker/state/<slug>.md`, checksum-matched. Use `scp`, or `ssh` **without** `-n` if
  you redirect a file into stdin: `-n` silently defeats `< file`.
- Append to `state/perfwar/FINDINGS.md` and to `state/orgs/protenix-trunk/STATUS.md` on both hosts.
  **Append only**, at the end, never a rewrite: STATUS.md has one owner and two writers corrupted a
  shared task file in this fleet once already.
- Keep gate phrases unbroken on one line: `% of the <noun>`, `the binding roof is`, `the limiter is`,
  `memory-bound`. A hard wrap fails the clause and it is the most common gate failure in this org.
- Every percentage must be followed within 140 characters by `of the compute|copy|read|write|
  bandwidth|roofline|fold|step|headroom` or by the word `against`. `% of the bank` alone does **not**
  satisfy the clause; write `% of the 1 461 760 B bank, measured against that bank`. And do not write
  any `N % of the write/compute/roofline` phrase with N below 70 unless a low-level limiter is named in
  the same breath.
- Say plainly that nothing here is merged and that the arms are an experiment, not production.

---

## 6. Results — TBD, to be filled by the execution pass

### 6.1 Verbatim throw, arm A, card 2

```
TBD-A-openfold3
```
```
TBD-A-boltz2
```

### 6.2 The predicate, per model, against the bank measured on card 2

| model | arm | measured bank | `held_per_bank` | CB region | sum | vs bank | predicted | measured |
|---|---|---:|---:|---:|---:|---|---|---|
| openfold3 512 | A | TBD-bank | TBD | TBD | TBD | TBD | FAIL | TBD |
| openfold3 512 | B | TBD-bank | TBD | TBD | TBD | TBD | ok | TBD |
| openfold3 512 | C | TBD-bank | ≤ TBD | 995 840 | TBD | under | ok | TBD |
| openfold3 512 | D | TBD-bank | ≤ TBD | 995 840 | TBD | under | ok | TBD |
| boltz2 512 | A–D | TBD-bank | TBD | TBD | TBD | TBD | as above | TBD |

### 6.3 `_BMM_CFG_REFUSED` after every arm

| arm | openfold3 512 | boltz2 512 | opendde 512 | expected |
|---|---|---|---|---|
| A | TBD | TBD | TBD | empty |
| B | TBD | TBD | — | **non-empty, 1 key** |
| C | TBD | TBD | TBD | empty |
| D | TBD | TBD | — | empty |

### 6.4 Parity

| model | size | arm A | arm B | arm C | arm D | identical? |
|---|---|---|---|---|---|---|
| openfold3 | 298 | TBD plDDT / TBD sha256 | TBD | TBD | TBD | TBD |
| openfold3 | 512 | did not fold | TBD | TBD | TBD | TBD |
| boltz2 | 298 | no plDDT / TBD sha256 | TBD | TBD | TBD | TBD |
| boltz2 | 512 | did not fold | TBD | TBD | TBD | TBD |
| opendde | 512 | TBD | — | TBD | — | TBD |

Isolated `torch.equal` probe, §5.2: **TBD-probe**.

### 6.5 opendde

TBD-opendde: does FIX-2 change anything for the model that already folded 512 aa clean, and is
`_BMM_CFG_REFUSED` empty there.

### 6.6 Merge recommendation for ask 4413

Read the live state first. Ask `4413` is recorded **open** in `state/pending-input/4413.md`
(15:49:44) while `c06bd76c` carrying both commits is an ancestor of `origin/main` (16:02:46). So this
leg is no longer evidence for a decision that is pending; it is **the per-model follow-up the ask
itself promised** (*"per-model byte-identical checks (openfold3, esmfold2, opendde) as follow-ups, not
preconditions"*). Write the recommendation against that framing, in one paragraph, following this
decision rule:

- **P1 confirmed** → the follow-up discharges the ask's own condition, and the fix is worth more than
  it claimed: two more models go from *does not fold* at 512 aa to *folds*, with parity unchanged.
- **P1 confirmed and P2 confirmed** → same conclusion, with the honest qualifier that FIX-D alone also
  suffices here, so FIX-2's demonstrated value is as the model-agnostic backstop rather than as the
  unique mechanism.
- **P1 falsified** → say what the predicate is missing, with the measured `held` and CB for the failing
  case, and flag that a second site throws behind the first. Do not repair it here.
- **P4 falsified** → this outranks everything: a bit-exactness deviation in shared code reached by
  three shipping models is a revert conversation, and it goes to Moritz as an `ask`, short.

Nothing in this leg is merged by this leg, and no merge is proposed by it. That call is Moritz's.

---

## 7. Decided against, so the execution pass does not relitigate it

- **Toggling the fixes in-process** by monkeypatching `batched_matmul` and
  `AttentionPairBias.__call__`. It is the literal reading of "four arms in one process" and it is
  wrong: it substitutes a hand-written copy of the function under test for the function under test.
  Reverse patches plus the `inspect.getsource` fingerprint give a stronger guarantee for less code.
- **`git checkout <old commit> -- tt_bio/tenstorrent.py`**, which is what `z-crashband-fix` did for its
  control. Main has moved three merges since (`f2e03908`, `a330d47a`, `211e7791`), so that would revert
  unrelated changes too and the arm would no longer isolate the two commits.
- **Preferring the smallest saturating `p`** in `_batched_matmul_search`. `p = 4` fits the live bank at
  512 aa and engages 64 of 110 cores instead of 32, so it is very likely a real third repair. It is
  forbidden by the brief's hard limits and it belongs to whichever leg owns that search. Recorded in §2
  and handed forward.
- **Reusing `z-permute-flip-land`'s 298 aa plDDTs and CIF shas as controls.** They were taken with
  `TT_BIO_REBLOCK_PERMUTE` effectively ON (`kernel_on: true` in every row), which is not the default
  this leg runs under.
- **A timed A/B of any kind.** `z-crashband-fix` measured the fold-wall floor at about 9 % of the fold
  on a loaded qb host, which is far above anything FIX-2 could cost, and a co-tenanted A/B on qb2 has a
  1-10 % noise floor on identical code paths. A wall-clock number here would read as having measured
  something. The cost question was already settled: zero throws at 298 aa, so zero device cost.
- **`--fast` arms, `release_gate.py`, and the `full_parity_gate` legs.** Out of scope. The brief asks
  four arms on two models at one size, and adding gate runs is how a leg spends a pass and delivers
  neither.
- **Fixing openfold3 or boltz2 by any other route.** Explicitly forbidden, and if FIX-2 does not close
  them the diagnosis *is* the deliverable.
