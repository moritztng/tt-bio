# The reading contract

Range `v0.68.0` -> **`v0.78.0`**, the shipped target. `v0.79.0-dev20260913` is tracked as a delta
column in `DIFF_PACKAGES.md` but is not the target: an unlock that exists only in a nightly has no
wheel and cannot be paid for. If a capability you find is nightly-only, say so in the row and grade
it POINTER.

Written once so it does not drift across thirty agents. Every package agent in the TTX diff fan-out
is held to this.

## What you are looking for

**A capability that did not exist in tt-metal 0.68.0 and maps onto a place our code works around its
absence.** Nothing else. Not "this got faster upstream", not "this was refactored", not "they added
an op". A changelog entry with no such mapping is noise and reporting it wastes the next agent's
read.

Two shapes qualify:

1. **A gate that now returns served where it returned declined.** An eligibility predicate, a
   `TT_FATAL`, a `validate()` branch, a silent fallback to a slower path.
2. **An API that now exists where we wrote a workaround.** A parameter, an overload, a program
   config field, a new op, a kernel helper.

## What counts as evidence

Every claimed unlock owes all four of these, in the report:

- **the gate or API**, named by file and line at both revisions;
- **the before and the after**, quoted. Not paraphrased — quoted, with the two `git grep` commands
  that produced them;
- **the mapping into our tree**: the file and line in `tt_bio/` that works around it today, with the
  workaround quoted;
- **the original measured value** of the lever it unblocks, taken from the dead-end catalogue or the
  state doc that first screened it, marked **op-level** or **fold-level**.

If you cannot produce all four, the row is a POINTER, not an UNLOCK. Report it as a pointer. A good
pointer is valuable; a pointer dressed as a finding costs the campaign a porting bill.

## Release notes are a search index, not evidence

Use them to decide what to grep for. Never quote one as a result. This project has already acted on
an external summary — "Triangle Attention falls back to a naive implementation without SDPA" — that
looked decision-relevant and did not apply, because we had already written our own persistent-mask
kernel. **Check what our tree actually does before believing a note about what upstream added.**

## Worked example: why a disappeared string is not a removed gate

The dead lever "pair-tensor L1 residency" died because `ttnn.matmul` refuses a sharded in0 operand
where `fuse_batch=False`. The naive check:

```
git -C .ttm.git grep -n "Batch fusion must be enabled" v0.78.0 -- ttnn/cpp/ttnn/operations/matmul/
  -> no output
```

Nothing. It looks removed. It is not. Widening the grep to `fuse_batch` across the whole matmul op:

```
# 0.68.0 -- three fatals
matmul_device_operation.cpp:493  TT_FATAL(program_config.fuse_batch, "Error: Batch fusion must be enabled.");
matmul_device_operation.cpp:588  TT_FATAL(program_config.fuse_batch, "Error: Batch fusion must be enabled.");
matmul_device_operation.cpp:776  TT_FATAL(program_config.fuse_batch, "Batch fusion is required when input A is sharded");

# v0.78.0 -- one fatal
matmul_device_operation.cpp:1443 TT_FATAL(program_config.fuse_batch, "{}: Batch fusion is required when input A is sharded", config_name);
```

Two of the three are gone from the op we call; the third survived with reworded text, and the two
that vanished are still present verbatim in `experimental/quasar/matmul/`, a part we do not own. So
the honest report is: *two of three `fuse_batch` fatals were removed from the non-Quasar matmul, one
survives, and which one our call site hits is unresolved.* That is a POINTER with a named next step,
not an unlock — and it is a good one. Note also `matmul_program_config.cpp:1248`, new in the range:
`fuse_batch_for_sharding = user_shard_spec.has_value() && b_is_unbatched`, which is new logic for
choosing `fuse_batch` when the operand is sharded.

Grep the string, then grep the symbol, then grep the whole repo, then check the path is one we
actually use. A gate that moved is not a gate that lifted.

## Our own contributions are in this diff, and they grade lower

Some upstream changes were made *for* TT-Boltz. They have the highest prior probability of mapping
onto our workarounds, and they are the easiest to double-count. If we already work around the gap
and the fold already runs at the speed it runs, then the unlock is **"delete our workaround"** —
a maintenance win, usually **not** a speedup. Grade it that way explicitly:

- `MAINTENANCE` — deletes our code, fold time unchanged by construction.
- `SPEEDUP` — enables a lever that a prior screen measured, with that number quoted.
- `BOTH` — deletes our code *and* the stock path is faster than our workaround. This one owes an
  argument for why, not just an assertion.

## Standing rules that are not negotiable

- **Eligibility is a Blackhole claim or it is not a claim.** Measured WH->BH transfer here is
  arithmetic k=0.623 +/- 0.090, layout k=1.329 +/- 0.157, placement k=1.161 +/- 0.047, and
  eligibility levers do not transfer at all. Our cell of record is a p300c.
- **Do not compose numbers across instruments.** This is the most expensive recurring error in this
  project.
- **A stall fraction is not a recoverable fraction. An op-level ratio is not a fold-level second.**
  1.6652x on the trimul was 0.445 s of a 17.340 s cell.
- **Do not re-run the wall-clock A/B.** `b2z-ttnn-upgrade` measured 0.9968x on 0.78.0, paired and
  interleaved. The direct-speed hypothesis is refuted; this campaign is about capability. If you
  find yourself timing folds, you have drifted.
- **No device, no builds.** The reading fan-out is card-free and that is the point. Anything needing
  a Blackhole measurement goes back to the orchestrator to be queued on qb2, serially, against the
  0.78.0 environment `ttx-version-recon` already built: `/home/ttuser/scratch/ttx/venv-new`, opened
  with `TT_MESH_GRAPH_DESC_PATH=<repo>/perf/ttx/mgd/bh_1x1.textproto TT_VISIBLE_DEVICES=<card>`.
  Do not rebuild it and do not bump the pin.
- **STOP is a pass.** "Nothing in this package unlocked anything" is a complete and valuable answer:
  it saves the porting bill. Say it plainly and say what you read.

## Report format

One file per package, `~/.coworker/state/ttx-pkg-<name>.md`. Flat. Owes:

```
PACKAGE:   <id and name>
PATHS:     <the prefixes you read, and any you skipped, with why>
VOLUME:    <lines / files / commits you actually covered, against the package's total>
UNLOCKS:   <one block per unlock; each owes gate/API, before -> after quoted with both greps,
            our workaround file:line, original measured value, and grade
            MAINTENANCE | SPEEDUP | BOTH>
POINTERS:  <candidates missing one of the four pieces, each with the exact next check>
RISKS:     <capabilities REMOVED or renamed that our tree depends on -- a porting bill, not an
            unlock. Name our file and line.>
NOTHING:   <what you read and found nothing in, so the next agent does not re-read it>
VERDICT:   GO | NO-GO | PARTIAL | BLOCKED | STOP
```

`RISKS` is not optional and is not a consolation prize. One is already known and confirmed:
`tt_metal/hw/inc/experimental/{noc,circular_buffer,tensor}.h`, included by
`tt_bio/kernels/rfd3_softmax/`, no longer exist as of 0.78.0. The first two moved to
`api/dataflow/`; the third's closest candidate is `experimental/udm/accessor/mesh_tensor_accessor.h`
and nobody has confirmed that is it.

A second case shows the same lesson from the other side. The `transformer/sdpa_windowed/` directory
is gone at 0.78.0, which reads as a removal until you grep for the word rather than the path: the
op was absorbed into `transformer/sdpa/` as `device/kernels/dataflow/windowed_mask_gen.hpp` and
`device/kernels/windowed_loop_geometry.hpp`. A vanished directory is not a vanished capability, and
this one is a candidate unlock for OF3's windowed atom attention rather than a bill.
