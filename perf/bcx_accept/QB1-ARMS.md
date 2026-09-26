# The qb1 arms: one lost to a thread race, one card recovered, and a grader split

## 1. `qb1_s11` deadlocked at startup and `ps` could not tell

`qb1_s11` (pid 47255, card 1 / `0000:41:00.0`) ran 1 h 46 m and produced nothing. Its log stops at
23:49:55Z:

    TT_FATAL: context_id -381514392 is invalid. (assert.hpp:104)
    compiling the next length bucket failed, leaving it to the trajectory that folds there
      (TT_FATAL @ metal_context.cpp:72 ... Device::init_command_queue_device_with_topology)

The competitor is a daemon `threading.Thread` started by `compile_next_length_bucket`
(`bindcraft/campaign.py:101-109`), in the same process as the main trajectory. It opens the device
to pre-compile the next length bucket's gradient graph.

`tt_bio/device_lease.py:377` serializes that with an `flock`, and on `qb1_s10` and `qb1_s12` the
lease did its job: both logged `is in use by worker:bcx-accept (pid <their own>); waited 120s.
Refusing to open it concurrently`, and both are still running. On `qb1_s11` the lease did not
refuse. The compile thread reached command-queue init, got a garbage `context_id`, and the process
has been in `futex_wait` on all 187 threads ever since. So the outcome is timing-dependent: the
window is the main thread having begun its device open but not yet taken the lease, and losing the
race is a **silent permanent deadlock rather than an exception**.

**Two ways this reads as healthy when it is not.** `ps` `%CPU` is cumulative over the process
lifetime, so the wedged arm still printed `100`; current per-thread CPU was `0.1`. And the clock
sampler is a separate writer, so `qb1_s11_clock.csv` kept growing at 1350 MHz for the whole
1 h 46 m. The orchestrator's tick read `ps` etime and recorded the arm as running. **Grade a
detached arm on the last write to its own run log, never on `ps` or on a sidecar the arm does not
write itself.** SIGINT was ignored for 25 s -- the main thread is blocked below the interpreter, so
Python's handler never runs -- which is itself a positive test for a real deadlock rather than
slowness.

Not the cause here, but adjacent and worth one line: a child that inherits the lease fd across
`fork()` re-acquires `LOCK_EX` successfully, because re-locking an open file description you already
hold is a no-op that returns success. Same-process-new-fd and child-new-fd both block correctly
(measured, all three cases). `tt_bio/main.py:62` does fork at import time, so the hole is reachable
in principle; it is not what `qb1_s11` hit, because the bucket compiler is a thread.

Owner: this is BindCraft 2's `campaign.py` racing tt-bio's lease. Filed here, not fixed here.

## 2. A `tt-smi -r` logs DPC containment as a matter of course, and the card is fine

After killing `qb1_s11` the card needed a reset, because a killed device holder is the state where
the next open hard-hangs the host. `tt-smi -r 0000:41:00.0` returned rc=0 and logged:

    pcieport 0000:40:01.1: DPC: containment event, status:0x1f01: unmasked uncorrectable error
    tenstorrent 0000:41:00.0: AER: can't recover (no error_detected callback)
    pcieport 0000:40:01.1: AER: device recovery failed

That is the exact signature that has been read as hardware death on this host. **It is an artifact
of the reset.** Dropping the link is an uncorrectable error from the root port's point of view, and
the tenstorrent driver implements no PCIe `error_detected` callback, so the generic AER handler has
nothing to call and always reports failure. Graded on that dmesg line alone, every successful reset
marks its own card dark.

What the card actually did, all measured after the reset:

    heartbeat     restarted from 0 and ticks at the healthy control's rate (+60 / 6 s)
    AER counters  all 0, fatal and non-fatal, 24 fields
    DevSta        CorrErr- NonFatalErr- FatalErr- UnsupReq-
    LnkSta        Speed 32GT/s (ok), Width x16 (ok)
    AICLK         800, which is the idle floor -- the other three read 1350 under load
    probe         open_device + 64x64 bf16 matmul + to_torch + close_device, twice,
                  rc=0, finite, rel err 0.0051, host alive, both sibling arms alive

The card is back in service and `qb1_s13` runs on it.

**Blast radius, and it settles a standing caution.** The brief carries "qb1 cards 0 and 1 are a
board pair, so a `tt-smi -r` on one resets the other". That is a p300c property and **qb1 is not a
p300c host**: `tt_card_type` reads `p150a` on all four nodes, and p150a is single-chip. The four
BDFs are `01:00.0`, `41:00.0`, `42:00.0`, `c1:00.0`, on separate root complexes. The reset was
observed to leave the other three cards untouched and both sibling arms running. There is no board
pair on qb1.

Also: `dmesg | grep -c AER` reads 51 on this host and every one of those lines is a boot-time
capability advertisement (`AER: enabled with IRQ`, `DPC: error containment capabilities`). Zero are
error events. A keyword count is not an event count.

## 3. The qb1 and qb2 draws are graded by DIFFERENT instruments

The pooling precondition asked for was the arithmetic one: an `--exact off` arm whose
`EXACT_SOFTMAX_STATS` moved is a failed run. That check is necessary and it is in `traj_arm.py`.
It is not sufficient, because it looks at the gradient path and the thing that splits these two
sets is the **grader**.

    qb1, traj_arm.py   design on card, acceptance decided by BC2's own JAX validation ensemble
                       (`tt_bio/bindcraft2.py:799`, `campaign_predictor(validation="jax")`, the
                       documented default)
    qb2, run_arm.py    does not call `campaign_predictor`; its validation ensemble folded on the
                       DEVICE pool, which is why that harness could crash with
                       `multimer_pool.py -> KeyError 'model_1_ptm' is not in the pool`

A crash of that shape is only reachable if validation is being asked of the card. So the 2 accepted
/ 5 completed in `RATE.md` were graded on device folds and the qb1 draws are graded on JAX folds.
**Whether a binder clears the seven filters is a property of the arithmetic AND of the model that
folds it**, so these acceptance verdicts are not one population. Report them as two lines until a
qb1 trajectory actually reaches validation and the grader can be compared, and do not add them into
a single accepted/completed ratio before then.

This does not touch the stage profile. Every stage reading is produced by the gradient loop on
card, upstream of any validation fold, so qb1 rows pool into the stage table cleanly and that is
where they are used below.

## 4. qb1 capacity, measured rather than planned

The four-wide fan the brief asked for assumed ~3 cores an arm on a 16-core box. qb1 also carries
`bcx-shipped`'s reference arm `ref_s2` at 750 % CPU, which the plan did not account for, so with
three BC2 arms plus that reference the box sat at loadavg 27 on 16 physical cores before adding
anything. `qb1_s13` restores the three-arm configuration `s10/s11/s12` was meant to be rather than
adding a fourth, and card 2 / `42:00.0` is left for `bcx-extrawire`'s sub-hour A/B.

Stamps now carry `commit` and a dirty-file count, which closes the "`arm_stamp.json` does not
record the tree SHA" defect: attributing an arm to a tree cut no longer means reading a mutable
worktree after the fact.
