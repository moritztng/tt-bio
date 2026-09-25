# NIH (NCI, Valkov lab) perf report — evidence and draft reply

Row `nih-perf`, 2026-09-25. Everything below was measured on our QuietBox 2 (`tt-quietbox2`,
4x Blackhole p300c) unless it says otherwise. Card 1, `TT_VISIBLE_DEVICES=1`. Engine tree is
the **v0.9.0 tag** (`1f2fcdef6dcb10df9c053a8a20f47e40f98709ed`), checked out as a nested
worktree at `v090/` and selected with `PYTHONPATH`, against the ttnn wheel we ship
(`ttnn==0.68.0`, `pyproject.toml` extra `tenstorrent`). Measurement script:
`perf/nihperf/fold_ops.py`, raw JSON in `perf/nihperf/results/`. That script is also the
attachment the draft email refers to.

## The one thing that matters

Their gap is **not** op count and **not** the hardware. It is a constant extra host cost on
every ttnn op dispatch, worth roughly +13 to +21 microseconds per op. That single constant
reproduces all four of their numbers, including why ESMC-600M only loses 18% while the folds
lose 41-49%.

## 1. Op count per call (exact, load-insensitive)

Counted with tt-metal's own per-enqueue counter, `ttnn._ttnn.get_device_operation_id()`, which
`enqueue_mesh_workload` increments exactly once per device op
(`ttnn/api/ttnn/device_operation.hpp:193`). Delta across one warm call. This is a count, not a
sample or an estimate, and it was byte-identical on every draw.

| model | input | device ops / call |
|---|---|---|
| esmfold2-fast | trpcage, 20 aa, single-seq | **13,234** |
| esmfold2 | trpcage, 20 aa, single-seq | **15,614** |
| boltz2 | trpcage, 20 aa, single-seq | **25,542** |
| esmc-600m | 8x ubiquitin (76 aa), batch 8 | **979** |

Protocol matches `scripts/perf_regression.py`: 1 recycling cycle, 10 sampling steps, 1 diffusion
sample, model loaded once in-process, 2 warmup calls excluded.

For reference, `main` today (commit `bcb51cf11`) issues 13,416 for esmfold2-fast, 1.4% above the
v0.9.0 tag.

Their stated "~18k ttnn ops per fold" is 36% above our 13,234 at the same tag on the same
fixture. Worth having them re-count with the line above, since if they are counting Python-level
ttnn calls rather than device-op enqueues they will get a bigger number for the same work.

## 2. Host per-op dispatch cost

Probe: `ttnn.relu` on one 32x32 bf16 tile, program cache warm, no host sync inside the loop, so
the loop measures the enqueue path and the device drains behind it. 40,000 ops per run, timed in
chunks of 100.

qb2 was loaded for this entire window (other fleet rows, 4-6 concurrent device jobs, loadavg
13-25 on an 8-core/16-thread host). Contention can only **add** time to a chunk, so the minimum
chunk is the least-contaminated estimate of the uncontended cost; the median chunk is what the
box delivered under that load. The minimum is stable to +/-0.05 us across runs while the median
swings by 4 us, which is exactly the signature of preemption rather than of a real effect.

| run | Inspector | min chunk us/op | median chunk us/op | loadavg |
|---|---|---|---|---|
| 1 | on | 8.308 | 9.480 | 21.2 |
| 2 | on | 8.386 | 12.536 | 21.7 |
| 3 | on | 8.294 | 11.755 | 20.9 |
| 1 | off | 8.096 | 11.722 | 21.3 |
| 2 | off | 8.048 | 13.753 | 20.8 |
| 3 | off | 8.102 | 11.270 | 20.9 |

**8.1 us per ttnn op enqueue with the Inspector off, 8.3 us with it on.** AICLK on card 1 read
1350 MHz throughout (sysfs `tt_aiclk`, sampled every 50 ms during the timed region; cards 0/2/3
were other rows). 99.9% of the loop wall was host enqueue, not device drain, so this is a host
CPU number and the device is nowhere near the limiter at this op size.

## 3. Their gap decomposed

Our per-call times are the committed baselines in `docs/perf_baselines.json` (see the caveat in
section 7). Theirs are the four numbers in their message. Op counts from section 1.

| model | ops | ours ms/call | theirs ms/call | delta ms | **delta us per op** |
|---|---|---|---|---|---|
| esmfold2-fast | 13,234 | 254.9 (3.92242/s) | 431.0 (2.32/s) | 176.1 | **13.3** |
| esmfold2 | 15,614 | 303.2 (3.2985/s) | 552.5 (1.81/s) | 249.3 | **16.0** |
| boltz2 | 25,542 | 568.3 (1.759625/s) | 1111.1 (0.90/s) | 542.8 | **21.3** |
| esmc-600m | 979 | 72.4 (110.519 seq/s) | 88.9 (90 seq/s) | 16.5 | **16.9** |

Four models, a 26x range in op count, deficits from 18% to 49%, and one constant explains all of
them: **every ttnn op costs them about 17 us more than it costs us.** Nothing here needs an
op-count difference, a clock difference or a hardware difference.

The cross-check that makes this convincing is ESMC-600M. At 979 ops per call, dispatch is only
979 x 8.1 us = 7.9 ms of a 72.4 ms call (11%), so an extra 17 us per op costs it 16.5 ms and it
loses 18%. esmfold2-fast is 13,234 x 8.1 us = 107 ms of a 255 ms call (42% dispatch), so the same
per-op penalty costs it 176 ms and it loses 41%. The model that dispatches least loses least, in
proportion.

Their own "~20 us/op" is consistent with this: our 8.1 us floor plus ~13 us. Their 20 us is the
whole per-op host cost; ours, measured the same way, is 8.1.

## 4. Build configuration — the leading suspect (NOT measured by us)

What our wheel actually is, read off the binary:

- `readelf -p .comment libtt_metal.so`: `clang version 20.1.8 (AlmaLinux OS Foundation
  20.1.8-3.el9)`, plus objects from `GCC 14.2.1` and `GCC 11.5.0`, linked with `mold 2.40.4`.
- No `.debug_info` section, zero `log_trace` strings in the binary. That is a **Release** build.

What a source build defaults to, from tt-metal at tag `v0.68.0`:

- `CMakeLists.txt:19-23` — for single-config generators, `CMAKE_BUILD_TYPE` defaults to
  **RelWithDebInfo**, not Release.
- `CMakeLists.txt:41` — `CMAKE_CXX_FLAGS_RELWITHDEBINFO = "-O3 -g -ggnu-pubnames -DDEBUG
  -fno-omit-frame-pointer"`. Note `-DDEBUG`.
- `tt_stl/tt_stl/assert.hpp:155-168` — `TT_ASSERT` expands to a real check **iff `DEBUG` is
  defined**, otherwise to `(void)(condition)`. So RelWithDebInfo leaves every `TT_ASSERT` on the
  host dispatch path live.
- `CMakeLists.txt:202-215` — `TT_METAL_ENABLE_LOGGING` defaults **ON** unless `CMAKE_BUILD_TYPE`
  is exactly `Release`, and compiles in `log_trace`/`log_debug` with
  `SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE`.
- `build_metal.sh:70` does default to `Release`. A bare `cmake -B build -S .` does not, and
  `--development` selects RelWithDebInfo.

So GCC 14 versus clang is probably a red herring: both are `-O3`. The build **type** is not. If
they configured with plain cmake, they are running asserts and trace logging that our wheel does
not have, on exactly the path that is costing them ~17 us per op. `ENABLE_TRACY` is a plain cmake option that nothing sets, so
the profiler is off unless they asked for it.

We have not measured the RelWithDebInfo penalty ourselves — we ship a wheel and do not build
tt-metal from source, and a two-way source build was out of reach this pass. It is a hypothesis
with a mechanism, not a number. The check costs them one command.

## 5. PCIe and host CPU — identical to ours

```
lspci -d 1e52: -vv    (all four Blackhole chips, tt-quietbox2)
  LnkCap: Port #0, Speed 16GT/s, Width x8, ASPM not supported
  LnkSta: Speed 16GT/s, Width x4 (downgraded)

lscpu
  Model name: AMD Ryzen 7 9700X 8-Core Processor   (8 cores / 16 threads)
```

Gen4 x4 per chip **is** the QuietBox 2 configuration, not a defect on their box, and every
baseline in `docs/perf_baselines.json` was measured on it. Their host CPU is the same part as
ours. Neither can explain the gap, because we have both and we do not have the gap.

On whether the link could affect dispatch latency at all: the probe in section 2 spent 99.9% of
its wall in host enqueue with the device draining behind, so at these op sizes dispatch is
host-latency bound, not PCIe bandwidth bound. A narrower link would show up on large tensor
transfers, not on an 18k-op stream of small kernels.

## 6. The Inspector

**Where the default comes from:** tt-metal, not tt-bio. `tt_metal/llrt/rtoptions.hpp:109-121` at
v0.68.0:

```cpp
struct InspectorSettings {
    bool enabled = true;                      // <- on by default
    bool capture_tensor_specs = true;         // <- on by default
    bool log_runtime_entries = false;         // the expensive YAML log, off by default
    bool rpc_server_enabled = true;
};
```

tt-bio v0.9.0 sets nothing, so their run and our baselines both had it on.

**What it actually costs per enqueue.** Their model of it ("flushes to disk every enqueue") is
not what the default config does, and the difference matters for diagnosis:

- `ttnn/api/ttnn/device_operation.hpp:199-213` — on every enqueue, if the Inspector is enabled,
  ttnn builds the operation name and, because `capture_tensor_specs` defaults **true**, reflects
  over the tensor arguments and copies a `TensorSpec` for each one.
- `tt_metal/impl/debug/inspector/inspector.cpp:366-399` — `emit_debug_entry` then takes a mutex
  and writes into an in-memory ring buffer. It writes to disk only if
  `TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES=1`, which defaults off.
- The disk flushes are real but they are on program and mesh-workload create/destroy, not on
  enqueue: `logger.cpp` does `ostream << ...; ostream.flush()` per event. With the program cache
  on those do not fire on the warm path, and tt-bio enables the program cache
  (`tt_bio/tenstorrent.py:4934`).

**Measured cost on our stack: 8.29 us/op on, 8.05 us/op off — 3%** (section 2). Their +26% is
8x that. If their Inspector really is disk-bound then something is creating and destroying
programs or mesh workloads on their warm path, i.e. the program cache is missing, which would be
a second and larger problem than the Inspector itself. Worth asking.

**Should tt-bio default it off: yes.** It is a debug facility, it is not free, and no inference
user wants it. Landed on `wk/nih-perf` in `tt_bio/main.py`: `TT_METAL_INSPECTOR=0` via
`setdefault`, inside the block that `--debug` skips, so `tt-bio --debug` keeps triage working and
an explicit `TT_METAL_INSPECTOR=1` still wins. Verified all three behaviours, and smoke-tested
with a real esmfold2-fast fold on the patched branch (`insp_env=0`, 13,416 ops, fold completed).
It cannot move numerics: the Inspector only writes logs and a ring buffer, it never touches a
tensor. The accuracy gate was not re-run for it.

Not yet in any release. It is on the branch for the orchestrator to merge; the next tag after
that merge carries it.

## 7. The baselines they are being compared against — a real problem

`docs/perf_baselines.json` resolves a model's baseline from
`cards.<card_type>.machines.<hostname>.models` first and falls back to `cards.<card_type>.models`.

- The `p300c` block's `machines` map contains exactly one entry: `tt-quietbox2`. **Ours.**
- So any p300c host that is not named `tt-quietbox2` silently falls back to the card-level block.
- Every cell in that card-level block is itself tagged `"machine_id": "tt-quietbox2"`,
  `"tt_bio_version": "0.7.2"`, `"date": "2026-09-01"`.

So they are running v0.9.0 and being gated, at a 15% threshold, against one specific machine of
ours measured at v0.7.2. The four values they quote (3.92242, 3.2985, 1.759625, 110.518954) are
exactly those cells. This is not a spec, and the file does not tell them that.

**We could not re-verify those baselines at v0.9.0 today.** qb2 carried other fleet rows for the
whole window: loadavg 13-25 on an 8-core host, 4-6 concurrent device jobs. The fold wall-clock is
the one number that load destroys, and it did:

| what | best draw | median | loadavg | AICLK card 1 |
|---|---|---|---|---|
| esmfold2-fast, v0.9.0, Inspector on | 0.3499 s (2.86/s) | 0.6484 s | 14.6 | 1350 |
| esmfold2-fast, v0.9.0, Inspector off | 0.4471 s (2.24/s) | 1.0226 s | 22.6 | 1350 |
| esmfold2-fast, main + fix, Inspector off | 0.3647 s (2.74/s) | 0.4717 s | 20.3 | 1350 |
| committed baseline | — | 0.2549 s (3.92/s) | quiet, v0.7.2 | — |

Best draw all day is 27% below the baseline and the median moves 2x with load, on the same tree
and the same card. That is contention, not a reading. The op counts and the min-of-chunks enqueue
cost survive a loud box; fold throughput does not, so no fold throughput number here should be
read as a verdict on v0.9.0.

A v0.7.2-versus-v0.9.0 A/B on the same box would have separated code drift from contention
without needing a quiet box. It is blocked: v0.7.2 cannot load today's ESMFold-2 HF checkpoint
(`DiffusionStructureHeadConfig.__init__() got an unexpected keyword argument 'architectures'`) —
the checkpoint moved and the old tag does not pin a revision. Doing it needs a pinned revision.

**Open, and it is ours not theirs:** reseed the p300c baselines on a quiet qb2 at the current
tag, and add a note to the file saying the card-level block is one machine's numbers so nobody
else reads it as a hardware spec.

## 8. Ranked suggestions, with what we measured for each

1. **Check the tt-metal build type.** `grep CMAKE_BUILD_TYPE build/CMakeCache.txt`. If it is not
   `Release`, rebuild with `./build_metal.sh --release`. Mechanism in section 4. Unmeasured by
   us, and the largest thing we can point at.
2. **Run the enqueue probe** (`perf/nihperf/fold_ops.py --probe-only`). Ours: **8.1 us/op**. Takes
   30 seconds, needs no model, and tells them immediately whether their host dispatch path is the
   problem or something inside the models is.
3. **Count their ops exactly**, `ttnn._ttnn.get_device_operation_id()` delta across a warm fold.
   Ours: **13,234** for esmfold2-fast. Settles the 18k question.
4. **`TT_METAL_INSPECTOR=0`.** Worth **3%** on our stack; they measured 26% on theirs, and that
   discrepancy is itself a clue (section 6).
5. **CPU governor / SMT.** Same 8-core 9700X as ours, and a latency-bound host loop cares.
   `cpupower frequency-info`. Not measured by us.
6. **Do not expect a trace-capture win on these two models.** tt-bio does use ttnn trace capture,
   but on the ESMC single-sequence path and the Protenix/BoltzGen diffusion loops, not on
   esmfold2 or boltz2 folds. There is no flag for them to flip here.
7. **Warn them off 4 concurrent folds on that host.** The esmfold2-fast fold's own host work ran
   3.1 cores busy on our box (2.06 CPU-s against a 0.67 s wall). That figure is inflated by our
   contention, but the thread count is real: four concurrent folds want ~12 cores on an 8-core
   part, so per-chip throughput will drop. Measured on tt-quietbox2, loadavg 17.

## 9. glibc and packaging

- ttnn 0.68.0 wheel tag: `cp312-cp312-manylinux_2_34_x86_64`.
- Actual symbol requirement, `objdump -T`: `libtt_metal.so` and `libdevice.so` import
  `GLIBC_2.34`; `_ttnncpp.so` `GLIBC_2.33`; `_ttnn...so` `GLIBC_2.32`.
- RHEL 8 is glibc 2.28. The wheel cannot load there. That is why they are building from source,
  and it is not a tt-bio limitation: tt-bio is pure Python (`pyproject.toml` build-system is
  plain setuptools, no compiled extension), so the entire floor is the ttnn wheel, which
  Tenstorrent publishes.
- tt-metal `INSTALLING.md` at v0.68.0 lists **Ubuntu 22.04** for Blackhole.
- A `manylinux_2_28` ttnn wheel is an upstream tt-metal packaging decision. **Moritz's call
  whether to push for it — no roadmap promise in the draft.**
- Recommendation: RHEL 9 is glibc 2.34, which matches the wheel tag exactly and is the smallest
  move for a Red Hat shop. Ubuntu 22.04 is what upstream tests Blackhole on.

---

# DRAFT EMAIL TO EUGENE — Moritz sends this, it has not been sent

Subject: Re: tt-bio throughput on our QuietBox

Hi Eugene,

Thanks for the detail in that report, it was enough to work from directly. I reproduced the
measurement side on our own QuietBox 2 today. Short version: I do not think your gap is the op
count, the PCIe link or the CPU. It looks like a constant extra host cost on every ttnn op, and
the most likely cause is how tt-metal was configured when you built it.

**Op count.** A 20-aa trpcage fold on esmfold2-fast issues 13,234 ttnn device ops at v0.9.0 on
our box. esmfold2 is 15,614, boltz2 is 25,542, and esmc-600m at batch 8 is 979. Those are exact
counts, not estimates, and they were identical on every repeat. You can get the same number on
your side without any instrumentation:

```python
from ttnn._ttnn import get_device_operation_id
before = get_device_operation_id()
...one warm fold...
print(get_device_operation_id() - before)
```

That is tt-metal's own counter, bumped once per device-op enqueue. If you see 13,234 then our op
counts agree and your ~18k is counting Python-level ttnn calls rather than device dispatches. If
you genuinely see 18k for the same fixture, tell me, because that would be a different bug.

**Per-op cost.** Our bare enqueue path costs 8.1 microseconds per op. I measured it with a tiny
program-cached op in a tight loop, no host sync inside the loop, taking the minimum over chunks
of 100 so that load on our box could not inflate it. You are reporting about 20.

Now put those together against your four numbers, using our op counts and our committed
baselines:

| model | ops/call | our ms | your ms | extra us per op |
|---|---|---|---|---|
| esmfold2-fast | 13,234 | 255 | 431 | 13.3 |
| esmfold2 | 15,614 | 303 | 552 | 16.0 |
| boltz2 | 25,542 | 568 | 1111 | 21.3 |
| esmc-600m | 979 | 72 | 89 | 16.9 |

One constant, roughly 17 microseconds of extra host cost per ttnn op, reproduces all four
deficits across a 26x range in op count. It also explains the thing that would otherwise look
odd: ESMC-600M only loses 18% because at 979 ops per call dispatch is just 11% of its time, while
esmfold2-fast is 42% dispatch and loses 41%. The model that dispatches least loses least, in
proportion. So the question is not where your ops went, it is why each one costs ~17 us more.

**The build.** This is where I would look first. Our wheel's `libtt_metal.so` is a Release build
with clang 20.1.8, no debug info and no trace logging compiled in. tt-metal's CMake, though,
defaults `CMAKE_BUILD_TYPE` to **RelWithDebInfo**, not Release, if you configure with plain
`cmake` (`build_metal.sh` does default to Release; a bare `cmake -B build` does not). Two things
follow from RelWithDebInfo at v0.68.0:

- Its flags include `-DDEBUG`, and `TT_ASSERT` expands to a real check only when `DEBUG` is
  defined, otherwise to nothing. So every assert on the host dispatch path goes live.
- `TT_METAL_ENABLE_LOGGING` defaults on for any build type that is not exactly `Release`, which
  compiles `log_trace`/`log_debug` into the binary at spdlog trace level.

Both of those sit on the path that is costing you the 17 us. So:

```
grep CMAKE_BUILD_TYPE build/CMakeCache.txt
```

and if it is not `Release`, rebuild with `./build_metal.sh --release`. I want to be straight that
I have not measured this penalty myself — we ship the wheel and do not build tt-metal from
source, so this is a mechanism I can point at in the source, not a number I can quote. GCC 14
versus our clang I would not worry about; both are -O3.

Before you rebuild anything, the fastest way to confirm the diagnosis is the enqueue probe. I have
attached the script. Run it with `--probe-only` and it will report microseconds per ttnn enqueue
in about 30 seconds, no model and no weights needed. Ours is 8.1. If yours comes back near 20,
that localises the whole thing to the host dispatch path and you will know a rebuild is worth the
time.

**PCIe and CPU.** Both are non-issues, and I can say that firmly because our box is configured
identically. Every Blackhole in our QuietBox 2 reads `LnkCap: Speed 16GT/s, Width x8` and
`LnkSta: Speed 16GT/s, Width x4 (downgraded)`. Gen4 x4 per chip is the QuietBox configuration,
not a fault on your machine, and all our published baselines were measured over it. Our host is
the same AMD Ryzen 7 9700X. On whether the link could be hurting dispatch latency anyway: in the
probe, 99.9% of the time is spent in the host enqueue with the device draining behind it, so at
these op sizes you are host-latency bound, not PCIe bandwidth bound. A narrow link would show up
on big tensor transfers, not on a stream of 13k small kernels.

**The Inspector.** Yes, it is on by default, and that default is tt-metal's, not ours:
`InspectorSettings::enabled = true` in `rtoptions.hpp`. You are right that it should not be on
for inference and we are changing that: tt-bio will set `TT_METAL_INSPECTOR=0` by default, with
`tt-bio --debug` putting it back for triage and an explicit `TT_METAL_INSPECTOR=1` still winning.
That is on a branch now and will be in the next release.

One correction on the mechanism, because it may matter for your diagnosis. With the default
settings it does not flush to disk per enqueue. `log_runtime_entries` defaults off; what happens
per op is that it builds the op name, copies a TensorSpec for each tensor argument
(`capture_tensor_specs` does default on), takes a mutex and writes into an in-memory ring buffer.
The per-event disk flushes are real, but they are on program and mesh-workload create/destroy,
which with the program cache warm should not fire on your steady-state path at all.

That matters because on our stack turning it off is worth 3%, and you measured 26%. If the
Inspector really is disk-bound on your box, that suggests programs or mesh workloads are being
created and destroyed on the warm path, i.e. the program cache is not holding. That would be a
bigger problem than the Inspector. If you still have the run, it is worth checking whether the
inspector log directory grew during a warm fold.

**glibc.** Honest answer: RHEL 8 cannot run our wheels and that is not going to change on our
side. The ttnn wheel is tagged `manylinux_2_34` and really does import `GLIBC_2.34` symbols;
RHEL 8 is 2.28. tt-bio itself is pure Python, so the floor is entirely the ttnn wheel, which
Tenstorrent publishes rather than us. Upstream lists Ubuntu 22.04 as the supported OS for
Blackhole. A `manylinux_2_28` build is an upstream packaging decision and I do not want to
promise one I do not control, but I am happy to carry the request if it would unblock you.

If you can move, RHEL 9 is glibc 2.34 and matches the wheel tag exactly, so it is the smallest
step from where you are and gets you off a source build entirely. Ubuntu 22.04 is the
best-trodden path if the distro is not fixed.

**One caveat on the baselines you compared against,** since you should know what you are being
measured against. The p300c numbers in `docs/perf_baselines.json` were seeded on one specific
machine of ours on 2026-09-01 at tt-bio v0.7.2. The gate looks for a block matching your hostname
first and falls back to the card-level block when it does not find one, and that fallback block
is that same machine's numbers. So you are running v0.9.0 against a v0.7.2 reading from our box,
at a 15% threshold. I am reseeding those and will make the file say so. The four values you quote
are real numbers from real runs, but treat them as "what one of our boxes did at v0.7.2", not as
a hardware spec.

I tried to re-verify them at v0.9.0 today and could not: our box was busy with other work all day
(loadavg 13-25 on 8 cores) and fold wall-clock is exactly the measurement that load ruins. The op
counts and the per-op enqueue cost above are unaffected, because I took minima over chunks and
contention can only add time. I will send you a clean set once the box is free.

Last thing, since you have four chips. The fold's own host-side work runs about 3 threads on our
box. Four concurrent folds would want roughly 12 cores from an 8-core 9700X, so expect per-chip
throughput to fall if you fan out to all four. Get the single-chip number right first.

Happy to get on a call if it is easier than email, and if you send me the output of the probe plus
your `CMakeCache.txt` build type I can probably tell you straight away whether the rebuild is
worth it.

Best,
Moritz

---

DRAFT-READY
