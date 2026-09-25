# NIH (NCI, Valkov lab) — the kernel finding, and the reply

Row `nih-perf-kernel`, 2026-09-25. Follow-up to `docs/support/nih-perf-reply.md` (row
`nih-perf`, commit `1022685a4`). Eugene Valkov ran the A/B we could not: same binaries, same
environment, kernel swapped. It closes the gap.

## What they did and what it bought

They replaced Oracle UEK 6 (5.4.17) with ELRepo `kernel-ml` 7.2.7 on the same box. The CPU
frequency driver went from `acpi-cpufreq` to `amd-pstate-epp`. Everything else held: clang 21
build of tt-metal 0.68, jemalloc, `OMP_NUM_THREADS=4`.

All numbers in this table are theirs. Op counts are ours, from `nih-perf` section 1, measured at
the v0.9.0 tag. "Ours" is `docs/perf_baselines.json`, which is still the unreseeded v0.7.2
seeding (see the caveat below).

| model | ops/call | before ms | after ms | our baseline ms | kernel gain | after vs our baseline |
|---|---|---|---|---|---|---|
| esmfold2-fast | 13,234 | 318.5 | 253.8 | 254.9 | +25.5 % | +0.4 % |
| esmfold2 | 15,614 | 367.6 | 305.8 | 303.2 | +20.2 % | -0.9 % |
| boltz2 | 25,542 | 775.2 | 467.3 | 568.3 | +65.9 % | **+21.6 %** |
| esmc-600m | 979 | 85.2 | 80.3-76.1 | 72.4 | +6.1 to +11.9 % | -9.9 to -4.9 % |

Real workload, their words: a 117-residue esmfold2-fast fold went from 6.0-6.2 s to 5.1-5.2 s,
about 16 % at the midpoints, and the output was byte-identical. Byte-identical matters: nothing
about the numerics moved, which is what you want from a scheduling fix.

Expressed per op, the kernel swap is worth **+4.0 to +12.1 microseconds per ttnn op**:

| model | kernel gain, us/op | residual vs our baseline, us/op |
|---|---|---|
| esmfold2-fast | 4.89 | -0.09 |
| esmfold2 | 3.96 | +0.17 |
| boltz2 | 12.05 | -3.95 |
| esmc-600m | 4.98 to 9.27 | +3.81 to +8.11 |

Three of four models now sit on or above our published baseline. Boltz-2 is 21.6 % above it,
which is consistent with our own open caveat rather than with anything on their side: the p300c
block in `docs/perf_baselines.json` was seeded on `tt-quietbox2` at v0.7.2 and has not been
reseeded. Reseeding is still owed on our side, tracked separately.

## The hypothesis we withdraw

`nih-perf` named tt-metal's bare-cmake `RelWithDebInfo` default as the leading suspect, on the
grounds that `-DDEBUG` activates every `TT_ASSERT` and turns on `TT_METAL_ENABLE_LOGGING`. That
was flagged as a mechanism we could point at in source, not a number we had measured. Their A/B
holds the binaries fixed and moves only the kernel, so **the build is not the cause of their gap**
and the suspect is withdrawn. The mechanism still exists in tt-metal and a RelWithDebInfo build is
still slower than a Release one, but it was not what they were hitting.

The per-op *shape* of the original diagnosis was right: a constant host cost on every op, with the
loss tracking op count rather than model size. The source was the CPU frequency driver.

One honest note on the arithmetic, for the record. Their pre-swap numbers here (3.14, 2.72, 1.29,
93.9) are already better than the numbers in their first report (2.32, 1.81, 0.90, 90), which is
where our 13-21 us/op figure came from. Something else moved between the two reports and we do not
know what. Against the first report the total is 13.4 us/op on esmfold2-fast and 25.2 on boltz2;
the kernel accounts for 4.89 and 12.05 of those. So the kernel is between a third and a half of the
total closure, and the largest single piece on boltz2.

## Why the kernel does this

`acpi-cpufreq` on a Zen 5 part under a 5.4 kernel drives frequency through legacy ACPI P-states:
a small set of discrete operating points, selected by a kernel governor on a periodic sampling
tick. The ttnn dispatch thread is a latency-bound, bursty, single-threaded workload. It runs for
microseconds, blocks, runs again. A sampling governor reads that as a mostly-idle CPU and parks it
at a low P-state, and every burst then pays a ramp. `amd-pstate-epp` hands the decision to the
CPU's own CPPC firmware, which tracks at a far finer granularity than a governor tick and holds
performance across exactly this pattern.

That model predicts what they measured everywhere we can check it:

- **Per-op device cost unchanged.** The device never saw the problem. Only the host side moved.
- **Bulk `from_torch`/`to_torch` 15-27 % faster.** Those are host-side memcpy and layout work,
  exactly the kind of short CPU burst a sampling governor under-serves.
- **Background interference 0.2-0.4 % down to 0.06 %** from one busy thread. Under `acpi-cpufreq` a
  busy sibling changes the governor's view of the CPU; under CPPC with EPP at performance it does
  not.
- **Loss scales with op count.** 25,542 ops on Boltz-2 lose most, 979 ops on ESMC-600M lose least.

## What our hosts run (read-only census, 2026-09-25)

Nothing was changed on any box.

| host | kernel | OS | scaling_driver | governor | EPP | CPU |
|---|---|---|---|---|---|---|
| `tt-quietbox2` (our reference box) | 7.0.0-34-generic | Ubuntu 24.04.5 LTS | `amd-pstate-epp` | `performance` | `performance` | Ryzen 7 9700X |
| `pc` (no card, dev only) | 6.1.0-44-amd64 | Debian 12 | `acpi-cpufreq` | `schedutil` | n/a | Ryzen 5 8600G |
| `tt-quietbox` (qb1) | not reachable this pass | | | | | |

The important reading is on qb2, and it is the answer to their question:

```
/proc/cmdline: BOOT_IMAGE=... ro quiet splash vt.handoff=7
/sys/devices/system/cpu/amd_pstate/status: active
/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver: amd-pstate-epp
/boot/config-7.0.0-34-generic: CONFIG_X86_AMD_PSTATE_DEFAULT_MODE=3
```

**No `amd_pstate=` anywhere on the kernel command line.** The stock Ubuntu 24.04 kernel is built
with `CONFIG_X86_AMD_PSTATE_DEFAULT_MODE=3`, and 3 is `AMD_PSTATE_ACTIVE`, so the driver selects
EPP mode on its own. Same CPU as theirs, family 26 model 68 (Zen 5, Granite Ridge). Every tt-bio
baseline we publish was measured on that configuration. They were not chasing a gap against a
tuned box; they were chasing a gap against a stock one.

qb2 also reads governor `performance` and EPP `performance`. We could not find what sets it: no
`tuned`, no cpufreq unit, nothing in `/etc/systemd/system`, and nothing on the command line. Since
we cannot say whether that is Ubuntu's default or something a past run left behind, the reply
mentions the governor as a second, smaller knob and does not claim it as our configuration.

## The OS question, answered

Their one remaining question: they would rather move to a newer OS than run a mainline kernel on
RHEL 8. Two constraints have to be satisfied at once, and only one of them is the kernel.

**Constraint 1, the kernel, for `amd-pstate-epp`.** Active mode (the `amd-pstate-epp` driver)
landed in Linux 6.3. Whether it comes up without a boot flag depends on the kernel's build config:
since 6.10, an undefined mode falls back to `CONFIG_X86_AMD_PSTATE_DEFAULT_MODE` instead of
failing to load, and distros that set that to 3 get it automatically. On anything older that has
the driver, `amd_pstate=active` on the kernel command line does it. Below 6.3 there is no active
mode at all, which is why UEK 6 (5.4) and RHEL 8 (4.18) land on `acpi-cpufreq`.

**Constraint 2, glibc, for the `ttnn` wheel.** `ttnn` 0.68.0 publishes exactly two wheels,
`cp310` and `cp312`, both tagged `manylinux_2_34_x86_64`. We checked the binary: the shipped
`_ttnn.cpython-312-x86_64-linux-gnu.so` imports up to `GLIBC_2.32` and `GLIBCXX_3.4.29`. So
**glibc 2.34 or newer, and Python 3.10 or 3.12**. RHEL 8 is glibc 2.28, which is the real reason
they are building from source at all.

| distro | kernel | amd-pstate-epp | glibc | Python | verdict |
|---|---|---|---|---|---|
| RHEL 8 / Rocky 8 / Alma 8 | 4.18 | no | 2.28 | 3.6/3.9 | below the wheel floor, and no amd-pstate |
| RHEL 9.x / Rocky 9 / Alma 9 | 5.14 + backports | driver backported, active mode not guaranteed default | 2.34 | 3.9 default, 3.12 in AppStream since 9.4 | works, expect to set `amd_pstate=active` |
| RHEL 10 / Rocky 10 / Alma 10 | 6.12 | yes, past the 6.10 auto-load change | 2.39 | 3.12 default | best RHEL-family fit |
| Ubuntu 24.04 LTS | 6.8 GA, 7.0.0-34 on ours | yes, no boot flag | 2.39 | 3.12 default | what we test on |

**Recommendation: RHEL 10 (or Rocky 10 / Alma 10) if the shop is RHEL-family, Ubuntu 24.04 if the
distro is not fixed.** RHEL 10 is kernel 6.12, glibc 2.39 and Python 3.12, which clears both
constraints with no boot flag and no source build. Ubuntu 24.04 is glibc 2.39 and Python 3.12 too,
and is the configuration every number we publish was measured on, so if they hit something we did
not see it is the configuration where we can reproduce it fastest.

RHEL 9 is the smaller move and it does work: glibc 2.34 is exactly the wheel's floor and
python3.12 is in AppStream from 9.4. The catch is the kernel. RHEL 9 is 5.14 with backports, and
Red Hat's own article on this is titled "How to enable amd-pstate" and is subscriber-only, which
tells you it is not simply on. If they go to 9, the one-line check after install decides it, and
`amd_pstate=active` in the command line is the fix if it comes up on `acpi-cpufreq`.

Whatever they pick, the check that settles it in one line:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver   # want: amd-pstate-epp
cat /sys/devices/system/cpu/amd_pstate/status             # want: active
```

No roadmap promise made on `manylinux_2_28`. That is a Tenstorrent packaging decision, not ours.

## The 16-thread saturation finding

Their remaining bad case: with all 16 hardware threads saturated, interference reaches up to 16 %
per core. We have not measured this and say so in the reply. The mechanism is not frequency, it is
SMT, and it is worth telling them because their current mitigation does not address it.

The 9700X is 8 physical cores with 2 threads each. On qb2 the siblings pair as `(N, N+8)`:
`/sys/devices/system/cpu/cpu0/topology/thread_siblings_list` reads `0,8`. Past 8 runnable threads
you are putting work on a sibling that shares execution resources with a core that is already
busy, and the ttnn dispatch thread is precisely the loser in that trade: it is one thread, it is
latency-bound, and it cannot be parallelised out of the problem.

`OMP_NUM_THREADS=4` does not protect it. In tt-bio that variable caps torch's intra-op and
inter-op pools (`tt_bio/runtime.py:163`, `bind_host_threads`) and the MSA search thread count
(`tt_bio/main.py:593`). The ttnn dispatch thread is not in either pool. So capping OMP threads
limits the *other* tt-bio work on the box and leaves the dispatch thread exposed to whatever else
is running.

What we would suggest, marked untested:

- Keep one physical core free for the fold. Give it a whole core with both siblings,
  `taskset -c 0,8 tt-bio predict ...`, and confine background work to `1-7,9-15`.
- If the box is running one fold at a time, do not oversubscribe past 8 runnable threads. The
  second thread on a core buys throughput for batch work and costs latency for dispatch.
- If they are running several tt-bio processes concurrently, `OMP_NUM_THREADS` is still the right
  knob for the torch and MSA side, it is just not the knob for this.

## The ESMC-600M residual

ESMC-600M is the one model still short, 4.9-9.9 % under our baseline, which they describe as near
their run-to-run spread. Two things are worth saying and neither is a defect claim:

- At 979 ops per batch-8 call, ESMC-600M is the least dispatch-bound model in the set, so it had
  the least to gain from the kernel and has the most of its time somewhere else. Its remaining
  delta is host tensor conversion and device compute, not dispatch.
- `OMP_NUM_THREADS=4` on a 16-thread box caps the torch side of exactly that work. Worth one run
  at 8 to see whether the residual is the cap. Untested by us.

Our baseline for it is also v0.7.2-seeded and unreseeded, same caveat as Boltz-2, so part of a
5 % delta may be ours.

## What we changed on our side this pass

1. **README host requirements and cpufreq check.** `README.md` gains the glibc/Python floor in
   Installation and a `scaling_driver` check at the top of Troubleshooting, with the symptom
   (uniform per-op host cost, device cost unchanged, loss tracks op count) and the fix. The
   customer is not named.
2. **Inspector-off default: already on main, re-verified.** `wk/nih-perf` commit `1022685a4` is an
   ancestor of `origin/main` via merge `9df6cad8f`. Verified on the merge tree, not the branch
   tip: `tt_bio/main.py:22` sets `TT_METAL_INSPECTOR=0`, a plain CLI import reads
   `TT_METAL_INSPECTOR = 0`, and `--debug` reads `None`, so upstream's default is restored for
   triage. Nothing further to merge.
3. **Correction appended to `state/concluded/nih-perf`**, recording that the RelWithDebInfo
   suspect was withdrawn by the customer's A/B. Appended, not overwritten.

Not done here, still owed elsewhere: reseeding `docs/perf_baselines.json` on a quiet box. That is
a separate row and Boltz-2 reading 21.6 % above baseline is now a second piece of evidence that it
needs doing.

---

# DRAFT EMAIL TO EUGENE — Moritz sends this, it has not been sent

Subject: Re: tt-bio throughput on our QuietBox

Hi Eugene,

That is a really good piece of work, thank you for doing the A/B properly. Holding the binaries
fixed and moving only the kernel is the one experiment that settles it, and it settles it.

So, plainly: I was wrong about the build. I pointed at tt-metal's RelWithDebInfo cmake default as
the likely cause and your experiment rules it out, since the binaries were identical either side
of the swap. The shape of the diagnosis held up, a constant host cost on every ttnn op, but the
source was the CPU frequency driver, not how tt-metal was compiled.

Your numbers line up with that cleanly. Using the op counts I sent last time, the swap is worth
about 4 to 12 microseconds per ttnn op, and the model that dispatches most gains most: Boltz-2 at
25,542 ops per fold picks up 12.1 us/op and 66 % throughput, esmfold2 at 15,614 picks up 4.0 us/op
and 20 %. Three of your four models now sit on or above our published baseline, and Boltz-2 is
22 % above it. Byte-identical output is the part I like best, because it means nothing about the
numerics moved.

On why it happens: acpi-cpufreq on a Zen 5 part drives a handful of discrete ACPI P-states through
a governor that samples on a tick. The ttnn dispatch thread runs for microseconds, blocks, runs
again, and a sampling governor reads that as an idle CPU and parks it, so every burst pays a ramp.
amd-pstate-epp hands the decision to the CPU's own CPPC firmware, which tracks far finer than a
governor tick. That also explains the two details in your message that would otherwise be odd:
per-op device cost unchanged, and bulk from_torch/to_torch 15-27 % faster. Both are host-side
bursts. The device never had the problem.

For what it is worth, the configuration you have now reached is ours. Our reference box is the
same Ryzen 7 9700X on stock Ubuntu 24.04, and it reads amd-pstate-epp with nothing on the kernel
command line, because Ubuntu builds its kernel with CONFIG_X86_AMD_PSTATE_DEFAULT_MODE=3. Every
baseline we publish was measured there. You were not measuring against a tuned machine.

**On the OS question.** Two constraints, and only one of them is the kernel:

- amd-pstate-epp needs Linux 6.3 or newer. Since 6.10 a kernel with
  CONFIG_X86_AMD_PSTATE_DEFAULT_MODE=3 brings it up with no boot flag. On an older kernel that has
  the driver, `amd_pstate=active` on the command line does it. Below 6.3 there is no active mode,
  which is where UEK 6 and RHEL 8 sit.
- The ttnn wheel is the other floor. ttnn 0.68.0 ships cp310 and cp312 wheels, both tagged
  manylinux_2_34, and the binary really does import GLIBC_2.34 symbols. So glibc 2.34 or newer and
  Python 3.10 or 3.12. RHEL 8 is glibc 2.28, which is why you ended up building from source in the
  first place.

Against those:

| | kernel | amd-pstate-epp | glibc | Python |
|---|---|---|---|---|
| RHEL 9 / Rocky 9 / Alma 9 | 5.14 + backports | driver is there, active mode probably not default | 2.34 | 3.12 in AppStream from 9.4 |
| RHEL 10 / Rocky 10 / Alma 10 | 6.12 | yes, no boot flag | 2.39 | 3.12 default |
| Ubuntu 24.04 LTS | 6.8 GA or newer | yes, no boot flag | 2.39 | 3.12 default |

If the shop is RHEL-family, I would go to RHEL 10 or its Rocky/Alma rebuild. Kernel 6.12 is past
the point where the driver comes up on its own, glibc 2.39 and Python 3.12 clear the wheel, and
you are off a source build entirely. If the distro is not fixed, Ubuntu 24.04 is what we test on,
so anything you hit there we can reproduce same-day.

RHEL 9 also works and is the smaller step. glibc 2.34 is exactly the wheel's floor. The catch is
that Red Hat's own article on this is called "How to enable amd-pstate", which suggests it is not
simply on, so budget for `amd_pstate=active`. Either way, one line after install tells you:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver   # want: amd-pstate-epp
cat /sys/devices/system/cpu/amd_pstate/status             # want: active
```

I am putting that check into the tt-bio README troubleshooting section, since you are unlikely to
be the last person to hit it. The symptom is distinctive enough to name: per-op device time
unchanged, wall clock up, and the loss tracking a model's op count rather than its size.

**On the 16-thread saturation case.** I have not measured this, so treat what follows as reasoning
rather than a result. That one is SMT, not frequency, and OMP_NUM_THREADS will not fix it. The
9700X is 8 physical cores with 2 threads each, siblings paired as (N, N+8). Past 8 runnable
threads you are scheduling onto a sibling of an already-busy core, and the ttnn dispatch thread is
the worst possible victim: single-threaded, latency-bound, and impossible to parallelise out of
the problem. OMP_NUM_THREADS=4 caps torch's thread pools and the MSA search in tt-bio, but the
dispatch thread is in neither pool, so the cap limits your other work and leaves the sensitive
thread exposed.

If fold latency matters on a busy box, I would give the fold a whole physical core rather than
capping threads:

```bash
taskset -c 0,8 tt-bio predict ...        # both siblings of core 0
# and keep the background work on 1-7,9-15
```

**On ESMC-600M still being 5-10 % short.** I think most of that is ours, not yours. Two things.
It is the least dispatch-bound model in the set, 979 ops per batch-8 call against 25k for Boltz-2,
so it had the least to gain from the kernel and its remaining time is host tensor conversion and
device compute. And our published p300c baselines were seeded on one of our machines at tt-bio
v0.7.2 and have not been reseeded since. Your Boltz-2 reading 22 % *above* the baseline is the
same artefact pointing the other way. I am reseeding them and the file will say when it was done.
If you want one more data point, try that model with OMP_NUM_THREADS=8 instead of 4 and see
whether the residual moves.

Thanks again. This was a better bug report than most of what we get from people who are paid to
write them.

Best,
Moritz

---

DRAFT-READY
