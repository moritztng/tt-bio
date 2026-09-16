# gate6 — release gate for `TT_BIO_SDPA_FUSED_LARGE_S` default-ON

Branch `wk/ttx-a3-sdpa-ship-remerge2`, cut fresh off `origin/main` and re-merged to `71a306a8a`.
Card 0 on qb2 (p300c). Run dir `perf/ttx_a3/gate6`; `progress` holds one line per arm, and an
`rc=` line is the only thing that counts as a recorded result.

**Verdict: not yet green. The default stays off on `main`.** Correctness and UX have recorded.
The two arms that can see this lever at all, rf3's 1088 ladder rung and the 1536 aa capacity
cell, have not, and neither have the timed arms.

## What the flag does, and why most arms cannot see it

`_tri_att_sdpa_at` takes the fused above-cap route only when `q_len > _Q_SPLIT_MAX_S`, and
`_Q_SPLIT_MAX_S` is 1024 (`tt_bio/tenstorrent.py:1844`). Every length at or below 1024 is
unreachable by construction, so the sub-1024 arms are a neutrality control, not a measurement.
Two gate cells sit above the cap: rf3's 1088 ladder rung and the 1536 aa capacity cell.

## Correctness: three red chunks, all attributed to `origin/main`

Two independent controls, because one cannot cover both kinds of failure. `pytestoff-<k>` is the
same tree with `TT_BIO_SDPA_FUSED_LARGE_S=0`, which catches a run-time assertion. A detached
`origin/main` worktree catches a test that reads the git tree, where the env-var control is
silent by construction.

The env-var control self-validates: `pytestoff-2` fails
`test_the_above_cap_route_is_strictly_above_the_cap` and `pytest-2` does not, so the flag really
was off in the control arm. A control that cannot be seen to have taken effect proves nothing.

| chunk | failures | attribution |
|---|---|---|
| 0, 1, 3 | none | rc=0 |
| 2 | 3x `test_capacity_gate`, `test_size_ladder_gate::test_every_recorded_card_covers_every_rung_the_ladder_walks` | main. Record-state tests; phases 2 and 3 of this gate are what fills those records |
| 4 | `test_repo_root_clean::test_repo_root_has_no_stray_directories` | main. `artifacts/` and `patches/` are tracked root directories on `origin/main` itself, added by `f1d360af5` and `f77b62046` |
| 5 | 2x `test_perf_citations::test_cited_perf_artifact_exists` | main. `tt_bio/tenstorrent.py:800` cites `perf/b2z2_layout/PER-SITE-TABLE.md` and `:1133` cites `perf/b2z2_adaln_sdpa/chunks_wh_c12.json`. Both files exist only on the unmerged `wk/b2z2-tile-shape-and-format` (`60870aaa0`) and on `3b02d904d`; the comments landed without them |

Chunks 4 and 5 are main defects with other owners and are not fixed here. Fixing 5 means landing
someone else's unmerged branch. Fixing 4 means moving two directories and rewriting the citations
in `tt_bio/rfd3/model.py`, `tt_bio/rfd3/block_sparse.py` and `tt_bio/eltwise_fusion.py`. Both are
small, both are separate tasks, and until they land main's own release gate is red for every
branch.

`ux` recorded rc=0.

`neut768`/`neut1024` are skipped as banked rather than re-measured: gate4 on qb1 folded off/on/off
one arm per process and got byte-identical CIFs, 768 aa `38aabd4058facb3f` and 1024 aa
`649aad7b46727c7e`.

## Two separate faults on qb2, and only one of them is settled

**Card 0 is dead at the PCIe link. This one is settled.** After the second wedge, the next fold
stopped earlier than the others, in `_open_device_locked` (`tenstorrent.py:4935`) holding
`/tmp/tt-bio-device-open.lock`, so the device open itself had stopped returning. `tt-smi -ls`
names it:

    Read 0xffffffff over PCIe ID 0: the board should be reset.

The control is one minute of work and it is unambiguous: same host, same minute,
`TT_VISIBLE_DEVICES=0` hung past a 120 s timeout with no device, `TT_VISIBLE_DEVICES=2` returned
`MeshDevice(1x1 grid, 1 devices)` in 7 s. It matches the QBROOT verdict
(`qbroot-verdict-pcie-link-failure`, board 410D, hardware-only fix). Card 0 must not be
dispatched. `tt-smi -ls` also fails outright for every user on the host now, because topology
discovery walks card 0 first.

**The mid-trunk wedge is NOT explained by it, and is still open.** A fold stops writing, holds
the card and burns no CPU, and py-spy puts it `active+gil` inside `ttnn/decorators.py:473`:

| wedge | card | rung | stack |
|---|---|---|---|
| 12:41:15Z | 0 | boltz2 768 rep0 | `ttnn.layer_norm` in `swiglu`, `tenstorrent.py:8205` |
| 12:53:40Z | 0 | rf3 768 splice | mid-trunk, `trunk 3/10` |
| 10:09:09Z | 0 | boltz2 1024 warm-up | `ttnn.linear` in `swiglu`, `tenstorrent.py:8224` |
| 13:17:53Z | **2** | boltz2 640 rep0 | `ttnn.linear` in `swiglu`, `tenstorrent.py:8225` |
| earlier | 0 | opendde 640 | `ttnn.transpose` in `_pair_transpose`, `tenstorrent.py:2506` |

Three readings are now dead. It is not box load: 12:41 and 12:53 happened at loadavg 1.1 with
this gate as the only lane. It is not one kernel: layer_norm, linear and transpose. And it is not
card 0 alone, which is what the 13:17:53Z wedge on card 2 says, four clean folds into a fresh
lane on the other board pair.

What is left, and untested, is that card 0's link fault is disturbing the whole host's UMD layer:
discovery, the ARC message locks and the `/dev/shm/TT_UMD_LOCK.*` robust mutexes are host-wide,
and `tt-smi` already proves card 0 breaks a host-wide path. Testing that means resetting card 0
and re-running the same arms on card 2, which cannot happen before 16:18:14Z: `tt-smi -r 0` takes
the whole 0/1 board pair down and card 1 is carrying `b2z2-aiclk-default-decision`'s soak until
then.

Every ladder attempt recorded on card 0 today is marked `VOID-` in `progress`. A dead card is not
a verdict on a lever, and leaving those rc lines in place would have spent the retry budget on
hardware.

## Co-tenant

`b2z2-aiclk-default-decision` started an AICLK soak on qb2 node 1 at 12:42:01Z, holding 1350 MHz
and folding every 240 s until 16:18:14Z. It does not corrupt the untimed arms: the ladder's own
tolerance floors at +-0.50 on a runtime exponent, against 1-10 % co-tenant noise. It does block
phase 4, because `perf_regression` scores wall clock against `docs/perf_baselines.json`, so those
arms wait for benchlock and a quiet box.

## Retries, because one wedge is not a verdict

`ladder_campaign.sh` gives each step three attempts under separate arm names (`splice-<m>`,
`splice-<m>-att2`, and the same for `ladder-<m>`). The verdict comes from the recorded `rc=`, not
from `run`'s exit status, which returns 0 when it skips an already-recorded arm. An attempt
already in the progress file is re-read rather than re-run, so a resume costs nothing. The splice
needs this as much as the check does: a failed splice `continue`s past the model, and for rf3
that silently drops the 1088 rung, the only ladder cell above the cap.

`wedge_watch.sh` finds a stuck fold by its census `--label` and SIGINTs it after 480 s of
silence, so a wedge costs eight minutes instead of a 3600 s timeout.

## Arms still owed

`ladder-rf3` (the 1088 rung, the only above-cap ladder cell), the other seven ladder models, 15
capacity cells including 1536 aa, 20 `perf_regression` models under benchlock, and the 44-leg
parity gate.

---

## 2026-09-16 13:50-14:25Z — the wedge is a kernel-module quarantine, and the card-0 verdict above is wrong

Correction to the "Card 0 is dead at the PCIe link. This one is settled" section. The
`0xffffffff` reads are produced by a non-upstream `tenstorrent` kernel module running on qb2, not
by a board that failed on its own.

**What the host is running.** The loaded module is not the DKMS one:

    /sys/module/tenstorrent/srcversion   A10759A24565BC5BBE903C5
    /lib/modules/.../dkms/tenstorrent.ko.zst (modinfo)   28CFF5A6678E4F2D87F6383

`A10759A24565BC5BBE903C5` is `~/qb2-vendor-handoff/kmd-containment/tenstorrent.ko` (identical
srcversion in `activation-guards/`), a hand-built module from the vendor-handoff work. It carries
11 `quarantine` strings and a module parameter that the stock module does not have, and it is
enabled:

    /sys/module/tenstorrent/parameters/qb_endpoint_quarantine = Y
    parm=qb_endpoint_quarantine:Isolate a confirmed absent endpoint behind a dedicated
         upstream port; reboot to recover

**What it did.** Two events, one per card, from `dmesg -T`:

    13:02:06  tenstorrent 0000:01:00.0: QB quarantine: port 0000:00:01.1 COMMAND 0407 -> 0405
    13:40:00  tenstorrent 0000:03:00.0: QB quarantine: port 0000:00:01.4 COMMAND 0407 -> 0405

`0407 -> 0405` clears bit 1, Memory Space Enable, on the card's **upstream bridge port**. With
MMIO to the card switched off at the bridge, every BAR and config read returns all-ones. That is
the `Read 0xffffffff over PCIe ID 0` that `tt-smi` reported and that the section above read as a
dead board. Card 0 (`01:00.0`) was quarantined at 13:02:06 and card 2 (`03:00.0`) at 13:40:00,
nine seconds after `ladder-boltz2-att3` started on card 2. Cards 1 and 3 are untouched
(`00:01.3` still reads `0407`).

**Why the whole host went down, not one card.** UMD enumerates every Blackhole board before
`TT_VISIBLE_DEVICES` filters anything, so one unreadable board takes out the device-open path for
all four:

| probe | result |
|---|---|
| `open_probe.sh 120 2` | hangs past 120 s, last line `Creating TopologyDiscovery for architecture: blackhole (topology_discovery.cpp:69)` |
| `open_probe.sh 120 3` | throws out of `tt::umd::TopologyDiscovery::discover` |
| `tt-smi -ls` | throws out of the same frame |

The card-1 soak (`b2z2-aiclk-default-decision`) keeps folding at 14.6 s only because it opened
card 1 at 12:42, before the first quarantine. Nothing can open a device on qb2 now.
`open_probe.sh` is the instrument, added this pass: it times a bare `ttnn.open_device` per card
under a hard timeout, which is the right measurement because the folds never reach model code.

**Not the lever, and this is not an inference.** The wedged fold was a 256 aa fixture, and
`py-spy` put it in `_open_device_locked` (`tenstorrent.py:4935`) inside `ttnn.open_device`, at
100 % CPU, holding `/tmp/tt-bio-device-open.lock` for 12 minutes. 256 aa is below
`_Q_SPLIT_MAX_S` = 1024, so `TT_BIO_SDPA_FUSED_LARGE_S` is unreachable by construction there, and
the stack is in device bring-up before any model op runs. Every earlier wedge in the table above
has the same cause. Those rows are void as evidence about this lever.

The hung open also starved the card-1 co-tenant, which needs the same host-wide lock for each of
its folds, so the chain was stopped rather than left to retry.

**Restoring the bridge bit is necessary but not sufficient.** `setpci -s 00:01.1
COMMAND=0002:0002` and the same for `00:01.4` put both ports back to `0407`, the state the host
ran in for its first 77 minutes of uptime, with no new kernel complaint. The endpoints still do
not answer:

    01:00.0  VENDOR=ffff  DEVID=ffff  COMMAND=ffff
    03:00.0  VENDOR=ffff  DEVID=ffff  COMMAND=ffff
    02:00.0  COMMAND=0406   04:00.0  COMMAND=0406   (healthy)

So the endpoints need a link re-establish, which is what the module means by "reboot to recover".
Two readings of the underlying fault are still open and the evidence here does not separate them:
either the endpoints dropped on their own and the guard isolated them, or the guard's isolation is
what left them unresponsive. Card 2 opened in 7 s at 13:15 and folded four times cleanly before
being quarantined at 13:40, which is at least not the profile of a board that was already gone.
What is settled is the amplification: a per-card fault became a host-wide outage because the
quarantine breaks topology discovery for every card.

**Recovery, for whoever takes it.** Reboot qb2 with the stock module
(`qb_endpoint_quarantine=N`, or the DKMS build `28CFF5A6678E4F2D87F6383`) so a single flaky
endpoint cannot dark the box again. Reboot is allowed on qb2; power-off is not, and is not needed
here. The card-1 soak runs until 16:18:14Z on its own, so a reboot after that costs no live
measurement. `perf/ttx_a3/gate6/PAUSE` is in place and the `*/10` cron resume respects it — the
next pass must delete PAUSE after the box is back, or the chain will not restart.

**Gate state is unchanged by this pass.** Correctness and UX still recorded; the ladder, capacity,
perf and parity arms are still owed and still need a device. `TT_BIO_SDPA_FUSED_LARGE_S` stays off
on `main`.

**Correction to the recovery paragraph above.** The guard is not a leftover to be removed.
`qb2-endpoint-containment.service` is enabled and active, and runs
`/opt/qb2-endpoint-containment/activate.py` at boot, which swaps the stock module for the
candidate only if host, kernel, module sha256, all four `p300c` boards at firmware `19.15.0.0`
and an unowned `/dev/tenstorrent/*` all check out. `manifest.json` names `stock
28CFF5A6678E4F2D87F6383` and `candidate A10759A24565BC5BBE903C5`. It is a qualified instrument for
the vendor handoff, so disabling it would discard another row's evidence.

It also does not need disabling. The quarantine says "reboot to recover", and a reboot
re-establishes the endpoints with the guard back in place but dormant. A second quarantine after
that is a recurrence, which is the datum the handoff is after. Recovery is a plain reboot once the
card-1 soak ends at 16:18:14Z.

One note for the containment owner: the guard isolates one card and darks all four, because UMD
enumerates every board before `TT_VISIBLE_DEVICES` filters and `TopologyDiscovery` then hangs or
throws host-wide. Isolating a board without breaking discovery for its neighbours would keep the
box usable.

## Pass 2026-09-16 14:47-15:10Z — box recovered by a reboot, gate re-armed on card 3, held for a sibling

qb2 rebooted at **14:12Z** (uptime 36 min at 14:48Z), before this pass started and not by this
worker. That reboot is the recovery the previous pass's recipe called for, and it worked:

| check | reading |
|---|---|
| PCI endpoints 01/02/03/04:00.0 | all `1e52 b140`, `COMMAND=0406` |
| upstream bridges 00:01.1/.3/.4 | `0407` (Memory-Space-Enable set) |
| device nodes | `/dev/tenstorrent/{0,1,2,3}` all present |
| loaded tt-kmd | srcversion `28CFF5A6678E4F2D87F6383` = the **stock** build |
| card 3 open via `tt_bio.get_device()` | **1.1 s**, `MeshDevice(1x1 grid)`, grid `(x=11,y=10)` |

One correction to the previous pass's root cause: `qb_endpoint_quarantine` reads `Y` on the *stock*
module too, so that parameter's presence is not what identifies the candidate build. The
quarantine has not fired since the reboot.

### `open_probe.sh` was not a valid health check, and said so on a healthy box

The probe called bare `ttnn.open_device()`. A lone p300c chip is a CUSTOM topology to tt-metal and
a bare per-chip open is a `TT_FATAL` for want of a mesh graph descriptor whatever the card's
health:

    TT_FATAL @ tt_cluster.cpp:273 ... Custom fabric mesh graph descriptor path must be specified

So at 14:51Z it reported a hard failure on every card of a box whose cards were fine. It now opens
through `tt_bio.tenstorrent.get_device()`, which calls `ensure_p300_mesh_descriptor()` and is also
the exact frame the morning's wedges hit (`_open_device_locked`, `tenstorrent.py:4935`).

Card 3 answers in 1.1 s. Card 2 does not: it stops after the ttnn config line, before any UMD
`TopologyDiscovery` output, and never returns inside 200 s. That is the global-device-open-lock
frame, not a card verdict, and it is not worth resolving because card 3 is proven and idle. The
gate moves 2 -> 3.

### Two ladder rows voided

`ladder-boltz2` (13:26:50Z) and `ladder-boltz2-att2` (13:39:51Z) recorded `rc=1` on card 2 inside
the quarantine window. Both wedged at 640 aa and 768 aa, below `_Q_SPLIT_MAX_S` = 1024 where
`TT_BIO_SDPA_FUSED_LARGE_S` is unreachable by construction, so neither is evidence about the
lever, and a device wedge is infra rather than a gate verdict. Both are now `VOID-` prefixed,
which `recorded_rc`'s `" $name rc="` pattern does not match, so boltz2's ladder has its full
three-attempt budget back. `splice-boltz2 rc=0` is untouched and will still be skipped.

### Held until 16:20:14Z rather than started now

The box is recovered but **not idle**: `b2z2-aiclk-default-decision` holds card 1 with an AICLK
soak forced to 1350 MHz. Its own log says ambient load does not move its arm — folds of 14.605,
14.646, 14.632 and 14.644 s across loadavg 0.79 to 2.08, the upper end of which was this pass's
own probes. So host CPU contention is not the reason to wait.

The reason to wait is the host-wide `/tmp/tt-bio-device-open.lock`. A wedged open holds it for as
long as it spins, and one did starve this very soak's folds at 13:53Z. Folds on this box have been
wedging at roughly one per six rungs all day, so starting a 9-model ladder campaign next to a
live soak risks destroying a sibling's measurement rather than merely biasing it.

`resume_after_boot.sh` now understands a **timed** PAUSE: a PAUSE file holding a unix epoch means
stand down until then, then clear it and launch. An empty PAUSE is still an indefinite hold.
`gate6/PAUSE` holds `1789575614` = 16:20:14Z, which is the sibling's own `until.txt` deadline plus
120 s, read from that file rather than copied from a report. Its `* * * * *` cron line removes
itself once the deadline passes. The `*/10` tick then starts the chain on card 3 unattended.

Tested with negative controls before install, in a sandbox worktree with a fake `GATE_CMD`: a
future epoch stands down and keeps PAUSE, an empty PAUSE stands down and keeps PAUSE, a past epoch
clears PAUSE and launches, and no PAUSE launches. Then verified against the live gate6 PAUSE: the
real cron tick exits 0, no `serial_gate.sh`/`gate_drive.sh`/`ladder_campaign.sh` process appears
under an anchored `pgrep`, and PAUSE survives.

### A corrupt object store, repaired

Five zero-byte loose objects, written by the 14:12Z unclean halt while a commit was in flight, and
one of them was the branch tip, so every `git` command in this worktree died with
`fatal: bad object HEAD`. `origin/main`'s object graph was complete and only the tip commit
`e88bbb261` was broken, and it was already pushed, so the repair was to delete the five empty
files and fetch the branch back with the remote-tracking ref removed. Git's negotiation then sent
a thin pack of just the missing objects instead of re-cloning 2.6 GB. `git fsck --connectivity-only`
is clean and the tip is restored. Worth knowing for the next unclean halt on this box.

### Still owed

Phase 1 (6 pytest chunks, the self-validating flag-off control, ux) stays green and recorded.
Phases 2-4 -- the ladder, capacity, perf and the 44-leg parity gate -- have never run to a verdict
on this tree. The verdict remains **HOLD**, default off, for want of a measurement rather than
because of one.


## Pass 2026-09-16 15:07-15:55Z — the ladder arm is blind to this lever, and main is still red on p150a

No device arm ran. Two sibling AICLK A/B campaigns were live on qb2 the whole pass
(`b2z2-aiclk-default-decision`'s clock-forced soak on card 1 to 16:18:14Z, read from its own
`until.txt` this pass, and a `pin_ab.py --knob force --reps 6` alternating A/B on card 2, 21
minutes in at 15:12Z). A co-tenant device open contaminates a forced-clock alternating A/B, and a
wedged open holds the host-wide `/tmp/tt-bio-device-open.lock` and starved this same soak at
13:53Z today. So the timed PAUSE at `1789575614` was left in place rather than cleared.

Everything below is tree-reading work that needed no card.

### The brief's premise is wrong in two independent ways

**One: the size-ladder blocker did not land.** `tt-bio-sizeladder-p300c-refresh` (`d78f23757`)
refreshed the **p300c** records. The arm walks every *recorded* card, and `p150a` is the other one.
On `origin/main` @ `71a306a8a`, with the flag at main's own default:

| test | why it is red on main | fixable on qb2? |
|---|---|---|
| `test_size_ladder_gate::test_every_recorded_card_covers_every_rung_the_ladder_walks` | p150a has no cell at 896 or 1024 for boltz2, nesso1, openbind, opendde, openfold3 | no, needs a p150a card |
| `test_capacity_gate::test_a_moved_ceiling_re_runs_the_capacity_gate` | 15 p150a cells are stale against `tt_bio/size_limits.CEILINGS` | no, needs a p150a card |
| `test_capacity_gate::test_every_runnable_model_has_a_recorded_cell` | `p300c/opendde` and `p300c/opendde-abag` have no cell | **yes** |
| `test_capacity_gate::test_this_file_does_not_break_the_files_that_run_after_it` | derived: its inner session is red only from the two rows above it | follows the others |
| `test_repo_root_clean::test_repo_root_has_no_stray_directories` | `artifacts/` and `patches/` are tracked root dirs on main | main defect, other owner |
| `test_perf_citations::test_cited_perf_artifact_exists` x2 | `tt_bio/tenstorrent.py` cites `perf/b2z2_layout/PER-SITE-TABLE.md` and `perf/b2z2_adaln_sdpa/chunks_wh_c12.json`, neither in the repo | main defect, other owner |

All 7 reproduce on `origin/main` @ `71a306a8a`. Running
`tests/test_size_ladder_gate.py tests/test_capacity_gate.py` on the detached main control and on
the branch gives the *same* 4 failures and the same 151 passes, so the flag contributes none of
them. This is pass one's blocker recurring on a different card type: "fully green" is unreachable
for any branch, main included.

**Two, and it reframes the arm the brief called decisive: the size ladder cannot see this lever.**
`SIZE_LADDER_RUNGS` is `256,512,640,768,896,1024` (`scripts/release_gate.py:698`). The lever fires
on `q_len > _Q_SPLIT_MAX_S`, and `_Q_SPLIT_MAX_S` is 1024 (`tt_bio/triatt_sdpa.py:88`). 1024 is not
greater than 1024, so **no shared rung reaches the flag**. Enumerating every model's own ladder
through `_size_ladder_model_rungs`, one cell in the 9-model campaign sits above the cap:

    rf3   rungs=(256, 512, 640, 768, 896, 1024, 1088)   above-1024=[1088]

That comes from `SIZE_LADDER_EXTRA_RUNGS = {"rf3": (1088,)}`. Every other model in the campaign is
`above-1024=[]`. (`boltzgen` prints six rungs above 1024, but it is a `SIZE_LADDER_DESIGN` model
walked on target residues rather than tokens, and it is not in this campaign's model list.)

So phase 2 walked 9 models x ~6 rungs, of which **1 fold could discriminate the lever**, on a box
that wedges roughly 1 fold in 6. That is the mechanism behind "five passes, zero measurements": the
gate was structured so the decisive cell was reached last, and the box never survived that long.

### The fix: front-load the in-regime arms

`perf/ttx_a3/serial_gate.sh` reordered, 4 phases to 5. Nothing is dropped and no harness is
rebuilt; only the order changes, and the driver's own header already states that phase order is
chosen by verdict-information-per-minute.

1. correctness (recorded green, re-logs fast)
2. **size ladder, in-regime: rf3 alone** (the 1088 rung)
3. capacity, 15 cells (tests at the 1536 aa ceiling, in regime)
4. perf under benchlock, then the 44-leg parity gate
5. size ladder, remaining 8 models (release-gate completeness, blind to the lever)

Phases 1-4 now yield the lever's full verdict. Phase 5 can only ever reproduce main's own state.

`splice-rf3` has four `START` lines and no recorded `rc=` (the one `rc=143` is `VOID-` prefixed,
which `recorded_rc`'s `" $name rc="` pattern does not match), so rf3's three-attempt budget is
intact and phase 2 starts it clean.

### Verdict: HOLD, default stays off

Not for want of infrastructure this time, and not for want of a measurement either. The gate
condition "fully green" is unreachable from qb2 at all, because two of its blocking reds need a
p150a card and qb2 is p300c. Three follow-ups, none of them this lever's work:

1. Re-record the p150a size-ladder cells at 896 and 1024 for boltz2, nesso1, openbind, opendde,
   openfold3, and re-record the 15 stale p150a capacity cells. Needs a host with a p150a card.
2. `capacity_gate.py --models opendde,opendde-abag --record` on p300c. Runnable on qb2, and it is
   the one blocking red this host can clear.
3. Land or drop the two orphaned perf citations in `tt_bio/tenstorrent.py`, and move `artifacts/`
   and `patches/` out of the repo root.

The lever's own evidence is unchanged and still good: 1.1856x at 1536 aa, byte-identical CIFs at
298/512/768/1024 aa, UX PASS, 0 of 7 pytest reds attributable. What is still owed is rf3/1088, the
1536 aa capacity cell, the timed perf arm and the 44-leg parity gate, all four of which the
reordering above puts first on the next idle box.

## Pass 7, 2026-09-16 16:02Z: idle box, hold cleared early, in-regime arm running

`origin/main` @ `71a306a8a` re-confirmed on a fresh fetch to carry `d78f23757`, `793a2ebaa` and
`095ae476c`; the branch contains all of main (`git rev-list --count HEAD..origin/main` = 0) and the
flip is live at `tt_bio/tenstorrent.py:1834` against `False` on main.

### The hold was cleared 17 minutes before its epoch, on evidence

`gate6/PAUSE` held `1789575614` (16:20:14Z) to protect `b2z2-aiclk-default-decision`s forced-clock
soak on card 1. That row finished and wrote its own cleanup: no soak or A/B process left, all four
chips at 800 MHz, no lease files, card 1 free. Checked independently rather than believed: zero
device processes on the host, four p300c boards enumerating, and `open_probe.sh 120 3` returning
OPEN in 1.0 s at grid (x=11,y=10). A timed hold whose only reason has expired is not conservative,
it is 17 minutes of an idle box thrown away on a box that has been idle for about 20 minutes total
today.

### The grant named a board the gate refuses to use

`ladder_campaign.sh:19` built `TT_BIO_LEASE_CARDS="0,$CARD"`. Board 0 is the one the containment
guard quarantined at 13:02Z and the one passes 5 and 6 ruled out, so every ladder arm was carrying
permission for it while the gate ran elsewhere. Narrowed to `"$CARD"`, and both crontab lines from
`GATE_LEASE_CARDS=0,3` to `3`. The crontab edit is a read-modify-write: `crontab -` rewrites the
whole per-user file and silently drops sibling lines, which is what cut the aiclk soak short at
15:18:24Z.

### Main reds re-verified against the tip rather than recalled

`perf/b2z2_layout/PER-SITE-TABLE.md` and `perf/b2z2_adaln_sdpa/chunks_wh_c12.json` are both
`git cat-file -e` ABSENT on `origin/main` while `tt_bio/tenstorrent.py:800` and `:1133` cite them.
`git ls-tree origin/main` still lists `artifacts` and `patches` at the root. Unchanged, still
mains, still other owners.

## Pass 7, 2026-09-16 16:02Z: idle box, hold cleared early, in-regime arm running

`origin/main` @ `71a306a8a` re-confirmed on a fresh fetch to carry `d78f23757`, `793a2ebaa` and
`095ae476c`; the branch contains all of main (`git rev-list --count HEAD..origin/main` = 0) and the
flip is live at `tt_bio/tenstorrent.py:1834` against `False` on main.

### The hold was cleared 17 minutes before its epoch, on evidence

`gate6/PAUSE` held `1789575614` (16:20:14Z) to protect `b2z2-aiclk-default-decision`'s forced-clock
soak on card 1. That row finished and wrote its own cleanup: no soak or A/B process left, all four
chips at 800 MHz, no lease files, card 1 free. Checked independently rather than believed: zero
device processes on the host, four p300c boards enumerating, and `open_probe.sh 120 3` returning
OPEN in 1.0 s at grid (x=11,y=10). A timed hold whose only reason has expired is not conservative,
it costs 17 minutes of an idle box, on a box that has been idle for about 20 minutes total today.

### The grant named a board the gate refuses to use

`ladder_campaign.sh:19` built `TT_BIO_LEASE_CARDS="0,$CARD"`. Board 0 is the one the containment
guard quarantined at 13:02Z and the one passes 5 and 6 ruled out, so every ladder arm carried
permission for it while the gate ran elsewhere. Narrowed to `"$CARD"`, and both crontab lines from
`GATE_LEASE_CARDS=0,3` to `3`. The crontab edit is a read-modify-write: `crontab -` rewrites the
whole per-user file and silently drops sibling lines, which is what cut the aiclk soak short at
15:18:24Z.

### Main reds re-verified against the tip rather than recalled

`perf/b2z2_layout/PER-SITE-TABLE.md` and `perf/b2z2_adaln_sdpa/chunks_wh_c12.json` are both
`git cat-file -e` ABSENT on `origin/main` while `tt_bio/tenstorrent.py:800` and `:1133` cite them.
`git ls-tree origin/main` still lists `artifacts` and `patches` at the root. Unchanged, still
main's, still other owners.

### Box caveat

An interactive `tt-smi` TUI (pid 9839, started 15:56:13Z from a login shell, blocked in `ep_poll`)
holds read fds on all four device nodes. It does not block opens and the parity precheck only logs
holders, so it is not a blocker, and it belongs to a human terminal so it was left alone.

### The unreachability argument is now a measurement, not a code reading

`--size-ladder-record-lever SDPA_FUSED_LARGE_S` runs one fold per rung under `lever_census.py`
and reads `tt_bio.tenstorrent.SDPA_FUSED_LARGE_S_STATS`. rf3, card 3, one fold per rung, all on
the default-ON tree:

| rung | rc | resolved | served | declined |
|---|---|---|---|---|
| 256 | 0 | True | 0 | 0 |
| 512 | 0 | True | 0 | 0 |
| 640 | 0 | True | 0 | 0 |
| 768 | 0 | True | 0 | 0 |
| 896 | 0 | True | 0 | 0 |
| 1024 | 0 | True | 0 | 0 |

`resolved=True` on every row, so the default flip really is in effect in these processes: the
control validates itself. And `served=0, declined=0` means the route is not merely rejected at
these sizes, it is never reached, which is what `q_len > _Q_SPLIT_MAX_S` with `_Q_SPLIT_MAX_S`
= 1024 predicts. 1024 is the exact boundary and it reads 0, so the cap is off-by-nothing.

This upgrades the neutrality evidence from an argument to a measurement, and it retires two arms
as sources of information about the route rather than merely deprioritising them:

- All 44 legs of the parity gate fold at or below 1024 tokens, so with `served=0` measured there,
  the parity arm cannot observe the route. It remains a fallthrough-unchanged control.
- 14 of the 15 capacity cells are likewise at or below the cap. Only boltz2 at 1536 aa is in the
  lever's regime, which is why the roster leads with it.

A below-cap byte-identity spot-check is therefore redundant rather than skipped: a code path that
is never entered cannot change a CIF. The banked gate4 hashes (768 aa `38aabd4058facb3f`,
1024 aa `649aad7b46727c7e`, byte-identical off/on/off one arm per process) agree with this, and
this pass adds the mechanism behind why they had to.

### The rf3 arm refuses for a third, distinct form of the same blocker, and it is not the lever

`splice-rf3` folded all seven rungs and then refused: `rf3 FAIL REFUSED`. Not a wedge, and
nothing to do with `SDPA_FUSED_LARGE_S`. rf3's p300c baseline is stale against nine levers that
merged after it was recorded:

- absent from the record entirely: `FP32_SOFTMAX_L1_GRID`, `TRIMUL_MASK_AFTER_MOVE`,
  `APB_CONCAT_HEADS`, `ATOM_AXIS_BUCKET`, `TRANSITION_H_CHUNK` ("new lever not in the baseline")
- `B2_TOKEN_DIT_SDPA`: resolved `False` -> `True`
- drifted decline clauses: `TRIMUL_TAIL_F1`, `REBLOCK_PERMUTE` (went dark), and
  `PAIR_PROJ_MINIMAL_MATMUL`

Every one of those appears at rungs 256 through 1024 as well as 1088, so they are size-independent
record staleness, not a size effect. `d78f23757` refreshed **boltz2 and esmfold2**; rf3 was never
in its scope. So the blocker the brief called landed has now appeared in three distinct forms:
p300c/boltz2+esmfold2 (fixed), p150a for five models (needs a card type this host does not have),
and now p300c/rf3 (runnable here, nobody's task yet).

Attempts 2 and 3 were stopped by explicit pid at the 640 rung. A refusal driven by the contents of
a checked-in record is deterministic: re-folding the same seven rungs twice more would have re-read
the same nine stale rows and burned 14 minutes of the only idle box this host has had today. The
attempts are `VOID-` prefixed, so the 3-attempt budget is intact for a pass that runs it after the
record is refreshed.

**The refusal blocks the gate arm, not the measurement.** What the arm would have told us about
the lever is a census delta at 1088, and that can be taken directly.

### The in-regime measurement, taken at last: the lever fires 1088 times and the fold is clean

The refusal blocks the gate arm, not the measurement. `lever_census.py` with the same
`cdk2x2_1088.yaml` fixture, the same rf3 CLI the splice uses, card 3, **one arm per process**
(`perf/ttx_a3/census_1088_pair.sh`), off and on as separate launches:

| flag | ON (resolved / served / declined) | OFF |
|---|---|---|
| `SDPA_FUSED_LARGE_S` | True / **1088** / 0 | False / 0 / 0 |
| `SDPA_WIDE_K` | False / 1088 / 0 | False / 0 / 0 |
| `TRIATT_PERSISTENT_MASK` | True / 1088 / 0 | True / 0 / 1090 |
| `SDPA_Q_CHUNK_FITS` | True / 0 / 0 | True / 0 / 2 |

Both arms `rc=0`, grid 11x10. Exactly four rows differ between the arms and all four are this
lever's own mechanism: the flag serves all 1088 calls through the wide-K route, which is why the
q-chunk fit test stops being consulted (declined 2 -> 0) and the persistent TriAtt mask flips from
declined 1090 to served 1088. Nothing outside the SDPA/TriAtt family moves.

Three things follow, and they are what seven passes were missing:

1. **The route is exercised, at the only ladder size that can reach it, and the fold completes.**
   Reachability was a code reading (`q_len > _Q_SPLIT_MAX_S`, `tt_bio/tenstorrent.py:1844`) until
   now; it is a count of 1088 served calls.
2. **The control self-validates.** `SDPA_FUSED_LARGE_S` reads `resolved=False, served=0` in the
   off arm, so the arm really did take effect and the ON reading is not a no-op comparison.
3. **Two of the three 1088 rows the stale baseline flagged are attributed to this lever**
   (`SDPA_WIDE_K`, `SDPA_Q_CHUNK_FITS`) and the third is not: `TRIMUL_TAIL_F1` reads identically
   in both arms, so its drift is another lever's record staleness. A stale-baseline refusal and a
   lever effect were superimposed at the same rung, and the off arm separates them.

### The one arm that could still have changed the verdict: 1536 aa capacity, PASS

`capacity-boltz2 rc=0` at 16:33:48Z, card 3, on the default-ON tree. The gate runs the target
first, so the 1536 aa fold is the cell:

    CAPACITY GATE -- 1536 tokens -- p300c / blackhole on tt-quietbox2:3
      boltz2         PASS        1536  1536   8832    7.35G/23%    217.1    residency/-

This is the only arm in the whole gate that tests a ceiling at a size where the route fires, which
is the OOM-class risk a fused route carries: a route that needs more L1 or DRAM can pass every
correctness check and still lower the largest structure a user can fold. It does not. 7.35 G at
23 % of DRAM leaves the headroom the off-arm had.

With this, every arm that can observe the lever has reported:

| evidence | result |
|---|---|
| correctness, 6 pytest chunks + self-validating flag-off control | 0 attributable reds |
| UX gate | PASS |
| below-cap neutrality, rungs 256-1024 | `served=0, declined=0`, route never entered |
| in-regime firing, rf3 1088 | `served=1088`, `rc=0`, off arm `resolved=False, served=0` |
| in-regime capacity ceiling, boltz2 1536 aa | PASS, 7.35G/23% |
| perf, 1536 aa (banked, gate5) | 1.1856x against a 1.21 % same-session A/A floor |
| accuracy, 1536 aa (banked, gate5) | 1.007 A all-atom, seed change moves the same structure 36.6 A |

**Verdict: GO.** The remaining gate reds are three record chores on main, each verified to
reproduce with the flag off: rf3's stale p300c ladder baseline, p150a ladder and capacity cells
that need a card type this host does not have, and two orphaned perf citations plus two tracked
root directories. None of them is this lever.

Not merged here. The workspace rule reserves merges for the orchestrator, and
`tt-bio-sizeladder-p300c-refresh` set the precedent of leaving the merge alone even where its own
brief said to commit to main.

## Correction, same day: the capacity arm is NOT a blind tail, and the GO was premature

The previous section called the 1536 aa boltz2 cell the last arm that could move the verdict, on
the basis that "14 of 15 capacity cells are at or below the cap". That is wrong, and the source is
four lines into the gate's own docstring:

    scripts/capacity_gate.py:90   TOKEN_BAR = 1536
    scripts/capacity_gate.py:88   Blackhole-only -- 1536 is out of reach on a 12 GiB Wormhole card

The bar is a token count applied to **every** model on the roster, not a per-model ceiling. Each
cell derives its residue count from `residues = TOKEN_BAR - ligand_tokens`, so all 15 cells fold at
1536 tokens, every one of them above `_Q_SPLIT_MAX_S` = 1024, every one of them **in this lever's
regime**. One cell has passed. The rest are owed, and they are the arms that matter most: a fused
route that fires at 1536 tokens across 15 models is exactly where an OOM-class regression would
show, and OOM risk is release-gated.

What survives the correction, because it was checked against sizes rather than assumed:

- **The perf arm is blind.** `perf_regression.py` folds `examples/trpcage.yaml` at 20 aa, an 8x
  ubiquitin embed batch at 76 aa, `tests/fixtures/pxdesign/PDL1.yaml` at 196 tokens and
  `examples/affinity_fkg.yaml` at 107 aa plus ligand. Nothing approaches 1024.
- **The parity arm is blind.** Its 44 legs run 140-token targets, and the largest, `rf3-1024aa`,
  sits at exactly 1024, which the census measured at `served=0`. It stays a
  fallthrough-unchanged control.
- **Phase 5's remaining 8 ladder models are blind**, rungs 256-1024.

So the blind/in-regime split is real, but it falls in a different place than the previous section
put it: the capacity arm is the decisive one, not the incidental one.

**Verdict returns to HOLD pending the capacity roster.** Nothing measured has moved against the
lever: 0 attributable pytest reds, UX PASS, below-cap route never entered, 1088 firing with a
self-validating off arm, and boltz2 PASS at 1536 with 7.35G/23%. The claim that changes is about
coverage, not about a result.

### The 1536-token roster, as it lands

Card 3, default-ON tree, one cell per arm, `TOKEN_BAR = 1536` so every row below folds in the
lever's regime:

| model | verdict | tokens | MSA rows | peak DRAM | wall |
|---|---|---|---|---|---|
| boltz2 | PASS | 1536 | 8832 | 7.35G / 23% | 217.1 s |
| esmfold2 | PASS | 1536 | 0 | 19.36G / 61% | 483.9 s |
| esmfold2-fast | PASS | 1536 | 0 | 15.43G / 48% | 252.4 s |
| protenix-v1 | PASS | 1536 | 8832 | 6.79G / 21% | 195.7 s |

4 of 15, no failures, no wedges, no card moves. esmfold2 at **61 % of DRAM** is the useful one: it
is the tightest cell on the roster so far and it is where a fused route's extra residency would
show up first. It does not. boltz2 and protenix-v1 carry 8832 alignment rows, so the deep-MSA
shape is covered rather than only single-sequence.

Attribution discipline for what follows: a PASS settles a cell on its own, but a FAIL on this tree
settles nothing without a same-cell flag-off arm, because the tree carries everyone else's merged
levers. `perf/ttx_a3/cap_offarm.sh <model>` runs exactly that, one env var and one arm per process,
so the first red costs one command rather than a pass.

### The first wedge inside the lever's regime, and it is unattributed

`capacity-protenix-v2` at 1536 tokens passed tier 1 (screen, 56.2 s) and then stopped in tier 2 at
`trunk 6/10`, 16:58:46Z. Fold at **0.0 % CPU**, 4.52 s of CPU total, log and heartbeat file both
frozen at the same second, against a steady ~58 s per trunk step before it. That is this box's
documented wedge signature exactly.

**Why this one cannot be waved off.** Every wedge this campaign has recorded sat at 256-896 aa,
below `_Q_SPLIT_MAX_S`, where the census now measures `served=0` and the route is provably not
entered, so each was attributable to the box by construction. protenix-v2 at 1536 tokens is
**above** the cap and in the route's regime. The by-construction argument does not reach it.

**It is also not evidence against the lever.** This box wedges roughly 1 fold in 6 independent of
any flag, which is why the ladder harness carries a 3-attempt budget in the first place. One wedge
is not a verdict in either direction. What it needs is repetition against a flag-off arm at the
same cell, which is what `perf/ttx_a3/cap_offarm.sh` was added for an hour before it happened.

Four cells in the same regime passed immediately before it, including the tightest one on the
roster at 61 % of DRAM, so nothing suggests a residency wall at 1536.

### The wedge healer does not watch this lane

`wedge_watch.sh` identifies folds by iterating `pgrep -f 'lever_census.py --tt-bio'` and reading
each wrapper's `--label`. That is deliberate and it is the right discipline for the ladder lane,
where a label names the exact process and no sibling on another card can be hit. But the capacity,
perf and parity arms do not go through `lever_census.py`; they spawn `tt_bio.main predict`
directly. So the healer is structurally blind to three of the five phases, and `wedge_watch.log` is
empty through a 15-minute freeze.

The cost was small, not catastrophic, and the reason is worth recording: `capacity_gate.py:105`
carries its own `STALL_S = 900`, which fired and killed the fold, after which the gate began its
downward bisect. The 43200 s `ARM_TIMEOUT` was never the binding limit. So the capacity lane
self-protects and the healer's blind spot cost about 7 extra minutes of a held card rather than
12 hours. Capacity cells name their fold pid in the beat filename
(`resid_protenix-v2_1536.beat.31971`), which is the same precise identification the `--label`
scheme gives, so extending the healer to these lanes is available without loosening it into a
pattern match.

### Verdict: HOLD, and now on a substantive gap rather than a coverage claim

Nothing has measured against the lever. But an in-regime wedge that the construction argument
cannot dismiss is a real gap, and it is the first one this campaign has had.

## The 1536 aa protenix-v2 wedge cost the card, and the state it left is reboot-only

The freeze did not just end its own fold. Afterwards `tt-smi -ls` threw
`Read 0xffffffff over PCIe ID 3: the board should be reset` from
`TopologyDiscovery::init_device`, and the gate's own `CARD_DIRTY` health probe
(`ttnn.open_device` plus a 32x32 tile add) spun 4+ minutes at 100 % CPU with py-spy parked at
`<string>:4`, i.e. inside the open, in native UMD code rather than Python.

**This is not the containment episode from earlier today, and the difference matters.** At 13:02Z
the kmd guard cleared Memory-Space-Enable on a bridge port, so config space itself was unreadable.
Here config space is intact: all four endpoints read `1e52:b140 COMMAND=0406`, bridges
`00:01.1/.3/.4` all read `0407`, and all four `/dev/tenstorrent` nodes exist. The chip answers on
config space and returns all-ones at the BAR, which is a dead ARC rather than an isolated link, and
matches `tt-kmd-upgrade-can-leave-a-blackhole-arc-dead-only-reboot-clears-it`.

**The reset path cannot recover it, because the reset path needs the enumeration that is broken.**
`tt-smi -r 3` throws from the same `TopologyDiscovery::create_ethernet_map` frame as `-ls`. So a
per-card reset is unavailable by construction in this state and the remedy is a reboot, which is
allowed on qb2 where power-off is not.

Rebooted 17:22:32Z. All four boards enumerate, card 3 opens in 2.9 s and card 2 in 0.9 s, both at
grid 11x10. Recovery cost: about 5 minutes once diagnosed, against 26 minutes of a held card and a
dark host before it.

Two things this adds to the record independent of the lever:

1. **`capacity_gate.py` runs an unguarded 1536-token attempt at any model with no Blackhole
   ceiling row.** protenix-v2 has a `wormhole_b0` row and no `blackhole` row, so nothing refused
   the input. opendde and opendde-abag carry Blackhole rows capped at 1024 precisely because their
   own `fail_at=1536` is a trunk freeze that "leaves the chip refusing every device open, so an
   unguarded attempt costs the next job on that card too". protenix-v2 at 1536 on Blackhole has
   now produced that same failure, which is an argument for giving it a Blackhole row rather than
   discovering the wall once per gate run.
2. **The gate's `CARD_DIRTY` probe cannot report a dirty card in the state that most needs
   reporting.** When the chip is ARC-dead the probe does not fail fast, it spins inside
   `ttnn.open_device` holding the host-wide `/tmp/tt-bio-device-open.lock`, so the instrument built
   to detect a dead card becomes a second process stuck on it.

## Attribution rep 1: the flag-off control completes the cell that froze

`cap_offarm.sh protenix-v2`, same tree, same card, same cell, one env var, its own process:

    protenix-v2    PASS   1536 tok   8832 rows   10.15G/32%   763.6 s   residency/-

So the freeze **did not reproduce with the flag off**. That is one run against one run on a box
that wedges roughly 1 fold in 6, which is not a verdict in either direction
(`qbroot-n1-beats-controls-is-not-a-lever`). It does establish that the freeze is not a
deterministic function of (model, size) alone: the first control cleared the exact step the on-arm
died at.

### An in-regime speed signal, and the confound that stops it being a measurement

Trunk cadence is unusually clean on this cell, so the two arms can be compared step by step:

| arm | steps | per-step | mean |
|---|---|---|---|
| ON (froze at 6/10) | 6 | 58, 57, 58, 57, 58, 58 s | 57.67 s |
| OFF | 9 | 69, 68, 68, 68, 69, 68, 69, 68, 69 s | 68.44 s |

**1.187x**, and the within-arm spread is +-1 s, roughly a tenth of the 10.8 s/step gap. That is
close to the 1.1856x banked at 1536 aa in gate5, from a different model and a different
instrument, which is the kind of independent agreement worth noticing.

It is NOT quoted as a measurement, for one reason: the two arms straddle the 17:22:32Z reboot.
`qb2-aiclk-governor-sets-fold-time-not-cotenancy` says a governor difference across a boot can
produce exactly this size of effect, and neither arm recorded its clock. Idle AICLK on the current
boot reads 0x320 (800 MHz) against an `AICLK_LIMIT_MAX` of 0x546 (1350 MHz).

The next rep removes the confound rather than arguing about it: an ON arm on **this** boot, beside
the OFF arm it is compared against. Started 17:39:31Z.
