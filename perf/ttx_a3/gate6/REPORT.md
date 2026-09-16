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
