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

## The wedges were card 0 dying, not load and not a kernel

The prior pass read three wedges at loadavg 4-9 as load-driven. Two more today killed that
reading: boltz2 at the 768 rung (12:41:15Z) and rf3 at the 768 rung (12:53:40Z) both stopped
writing with this gate as the only lane on the box and loadavg 1.1.

py-spy named a different ttnn op each time, which is the tell that it is not a kernel:

| wedge | rung | stack |
|---|---|---|
| 12:41:15Z | boltz2 768 rep0 | `ttnn.layer_norm` in `swiglu`, `tenstorrent.py:8205` |
| 10:09:09Z | boltz2 1024 warm-up | `ttnn.linear` in `swiglu`, `tenstorrent.py:8224` |
| earlier | opendde 640 | `ttnn.transpose` in `_pair_transpose`, `tenstorrent.py:2506` |

All three are `active+gil` inside `ttnn/decorators.py:473`: the call entered the device and never
came back. After the second one the next fold stopped even earlier, in `_open_device_locked`
(`tenstorrent.py:4935`) holding `/tmp/tt-bio-device-open.lock`, so the open itself had stopped
returning. `tt-smi -ls` then named it:

    Read 0xffffffff over PCIe ID 0: the board should be reset.

**qb2 card 0 is down at the PCIe link.** The control is clean and cheap: on the same host, at the
same minute, `TT_VISIBLE_DEVICES=0` hung past a 120 s timeout with no device and
`TT_VISIBLE_DEVICES=2` returned `MeshDevice(1x1 grid, 1 devices)` in 7 s. The gate moved to card
2 on grant `0,2` at 13:15:28Z and walked boltz2's 256 and 512 rungs in under 90 s, against the
4 minutes card 0 needed for a 256 warm-up before it hung for good.

Every ladder attempt recorded on card 0 today is marked `VOID-` in `progress`: a dead card is not
a verdict on a lever, and leaving those rc lines in place would have burned the retry budget on
hardware. This also matches the QBROOT verdict (`qbroot-verdict-pcie-link-failure`, qb2 board
410D, hardware-only fix) rather than anything in this tree.

A `tt-smi -r 0` takes the whole 0/1 board pair down, and card 1 is carrying
`b2z2-aiclk-default-decision`'s soak until 16:18:14Z, so the reset is not this task's to run.

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
