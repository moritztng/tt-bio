# mgx-design-accuracy — is a design returned at 1536 as good as one returned at 512?

MGX row (`state/mgx/CHARTER.md`, C2 + amendment A4). Host `whglx` / `j10glx02`. Branch
`wk/mgx-design-accuracy`, worktree `/home/moritz/.coworker/wt/mgx-design-accuracy` on pc,
device-side checkout `whglx:~/wt-mgx-design-accuracy`. **Current state, not a transcript.**

The row opened on two numbers from `state/mgx-design-ceiling.md`: BoltzGen's one 1536 design
scored scRMSD **11.480 A** against **4.054 A** for its one 512 design, with **7.1x** the clash
fraction. n=1 per size, handed over explicitly as a sized question.

**Both numbers are withdrawn, and the fixture they were measured on cannot answer the
question.** `perf/mgxaccuracy/` carries the harness and the artifacts.

## CATCHER

CATCHER: **every catcher this row has been handed is discharged and every one PASSES.** The
sample-width pair owed by the 2026-09-24 00:5x note ran first this pass, from
`~/wt-mgx-catcher` at `d1133c67e` (= `origin/main` `1f338604e` merged into this branch).

    merge        model     card  rc  wall_s  AICLK med  metric
    cc908c377    boltzgen   21    0   632.6    1000     scRMSD   2.37 / 11.35 A
    720b43a5d    pxdesign    3    0   246.8    1000     fit_rmsd 0.1446 / 0.158 A
    720b43a5d    boltzgen    5    0  2082.7    1000     scRMSD   median 4.749 A, n=8
    1f338604e    pxdesign   30    0   230.6    1000     fit_rmsd 0.1446 / 0.158 A
    1f338604e    boltzgen   31    0   576.5    1000     scRMSD   12.117 / 13.912 A
    811ac1316    pxdesign    5    0   246.0    1000     fit_rmsd 0.1446 / 0.158 A
    811ac1316    boltzgen    2    0   543.0    1000     scRMSD   2.131 / 0.552 A
    6ea518246    rfd3       29    0    45.1    1000     digest   7ab5705258251e10  MATCHES
    6ea518246    pxdesign    -    0   227.0    1000     fit_rmsd 0.1446 / 0.158 A
    6ea518246    boltzgen    5    0   492.0    1000     scRMSD   8.786 / 7.735 A

**The rfd3 entry is the strongest catcher this row has run, because it checks a HASH.**
`wk/mgx-trace-region` deletes rfd3's trace code, so the orchestrator asked for one design
whose output equals that row's own eager digest. Reproduced bit-for-bit at
**7ab5705258251e10**, with the invocation copied verbatim out of their `runs.jsonl`
(`examples/rfd3_binder.json --seed 0 --num_timesteps 20`) rather than approximated -- a
similar run would have been a different measurement wearing the same name. It ran through
`perf/mgx_trace_region/chain.py`, which already does lease acquisition and
`sha256(bytes)[:16]`; that tool now takes `MGX_CHAIN_HOLDER` so a second row can use it
without writing another row's identity into the lease dir.

**pxdesign has now read 0.1446 / 0.158 / 0.158 at FOUR consecutive merge points** --
720b43a5d, 1f338604e, 811ac1316, 6ea518246 -- across msa-depth, sdpa-mask, sample_chunks,
trace-region and affinity-scale. Every one of those touches a file pxdesign rides.

**pxdesign has now read the SAME THREE NUMBERS across four merge points.** 0.1446 / 0.158 /
0.158 on 720b43a5d, on 1f338604e and on 811ac1316 -- the last of which carries msa-depth
(`protenix.py`, which pxdesign rides directly), sdpa-mask (`tenstorrent.py`, which every model
rides) and `msa: empty`. Three merges, one catcher pair, no movement to four decimals.

**The pxdesign catcher is the informative one and it did not move at all.** `fit_rmsd` on the
1224-token production case reads 0.1446 / 0.158 / 0.158 on `1f338604e`, which is the same
three numbers to four decimals as on `720b43a5d`. Against its own 15 A gate the case clears
by 95x. Load was 59.2 of 64, so the seconds are indicative only; the metric is not.

**And the merge is inert on both cases by MECHANISM, not just by a matching number.** Two
independent checks, because n=2 on a stochastic designer could not have carried this alone:

  * `sample_chunks.py` only changes behaviour when a chunk is REFUSED, and it prints when it
    narrows. `grep -c "chunk refused"` over both real job logs
    (`log_{boltzgen_512_d2_bgall,pxdesign_1024_d2_s400}_sw_catcher.txt`) returns **0**.
  * with no refusal the path is byte-identical work: `git diff 720b43a5d 1f338604e --
    tt_bio/boltz2.py` moves `resolve_sample_chunk_width` to `sample_chunks.py` with its logic
    unchanged and refactors the chunk loop into `_denoise_chunk` at the same width, same
    padding of the short tail, same order. pxdesign coming back bit-identical is that
    conclusion measured on the shared `protenix.py` / `tenstorrent.py` path.

So **the boltzgen catcher's 12.117 / 13.912 A is not a merge effect** — it cannot be one. It
is two draws from the banked after-fix 512 cell (n=8, median 4.749, range 0.693-12.925), and
they are two HIGH draws. BoltzGen's design sampling is not seeded in this harness, so that is
sampling, and n=2 against n=8 resolves no shift either way (`results/power.txt`).

**The next catcher settled it by landing at the opposite end.** The same case on `811ac1316`
drew **2.131 / 0.552 A** — both LOW, one of them below the n=8 cell's own minimum — where the
`1f338604e` pair drew 12.117 / 13.912, both high. Two consecutive n=2 draws of one nominally
identical configuration spanning 0.55 to 13.91 A is what a distribution with that range does,
and it is the direct evidence that neither pair was telling us anything about a merge. It also
means a catcher on this case can read anywhere between "0% designable" and "100% permissive"
by luck alone: this one reports pass_permissive 1.0 and the previous one 0.0. **The
eight-design cells carry every quality claim in this row; a catcher never does.**

### Every device-side boltzgen number in this row predates the engine it will ship on

`cc908c377` changes `TokenDistanceRecycle` numerically — the pairformer adds its residuals
into `v` in place, so `v + pairformer(v)` was twice the pairformer's OUTPUT where the
reference takes `v + update`. Whether a run is reached by that is a property of the
**checkpoint**, and the run's own `design.yaml` says nothing: grepping it for
`token_distance` returns nothing, which reads like off and is not. Read out of the
checkpoints the runs load (qb2, torch mmap, no device): `boltzgen1_adherence` and
`boltzgen1_diverse` both carry **`use_token_distances=True`**, blocks 2, dim 64
(`results/token_distance_fires.txt`).

So **every boltzgen scRMSD below — both 512 cells and both 1536 arms — is pre-fix.** The
upstream CPU reference is torch and is unaffected; pxdesign is not reached. The two 1536 arms
still running are now the **BEFORE** side of a before/after rather than the answer, which is
a better use for them than the answer would have been: the fix lands in exactly the module
that injects the target's token distances, which is the conditioning a binder design depends
on, and its error would plausibly grow with token count. That makes it the first named
candidate for any 1536 quality loss, and it is testable by effect rather than by diff.

## QUALITY

**A THIRD 512 CROP LANDED 03:2xZ AND FLIPPED THE READING: a GOOD crop is the exception, not
a bad one.** Three single-chain windows of chain A, 100 residues apart, n=8 each, one engine:

    crop            n    min     median     max    <=4A
    offset   0      8   0.823     4.631   13.493    50%
    offset 100      8  13.024    16.021   17.236     0%
    offset 200      8  13.283    17.300   20.361     0%

Two of three sit at 16-17 A with ZERO designs under the permissive bar and agree with each
other to 1.28 A; offset 0 is the outlier. With two crops the natural reading was "offset 100
is a hard one" and it was wrong: the typical window of this protein designs badly and offset 0
happens to be a good surface. Spread across crops 12.67 A
(`results/three_crops_512.txt`). `docs/design-throughput.md` now carries the three-row table
and tells a user to search the target rather than pile designs onto one crop -- eight designs
against a bad window returned zero binders under 4 A, twice over.

**THE SIZE SIGNAL DOES NOT CLEAR ITS OWN ACROSS-TARGET FLOOR, measured 01:35Z.** A second
512-residue crop of the SAME protein -- offset 100 along its own sequence, so 412 of its 512
residues are shared with the offset-0 crop, same engine, designer, binder length, metric and
device refolder, n=8, single-chain at both offsets -- reads:

    crop             n     min   median     max   <=2A  <=4A
    512 offset 0     8   0.693    4.749  12.925   25%   38%
    512 offset 100   8  12.249   17.785  28.623    0%    0%
      -> median gap 13.04 A, U = 2 of 64, rejects at 0.05

    target effect at FIXED size 512                  13.04 A
    size effect at offset 0, upstream refolder       12.34 A  (19.347 - 7.012)

**Changing WHICH 512 residues you point at moves the median more than going from 512 to 1536
does.** The pre-registered rule was that a 1536 median must sit above the TOP of the 512
across-target band; that band is now 4.75-17.79 by median and 0.69-28.62 by range, and 19.35
sits 1.6 A above the top median and well inside the offset-100 range. The old 2.10 A floor
came from two targets that happened to be close -- it was a draw, not a floor
(`results/across_target_floor_512.txt`).

**No measurement below is retracted. The INFERENCE that the 512-to-1536 difference is about
SIZE is.** Every number stands exactly as measured; what falls is reading them as a size axis.

**The 2x2 that separates size from target is running** -- {offset 0, offset 100} x {512,
1536}, all after-fix, n=8 each. Two cells are in (4.749 and 17.785), two are on cards 3 and
23. With all four, the size effect is the shift across the two sizes averaged over offsets and
the target effect is the shift across offsets averaged over sizes. **Until they land, no size
verdict goes in writing.**

Both 1536 arms reached 8 of 8 at step 4 and were harvested at 23:13Z (`results/size_1536.jsonl`). On
the valid target, gpb dimer A1-1536 + 80 binder:

    side                              size   n     min  median     max   <=2A  <=4A
    device (card designs, card refolds)  512   8    0.86    8.46   12.90   25%   38%
    device (card designs, card refolds) 1536   8   19.31   21.52   22.48    0%    0%
    upstream fp32 refolds THOSE designs  512   8    0.98    7.01   12.32   13%   38%
    upstream fp32 refolds THOSE designs 1536   8   16.45   19.35   22.21    0%    0%

**The ranges do not overlap at all** — the worst 512 design beats the best 1536 one. Pooled
over both targets the device reads median 8.55 at 512 (n=17) against 14.26 at 1536 (n=17),
U=39, z=3.63. **That separation is real and it is between these particular crops; it is no
longer evidence about SIZE.** An earlier version of this paragraph read the 1536 median as
sitting 11.0 A above the top of a 512 across-target band of 8.46-10.56 and called that band
the floor the pre-registered rule requires. The band was two targets that happened to be
close: the offset-100 crop puts a third 512 cell at 17.79 A, so the real band reaches at
least 17.79 by median and 28.62 by range, and 19.35 does not clear it.

**JOINT 1 holds as an INSTRUMENT result and not as a size result: the offset-0 512-vs-1536
gap survives holding the SCORING FOLD constant, at n=8 per side.** It rules out the refolder
as the cause of that gap. It does not establish that the cause is size, because the
across-target floor above is larger than the gap.** The 512 upstream-refold row is new this pass (`out_devrefold512_all8`, harvested
00:37Z) and it is the row that removes the last shared-instrument objection. Both rows are
the device's own designs; the only thing that folds them is upstream BoltzGen, torch fp32 on
qb2 CPU, same checkpoint, same protocol, no card. Same target, crop, binder length, designer,
metric and N,CA,C,O backbone on both sides.

    upstream refolder, device designs    512   8    0.98    7.01   12.32
    upstream refolder, device designs   1536   8   16.45   19.35   22.21
    -> zero overlap, Mann-Whitney U = 0 of 64, exact two-sided p = 2/C(16,8) = 1.554e-4

**Both sides are PRE-FIX, by timestamp and not by assumption**: the 512 designs were staged to
qb2 at 23:21Z and the after-fix 512 run did not finish until 23:49:50Z, so the comparison
cannot be picking up the token-distance fix on one side only. The two refolders differ by
-1.44 A at 512 and -2.17 A at 1536, both inside the 0.9-1.9 A repeat spread of a single
upstream refold, so the instrument reads the same on both sides and the 12.33 A between them
is not the instrument (`results/size_refolder_held_constant.txt`).

So no device-side refolding artifact explains the 1536 loss, and the remaining candidates are
the design step itself and the target.

Two things still stop this being the answer, and the second is the interesting one:

  1. one valid target per size, so a size effect and a target effect are still the same
     number. `plans/bg_floor_1536.txt` is the second valid 1536 target that would separate
     them; it is queued behind rfd3.
  2. **every one of those designs is PRE-FIX.** `cc908c377` corrects `TokenDistanceRecycle` —
     the module that injects the TARGET's token distances, which is the conditioning a binder
     design is built on — where the old code returned twice the pairformer's output instead
     of its input plus the update. An error there is exactly the shape that grows with token
     count, and both shipped checkpoints have the module on. `plans/postmerge_720b43a5d.txt`
     runs the same fixture at both sizes on the fixed engine and decides between "1536 is
     harder for boltzgen" and "the port had a conditioning bug that got worse with size".
     **It is running now on cards 5 and 3.**

The remaining gap in the reference is upstream's OWN 1536 design, and it is the long pole:
`out1536` on qb2 has been in its design step **10.4 h** and has written zero structures. A
1536 answer that needs it may not be reachable on CPU at all, which is itself a finding and
is why the `out_dev1536` split above was built to not depend on it.

QUALITY: **upstream torch fp32 on CPU is no better than the device at 512 residues, and one
draw of the reference cannot say which is better either way.** Upstream's unmodified six-step
pipeline reads **7.17 A** on the same fixture and spec; a repeat of that same quantity on the
same design reads **11.04 A**. Both sit inside the device's **3.30-13.78 A** range over nine
designs on that target (median **10.56 A**), and the reference's own **3.87 A** run-to-run
spread is larger than the 2.10 A across-target floor at this size. So the ~10 A designability
of a real 512-residue target is BoltzGen and the target, not this port.
Against that, the two published gaps are withdrawn: scRMSD 10.56 A at 512 (n=9) against
11.480 A at 1536 (n=1) is an ordinary member of the 512 spread, not a 2.8x gap; clash fraction
median 0.00107 at 512 (n=9) against 0.00197 at 1536 (n=8) is 2.04x rather than 7.1x and does
not separate (U=15, z=1.79). **n=8 per target refutes the published ratios and does not establish
their absence** — the two are different questions and the second needs more designs. Refuting
them needs no power at all: the 512 target's own scRMSD spans a factor of 4.2 across eight
draws and its clash fraction a factor of 5.0, so both single-draw ratios sit inside one
target's spread as a matter of arithmetic. Detecting a real shift is where n bites, and
`perf/mgxaccuracy/power.py` resamples the seventeen 512 designs measured here through the same
tie-aware Mann-Whitney the report prints:

    1536 arm is the 512 distribution x    1.25   1.50   2.00   2.80
    power at n = 8 per size                16%    31%    45%    65%
    n needed for 80% power                   64     32     24     12

**So the eight designs per size now in flight cannot resolve even a doubling of the median at
better than a coin flip.** A 1536 result that lands inside the 512 band will therefore mean
"no effect this study could see", not "no effect", and this row says the first. Twelve designs
per size would have resolved the 2.8x gap it opened on; thirty-two would resolve 1.5x. **pxdesign at
1536 fails its own gated metric (fit_rmsd 95.183 A against a 15 A gate, n=8) and returns a
binder with zero contacts to the target — traced to the fixture, not the size.** rfd3's clash
fraction is 19x LOWER at 1536 than at 512.

**pxdesign is now ANSWERED on a valid target and it holds:** `fit_rmsd` median 0.1128 A at 592
tokens, 0.1423 at 1224, **0.1747 at 1616** (8 of 8 designs each, AICLK median 1000 MHz DURING,
`perf/mgxaccuracy/results/valid_target.jsonl`). There is a small resolved size trend — 1.55x
over 2.7x the tokens, 9.1x the 0.0068 A spread between two different 512-residue targets — and
at its worst it is **86x under** the 15 A `PXDESIGN_MAX_FIT_RMSD` gate, against 95.183 A at the
same 1616 tokens on the disconnected fixture. **rfd3 clears 512 on the same valid target**: 4 designs, rc=0, 387.8 s,
AICLK median 1000 DURING (n=152), load median 49.8 of 64, on `07bd40614`
(`results/rfd3_valid.jsonl`); the 1536 entry of the same plan runs behind it on the same
card so the pair shares a target. **Boltzgen's designability on a valid target is
the one QUALITY arm still owed**; two jobs hold chips for it.

**boltzgen's 512 control on a valid target has landed, and it shows designability is
target-bound rather than size-bound.** gpb dimer A1-512 + 80 binder, 8 of 8 ranked, scRMSD
min 0.865 / **median 8.455** / max 12.897, **25 % designable at 2 A, 38 % at 4 A**; wall
2433.2 s, AICLK median 1000 MHz DURING (n=898), load median 85.9 with an 821.9 peak, so its
seconds are an artifact and its scRMSD is not. Beside the other targets at comparable size:

| target | n | scRMSD median | <=2 A |
|---|---|---|---|
| 7ROA chain A, ~120 res (`docs/implementation-parity-data/boltzgen.json`) | 16 | **0.78** | 93.75 % |
| gpb dimer A1-512 + 80 | 8 | **8.455** | 25 % |
| big_1831 A1-512 + 80 | 9 | **10.56** | 0 % |
| big_1831 A1-512 + 80, **upstream fp32 on CPU** | 1 | **11.04** | 0 % |

**Every one of those designs is docked — designability and docking are different axes and
only one of them is weak.** `contact.py --complex` on all eight gpb-dimer 512 designs: closest
heavy-atom distance **1.29-2.41 A**, 478-1199 contacts under 5 A, 8 of 8 IN CONTACT. So
BoltzGen is placing binders on the surface it was given and the ~8.5 A median scRMSD is
self-consistency — the designed sequence does not refold to the designed backbone — not a
binder left floating in space. That distinction matters for a user: the failure mode here is
"order this and it may not fold as drawn", not "the model ignored your target".

**And at 1536 every design is docked too, on BOTH targets — including the one pxdesign
fails on.** All 16 designs, measured on pc: on the valid gpb-dimer 1536 crop, closest heavy
atom **1.15-2.09 A** with 2255-2560 contacts; on `big_1831`'s 1536 crop — two components,
zero inter-chain edges, where pxdesign returns a binder **14-41 A away with zero contacts** —
**0.75-1.85 A** with 916-1679 contacts (`results/contact_1536.txt`).

**That narrows the fixture's disqualification rather than widening it.** A disconnected
conditioning graph kills a model that sees the target as a distogram clamped at 22 A; boltzgen
conditions on coordinates and docks straight through it. So `big_1831` at 1536 is disqualified
for pxdesign-like conditioning and is NOT automatically disqualified as a boltzgen target —
which is P4b's docking half, answered, and it makes the `q` arm's scRMSD worth reading rather
than discarding.

**Careful with that table: 7ROA is NOT at this size, so "a ~10 A spread at one size" — which
an earlier version of this section said — is wrong.** 7ROA is ~120 residues and a far easier
problem. The fixed-size floor is the two targets that ARE at 512:

    across-target spread at 512 residues   8.46 vs 10.56  ->  2.10 A   (9 + 8 designs)

**SUPERSEDED 2026-09-24 01:35Z: 2.10 A was not the floor, it was a draw.** A third 512 cell --
the same chain cropped 100 residues along -- reads median 17.785 A (n=8, 12.249-28.623)
against 4.749 A at offset 0 on the same after-fix engine, U=2 of 64. **The across-target
spread at a FIXED 512 residues is 13.04 A** (11.39 A on the same-tree repeat), not 2.10.
That is larger than the 12.34 A
the row had been reading as a size effect (`results/across_target_floor_512.txt`). Every
downstream use of "the 2.10 A floor" in this document is stale for that reason; the sentences
are left as they were written so the reasoning stays legible, but the number to judge a 1536
median against is the 13.04 A band, not 2.10.

So the honest reading is two separate facts. Target DIFFICULTY varies enormously (0.78 A on a
small easy target against 8-11 A on a 512-residue one), and at a FIXED 512 residues two
different targets differ by 2.10 A -- three differ by 13.04 A. The 1536 rung has to be judged
against the wider band, which makes it a much LOOSER test than this section originally
claimed, and `perf/mgxaccuracy/report.py` prints the band it was given.

### Quiet window: the gate moved to the host that can see it

Orchestrator note 2026-09-23 16:10 UTC. The file
`/home/moritz/.coworker/state/mgx/quiet-window` **does not exist yet** (checked this pass; it
opens after the bigalloc merge lands). Two things follow that are not just compliance:

**The fan could not have honoured it.** `fan.py` runs ON WHGLX, and whglx has neither
`~/.coworker` nor `/home/moritz/.coworker` — both checked, both absent. A long-lived launcher
there would have launched straight through a window it cannot see. So the fan was **stopped**
(explicit pid 2141185) and the gate moved to pc: `perf/mgxaccuracy/launch_if_clear.sh` refuses
with exit 1 and prints the window's own open/close times when the file is present. Verified in
both directions, and verified the refusal really exits 1 rather than merely printing — a pipe
to `tail` hid the status on the first check.

**Nothing was killed that was running.** Both 1536 boltzgen jobs survived the fan's death and
are still going, which is what the note asks for. `fan.py` starts each job with
`start_new_session`, so the driver dying does not take the rung with it.

**Waiting for a chip no longer means losing it.** Stopping the fan left this row checking
whglx once per pass and finding 0 of 27 free twice, which is how a pass ends with nothing
launched. `perf/mgxaccuracy/watch_and_launch.sh` puts the wait on pc instead: it polls whglx
for a genuinely free chip, **re-reads the gate on every iteration** rather than once, launches
one fan and exits, and has a deadline so it cannot outlive the question. Running now for the
rfd3 512 arm. `fan.py --lines` closes the last hole — a window that opens mid-job — by letting
a pass launch named plan entries and let the fan exit instead of leaving it queued.

**Nothing of this row's needs to jump the window.** The queued work is the rfd3 pair and any
boltzgen requeue; none is time-critical, and the two arms that matter are already mid-run.

**rfd3's 512 arm is being launched by the watcher and its 1536 arm is not**
(`perf/mgxaccuracy/plans/rfd3_valid.txt`, entries 1 and 2). Its old 512 arm was already valid
as measured — `A1-432` of one chain, conditioning graph one component — but it was measured on
a DIFFERENT target from the 1536 rung, and this row's whole finding is that the target
dominates. So the pair is re-run on the gpb dimer, 512 first. The plan
also records up front that rfd3's per-CHAIN geometry flags at 1536 will recur as an INSTRUMENT
artifact: the trailing designed length lands in the last named chain, so the checker scores the
target fragment plus the binder as one broken chain. Its clash FRACTION is a global count and is
unaffected.

### CONFIRMED on device: pxdesign holds to 1616 tokens, and the merge works

This row's pxdesign fix is on `main` (`0c34f135a`, `d6ddfa5f9`), merged here as `a9b76ef93`.
`main` has since advanced to `1b423e9e4` (esmfold2-only), merged here as `dc7ae0a1d`; nothing
below is re-recorded because `git diff --stat d6ddfa5f9 origin/main` over `tt_bio/pxdesign/`,
`tt_bio/protenix.py`, `tt_bio/boltzgen/` and `tt_bio/rfd3/` is **empty** — checked, not taken
from the note. Both pre-registered predictions in `perf/mgxaccuracy/PREDICTION.md` are
confirmed (`results/valid_target.jsonl`):

| target (all `gpb_dimer_1646`) | n_token | n | fit_rmsd min / median / max | s/design | AICLK DURING |
|---|---|---|---|---|---|
| A1-512 + 80 binder | 592 | 8/8 | 0.1079 / **0.1128** / 0.1456 | 42.7 | 1000 (n=132) |
| **chain B** 1-512 + 80 (offset 823) | 592 | 8/8 | 0.1069 / **0.1196** / 0.1435 | 51.6 | 1000 (n=156) |
| 1024 + 200 binder | 1224 | 8/8 | 0.1313 / **0.1423** / 0.1668 | 110.3 | 1000 (n=328) |
| 1024 + 200, card 9 | 1224 | 8/8 | 0.1313 / **0.1423** / 0.1668 | 101.6 | 1000 (n=309) |
| 1536 + 80 binder | 1616 | 8/8 | 0.1589 / **0.1747** / 0.1939 | 168.6 | 1000 (n=514) |

**P1, the row's central pxdesign result: 0.1747 A against 95.183 A at the SAME 1616 tokens.**
Same model, same 400 steps, same batch of 8; the only difference is the target's sub-22 A
conditioning graph — one component with 8316 inter-chain edges against two components with
zero between them. 545x, predicted under 2 A before the run.

**P2: the live-reachable region was never broken.** 1224 tokens is the most a user can submit
(1024 target cap + 200 binder) and it reads 0.1423 A against the 15 A gate.

**The merge is confirmed in the form asked for, at quiet load** (`results/mergeconf_quiet.jsonl`):
one design at the 1224-token production case, card 21, **130.5 s** wall, AICLK 1000 DURING
(n=51), load 60.5, fit_rmsd 0.1514. Before the fix this rung died at 50-55 s on the
AttributeError, so completion is the confirmation. A first attempt inside a load spike read
161.9 s — **1.24x load inflation**, which is why it was re-run rather than quoted.

**An unplanned check falls out of the repeat.** `fit_rmsd` is 0.1514 A to four decimals on
**different cards** (6 and 21) at **3x different load**, and the 8-design 1224 runs agree to
four decimals across cards too. So the quality metric depends on neither host load nor which
card ran it — the standing "output must not depend on which card ran it" hard stop, tested.

**There IS a small size trend and "flat" was too strong**, but it is not a problem:

    across-target spread at 592 tok   0.0068 A
    size trend 592 -> 1616            0.0619 A   (1.55x over 2.7x the tokens)
    trend / spread                    9.1x       -> resolved, and monotone over 3 rungs

At its worst 0.1747 A is **86x under** the 15 A `PXDESIGN_MAX_FIT_RMSD` gate. Every pxdesign
design measured on a valid target, two 512-residue targets and three token counts, sits in
**0.1069-0.1939 A**.

**Load binds the seconds and not the quality.** whglx spiked to 427 and to 510 during this
row's runs, twice briefly refusing ssh with `kex_exchange_identification`; the ceiling row
measured 9.0x inflation at load 467 with a clean 1000 MHz clock, and `scripts/speed_bar.py`
cannot see host load. `fit_rmsd`, scRMSD and clash fraction are load-insensitive, so a loud
box delays this row's answer without changing it.

### The published gaps are single-draw artifacts

Neither arm needed a chip. Both came off runs `mgx-design-scale` had already paid for and
never scored.

**scRMSD** — harvested with `perf/mgxscale/job.py designability()`, which calls
`scripts/boltzgen_designability.py score()` rather than re-deriving the column or the bars
(`perf/mgxaccuracy/results/prior_runs.jsonl`):

| target | pipeline | n | scRMSD min / median / max | <=2 A | <=4 A |
|---|---|---|---|---|---|
| 512 (big_1831 crop, offset 0) | full, budget 1 | 1 | - / **4.054** / - | 0 % | 0 % |
| 512 (**same fixture**, 4121 atoms) | full, budget 4 | **8** | 3.302 / **10.590** / 13.778 | 0 % | 12.5 % |
| 1536 (big_1831 crop, offset 0) | full, budget 1 | 1 | - / **11.480** / - | 0 % | 0 % |

The 4.054 A "control" is the **second-best of eight**. 11.480 A sits near the 70th percentile
of the same 512 target's own spread. Card 2, AICLK median 1000 MHz DURING (n=835), load median
73.8 on 64 cores.

**Clash fraction** — the eight 1536 designs from the ceilscale job were on disk and unscored.
Scored with the same `perf/wh-correctness/check_structure.py --kind design` that
`scripts/release_gate.py` imports for its GEOMETRY leg, one invocation covering both sizes
(`perf/mgxaccuracy/results/geom_n8.jsonl`):

| target | n | clash fraction min / median / max | flagged |
|---|---|---|---|
| 512, `--steps design` | 9 | 0.00043 / **0.00107** / 0.00215 | 4 of 8 batched, 1 of 1 control |
| 1536, `--steps design` | 8 | 0.00077 / **0.00197** / 0.00316 | 5 of 8 |

**2.04x, not 7.1x; U=15, z=1.79, not separated**, and the spreads overlap (512 max 0.00215
against 1536 min 0.00077). The ceiling row's 0.00021 control is below all nine 512 designs.

`perf/mgxaccuracy/report.py --metric {scrmsd,clash_frac}` prints per-target distributions, then
the fixed-size across-target spread, and only then the size comparison.

### pxdesign's own metric fails at 1536, and it is the fixture

`fit_rmsd` is pxdesign's end-to-end conditioning signal — the residual of one rigid fit of the
model's reconstruction of the CONDITIONED target tokens onto what the featurizer conditioned on
(`tt_bio/pxdesign/write.py:47`), gated at `PXDESIGN_MAX_FIT_RMSD = 15.0` in
`scripts/release_gate.py`. Harvested across every pxdesign run on the box
(`results/pxdesign_fit_rmsd_ladder.txt`): 0.156 at 336 tokens, 0.0755 at 592, 0.452 at 1024
(824+200), 0.125 at 1024 (960+64) — and **95.183 at 1616 on `big_1831`**, 6.3x the gate, on all
eight designs, no overlap with anything below.

**What a user got there, measured** (`contact.py`): the binder is **14.43-41.11 A from the
target with ZERO atom pairs within 5 A**, against 1.35-1.94 A and 77-166 contacts at 512 — a
well-formed 80-residue backbone floating in space. The geometry instrument called it clean: it
scores the 321 atoms pxdesign writes, which is the binder alone.

**The carrier is conditioning-graph CONNECTIVITY, not saturation and not token count.**
pxdesign sees the target only as a 64-bin distogram over 2-22 A
(`tt_bio/pxdesign/featurize.py:40`). The 512 crop reading 0.0755 A is already 73.5 % saturated
and the VALID 1536 crop is 90.0 % against the invalid one's 90.9 %, so saturation separates
nothing. Components do: `big_1831` 1536 is **2** components (1008 + 528) with **0** inter-chain
edges, every valid crop is **1** — gpb dimer at 512, 1024 and 1536 with 0, 4715 and 8316
inter-chain edges.

**On the invalid fixture this is a proof, not a hypothesis.** Bins are (22-2)/63 = 0.317 A
wide. `rigidity.py` moves chain B rigidly (`results/rigidity_1536.txt`): the gpb dimer's 1536
crop has 8316 resolvable inter-chain pairs and a 0.1 A translation moves 13.4 % of them to
another bin, a 10 A translation 97.9 %, a 30 degree rotation 97.2 %. `big_1831`'s 1536 crop has
**0**, and **0** change under every one of those motions. The relative placement is not a
function of the input, so no model recovers it and a fit that tries lands at the scale of the
separation — 95.183 A against 246.9 A. **No device run is needed and none can falsify it.**

Sufficiency on the valid target was the part that WAS a prediction, pre-registered in
`PREDICTION.md` — and it is now confirmed at 0.1747 A (see above). Severity resolved with it:
`aiand-bio/japanfold/catalog.py:21` caps a user at 1024 target residues, so 1616 tokens is
unreachable in production, and the 1224 tokens that ARE reachable read 0.1423 A.

### CLOSED: pxdesign was crashing on main at every size, and this row found and fixed it

`1c393dda8` (bigalloc's, on `main` 05:55 CEST) added `self._paircond_rows_refused` to
`Protenix.__init__` and reads it from `_diffusion_pair_cond`. `ProtenixDesign(Protenix)`
inherits that method and **deliberately does not call** `Protenix.__init__`
(`tt_bio/pxdesign/model.py:73` — no trunk, no confidence head in this checkpoint). So the
subclass ran the new code without the new state.

**Not size-gated.** `tt_bio/tenstorrent.py:5252` reads the memo *before* attempting any
allocation, and pxdesign's fp32 default always takes that branch, so every `design()` died.
Observed at 1224 and 1616 tokens, 50-55 s in, one card each. Nobody had seen it because every
design row's checkout predated the commit (`mgx-design-scale`'s is `e3697d314`, confirmed not
to contain it). The live `japanfold serve` started Sep 22 23:05, pre-regression, so it was
never a running outage — it would have landed on the next restart or deploy.

**Fixed and confirmed.** One line in pxdesign's own file, so the shared pair path this row may
not touch stayed untouched; cherry-picked to main as `0c34f135a` + `d6ddfa5f9`, and
`0a13523df` adds a host-only AST guard (attributes assigned in `Protenix.__init__` ∩ read by
methods `ProtenixDesign` does not override − assigned in its own `__init__`), verified to fail
naming `_paircond_rows_refused` when the fix is removed. Confirmed on hardware by the
merge-confirmation run above.

### Every 1536 reading above is on the invalid fixture

Worth stating plainly because it cuts both ways. The withdrawals **stand**: a ratio that fails
to separate at n=8 fails to separate whatever the input was, and the 512 rungs are compact
single-chain crops either way. But **no positive statement about quality at 1536 follows from
any of it** — the boltzgen 11.480 A, its 0.00197 clash median, and rfd3's 19x-lower clash
fraction are all measured against two proteins 246.9 A apart. They are not evidence that 1536
is fine. That is what the requeued arm is for, and until it lands this row has no 1536 quality
number on a valid target for any of the three designers.

### rfd3 and pxdesign at both sizes on the geometry instrument

Same invocation, `perf/mgxaccuracy/results/geom_others_n8.jsonl`:

- **pxdesign: 0 of 8 flagged at 512 and 0 of 8 at 1536**, 0 clashes on every design. Reads as
  no size effect, and `fit_rmsd` above is why that reading is not enough.
- **rfd3: clash fraction is 19x LOWER at 1536** — median 0.00116 (n=2) against 0.02184 (n=4) at
  512, where its own `size_limits.py` band at 640-1024 is 0.016-0.051. Its two 1536 flags are
  the chain-B assignment artifact `state/mgx-design-ceiling.md` already root-caused, not new.

Neither ships a designability metric and tt-bio ships no inverse folder for them. BoltzGen's
`inverse_folding` + `design_folding` steps chain on a directory of `<id>.cif` plus an `<id>.npz`
carrying five per-token arrays (`design_mask`, `mol_type`, `ss_type`, `token_resolved_mask`,
`binding_type`; binder tokens come FIRST), so synthesising that metadata for a foreign backbone
is the whole cost.

**One of the two blockers is now gone.** The old note said validating a synthesised npz needs
a chip; it does not. `ref/designfolding_only.sh --from DIR --name N` runs BoltzGen's step 4 on
the upstream CPU package against any directory of `<id>.cif` + `<id>.npz`, at 17 min per
80-residue design and no device at all — verified by feeding it the device's own designs,
which it accepted unchanged. So the work left is the npz synthesis and it can be written and
debugged entirely on qb2. Sized and named, not started.

### In flight, and what is NOT

At 01:05Z. Everything below is running unattended; nothing needs a decision to continue.
The gate was re-checked immediately before each launch this pass and was absent every time.

    whglx  card3  boltzgen 1536 gpb, 8 designs, AFTER-fix   pid 451967, 8/8 designed and
                                                            inverse-folded, in step 3  <- DECIDES THE ROW
           card13 boltzgen  512 gpb offset 100, 8 designs   across-target floor, after the crop fix
           card23 boltzgen 1536 gpb offset 100, 8 designs   across-target floor, after the crop fix
    qb2    out1536  upstream six-step, now PAST its design step and inside the 1616-token
                    whole-complex fold at 11.8 h. No longer the long pole: its scRMSD was
                    taken off step 4 directly and is banked.

Harvested this pass, all banked in `results/`:

    out_devrefold512_all8   upstream refolds the device's 8 gpb-512 designs   7.012 A  n=8
    out1536_df              upstream's OWN 1536 design, step-4 only           7.907 A  n=1
    out_1536_r2             the same design refolded again                    6.668 A  n=1
    catcher_sw.jsonl        pxdesign + boltzgen catchers on 1f338604e         both PASS
    rfd3_valid.jsonl        rfd3 512 (4 designs) and 1536 (2 designs)         both PASS, rc 0

Cards 30 and 31 were released when the two catchers finished; card 6 and card 5 released when
rfd3 and the 512 arm finished. The row now holds ONE chip, card 3, well under its cap of 3.

Three chips is this row's cap and it is holding exactly three, none of them 1/24/25/26/27 and
none chip 4. **The lease dir churns faster than a census survives**: cards 11 and 28 read
released with dead pids at 23:12Z and were both taken by other rows within six minutes, while
3, 5 and 6 freed up in the same window. The fan picks at acquire time against the lease dir,
which is the only reading that is not already stale — do not pre-select a chip by hand here.

`out_devrefold512_all8` is the arm that closes joint 1 of the size claim. It gives the
upstream refolder eight designs at 512 to set against its eight at 1536, so the
refolder-matched comparison stops being n=2 against n=8. ~17 min per design, ~2.3 h, no chip.

**Progress is read off the pipeline step, not off elapsed time, and the step is finer than
"refolding".** `refold_cif/` is step 3, the whole 1616-token complex refolded, which feeds
`bb_rmsd_design`; `refold_design_cif/` is step 4, the designed binder refolded ALONE at 80
residues, and that is the scRMSD this row is about. Reading the wrong one answers a different
question — the first version of `scrmsd.py` did, and was caught only because it refused to
find an 80-residue binder in a 592-residue file. Step 3 costs ~13 min per design at 1616
tokens, so ~1.7 h for eight; step 4 after it is minutes.

**The scRMSD no longer waits for the analysis step.** `perf/mgxaccuracy/scrmsd.py` recomputes
`designfolding-bb_rmsd` from `refold_design_cif/` plus the designed backbone, and `--check`
holds it against the pipeline's own column on the completed 512 run: **8 of 8 designs match
to five decimals, max gap 0.00000 A**, with the N,CA,C,O backbone (N,CA,C or CA alone miss by
0.22 A, so the atom set is measured rather than assumed — `results/scrmsd_step4_check.txt`).

**main moved again and the boltzgen path DID change this time — read what the change is
before deciding it matters.** `origin/main` is now `5868fd48b`, merged here as `b456022f8`;
`git diff origin/main HEAD -- tt_bio/` is empty, so this branch carries no engine divergence.
The whole boltzgen delta is six lines in `tt_bio/boltzgen/model/models/boltz.py`
(`58bc87b94`): two `reset_static_cache()` calls, one after the trunk and one over the
structure module's children, both placed AFTER the consumer of the staged tensors. It frees
DRAM, it does not feed anything. So the two arms now in flight, which run on the pre-change
`7cf87b844`, do not need re-measuring for accuracy — but their numbers are attributable to
that tree and this paragraph is where that is written down. The brief's "merge main before
your next device run" binds the rfd3 pair and anything else this row starts.

**The whglx checkout is deliberately NOT at this branch's head.** It sits at `7cf87b844`,
before the `origin/main` merge, whose diff touches `tt_bio/worker.py` and
`tt_bio/tenstorrent.py`. BoltzGen spawns a fresh subprocess per step, so a `git reset --hard`
now would import new engine code into a measurement already in flight. Tools this row needed
there (`scrmsd.py`, `contact.py`, `fan.py`, the rfd3 plan) were copied in individually.
Update the checkout once the two arms are collected, not before.

**A numpy-only reading does not belong on the device host.** `contact.py` on all 8 designs
takes 0.45 s per file on pc and had not finished in 11 minutes on whglx at load 64 with 477
of 566 GB in use. The 1536 docking numbers above were measured on pc for that reason.

**The rfd3 watcher was stopped, and the reason is the new brief constraint rather than a
failure.** It would have launched onto the whglx checkout at `7cf87b844`, and
`tt_bio/main.py`, `tt_bio/tenstorrent.py` and `tt_bio/worker.py` have all moved since — every
model rides those three, rfd3 included. So the number would have been attributable to a tree
that is not main. Killed by explicit pid 3996189; it had found 0 of 27 chips free for 53
minutes, so nothing was lost. `launch_if_clear.sh` now **refuses** when the remote checkout
does not contain `origin/main`, asked of the checkout that will execute the job rather than
of this worktree — verified refusing against whglx right now, and the ancestry logic verified
on both branches. rfd3 goes after the two arms are collected and the checkout is updated.

**A size comparison that does NOT need upstream's 1536 design, and is worth queueing.**
`out_dev1536` scores the device's 1536 designs with upstream's refolder. The same thing
exists at 512 for only TWO of the eight device designs (the extremes, 1.736/0.874 and
13.688/11.756). Refolding the other six at 512 — six times ~17 min on qb2, no chip — would
give **device designs at both sizes, scored by the same upstream refolder**, which is a size
comparison with the scoring fold held constant and upstream's expensive 1536 DESIGN step
taken out of the critical path entirely. Run it after `out_dev1536` finishes rather than
beside it: qb2 is 16 cores and already at load 27.

**Queued, not started:** rfd3 512 and 1536 (both entries), and `plans/bg_floor_1536.txt`, the second
valid 1536 target that would give the 1536 rung an across-target floor of its own.

## REFERENCE

REFERENCE: **there is NO upstream design reference on the valid target, at either size.
The 1536 one this row reported last pass is WITHDRAWN: it was designed against the
disconnected fixture the row itself disqualified.** Both sides' specs name a file
`bgt1536.cif` and the two files differ:

    fixture                                 chains        atoms   source
    qb2   ~/bgref-work/fx/bgt1536.cif       A=1008 B=528  12405   built 12:38, old rung
    whglx fx_boltzgen_1536_.../bgt1536.cif  A=823  B=713  12464   committed valid target

`gpb_dimer_1646.cif` carries A=823 B=823, so a 1536 crop in file order gives A=823 B=713 --
the device's. A=1008 cannot come from it. And qb2's fixture measures **162.40 A minimum
inter-chain distance, 250.81 A centroid separation**: the rung whose two proteins condition
nothing of each other, which makes an 80-residue binder against it an easier and different
problem. The qb2 fixtures are stamped 12:38 on 09-23 and the valid target was committed at
15:08 the same day, so the reference was started before the row knew the rung was invalid and
was never rebuilt (`results/upstream_reference_wrong_fixture.txt`).

**Withdrawn with it:** "upstream does not degrade from 512 to 1536", "the 1536 loss is the
port's, not BoltzGen's", and the four-row table that set upstream's old-target 11.04 A beside
the device's valid-target 7.01 A. Those two cells are different targets and are not a column.

**Not touched by this:** joint 1, where both sides are the DEVICE's own designs on the valid
target and no upstream fixture enters -- 7.01 A at 512 against 19.35 A at 1536, n=8 each,
U=0 of 64, p=1.554e-4. The device's size effect on a valid connected target stands.

**`docs/design-throughput.md` is UNAFFECTED and must not be "corrected".** Its upstream row is
labelled "the same 512 residues" and sits directly under "first 512 residues of a 1008-residue
chain", so it is paired with the 10.56 A device figure on that same old target -- a matched
comparison, checked line by line this pass. The doc carries no 1536 quality number at all,
which is why the withdrawal never reached a user.

**So joint 2 is OPEN.** The brief's test -- a worse device number at 1536 is a defect only if
upstream does better at 1536 -- is unanswered for the target the size claim is measured on.
Closing it needs an upstream BoltzGen design against `gpb_dimer_1646.cif` at 1536: ~12 h of
design step per design on 16 cores, or the GPU in ask 9957.

The 512 half of the reference is unchanged and is quoted below as it stood -- and it is
sound, because there the two sides genuinely were the same old 512 crop.



The unmodified six-step
upstream run finished — `REF_DONE size=512 rc=0 wall_s=17310`, 4.81 h for ONE design on 16
cores — and its own analysis table reads **7.16724 A**. A step-4-only run on the SAME design,
same config, same checkpoint, reads **11.04058 A** (`results/reference_512.txt`):

    upstream six-step, its own analysis CSV      7.17 A
    upstream step-4 only, same design           11.04 A   -> spread 3.87 A on ONE design
    device (TT), same target, n=9 median        10.56 A   (range 3.30-13.78)

**Both upstream draws fall inside the device's range**, and 3.87 A is larger than the 2.10 A
across-target floor at this size. An earlier version of this section quoted 11.04 as *the*
reference and read it against 10.56 as if the ordering meant something; it does not.

**And the refold noise is a large part of what looks like design-to-design variation.**
Three independent repeat pairs of the same refold on the same design now exist: 0.87/1.74,
11.76/13.69 and 7.17/11.04 A — spreads of **0.86, 1.93 and 3.87 A**. The device's nine
designs at 512 span 10.5 A in total, so a meaningful share of that spread is the scoring
fold rather than the designs differing. It does not change any median, and it makes every
single-draw comparison this row withdrew look worse rather than better.

**`scrmsd.py` is validated a second time by this run, independently.** On a different host,
a different refolder and a CPU fp32 path it reproduces the pipeline's own
`designfolding-bb_rmsd` to **0.00001 A** with the N,CA,C,O backbone (N,CA,C misses by 0.024,
CA alone by 0.075). The first validation was 8 designs on the device at 0.00000 A. So reading
the metric at step 4 instead of waiting for analysis is sound on both sides of the
comparison, which is what makes an upstream 1536 number reachable at all.

At 17:20Z on qb2 (16 cores, load 28.6):

    out512        six-step run   step 1 design 2:01:29 for ONE design; step 3 folding 2.2 h
                                 in with nothing written; now nice 19
    out512_df     step 4 only    DONE — 11.04 A, the row above
    out_devrefold step 4 only    the DEVICE's own 512 designs, refolded by upstream fp32
    out1536       six-step run   step 1 design, 4.2 h in, 446 % CPU, state R, nice 0

**What changed, and why 1536 is now reachable.** Designability needs step 1 and step 4.
Between them sits step 3, `folding`, which refolds the WHOLE COMPLEX and is the most expensive
step at a large target — measured, not guessed: the 512 reference has been in it 2.2 hours on
one design with nothing written, and at 1616 tokens it is what put an upstream 1536 number out
of reach. **Step 4 does not consume step 3's output.** `folding.yaml` and `design_folding.yaml`
differ by exactly two lines, `return_designfolding: true` and `designfolding: true`, and name
the same `design_dir`. Diffed, not assumed. So
`perf/mgxaccuracy/ref/designfolding_only.sh` stages a copy of the inverse-folded designs and
runs step 4 alone, and `scrmsd.py` reads the metric analysis would have written. **This is not
a shortcut on the model's own work**: the design is sampled at upstream defaults
(`sampling_steps 500`, `recycling_steps 3`) and refolded by the same checkpoint at fp32. What
is dropped is a complex-prediction metric this comparison never reads. The staged copy is what
lets it run beside the six-step run — both would otherwise write `metrics_tmp/data_<id>.npz`.

**The step-4 cost is set by the BINDER, not the target — 16 min 47 s for one 80-residue
design, measured — so the device's designs can be refolded by upstream's fp32 refolder at any
target size.** That is running now on two whglx 512 designs (`out_devrefold`) and is the one
instrument available without a GPU that separates *the designs are worse* from *the refolder
that scores them is worse*. The device's inverse-folded output loads into the upstream
pipeline unchanged; verified.

**The refolders were compared directly, and the control is the part that matters.** The
scRMSD is produced by a fold the DEVICE ran, so "the designs are worse" and "the refolder
that scores them is worse" have the same symptom. Step 4's cost is set by the binder — 80
residues, ~17 min on CPU, unchanged by target size — so upstream fp32 refolded the device's
OWN designs, the two extremes of its 512 distribution, and then refolded them AGAIN
(`results/refolder_pair_512.txt`, pre-registered as P5 before any number existed):

    design      device   upstr r1   upstr r2   upstr mean   dev - mean
    bg512        0.865      1.736      0.874        1.305        -0.44
    bg512_1     12.897     13.688     11.756       12.722        +0.18

**Read the repeat column first.** Two upstream runs of the SAME design, same config, same
thread count, same checkpoint, differ by **0.86 A** on one and **1.93 A** on the other. An
fp32 CPU diffusion refold is not reproducible run to run — thread scheduling changes the
reduction order and the sampler amplifies it. Against that spread the device is
**indistinguishable** from upstream: its values sit in or on the edge of upstream's own
range, and the residuals against the upstream means have **opposite signs**, so there is no
direction to the difference.

An earlier version of this section read the r1 column alone as a +0.83 A additive offset and
reasoned about what it would mean for the reference comparison. The control that withdrew it
was already queued when the claim was written, which is the only reason it did not ship
(`7986b542f`). **What survives is stronger:** the 11.04-against-10.56 comparison needs no
refolder correction, and one upstream draw carries 0.9-1.9 A of noise — an n=1 reference
bounds the LEVEL, not a difference. P5's verdict clause holds under either refolder.

**First 1536 cross-refold, n=1, no conclusion drawn.** Upstream fp32 refolding the device's
first gpb-1536 design reads **18.534 A**, against a 512 range of 0.86-13.78 across seventeen
designs. That is outside the 512 range — and one design is not a median, which is the exact
error this row exists to correct. Seven more are folding (`out_dev1536`, ~17 min each). Read
it only when the set is in, beside the device's own step-4 numbers on the same eight designs.

**The device and the reference do the same work per design, checked on disk rather than
assumed.** Both `design.yaml`s read `sampling_steps: 500`, `recycling_steps: 3`,
`diffusion_samples: 1`. The `s400` in the device directory names was a harness artifact —
`--steps` reaches rfd3 and pxdesign and never boltzgen — and the tag no longer prints it.

**The 512 six-step run was not killed, it was reniced.** Its step 3 runs every thread at nice
19 — `renice` on the pid alone leaves the other nine threads at 0, checked in
`/proc/PID/task/*/stat` — so the 1536 design step gets the cores while the 512 run keeps its
claim to being upstream's pipeline unmodified.

**Next on the reference:** when `out1536/intermediate_designs` has its cif and step 2 has run,
`bash perf/mgxaccuracy/ref/designfolding_only.sh --size 1536` gives the upstream 1536 scRMSD
without waiting on a 1616-token complex fold. n=1 at each size either way — the ceiling CPU
imposes. **Ask 9957 (vast.ai GPU) is still OPEN and unanswered since 02:27Z**, asked by
`mgx-reference`; its numbers are stale (the box it describes stopped ~03:35Z) but the decision
it needs is the same one this row would use. A second ask would be a duplicate, so this row
records the need against 9957 rather than filing one.

**Read what these two references are before using them.** They run `bg512.yaml` and
`bg1536.yaml`, the ORIGINAL `big_1831` crops — so the 1536 one is on the two-body fixture. That
is deliberate and it answers the charter's branch-4 question: whether upstream is *equally* bad
where the device read 11.480 A, i.e. whether that number was device-specific. Together the pair
gives **the reference's own 512-to-1536 ratio on the same fixture the device numbers came
from**. What it does NOT give is a 1536 reference on a valid target.

What already exists at ~120 residues: upstream **boltzgen 0.3.2 on a rented RTX 3090**, n=16 on
7ROA, median **1.05 A / 68.75 % designable** against the device's **0.78 A / 93.75 %**
(`docs/implementation-parity-data/boltzgen.json`). At a size that has worked for months the
device is on the favourable side of upstream, which is the same direction as the 512 result
above.

vast.ai is blocked on ask 9957, so 512 and 1536 have to be CPU. tt-bio's vendored BoltzGen has
no torch fallback (`tt_bio/boltzgen/model/modules/diffusion.py:93` binds `TTScoreModelAdapter`
unconditionally), so the reference is the upstream pip package. Its identity is committed in
`perf/mgxaccuracy/ref/`: boltzgen 0.3.2 on torch 2.14.0+cpu, HuggingFace snapshot
`c1be29e1f82ffcc72264f64b993c43fb4e0d17f0` — the same version AND snapshot the committed 3090
reference used, so the two differ in hardware and dtype and nothing else. Fixtures md5-checked
against the files whglx read. One patch was needed to run at all
(`ref/cpu_capability.patch`: `torch.cuda.get_device_capability()` is called unconditionally
before the only branch that consumes it) and it changes no arithmetic. Three Lightning settings
are overridden per step and all three move the reference toward exact: `accelerator` gpu->cpu,
`precision` bf16-mixed->32, `matmul_precision` high (TF32)->highest. `sampling_steps 500`,
`recycling_steps 3`, `diffusion_samples 1` stay at the upstream default.

## ATTRIBUTION

ATTRIBUTION: **the 1536 loss is in the DESIGNS, not in the fold that scores them — upstream
torch fp32 refolding the device's own eight 1536 designs reads 19.35 A against the device
refolder's 21.52 A on the identical eight.** Two independent refolders, one of them upstream's
unmodified CPU pipeline at fp32, agree these designs do not refold to their own backbones. That
retires the whole class of explanations in which a device-side refolding artifact manufactures
the gap between these two crops. **It does not show the gap is about SIZE** -- the
across-target floor at a fixed 512 residues is 13.04 A (11.39 A same-tree), comparable to
the 12.34 A gap itself -- and note the after-fix device size gap at offset 0 is 14.77 A, so
NEITHER effect dominates the other; they are the same order
(`results/across_target_floor_512.txt`), so what is attributed here is "these 1536 designs do
not refold", not "1536 designs do not refold".

**The upstream 1536 DESIGN this row reported last pass is WITHDRAWN and gives no direction.**
It was designed against the disconnected fixture (A=1008 B=528, 162.40 A apart), not the
valid target the device ran, so its 7.91 A is a different and easier problem and cannot be
set against the device's 19.35 A. There is currently NO upstream design reference on the
valid target at any size, so whether the 1536 loss is the port's or BoltzGen's is OPEN
(`results/upstream_reference_wrong_fixture.txt`).

**What is left to attribute is the design step, and it has one named candidate under test
right now.** `cc908c377` corrects `TokenDistanceRecycle`, which injects the target's token
distances into the pair representation: the old code returned twice the pairformer's output
where the reference takes its input plus the update. Both shipped checkpoints carry
`use_token_distances=True`, read out of the checkpoints themselves rather than the step yaml,
which says nothing (`results/token_distance_fires.txt`). Every boltzgen number above predates
it. An error in that module is the shape that grows with token count, and the experiment is
the same fixture at both sizes on the fixed engine — running on cards 5 and 3, BEFORE banked
at 8.46 / 21.52 A. **If the fix is the mechanism the 1536 median falls and the 512 one does
not.** This is attribution by effect, which is the only kind available without a GPU.

**Second, and independent of all of the above: the 1536 rung of the shared design ladder is
two proteins 246.9 A apart, so no quality question can be asked at that rung, and the row's
opening evidence was measured there.** `perf/bhdesign/make_big_target.py` reaches a large token count by translating real
chains 150 A apart along +x, and justifies it in its own docstring as "a two-chain target,
which is what a real complex is". Measured on the fixture it produced:

    big_1831.cif      chain A-B centroid 246.9 A, closest atoms 162.40 A apart
    chain A is 1008 residues, so every rung <=1008 is a compact crop of ONE chain
    512 crop:  Rg  23.4 A, max radius  39.4 A      <- a target
    1536 crop: Rg 123.3 A, max radius 201.2 A      <- two targets in one file

A real complex has an interface, and the measured discriminator is the connectivity of the
sub-22 A conditioning graph rather than the saturated fraction. `big_1831.cif`'s 1536 crop is
two components of 1008 and 528 with **zero** inter-chain edges under 22 A; the gpb dimer's is
one component with 8316. On an input whose conditioning graph is disconnected, returning an
undocked binder is correct behaviour, not a defect.

**So the fixture answers the capacity question soundly and cannot answer the quality question.**
`mgx-design-ceiling`'s "pxdesign PASS to 1536" and "boltzgen PASS at 1536 residues" are passes
on the token axis, which is what that row asked; they are not statements about design quality,
and this row should not have inherited the fixture for one.

**Landed so a later reader cannot repeat it.** `perf/bhdesign/make_big_target.py`'s docstring
now carries the measurement and names the replacement, and the script runs the connectivity
check on its own output and prints `WARNING: CAPACITY FIXTURE ONLY` when the graph is
disconnected — inert on the artifact, since the regenerated `big_1831.cif` is md5-identical to
the committed one. `docs/design-throughput.md` gained the user-facing half in two sections
("Give PXDesign one body, not two", and the designability table), and its "all three hold at
1536" line now says that is a capacity and rate claim rather than a quality one.

`perf/mgxaccuracy/make_contact_target.py` builds the other kind and **REFUSES** one whose
conditioning graph is disconnected, on the model's own 22 A threshold rather than a contact
radius chosen here — run on `big_1831.cif` it refuses at 2 components with 0 inter-chain edges,
with the 5 A contact test kept as a second clause. The rebuilt target is md5-identical to the
committed one. Its output, cut from the 1GPB **biological assembly** (the deposited ASU is a
single chain, which is why `perf/ceilrfd3/targets/gpb_823.cif` is single-chain and the ladder
had nothing above 1011 to crop): `perf/mgxaccuracy/targets/gpb_dimer_1646.cif`, 1646 residues,
2 chains, interface min atom distance **1.88 A**, Rg 38.4 A. Crops: 512 -> Rg 25.2 A, 1024 ->
Rg 31.1 A, 1536 -> Rg 37.4 A with both chains present, and offset 100 at 1536 is also one
component (5951 inter-chain edges), which is what `plans/bg_floor_1536.txt` would use.
Coordinates are not translated; the interface is the point.

**The guard refuses the synthetic fixtures and nothing else, and the pathology is broader than
one file.** Run over every structure CIF in the repo (`perf/mgxaccuracy/calibrate_guard.py`,
output in `results/guard_calibration.txt`): **41 of 44 are one component**, including every
deposited structure — multi-chain antibody complexes, hemoglobin at 574 tokens with 12667
inter-chain edges, the 9xxx benchmark set. The three refused are `big_1831` (2 components),
`big_3662` (4) and `big_7324` (8), all with **zero** inter-chain edges, and all three are
outputs of `make_big_target.py`. So the guard separates synthetic from real rather than
multi-chain from single-chain.

### Instrument defects found by running, all fixed

Each cost a chip or a wrong number before it was caught. Commits carry the detail.

- **`free_cards()` handed out an occupied chip, three times over.** The flock alone, the lease
  alone and the env pin alone are each wrong: a driver that takes the lock and hands the chip
  to a spawn child leaves the lock free (card 0 read takeable while `mgx-combos` was computing
  on it), and card 31's lease read `released` while `mgx-bigalloc` held `/dev/tenstorrent/7`
  open — node 7 IS logical 31, so the charter's non-identity map is right. Busy is now the
  union of four signals including an **open device fd**, which is the only one that sees a
  chip released by one row and taken by another. `perf/mgxscale/cards.py` prints all four.
- **A wrong interpreter burned three chips** into `FAIL`/`unknown` rows that read like model
  failures. `job.py preflight()` now refuses before a card is taken.
- **`ps -eo pid= -p PID` is a vacuous liveness check** — `-e` overrides `-p`, so it always
  exits 0. Two fans ran on one plan because of it.
- **The report pooled two different targets** at one size into a single cell, and later pooled
  clash fractions with scRMSD — nine zero-valued geometry designs pulled a 512 scRMSD cell to
  a pooled median of 3.68 A under the scRMSD label. Cells key on the target; each row carries
  one metric and only that metric's request loads it.
- **Two scripts disagreed on one cell** (10.59 against 10.56) because one dropped n=1 rows and
  the other kept them, an inclusion rule written down nowhere. Unified on every draw.
- **The scRMSD scorer first read the wrong refold directory.** `refold_cif/` is the step-3
  whole-complex refold; `refold_design_cif/` is the step-4 isolated one the bar is about.
- A numpy `int64` in the designability harvest would have raised at serialisation, after the
  device work was spent. And every row now records `fit_rmsd`: a harness that records seconds
  and not that signal reads eight undocked binders as a clean eight-design row.

### Two fleet-level findings this row cannot fix from inside

**For the orchestrator, and for any row quoting a 1536 number.** No `big_*` fixture may carry
a QUALITY claim above **1008 residues**, the length of its first chain. Every crop at or below
that is a compact piece of one protein and is sound; every crop above it is that protein plus a
second one translated 150 A away, and the measured consequence is in ATTRIBUTION. Capacity,
token-count and rate claims at 1536 on those fixtures stand — `mgx-design-ceiling`'s passes and
`docs/design-throughput.md`'s rates are unaffected. What does not stand is any statement about
how good the design is. `perf/mgxaccuracy/targets/gpb_dimer_1646.cif` is the replacement and
`make_contact_target.py` refuses to build another bad one.

**The chip grant is not a lease.**

`fleet.log` records this row launched on **whglx-card0** at 14:18:38 CEST. Card 0 was never
free: `worker:mgx-combos` was computing on it at 12:28Z and by 12:53Z it had passed to
`mgx-diffusion` (`perf/mgx-diffusion/hold.py 0` plus a ladder and three `tt_bio.main predict`
processes, all pinned to card 0). Two other rows used the chip this row was granted inside half
an hour, because each picks its chip with a probe that cannot see the others. This row declined
the grant rather than colliding — the fleet's card number is not a lease — and its fan takes
whichever chip is genuinely free instead.

## What the next pass does, in order

1. **Check the gate first.** `ls /home/moritz/.coworker/state/mgx/quiet-window`. If present,
   start nothing — not a device job, not a CPU job. Launch only through
   `perf/mgxaccuracy/launch_if_clear.sh`, never by sshing a fan directly; whglx cannot see
   the gate, and the launcher also refuses when the remote checkout does not contain
   `origin/main`.

2. **Collect the after-fix 1536 arm. It decides the row and nothing else is blocking.**

       whglx card3  boltzgen 1536 gpb, 8 designs, AFTER-fix, pid 451967
       BEFORE on the identical fixture: 21.52 A device-refolded, 19.35 A upstream-refolded, n=8

   `bash perf/mgxaccuracy/collect_1536.sh`. Read the step count first —
   `ls OUT/intermediate_designs_inverse_folded/refold_design_cif | wc -l` against 8 — so a
   partial harvest cannot pass for a complete one. At 01:05Z it had 7 of 8 designs at step 1
   and had not started step 4. **Its SECONDS are void**; whglx load was 54 at launch.

3. **Complete the 2x2, then read size and target separately. This is now the row's core
   experiment.** All four cells are after-fix, n=8, same designer/binder/metric/refolder:

       offset   0   512    4.749   DONE
       offset 100   512   17.785   DONE
       offset   0  1536       ?    whglx card 3, in step 3
       offset 100  1536       ?    whglx card 23

   `python3 perf/mgxaccuracy/twobytwo.py perf/mgxaccuracy/results/size_1536.jsonl` computes
   it: size effect = shift across sizes averaged over offsets, target effect = shift across
   offsets averaged over sizes, plus the interaction and a Mann-Whitney on each pooled half.
   It exits 1 and reports NOTHING until all four cells are in. Report BOTH effects with their
   medians. If the target effect is the larger one again, the row's answer is that target
   choice dominates size for BoltzGen designability -- a real, user-actionable finding that
   the docs already point at.

   **THE FOUR CELLS DO NOT ALL RUN THE SAME TREE, found 02:0xZ.** The card-3 arm runs from
   `~/wt-mgx-design-accuracy` at `07bd40614` and the offset-100 cells from `~/wt-mgx-catcher`,
   and `git diff 07bd40614 d0c4f33e4 -- tt_bio/` is 422 insertions over 14 files including 128
   changed lines of `tenstorrent.py`, which every model rides. Per cell:

       512  offset   0   720b43a5d
       512  offset 100   d1133c67e
       1536 offset   0   07bd40614   (confirmed to CONTAIN cc908c377, so genuinely after-fix)
       1536 offset 100   d0c4f33e4

   **MEASURED 02:34Z, and it is inert.** The same crop on both trees, n=8 each: 512 offset 0
   reads median **4.748 A on 720b43a5d and 4.631 A on the current tree, a 0.117 A shift**, and
   Mann-Whitney does not reject (U = 28 of 64, where 32 is no difference at all). The engine
   delta is **111x smaller** than the 13.04 A target effect, so the cross-tree construction
   does not threaten it and the docs sentence stands (`results/engine_delta_is_inert.txt`).
   The offset-100 re-run is queued behind it on card 2 and will make the pair fully same-tree.

   **That control also measures the row's precision, which is worth more than the control
   itself:** run-to-run reproducibility of an n=8 MEDIAN is 0.117 A, while the individual
   designs inside those same cells span 0.693 to 13.493 A. Eight-design cells carry quality
   claims; single draws and catchers do not, and this is the number that says so.

   **THE CONFOUND IS BEING REMOVED, not just named: an 823-residue cell is running.** Chain A
   is 823 residues, so ALL of chain A is a single-chain target, and against the offset-0 512
   cell it is a nested, same-protein, same-chain-count size comparison:

       512 residues (1-512 of chain A), single chain   median 4.631 A, n=8, banked
       823 residues (1-823 of chain A), single chain   card 2, launched 03:2xZ, ~45 min

   The fixture the job actually built was checked rather than assumed:
   `chains {'A': 823}, 6691 atoms, coord_sha 7020d95a`, one chain and no chain B
   (`plans/size823_onechain.txt`). Whatever it shows is about SIZE within one chain.

   It does NOT replace the 1536 cells. A user who brings a 1536-residue target brings the
   second chain with it, so the 2x2 stays the number describing what the service does at that
   size; 823 is the clean mechanism question sitting underneath it.

   **The SIZE axis of this 2x2 carries a named limitation; the TARGET axis does not.** Chain A
   of the source is 823 residues, so a 512 crop in file order never reaches chain B: both 512
   cells are single-chain and both 1536 cells are two-chain contacting dimers (all four
   fixtures checked with `fixture_id.py` at 01:57Z, both 1536 crops at 1.88 A minimum
   inter-chain distance). So the size main effect is **"512 single-chain vs 1536 two-chain"**,
   not 512-vs-1536 residues in the abstract, and it must be written down that way. The target
   main effect compares offsets at a fixed size where the chain count matches, so the 13.04 A
   figure is clean. Pre-registered before the last two cells landed
   (`results/across_target_floor_512.txt`).

   **Append the two 1536 cells with `engine` AND `target` set, or the script will refuse
   them, which is deliberate.** Size+offset does not identify a cell: `size_1536.jsonl`
   already holds a 1536 offset-0 row from the disqualified `big_1831` fixture with the same
   key as the valid one (11.205 A against the valid 21.516 A), and the pre-fix and after-fix
   cells collide the same way. The script caught the first of those on its first run.

   The after-fix arm also answers, separately, whether `cc908c377` moved 1536 quality: BEFORE
   on the identical offset-0 fixture is 21.52 A device-refolded / 19.35 A upstream-refolded,
   n=8. Report that as its own line; it does not need the 2x2.

4. **Do not publish a 1536 quality number to users until step 3 lands.** `cc908c377` is on
   main, so every 1536 number now banked describes an engine nobody will run. This is the one
   place the row could still mislead a competition entrant, and the doc currently has no 1536
   row at all, which is the safe state to be in while waiting.

5. **Joint 2's reference is RUNNING on the valid target, launched 01:2xZ. It is the row's
   critical path and it needs ~12 h.** `qb2:~/bgref-work/run_valid.sh 1536 6`, detached, log
   `~/bgref-work/ref_valid_1536.log`, output `~/bgref-work/out1536_valid`. Harvest with
   `scrmsd.py ~/bgref-work/out1536_valid --json --model boltzgen --target gpb_dimer_1646
   --target-res 1536 --side upstream-design-upstream-refold`.

   **The fixture was checked BEFORE launch, which is the whole point.**
   `perf/mgxaccuracy/fixture_id.py` on the newly built `~/bgref-work/fx_valid/` reports
   coord_sha `34d0b6f4a...` at 1536 and `7c38b3080...` at 512, both **matching the device's
   fixtures exactly**, and the 1536 one reads `min_interchain 1.88 A ... CONTACTING`. The
   withdrawn reference failed precisely here.

   `--steps design inverse_folding design_folding analysis` omits `folding`, the whole-complex
   refold. The log confirms it took: it prints `[Step 1/4]`, not 1/6. This is not doing less of
   the model's own work -- the design runs at upstream defaults (sampling_steps 500,
   recycling_steps 3) and `design_folding` reads `design_dir`, not the folding output. What is
   dropped is a complex-prediction metric this comparison never reads, and dropping it is what
   makes an upstream 1536 number reachable on CPU at all: the old run spent 68 min in that step
   and wrote nothing.

   **Cores for it came from killing the old reference**, which was folding the DISQUALIFIED
   fixture at 561% for no reader: pids 1235144/1235147/1925298/1925664/1925666, all confirmed
   gone with no orphans. qb2 was at load 29 on 16 cores with three other rows also on it.

   **The 512 counterpart is NICE-TO-HAVE, not required, and that is why it keeps being held.**
   Joint 2 asks whether the device is worse than upstream AT 1536. That needs upstream@1536
   against device@1536 on the same target, which is exactly the run in flight -- the fixture
   coord_sha was matched before launch, so the comparison is clean on its own. The 512
   reference would add upstream's own size trend, which is context rather than the answer.
   Held at 03:3xZ with qb2 at load 24.5 on 16 cores and the 1536 run at 341%: adding a second
   4-5 h CPU job would slow the one that answers the question. Start it once qb2 is quiet, or
   take ask 9957 (vast.ai GPU, unanswered since 02:27Z 09-23) for both.

   Health at 03:35Z: worker 2095799 at 341% CPU, 2h13m elapsed, still `[Step 1/4] design`,
   zero structures written -- which is expected, since the design step on the old fixture took
   10.4 h and writes nothing until it finishes.

6. **The second valid 1536 target is UNBLOCKED and running.** `crop_cif` rebased the atom
   serial but left `label_seq` at the parent's values, so a crop at a nonzero offset started
   at seq_id 101 while BoltzGen indexes `entity_poly_seq` positionally, and the parser
   asserted before the device opened. Fixed in `1f6e813dc`: label_seq is numbered 1..n per
   chain, `auth_seq_id` untouched. Offset 0 is **byte-identical** pre vs post at both sizes,
   so nothing already shipped moves, and `tests/test_crop_cif_offset_seq_numbering.py` pins
   the invariant (6 passed on qb2; with the fix reverted exactly the two offset-100 cases
   fail). `plans/bg_floor_1536.txt` relaunched on cards 13 and 23 and is past the point where
   it died. Collect it with the card-3 arm. Note the offset axis had never run on ANY row
   sharing `perf/bhdesign/ladder.py`, so this unblocks more than this row.

7. Do not quote a size ratio measured on any `big_*` fixture above its first chain's 1008
   residues. All three are disconnected; the orchestrator note is in ATTRIBUTION.

8. **Chips: pick at acquire time, never from a census.** Cards 11 and 28 read released with
   dead pids and were gone to other rows six minutes later. The fan resolves the lease dir
   when it acquires, which is the only non-stale reading. Cap is 3, never 1/24/25/26/27,
   never chip 4. The row holds one chip right now.

VERDICT: PARTIAL — **crop choice and target size are the SAME ORDER of effect for BoltzGen
designability, and both are huge: moving a 512-residue crop 100 residues along one chain
moves the median 11.39 A same-tree (4.631 -> 16.021, n=8 each, MW rejects) or 13.04 A on the
earlier cross-tree pair, while going 512 -> 1536 on one crop moves it 14.77 A (4.631 ->
19.398, after-fix).** An earlier version of this verdict, and of
`docs/design-throughput.md`, said crop choice mattered MORE than size. That ordering is
withdrawn: it compared the 13.04 A target gap against a 12.34 A size gap computed on the
UPSTREAM refolder, and the after-fix device number at 1536 is 14.77 A, which exceeds the
target gap either way. The doc now says the two are the same order and neither is
second-order, which is what the data supports and keeps the user advice intact.

**What is NOT withdrawn is the thing that broke the original size story.** The row opened
treating 512-vs-1536 as a size axis. At a FIXED 512 residues two crops of one chain differ by
11-13 A, so a single 512-vs-1536 contrast on one crop can never have been a size measurement;
it was always a size effect plus an unmeasured crop effect of comparable magnitude. That is
why the 2x2 exists and why no size verdict goes in writing before its last cell lands.

**AND THE NAMED MECHANISM IS REFUTED, measured 02:48Z.** `cc908c377` (TokenDistanceRecycle,
which injects the target's token distances and was this row's first candidate because its
error plausibly grows with token count) moves the 1536 offset-0 median from **21.516 A to
19.398 A, n=8 per engine on the identical fixture, and Mann-Whitney does not reject (U=44 of
64)**. Zero of eight designs clear the 4 A permissive bar on either engine. The fix helps by
about 2 A and does not rescue 1536 (`results/token_distance_fix_refuted.txt`). The one thing
that moved is the minimum, 19.311 A pre-fix against 6.650 A after; one design in eight is not
a result at this spread but it is the only hint the fix does anything at this size.

**So a 1536-residue BoltzGen design returned today still does not refold to its own backbone,
and this row has not located a port-side cause.** The remaining suspects are the brief's, in
its order: the row-blocked and refused-upload paths, narrow-q, bf16 accumulation over the
token axis. But the across-target floor makes "the 1536 loss" the wrong noun to hunt: at a
FIXED 512 residues two crops differ by 13.04 A, so what needs explaining may be crop
difficulty rather than anything size-specific in the port.

**The experiment that separates the two is running and is the row's remaining work:** the 2x2
{offset 0, offset 100} x {512, 1536}, after-fix, n=8 per cell. Two cells are in, two are on
cards 3 and 23. No size verdict goes in writing before they land.

**Joint 2 is separately open**: the upstream reference this row compared against had been
designed on the disqualified disconnected fixture, so it was withdrawn, and a replacement on
the verified-identical valid target is ~12 h into its design step on qb2.

**Joint 1 closed this pass.** Holding the scoring fold constant — upstream torch fp32 refolds
the device's own designs at both sizes, n=8 each — gives 7.01 A at 512 against 19.35 A at
1536 with zero overlap, Mann-Whitney U=0 of 64, exact two-sided p=1.554e-4. Both sides are
pre-fix by timestamp, checked. That removes the last shared-instrument objection: no
device-side refolding artifact explains the 1536 loss.

At 512 residues the device is not worse than upstream and the question's original
premise is gone there: upstream's own unmodified pipeline reads 7.17 A on the same target
against a device median of 10.56 A (n=9, range 3.30-13.78), a repeat of that same upstream
quantity reads 11.04 A, and both upstream draws fall inside the device's range — the
reference's 3.87 A run-to-run spread on ONE design is larger than the 2.10 A across-target
floor at this size, so no ordering can be read off it.** So the ~10 A designability of a real 512-residue target is
BoltzGen and the target, not this port, and `docs/design-throughput.md` now tells a
competition user to crop the target, with the numbers.

**Both halves of the opening evidence are withdrawn.** The 2.8x scRMSD gap was two single
draws from one distribution — at n=9 the same 512 target spans 3.30-13.78 A around 10.56, so
11.480 A at 1536 is an ordinary member. The 7.1x clash fraction is 2.04x at n=8 per size and
does not separate (U=15, z=1.79). And the fixture both were measured on is disqualified for
the question: the ladder's 1536 rung is two proteins 246.9 A apart whose sub-22 A
conditioning graph has zero edges between them, which no model can resolve and no device run
can falsify.

**pxdesign is ANSWERED and holds:** `fit_rmsd` 0.1128 A at 592 tokens, 0.1423 at 1224, 0.1747
at 1616 on a valid target — a resolved 1.55x trend that is still 86x under its own 15 A gate,
against 95.183 A at the same 1616 tokens on the disconnected fixture. The live-reachable
1224-token case was never broken. Along the way this row found and fixed a crash that had
been failing every pxdesign run on `main` since 05:55 CEST; the fix is on main and confirmed
on device at quiet load.

**boltzgen docks at 1536 on both targets** — all 16 designs in contact, 0.75-2.09 A — which
also shows the disconnected fixture breaks distogram-conditioned models and not
coordinate-conditioned ones.

**At 1536 the answer is no longer withdrawn, it is measured, and it is bad.** On the valid
target the device reads median 21.52 A over eight designs (19.31-22.48) against 8.46 A over
eight at 512 on the same target, with zero overlap and the 1536 median 11.0 A above the top
of the 2.10 A across-target 512 band. Upstream torch fp32 refolding those same eight designs
reads 19.35 A, so the loss is in the designs and not in the fold that scores them. **A
1536-residue BoltzGen design returned today does not refold to its own backbone**, and that
is the row's headline whatever the cause turns out to be.

**What decides the row is running, and it is one named mechanism against one banked BEFORE.**
Every boltzgen number here predates `cc908c377`, which corrects the module that injects the
target's token distances — the conditioning a binder design is built on — and whose error is
the shape that grows with token count. The same fixture at both sizes on the fixed engine is
on cards 5 and 3 now. If the 1536 median falls and the 512 one does not, this row located a
conditioning bug by effect and the fix is already on main. If it does not move, the mechanism
is refuted and the honest close is that BoltzGen loses designability at 1536 regardless of
port, documented for a competition user in one sentence with the numbers.

**What is still owed, and it is running, not planned:** the after-fix boltzgen pair at 512 and
1536 (the deciding experiment), the rfd3 pair on a valid target, and upstream's refold of the
eight device 512 designs, which closes the refolder-matched comparison at n=8 against n=8.
Upstream's OWN 1536 design is the one arm that may never arrive: 10.4 h in its design step
with zero structures written, and ask 9957 (vast.ai GPU) unanswered since 02:27Z. The
attribution split was built to not depend on it. The honest ceiling is stated rather than
implied: n=8 per size gives 45 % power against a doubling of the median, and with one valid
target per size a size effect and a target effect are still the same number.