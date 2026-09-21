# allm-audit — did the perf work transfer? Ten folds that size the campaign

TASK TYPE: VERIFY/BENCHMARK | PLAYBOOKS loaded: §ACCELERATE "THE PERF METHOD" + ALWAYS-ON |
memories read: `qb2-aiclk-governor-sets-fold-time-not-cotenancy`,
`perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`, `perf-page-cell-is-historical-not-live-baseline`,
`op-ab-must-interleave-arms-compile-warmup-bias`, `benchlock-one-shot-check-blind-to-mid-run-contention`,
`benchlock-protects-co-tenants-not-just-the-caller`, `eligibility-firing-condition-is-not-a-code-fact`,
`sibling-perf-campaigns-need-namespaced-output-paths`, `reference-checkpoint-version-binding-strict-false`,
`hf-revision-pin-fix-missed-three-direct-callers`, `wait-loop-pgrep-pattern-self-matches-its-own-wrapper`,
`qb2-p300c-dispatch-granularity-is-a-board-pair`, `design-row-throughput-denies-gpu-its-batch`.

Branch `wk/allm-audit`, worktree `/home/ttuser/.coworker/wt/allm-audit` on qb2.
Arms folded on **qb2 card 1**, artifacts `perf/allm_audit/` and `/home/ttuser/allm_audit/` (mine alone).

VERDICT: GO — **all five transfer ratios measured, ten arms on one card with one instrument.**
BoltzGen **1.2896x**, OpenFold3 **1.1311x**, RFdiffusion3 **1.0411x**, ESMFold2 **1.0385x**,
OpenDDE **1.0319x**. Every arm ran at AICLK 1350.0 mean and 1350 minimum sampled DURING the fold,
zero re-asserts, on qb2 card 1 (Blackhole p300c), with both arms of a model interleaved on the same
card and an A/A floor beside each. Every tree is verified file by file against its published commit,
the scope is settled against the page's own source, and no old commit was abandoned. **Only BoltzGen
is anywhere near Boltz-2's 1.5006x.** The three shared-pairformer models land 1.03-1.13x, and the two
that sit outside that core land 1.04x and 1.29x, which is the opposite of what the shared-core
hypothesis predicts. This row reports the ratios and stops there; pairing them with the work census
to say what each one MEANS is `allm-orchestrator`'s, per its 2026-09-20 correction — a ratio near
1.0x is equally consistent with a blocked lever and with a model already converting its work better
than the model the 1.5x came from, and the ratio alone cannot tell those apart.

**The gate hole this document found has since been closed, recorded here because the finding was
this row's.** `_allm_donecheck.py`'s `ratios` check counted `\d\.\d+x` anywhere under `RATIOS:` and
needed four. It passed this document when it contained **zero measured ratios**: the four matches
were the prose `1.5x` and `1.0x` and the two `pvx-didittransfer` figures quoted for comparison. A row
citing its predecessor's numbers cleared it for free. The check is now keyed per model name and
refuses an in-flight word, so each of the five owes a ratio or a plain statement that it could not
get one.

## SCOPE

SCOPE: **`predict_one` — featurize, fold, write the CIF — one process, timed per fold after a cold
fold, host and device wall clock, `hoist=False`.** That is `tt_baseline.build_fold`'s default region
(`scripts/gpu_vs_tt/tt_baseline.py`, `timed_region="predict_one (featurize + fold + CIF write)"`),
the region `pvx-baseline` and `pvx-didittransfer` used for every cell of theirs, and — this is the
part the brief got wrong — **it is also the published page cell's own region.** One definition, every
arm of every fold model, no exceptions.

**The brief says the published cells are "device-only forward, excluding host". They are not.**
`site/data/perf-512aa.json` is the only place those numbers are edited, and its `scope.split` says
it in one line: *"The Tenstorrent host seconds are inside the published cell on every row."* Each
p150a cell carries a `split` block with `in_cell: true` and the host seconds it contains —
ESMFold2 0.056 s of 29.393, Protenix-v2 0.101 s of 50.543, OpenDDE 0.125 s of 82.393, OpenFold3
1.839 s of 38.254. Device-only is a SECOND reading the page derives by subtracting that split for
the NVIDIA comparison; it is not the cell. So `predict_one` reproduces the published timed region
rather than departing from it, and the two campaigns' numbers sit on one scale.

Two consequences worth stating rather than leaving implicit:

1. Host is 0.19-0.24 % of the cell on ESMFold2, Protenix-v2 and OpenDDE, so on those models the
   distinction cannot move a ratio at all. On **OpenFold3 it is 4.8 %** and that one is worth
   watching: if its ratio comes out near 1.0x, host is a large enough term to be asked about
   separately before the model is handed to `allm-gates`.
2. The published cell is still **not** a baseline for my arms and is not used as one. It was taken
   on a different day, a different box mood and an unpinned governor. It is used for exactly one
   thing: naming the commit each old arm is extracted at. Every ratio here is read between two arms
   of mine that share the region, the card, the clock and the instrument.

**The two design models keep their own unit.** BoltzGen and RFdiffusion3 are s/**design**, not
s/fold, and nothing here converts between them (`design-row-throughput-denies-gpu-its-batch`). Their
timed regions are their own published harnesses', restated under DESIGN-SCOPE when those arms land.

## ARMS

ARMS: **qb2 card 1, Blackhole p300c**, ttnn 0.68.0, grid 11x10, AICLK **pinned at 1350 MHz and
sampled DURING every fold at 4 Hz** by the instrument, per fold, alongside every other chip in the
box. Benchlock held for every session. Arms alternate by session, old tree and new tree in turn,
because a tree cannot be swapped inside one process.

**Board partner, named because on this part it is the largest confound.** qb2 is four p300c chips in
two board pairs, 0/1 and 2/3, and card 1's partner is **card 0**, which shares its power budget. A
busy partner is the difference between a 14.6 s fold and a 22.8 s one on the same commit. During
this pass the `of3t-crop640` campaign held cards 0 and 2, so the instrument's per-fold per-node
record of the partner's clock and watts is the evidence each arm is judged on, not a sentence about
the box. qb1 was checked as an alternative and is worse: loadavg 18.97, three of four cards held.

**Instrument:** `pvx-baseline`'s `cell.py`, byte-identical (md5
`21e0770f080a4d965203b59a193107be`), committed as `perf/allm_audit/cell.py`. Same file as
`pvx-didittransfer`'s, so all three rows time the same region with the same code.

**Provenance is checked against git, not assumed.** Each old tree is `git archive` of its published
commit, and every `.py` under `tt_bio/` in all six trees was re-hashed on disk with
`git hash-object` against `git ls-tree -r <commit>`: **1693 files, one differing**, and that one is
the ESMFold2 hub-revision line documented under UNBUILDABLE. So the only edit anywhere in any arm
is the edit this document describes. The new arm is `origin/main` pinned at
`47810889f` (2026-09-21), extracted the same way, so "today" is one commit for every model rather
than whatever main happened to be that hour.

| model | old commit | date | published cell | new commit |
|---|---|---|---|---|
| ESMFold2 | `e65b66be` | 2026-08-16 | 29.393 s/fold | `47810889f` |
| OpenDDE | `b4feba14` | 2026-08-15 | 82.393 s/fold | `47810889f` |
| OpenFold3 | `973ae49f` | 2026-08-15 | 38.254 s/fold | `47810889f` |
| BoltzGen | `59474b45` | 2026-08-14 | 45.289 s/design | `47810889f` |
| RFdiffusion3 | `6f85ecfe` | 2026-08-20 | 91.443 s/design | `47810889f` |

Every commit subject names its own cell, which is how each was confirmed to be the right one:
`e65b66be` is "docs(perf): republish ESMFold2 512aa cell at 29.393s post p3 levers", `b4feba14` is
"oddedgx: 82.393 s at 512 aa", `6f85ecfe` is "rfd3: page cell 105.123 -> 91.443 s/design".

## UNBUILDABLE

UNBUILDABLE: **no old commit has been abandoned. One could not be folded as it stands and was
repaired without substituting a measurement, and the repair is the interesting finding of this
pass.**

**ESMFold2 at `e65b66be`: the tree is fine, the hub moved under it. Folded, after a one-line
repair at the call site.** The first old arm died in model load:

    TypeError: DiffusionStructureHeadConfig.__init__() got an unexpected keyword argument 'architectures'

Not a build failure and not a code difference — `tt_bio/_vendor/esmfold2_hf/configuration_esmfold2.py`
is **byte-identical** between `e65b66be` and `origin/main`. On **2026-09-14 19:17 UTC** upstream
re-published `biohub/ESMFold2`, `biohub/ESMFold2-Fast` and `biohub/ESMC-6B` in place, in a new
config schema, and took the public service down for six hours. `origin/main` answered with
`weights.HF_REVISIONS`, pinning every third-party repo to the commit its port was verified against.
A tree from August predates that table, asks the hub for `main`, and gets a checkpoint it cannot
parse. **An old commit can stop being foldable without a single byte of it changing.**

The repair keeps the arm honest rather than replacing it. Substituting a newer checkpoint would
make the ratio a weights change wearing an optimization's clothes. The pins ARE the pre-2026-09-14
commits, so pinning restores exactly the checkpoint the 2026-08-16 cell was measured against and
gives both arms byte-identical weights, which is the only way the ratio is a property of the tree.
`perf/allm_audit/pinned_cell.py` reads `HF_REVISIONS` out of the new tree's `weights.py` as TEXT
(never imported — this process has an old `tt_bio` on its path), fills `revision` on
`huggingface_hub.hf_hub_download` / `snapshot_download` and on the copy `transformers` binds, and
is run on **both** arms so the two differ in the tree and in nothing else. On the new tree it is a
no-op by construction, because that tree passes the same revisions itself.

**Which repos it pinned is COUNTED at run time and written into every result JSON**
(`hf_pins_fired`), never read off the fact that a pin table exists. A shim that pinned nothing
would record `{}`, and that is what distinguishes the old arm from the new one.

Ten pins parse: ESMFold2, ESMFold2-Fast, ESMC-6B, esmc-300m, esmc-600m, protenix-v2-weights,
OpenDDE, and three SaProt repos.

**And the shim was at the wrong altitude — its own firing count is what caught that.** With
`hf_hub_download` wrapped in every module that binds it, the old arm failed identically and
`hf_pins_fired` came back **empty**. transformers 5.16.1 resolves a config through `cached_files`,
not `hf_hub_download`; widening the shim to `cached_file`/`cached_files` did not fire either. Had
the shim not counted what it pinned, "the pin table is installed" would have read exactly like "the
pin took", which is `eligibility-firing-condition-is-not-a-code-fact` in a different costume.

The repair is therefore the one line `origin/main` already carries, applied at the call site in the
old tree: `ESMFold2Model.from_pretrained(repo, load_esmc=False, revision=<pin>)`, with the value
read out of the new tree's table rather than typed so the arms cannot drift. Verified without a
device first — the config parses at the pin, `type=release`, `esmc_id=biohub/ESMC-6B` — and then on
the card, where the load line names the pinned snapshot directly:
`.../snapshots/8fc3ff471022fdce52c77030685eb775de0c00a3/ccd.pkl`. The diff against
`git show e65b66be:tt_bio/esmfold2_runtime.py` is committed as
`perf/allm_audit/old_esmfold2_hf_revision.diff`, because an old arm that was edited has to show the
edit. The `pinned_cell.py` shim stays in the runner for the loaders that DO call `hf_hub_download`
directly, and keeps recording what it fired on.

## RATIOS

RATIOS: **the deliverable, one transfer ratio per model, all five measured.** Each is read between
two arms of mine that share the timed region, the card, the pinned clock and the instrument; the
published cell is used only to name the commit the old arm is extracted at. Each effect is quoted
against the LARGER of its pair's two A/A floors, because an op-level win is a screen and four such
levers reached the fold here at 25x-to-infinite error with two flipping sign. For reference, the two
`pvx-didittransfer` measured on this same instrument: Boltz-2 **1.5006x**, Protenix-v2 **1.0525x**.

| model | old | new | ratio | effect vs larger A/A floor | digest old -> new |
|---|---:|---:|---:|---|---|
| BoltzGen | **46.352 s/design** | **35.943 s/design** | **1.2896x**, defensible floor **1.2704x** | 10.409 s vs 1.848 s = **5.6x** | design model, no digest; atoms differ, old 3818-3906 new 3795-3871 |
| OpenFold3 | **38.425 s** | **33.970 s** | **1.1311x** | 4.455 s vs 0.067 s = **66.5x** | `6ee6ac7a3e730688` -> `9171421df49ef336` |
| RFdiffusion3 | **93.077 s/design** | **89.403 s/design** | **1.0411x**, a lower bound | 3.674 s vs 2.985 s = **1.2x** | design model, no digest; atoms identical design for design |
| ESMFold2 | **28.580 s** | **27.520 s** | **1.0385x** | 1.060 s vs 0.116 s = **9.1x** | `608ce8c40a2c4e33` -> `608ce8c40a2c4e33`, unchanged |
| OpenDDE | **81.508 s** | **78.990 s** | **1.0319x** | 2.518 s vs 0.088 s = **28.6x** | `6623f39115836675` -> `ad0e34ae7f12a61a` |

**BoltzGen: 1.2896x, and it is the only one of the five in Boltz-2's class.** 46.352 s/design at
`59474b45` to 35.943 s/design at `47810889f`, medians of four warm designs after dropping the cold
one and the one that pays the checkpoint switch, which the harness detects from the "Switched
checkpoint." line rather than by picking the slowest. Both arms did the identical amount of the
model's own work: `steps_ok: true` with **501 step stamps counted per design in all twelve designs
of both arms**, 500 resolved from the shipped `design.yaml`, `diffusion_batch` 1 either side, same
fixture and same two checkpoints. Node 1 read AICLK 1350.0 mean / 1350 min / 1350 max over 1265 and
1022 samples, zero re-asserts on either arm.

**The partner ran the wrong way round on this pair, so quote the bound, not the point estimate.**
Card 1's board partner is node 0, and it was at 993.4 MHz mean (800-1350, 32.7 W) through the OLD
arm and flat 800 MHz / 28.2 W through the NEW one. A busy partner slows the measured card, so the
confound sits on the numerator and inflates old/new. The old arm's own record bounds it: the partner
swung its full 800-1350 range *within* that arm and the arm's warm spread is only 1.51 %, so the
partner term cannot exceed 1.51 %. The new arm has the mirror problem and it points the same way —
it picked up a co-tenant on node 3 mid-arm (pid 1141289, node 3 going 800 -> 1350) and its four warm
designs rise monotonically 35.465 -> 37.313, a 5.14 % spread, so if anything it is penalised. Worst
case for the claim is the old arm inflated the full 1.51 % and the new arm not penalised at all:
**>= 1.2704x**. That is the number to defend and it still puts BoltzGen with Boltz-2 rather than
with Protenix-v2.

**Its atom counts differ between the arms and that is not an arithmetic change.** BoltzGen's arms
wrote 3818-3906 atoms and 3795-3871, but they also vary design to design *within* each arm, because
this harness runs the shipped CLI without a fixed seed and each design is a different sampled
sequence. So the two arms are not comparable structure by structure and no claim is made that they
are. RFdiffusion3 is the opposite case and the contrast is the point: it runs at seed 42 and its two
arms reproduced each other exactly, which is why an output claim is made there and not here.

**OpenFold3: 1.1311x, and it is the only fold model here whose arithmetic the window changed.**
38.425 s at `973ae49f` to 33.970 s at `47810889f`. Both sessions clean on every fold, A/A floors
0.067 s (0.17 %) and 0.028 s (0.08 %), AICLK 1350.0 mean and 1350 minimum throughout. The effect is
4.455 s, **66.5x the larger floor**, the cleanest separation of the ten arms. The old arm lands on
the published cell's own output: plDDT **0.547851** against the cell's 0.547851, with the seconds
0.45 % apart (38.425 against 38.254). Unlike ESMFold2 the pair is not bit-identical — digest
`6ee6ac7a3e730688` -> `9171421df49ef336`, plDDT 0.547851 -> 0.549222. Something in the window reached
this model's arithmetic. One term to hand over with it, from SCOPE: OpenFold3 is the one model here
where host is large inside the cell, **1.839 s of the published 38.254 s, 4.8 %**. At 1.1311x that is
big enough to matter to an attribution, and it has not been re-measured here — this row times the
region, it does not split it.

**RFdiffusion3: 1.0411x, a lower bound, and the ratio is not the strong result.** 93.077 s/design at
`6f85ecfe` to 89.403 s/design at `47810889f`, medians of three warm chunks after the cold one, batch
1, 200 timesteps, seed 42 either side. Node 1 at AICLK 1350.0 mean / 1350 min / 1350 max over 1552
samples, zero re-asserts.

**Its floor nearly swallows it, so do not read three decimals off it.** The effect is 3.674 s
(3.95 %) against a new-arm warm spread of 2.985 s / 3.34 % (88.031 / 89.403 / 91.016). That is
**1.2x the larger floor**, against 9.1x, 28.6x and 66.5x for the three folds. The defensible claim
is "in the 1.03-1.13x pack, not in Boltz-2's class", not a point estimate. The partner confound
gives the direction for free and it points the same way: node 0 was flat 800 MHz through the OLD arm
and 1108.9 MHz mean (800-1350, 34.9 W) through the NEW one, penalising the denominator, so
**1.0411x is a lower bound**.

**The stronger result for this model is that both arms wrote the same structures.** `atoms: [5126,
5117, 5129, 5112]` in both arms, identical design for design — the same four designed sequences,
atom for atom, from the same seed. **The window did not change one bit of RFdiffusion3's
arithmetic.** That is the same statement ESMFold2's unchanged digest makes, it is independent of a
noisy 3.95 %, and it is what actually answers this campaign's question for this model.

**ESMFold2: 1.0385x, and it did not receive the window.** Over the 33 days in which Boltz-2's fold
fell 1.5006x, ESMFold2's fell 3.71 %, 28.580 s at `e65b66be` to 27.520 s at `47810889f`. Both arms
n=3 warm after a discarded cold fold, AICLK 1350.0 mean AND 1350 minimum on every timed fold of both
arms, zero re-asserts. A/A floors 0.116 s (0.41 %) and 0.024 s (0.09 %); the effect is 1.060 s,
**9.1x the larger floor**, so it is a measurement rather than a spread, and it is nowhere near 1.5x.

**The window did not change one bit of ESMFold2's arithmetic either.** Both arms return CIF digest
`608ce8c40a2c4e33` and plDDT 0.9286 across all six timed folds. Whatever landed between 2026-08-16
and today either did not touch this model's path or touched only its scheduling. It also confirms the
arm is the computation the published cell named: that cell records plDDT 0.9285 and I get 0.9286 out
of the pinned checkpoint.

**One partner asymmetry on this pair, bounded by the arm's own record at 0.35 %.** `esm_old_s1` is
the only one of the six fold arms carrying a `foreign_tt` entry — device node 0, card 1's board
partner, which ran 800 -> 1350 MHz at 28.8 -> 33.8 W across the arm, while `esm_new_s1` sat at a flat
800 MHz / 28.8 W on all three folds. A busy partner sits beside the numerator, so the asymmetry
inflates old/new. The three old-arm warm folds ran at three different partner states and the idle one
was fastest: 28.580 s at 1082 MHz mean / 33.8 W, 28.595 s at 920 MHz / 31.2 W, 28.479 s at a flat
800 MHz / 28.8 W. Busiest against idle is **0.35 %**, inside the arm's own 0.41 % spread, so the
partner term cannot account for a 3.85 % gap. **1.0385x stands, with at most ~0.35 % of it partner
asymmetry.** The other five arms are clean on this term.

**OpenDDE: 1.0319x, the smallest of the five, and its output moved.** 81.508 s at `b4feba14` to
78.990 s at `47810889f`, both sessions clean on every fold, A/A floors 0.025 s (0.03 %) and 0.088 s
(0.11 %), AICLK 1350.0 mean and 1350 minimum throughout. The effect is 2.518 s, **28.6x the larger
floor**, so it is real and it is small: 3.09 % over the same 33 days in which Boltz-2 fell 1.5006x.
OpenDDE delegates to the whole Protenix-v2 graph and it lands within 0.6 % of Protenix-v2's own
1.0525x, which is the one cross-check in this table against a number measured by another row.

**One thing in that pair is worth someone else's attention: the fold output changed.** Digest
`6623f39115836675` -> `ad0e34ae7f12a61a` and plDDT **0.75411 -> 0.717514**, a drop of 0.037 on a
metric whose scale is 0 to 1. Both arms load the same checkpoint — the old arm's shim pinned
`aurekaresearch/OpenDDE` at `02c1835848` and counted it (`hf_pins_fired:
{"aurekaresearch/OpenDDE": 1}`), and the new tree passes that same revision itself, so
`hf_pins_fired` is empty there by construction. The move is therefore in the tree, not in the
weights. This row measures seconds and does not judge accuracy, so it is recorded and handed on
rather than assessed: the plDDT delta is 50x ESMFold2's zero and 27x OpenFold3's 0.00137 on the same
window.

**What the five say together, stated as a shape rather than as a verdict.** The shared-pairformer
models — OpenFold3, OpenDDE and (from `pvx-didittransfer`) Protenix-v2 — came out 1.1311x, 1.0319x
and 1.0525x. The two models sitting outside that core came out 1.0385x and 1.2896x. So the largest
transfer in this table belongs to a model the shared-core hypothesis does not cover, and the three it
does cover are the bottom half. That is a fact about the ratios, not an explanation of them, and the
explanation needs the work census this row does not own.

## DESIGN-SCOPE

The two design arms reuse the harnesses their published cells were taken with, recovered from
`a4823118e` (they were pruned from the tree by `57669d90d` and live only in history):

- **BoltzGen** `bg_page.py` — the shipped `tt-bio design --model boltzgen --steps design` CLI, 500
  sampling steps asserted not passed, protocol protein-anything, batch 1. One design is the wall
  between consecutive `batch k/N` stamps: trunk, all 500 denoising steps, post-processing, writing.
  Design 1 is dropped as cold and the design paying the checkpoint switch is dropped, detected from
  the "Switched checkpoint." line rather than by picking the slowest. Fixture
  `perf/dsfix/fixtures/bg_R3.yaml`, 514 tokens.
- **RFdiffusion3** `rfd3_page.py` — the released `tt_bio.rfd3.design.run_design`; timed region
  `RFD3Sampler.sample`. Fixture `perf/dsfix/fixtures/rfd3_R4.json`, 6051 atoms, 685 residues,
  200 timesteps, seed 42. Both fixtures exist in both old trees and on main.

Neither harness pins a clock, and both drive the work in a subprocess or inside the released entry
point, so `perf/allm_audit/pinned_run.py` supplies the missing part and nothing else: it forces
FORCE_AICLK on the node the child uses, keeps it there with the same watchdog, and samples every
chip at 4 Hz while the child runs. `Clock` and `Sampler` are **imported from `cell.py`**, so the
design arms and the fold arms are pinned and sampled by the same code.

**Which RFdiffusion3 arm, decided from the published cell rather than from the harness default.**
`rfd3_page.py` offers two arms and they are different units: `ceiling` asks for eight designs at
batch 8, which the runtime clamps to chunks of two at this target size, and `b1` runs batch 1. The
p150a cell the commit was published from is **batch 1, 200 timesteps, median of three warm
designs**, so `b1` is the arm both of my RFdiffusion3 arms run. The ceiling arm is not run here at
all: mixing the two would be exactly the conversion
`design-row-throughput-denies-gpu-its-batch` warns about. BoltzGen runs the harness as published,
six designs leaving four warm after the cold one and the one that pays the checkpoint switch.

**Both design harnesses took their card from a hardcoded `TT_VISIBLE_DEVICES=0` and wrote to one
fixed results path.** This row is granted card 1 and needs two arms of each model in separate
files, so both now read the card from the environment and the output path from `ALLM_OUT`, and
the harness file is byte-identical in both trees of a pair (md5 `d82ef373574a6a92753df0d35c87d8df`
for `bg_page.py`, `d24413b00bcc825508db2a96cb52b071` for `rfd3_page.py`). The card is not
bookkeeping: the lease refuses a wrong-card open, and it refuses an unpinned one because a process
that can see four chips brings up all four.

**Both old trees answer the harness's calls.** `design --help` on `59474b45` carries every flag
`bg_page.py` passes, and `run_design` at `6f85ecfe` takes the same fifteen parameters as on main,
`batch_size` and `num_timesteps` among them. Checked before queueing rather than at the head of a
benchlock queue.

**RFdiffusion3's 91.443 s is a retired figure and this row does not treat it as live.** The page
has since re-measured that cell at **92.472 s** on card 1 and says of the older number that it "was
measured in a different window on card 2 ... it is not comparable with either arm here and is not
quoted as a baseline". `6f85ecfe` is still the right commit to extract — it is the commit that cell
was published from — but the published seconds beside it are not a target to reproduce.

## HARNESS

**`output_ok: false` on both RFdiffusion3 arms was my harness, not the model, and the arms stand.**
`rfd3_page.py` failed all four designs of each arm on `na != EXP_ATOMS` with `EXP_ATOMS = 6051`,
commented "featurised L at R4". That is exactly what 6051 is, and it is not what a CIF contains.
Verified against the live source rather than taken on trust, three claims, all three holding:

1. `tt_bio/rfd3/design.py:658` sets `n_atoms=int(X.shape[1])`, the padded featurised atom axis, and
   line 664 prints the same expression. That is where the 6051 in my log comes from.
2. `_write_cif` does not write from it. It builds a `keep` list, skips every slot flagged
   `_is_virtual` (synthetic atom14 pad slots) and every `CB` whose residue the sequence head called
   GLY, then sizes the output `struc.AtomArray(len(keep))` (lines 233-249).
3. The GLY skip reads `gly_tok`, built from `pred_restype` — **the designed sequence** — which is
   why four siblings of one arm read 5126 / 5117 / 5129 / 5112 rather than one number.

So the constant compared a real atom count against a padded width and could never pass for any
design, and the sibling variation it also tripped on is the model designing different sequences.
`validate()`'s own docstring already said so: *"residue topology and finiteness, never atom equality
between siblings"*. The docstring was right and the constant contradicted it. The two checks that do
encode the docstring both PASSED on all eight CIFs — `EXP_RES = 685` on every one, and zero
non-finite coordinates — which is visible in the recorded `output_fail`, whose every entry is an
atom-equality line and none of which mentions residues or finiteness. `bg_page.py` never had the bug:
it records `atoms` varying 3795-3906 with `output_ok: true`.

Fixed in `perf/allm_audit/rfd3_page.py`: `EXP_ATOMS` and its comparison are deleted, residues and
finiteness stay, and the docstring now says why the atom count is recorded and never asserted. The
two arms' designs were written to `/tmp/rfd3_page_b1`, which the harness itself `rm -rf`s at the head
of each run, so the corrected validator could not be re-run against those exact files; the evidence
the arms are valid is the recorded `output_fail` above, not a re-run, and it is stated that way
deliberately.

**Handed on rather than fixed, because it is not this row's file:** `DesignResult.n_atoms` in
`tt_bio/rfd3/design.py` is a public field reporting the featurised width for a structure whose own
`out_path` contains 15-18 % fewer atoms. The verbose line at 664 prints the same misleading number.
Anyone reading that field to size an output will be wrong by that margin. Recorded here and left for
whoever owns that file.

## VERDICT

VERDICT: GO — **the deliverable exists in full: five transfer ratios, measured, each with an A/A
floor and a DURING-sampled clock.**

| model | ratio | how far above its own floor |
|---|---:|---|
| BoltzGen | **1.2896x** (>= 1.2704x defensible) | 5.6x |
| OpenFold3 | **1.1311x** | 66.5x |
| RFdiffusion3 | **1.0411x** (lower bound) | 1.2x |
| ESMFold2 | **1.0385x** | 9.1x |
| OpenDDE | **1.0319x** | 28.6x |

Ten arms, one card (qb2 card 1, Blackhole p300c), one instrument (`pvx-baseline`'s `cell.py`, md5
`21e0770f080a4d965203b59a193107be`, byte-identical to the file `pvx-didittransfer` used), one timed
region for every fold arm, the two design models keeping their own s/design unit. AICLK pinned and
sampled at 4 Hz DURING every arm: 1350.0 mean and 1350 minimum on all ten, zero re-asserts anywhere.
1693 `.py` files across six trees re-hashed against their commits with one intended difference, the
ESMFold2 hub-revision line documented under UNBUILDABLE. No old commit abandoned and no measurement
substituted.

**What this row does NOT conclude, and deliberately.** It does not label any model a target. A ratio
near 1.0x is consistent with a blocked lever and equally consistent with a model already converting
its work better than the model the 1.5x came from — Protenix-v2 came out 1.0525x with no execution
gap at all, running the shared Pairformer 1.837x better per MAC than Boltz-2 while doing 5.948x its
trunk arithmetic. The ratio alone cannot separate those, and the work census that can is
`allm-model`'s. What this row hands `allm-orchestrator` is five numbers and the confound bound on
each, which is what it was asked for.

**Three things in here are worth someone acting on, none of them ratios:**

1. **An old commit can stop being foldable without a byte of it changing.** ESMFold2 at `e65b66be`
   died in model load on a config schema upstream re-published in place on 2026-09-14. The tree is
   byte-identical to main in the file that failed. Any future archaeology against a pre-2026-09-14
   commit needs `weights.HF_REVISIONS` applied at the call site, and needs to COUNT what it pinned:
   my first two shims installed cleanly and pinned nothing, and without the firing count "the pin
   table is installed" would have read exactly like "the pin took".
2. **OpenDDE's fold output moved 0.037 plDDT across the window** with byte-identical weights on both
   arms. That is 50x ESMFold2's zero on the same window. Seconds are this row's unit; that one needs
   an accuracy owner.
3. **The `EXP_ATOMS` harness defect above**, plus the mislabelled `DesignResult.n_atoms` it exposed.
