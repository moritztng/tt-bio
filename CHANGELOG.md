# Changelog

All notable changes to TT-Bio are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut from a commit that has passed the on-hardware test suite (see `RELEASING.md`).

## [Unreleased]

### Changed

- The fused attention's ragged-tail mask (`TT_BIO_SDPA_RAGGED_PAD`) now defaults **on**. An unmasked
  ragged tail makes the softmax denominator sum padded columns, which is wrong arithmetic rather
  than a tuning choice, and a guard you have to know to ask for only protects people who already
  knew. With every model bucketing (below) it fires on nothing on any shipped path, so it changes no
  published number; it protects whatever still reaches the op ragged. `TT_BIO_SDPA_RAGGED_PAD=0`
  restores the old behaviour for bisecting.

### Fixed

- Models were padding the atom axis and not the token axis. At any token count that is not a
  multiple of the bucket, the ragged tail reached the triangle attention as real key columns rather
  than masked ones, and both the stock and the fused attention read them: on a probe, relative error
  against the aligned answer is 0.914 ragged against 0.038 padded. A 98-residue fold presented 1208
  ragged calls out of 1208 on Protenix-v2 and 1216 on OpenDDE, which carries a third axis in its
  structural-token refiner at roughly twice the residue count and so is essentially never aligned.

  **Every shipped model now buckets its token axis to a multiple of 32**, masked, sliced back on
  exit, through one shared helper (`token_axis.bucketed_pairformer`) with the per-model facts held
  as data in the `TOKEN_AXIS` table rather than as copies of the code. RF3 was the last holdout and
  its two Pairformer sites now route through the same helper, taking 104 ragged fused-attention
  calls per fold to 0 with a bit-identical answer. `TT_BIO_TOKEN_BUCKET=0` restores the old ragged
  path fleet-wide for an A/B and `TT_BIO_TOKEN_BUCKET_MULTIPLE` re-runs every model at another
  width in one variable; `TT_BIO_PROTENIX_TOKEN_BUCKET` and `TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE`
  still work for that family and are ANDed with the global.

  Where the pad is 0 the fold is byte-identical, so a token count already on the bucket costs
  nothing and changes no number: at 512 residues both arms return CIF `5e404779d791fa8f` on
  Protenix-v2 and agree to -0.101 s over eight interleaved pairs on OpenDDE. Where the pad is not 0
  you pay for the columns you added: 298 residues round to 320 and **cost** 4.8 % on Protenix-v2 and
  about 6 % on OpenDDE, on warm medians. (A previous draft of this entry reported those two as a
  speed-up. That was a sign error: the committed paired runs read 26.578 s bucketed against 25.357 s
  ragged on Protenix-v2, and 37.199 s against 35.040 s on OpenDDE, so the bucket is the slower arm
  at that length. It buys correctness, not throughput.)

  The multiple is 32 rather than 64 on measurement, taken on Boltz-2 with the arm order alternated:
  64 is 20.1 % slower at 76 aa, where it pads to 128 against 32's 96 and pays 2.37x the triangle
  work, and 4.18 % faster at the 20 aa smoke fixture, where both are one tile of real work and the
  fold is dispatch-bound. Nothing runs 20 aa in production, so the 76 aa reading decides it. The
  narrower pad is not automatically the faster one, because the two widths do not select the same
  kernel. `docs/size-generality.md` has the size-by-size reading.

## [0.7.0] - 2026-08-24

### Added

- `tt-bio predict --model openbind` folds with OpenBind-0, the same OpenFold3 stack on upstream's
  v0.5.0 checkpoint, tuned for protein-ligand co-folding. It takes ligands by SMILES or CCD code
  alongside protein, RNA and DNA chains, with an MSA on by default and optional per-chain
  templates. Accuracy is measured against the upstream v0.5.0 CPU reference, five seeds a side:
  ubiquitin (L76, MSA) all-atom RMSD 0.969 A inside a 1.033 A reference noise floor, and
  FKBP12 + SB3 (1FKG, L107 protein + 33 ligand atoms by CCD) 0.602 A inside a 0.551 A floor, where
  the residual is the ligand pose and not the fold. Binding affinity is not predicted, and covalent
  bonds and cyclic chains raise rather than fold something else. The weights are a separate
  checkpoint from `--model openfold3` and are not downloaded: set `TT_BIO_OPENBIND` or put
  `of3-ob-2025-06-30-174k.pt` in `~/.boltz`. See `docs/openfold3-port.md` and `docs/weights.md`.

  `--model openfold3` is unchanged. It keeps the preview2 checkpoint and the featurizer preview2
  was trained on: the four MSA fixes v0.5.0 shipped are keyed on the checkpoint, not applied to
  both, and a preview2 fold is byte-identical before and after.

- **PXDesign** binder design: `tt-bio design --model pxdesign target.yaml`. Give it a target
  structure, the chains to condition on and a binder length, and it writes one CIF per design,
  each placed in the target structure's own frame so it opens alongside your input. The binder
  is written as GLY because PXDesign generates a backbone with no sequence. The generator
  checkpoint (556 MB, Apache-2.0) downloads on first use. Selecting designs, which upstream
  does with a Protenix and an AF2-IG filter, is not on the CLI yet.

- `pxdesign-featurizer` joins `full_parity_gate.py`: 25 bit-exact arms against a committed
  capture of the upstream featurizer, card-free. The two `af2ig-trunk` legs now read
  `AF2IG_PARAMS` instead of a hard-coded home directory, so a release host that keeps the
  AlphaFold parameters elsewhere reports where it looked instead of a silent GAP.

- `rf3-1024aa` joins `full_parity_gate.py`, the gate of record. RF3 had no leg there at all: its
  accuracy coverage lived only in `release_gate.py`, so the command that must pass before a tag
  folded ten models and not this one. The leg folds the 997 aa 7EIP anchor and gates its CA-RMSD
  to the deposited crystal at 4.0 A, calling `release_gate`'s `run_rf3_1024aa` in-process the way
  the BoltzGen and OpenDDE-abag legs call theirs, so there is one implementation of the leg and
  one reference cache. Measured 1.958 A over 966 residues in 5.4 minutes.

### Fixed

- RF3's accuracy cell looked for its checkpoint at `~/.cache/tt-bio/rf3/`, a second hard-coded
  path that `tt_bio.weights` does not use. On a host where `tt-bio weights fetch rf3` had already
  put the checkpoint, both release-gate legs that shell to the cell failed with
  `FileNotFoundError` after paying for featurization. It now asks `tt_bio.weights` where the
  weights are (`$RF3_CKPT`, `$TT_BIO_CACHE`, `$BOLTZ_CACHE`, `~/.boltz`), and
  `full_parity_gate.py --check` reports a missing RF3 checkpoint card-free instead of erroring
  mid-run.

- OpenFold3 and OpenBind: a CCD ligand kept its chirality. The reference-molecule builder never
  told RDKit to read stereochemistry off the 3D coordinates before discarding them, so the
  molecule handed to the conformer generator had no chiral tags and ETKDG drew a random handedness
  per stereocentre, which then reached the model as an input feature. SB3, SAH and ATP were all
  fully unassigned. Run-to-run ligand-pose spread on the FKBP12 leg fell from 0.630 A to 0.183 A.
  Polymer folds are untouched: the guard fires only on an all-ligand atom array.

- `full_parity_gate.py`: a port leg whose scorer runs on a card reported ERROR instead of a
  verdict. tt-metal writes its log lines to stdout, the scorers print their JSON report to the
  same stream, and the gate ran `json.loads` over the whole thing. `af2ig-trunk-device` hit this
  on every run on an 11x10 Tensix grid, where the triangle-multiply L1 retry always fires at 208
  tokens: the parse failed and the error the gate printed was tt-bio's own notice saying the clash
  was handled and the result unchanged. The leg was unaffected on the 13x10 grid it was registered
  on, so the blind spot was one board class wide and invisible from the other. The gate now takes
  the report out of stdout and lets device log lines around it be log lines. A report truncated by
  a scorer that died mid-print is still an ERROR.

- `full_parity_gate.py --workers qb2:2` run on qb2 itself now dispatches locally. It compared
  the host token against the machine's own hostname (`tt-quietbox2`), classified the box it was
  running on as remote, and ran every device fold through `ssh qb2` — an alias that exists only
  in a workstation's `~/.ssh/config`. All 21 device legs of a full gate died in under a second
  each on `Could not resolve hostname qb2`, leaving a GATE FAIL that was pure plumbing. The
  fleet short names `qb1` and `qb2` now resolve to their own boxes; genuinely cross-host
  `--workers` entries still dispatch over ssh.

### Gates

Gated on a Tenstorrent Blackhole p300c (tt-quietbox2, Python 3.10.12, TT-NN 0.68.0), in a venv
built from a wheel of this tree, with every gate run against the checkout it is tagging.

- Accuracy floors (`release_gate.py`): all eight structure models clear their RMSD and TM floor.
  OpenBind-0 folds 1.693 A at TM 0.894 against a 3.5 A / 0.70 floor. PXDesign's fit RMSD is
  4.909 A against a 15.0 A floor, with its coordinate digest matching. Every model carried over
  from 0.6.8 reproduces its 0.6.8 number to the digit: Boltz-2 1.700, ESMFold2 1.772,
  ESMFold2-Fast 1.804, Protenix-v2 1.374, OpenDDE 1.418, OpenFold3 1.662, RoseTTAFold3 1.239 A.
  BoltzGen 0.830 A scRMSD at a 100% pass rate, OpenDDE-AbAg DockQ 0.873, Nesso-1 worst scalar
  3.771xR, ESM-C 300M/600M per-residue PCC 0.99961 / 0.99964.
- Parity gate (`full_parity_gate.py`, 40 legs): 32 PASS, 1 PASS-caveated, 4 GAP, 2
  BLOCKED-REGEN, 1 FAIL, 0 DRIFT elsewhere. All four GAPs and both BLOCKED-REGEN legs are the
  same ones 0.6.8 shipped, with the same verdicts; a GAP that reproduces its committed record is
  a reproduced verdict, not a failure. The FAIL is `af2ig-trunk-device`, described under Known
  gaps. New this release and green for the first time: OpenBind's two structure legs
  (ubiquitin all-atom 0.969 A, FKBP12+SB3 0.603 A) and the 25 bit-exact `pxdesign-featurizer`
  arms.
- Packaging (`packaging_smoke.py`): 61/61 data files and 43/43 declared runtime dependencies ship
  in both the wheel and the sdist, and land on disk after a clean install.
- Host test suite: 1120 passed, 52 skipped, 1 xfailed.
- UX regression (`ux_regression.py`): every surface cleared progress, argument parsing and the
  results manifest.
- Performance regression (`perf_regression.py`, 15 legs): 14 PASS, 1 FAIL. Run on qb2 card 2
  after the tag, against `71d83a94` (this tagged tree plus a perf-page merge whose whole diff is
  a new contention exit code, no timed path touched). Every model that has a p300c baseline.
  Boltz-2 reads 1.742 structures/s against a 1.498 baseline, OpenFold3 2.465 against 2.142,
  Protenix-v2 3.214 against 3.195, OpenDDE 2.640 against 2.683, BoltzGen 0.01871 designs/s
  against 0.01706, RFdiffusion3 0.2313 against 0.2653, PXDesign 0.06846 against 0.06694. The one
  FAIL is `boltz2-affinity` at -35.3%, and the pre-0.7.0 merge base fails the same baseline by
  the same margin, so that line is a stale baseline rather than this release's code. See Known
  gaps.
- The `biotite<1.7` pin was checked on the interpreter its break appears on. A clean Python 3.12
  install of this wheel resolves biotite 1.6.0, where both symbols the vendored AtomWorks tree
  reaches are present; biotite 1.7.1 on the same interpreter has dropped `BondList._bonds` and no
  longer exports `connect_via_residue_names` from `biotite.structure.bonds`. So the pin is what
  keeps a clean 3.12 install able to featurize RoseTTAFold3.

### Known gaps

Named rather than dropped, because a release that does not say what it did not check is not
gated.

- `af2ig-trunk-device` FAILs against its committed floor, and the cause is the committed floor
  rather than the port. The leg read 13 of 94 taps missing at minimum PCC 0.9960112623 and
  envelope 13.794076; the record it is scored against holds 9 taps. Both numbers are already
  root-caused: the AF2-IG port established that an 11x10 Tensix grid gives 13 taps at PCC
  0.9960112623 where a 13x10 grid gives 8, and pinned it by forcing a 13x10 board down to 11x10,
  which returned the 11x10 figures to all ten printed digits. This run reproduces that same
  11x10 value digit for digit. The committed record is a 13x10 measurement taken when the
  template stack still ran on the host, and the leg now runs it on the card, which the port
  measured as amplifying the gap from 2.6e-10 to 1.5e-3. So the record is stale in two respects,
  board grid and template placement, and needs re-recording per grid and per arm — the port's
  call, not a release action. AF2-IG has no CLI path in 0.7.0: it is the filter half of PXDesign's
  design selection, which has not shipped, so no user path is affected.
- Size-generality ladder: not run. Its baseline exists only for the p150a, and the p150a in the
  fleet was unavailable for the whole release window. A ladder baseline recorded on the p300c
  from this release's own runs could not detect drift in the code that recorded it.
- The `boltz2-affinity` perf baseline is not satisfiable on this host, so its FAIL scores the
  baseline rather than the code. It was seeded 2026-07-19 on 0.3.1 as a single untimed draw, was
  never reseeded when the affinity trunk moved from fp32-on-host to fp32-on-device, and the two
  p300c cards in tt-quietbox2 disagree by roughly 2x on it. It is also the only p300c leg with no
  machine-specific entry, so it falls back to that July card-level figure. Two same-card A/B
  rounds against the pre-release merge base `6fc864c9` put the two trees within 4.0% and 1.2% of
  each other, in opposite directions, while each tree moved 31 to 38% between rounds on its own
  unchanged code as the box filled up. Both arms fail the 69.8 s baseline, by 33 to 51%. So the
  leg needs a per-card reseed, and a protocol that can outvote a 38% swing before it can carry a
  15% threshold on this host at all.
- Three perf baselines are too stale to fail, so their PASS carries no information. `esmc-300m`,
  `esmc-600m` and `esmc-6b` read +266%, +322% and +100% against p300c baselines seeded between
  0.2.5 and 0.3.1, which predate fused RoPE, ESM-C trace capture and the fused TriMul and
  TriAtt kernels. esmc-600m could lose three quarters of its throughput and still report PASS.
  Boltz-2, ESMFold2 and OpenFold3 are the mild version of the same drift: all three now read at
  or past +15%, so under 15% of today's speed is left as regression headroom. Reseeding needs a
  quiet card of this type.
- `rf3`, `esmc-300m-single` and `nesso1` have no p300c baseline at all, so the perf gate returns
  NO BASELINE for them on this card type and cannot score them.
- OpenBind-0 and PXDesign still have no cell on the benchmark page. Measuring them was in flight
  when this was cut. The page names both as unmeasured rather than projecting a number.
- `tt-bio design --model pxdesign` is not exercised end to end by any gate leg. Its accuracy is
  covered by a fit-RMSD floor, a coordinate digest and 25 bit-exact featurizer arms; the CLI path
  around it, from argument parsing through weight resolution to the results manifest, is not.

## [0.6.8] - 2026-08-24

### Added

- Nesso-1 protein-ligand binding affinity, via `tt-bio affinity`. It has no structure module,
  so it returns an affinity value and a binder probability instead of coordinates, and it is
  much cheaper than folding for that question: 33 s end to end for one 512 aa complex on a
  single Blackhole card against 386 s for the Boltz-2 affinity path on the same input. Point it
  at a directory to screen a ligand series with the model resident across inputs. On DAVIS it
  reaches 0.662 mean within-target Pearson against measured Kd, close to the 0.636 the upstream
  implementation gets on an H200. The trunk runs bf16 by default; `--trunk fp32` is the more
  faithful arm below ~150 tokens and runs out of DRAM around 1000. Proteins and ligands only,
  one ligand per input. See `docs/nesso1.md`.

### Performance

- RoseTTAFold3 folds 1.63x faster at 512 aa (80.28 -> 49.29 s) and 2.05x at 768 aa
  (207.28 -> 100.95 s), with no flag to set. Triangle attention now runs on the fused
  attention kernel instead of the materialised fp32-softmax chain. That route used to be the
  less accurate one, which is why it was off; masking the ragged key tail fixed its accuracy,
  so it is now both the faster and the more accurate route and there is nothing left to trade.
  Predictions move slightly and they move toward the reference: CA-RMSD 0.2030 -> 0.1780 A on
  7ROA (117 aa) and 0.0955 -> 0.0920 A on ubiquitin (76 aa), same seeds, both further inside
  their reference noise floors than before. Sequence lengths that are already a multiple of 32
  are bit-identical to 0.6.7. See `docs/implementation-parity.md`.

### Gates and documentation

- RoseTTAFold3 is now covered by the release gate's fold leg and by the size ladder. It shipped
  as a `predict --model rf3` choice in 0.6.6 with no correctness coverage in either gate, while
  carrying RF3-scoped performance levers. That combination is how a lever gets tuned at one
  sequence length and left dark at every other one.

- RFD3 now has a correctness leg in `scripts/release_gate.py` (`--model rfd3`, and in the default arm set).
  It had three legs already and none of them could see a broken design: the featurizer leg in
  `full_parity_gate.py` scores an input (43/43 feature keys bit-exact), the perf leg scores a
  wall-clock, and the UX leg scores the CLI. All three stay green when the structure leaving the
  far end is garbage, which is where both of RFD3's escaped defects lived. The new arm scores the
  delivered mmCIF: strict parse, backbone geometry, heavy-atom clashes, a real sequence at the
  designed positions, and byte-identical coordinates from a repeated seed in a fresh process.
  Geometry is gated as a clean rate over four designs rather than per design, because an
  occasional broken backbone is real RFdiffusion-family behaviour and the field's answer is to
  generate several and filter.

  Every number is computed over the designed residues only, recovered by re-featurizing the spec
  on the host. For a binder RFD3 merges the designed residues into the target's own chain and
  nothing in the output marks which is which, so a chain-level number averages the generated
  residues against the copied target and passes by dilution. New fixture
  `examples/rfd3_binder.json`, the target `examples/binder.yaml` already uses.

  Not designability: upstream RFdiffusion3 evaluates with ProteinMPNN/LigandMPNN sequences plus
  AF3 rather than its own sequence head, tt-bio ships no MPNN, and `docs/rfd3-design.md` already
  tells users to redesign the built-in sequence before ordering.


## [0.6.7] - 2026-08-23

### Fixed

- OpenFold3: a YAML `msa:` pointing at your own alignment file crashed the fold with
  `IndexError: list index out of range`. The vendored parser keeps only files whose stem is
  one of its known MSA sources and dropped everything else, so there was nothing left to
  index. Your file is now exposed to the parser under the canonical name, bytes untouched.
  Present in every release that shipped the OpenFold3 `msa:` key, 0.6.6 included; the
  committed `examples/prot_custom_msa.yaml` was one of the inputs that crashed.

- OpenFold3: `cyclic: true` on a chain now raises instead of folding it as a linear chain and
  reporting success. The vendored tree carries neither `Chain.cyclic` nor the `cyclic_mask`
  feature it derives, so the flag reached nothing. Use `--model rf3` or `boltz2` for cyclic
  chains.

- OpenFold3: a CCD ligand's reference conformer was built without stereochemistry, so the
  generator picked a handedness per centre at random and that arbitrary choice became a model
  input. Chiral centres are now assigned from the CCD entry first, so the conformer keeps the
  handedness the code names. `--model openfold3` is polymer-only today and refuses ligand
  chains, so no fold in this release reached it; the other models build their ligands on their
  own paths and were never affected.

- OpenFold3: an MSA deeper than its per-source cap is now truncated, the way the reference
  truncates it. The vendored parser dropped the truncated copy and returned the full
  alignment, so a deep alignment reached the featurizer whole and the model saw a different
  set of rows than the reference did. Nothing changes below the caps: all seven OpenFold3
  parity legs sit under them and reproduce their committed numbers.

- Protenix-v2 and OpenDDE fold more accurately. Two bugs in the pair trunk both models share
  are fixed: the mask marking which residue pairs are real reached only one of the two triangle
  multiplications, and `OuterProductMean` added its output bias without the scale the reference
  applies. Every Protenix-v2 and OpenDDE structure leg in the accuracy gate now lands inside the
  reference's own seed-to-seed spread; some fell outside it before. The other models on that
  trunk reproduce their published numbers unchanged. See `docs/implementation-parity.md`.

- RF3 folds are back to full speed. 0.6.6 turned on the accurate softmax for Protenix-v2 and
  OpenDDE, and it reached two extra sites inside RF3's pairformer that were never meant to get
  it: 512 aa went from 82.5 s to 111.8 s. The setting is scoped now and the structure is
  bit-identical to what 0.6.5 produced.

- RoseTTAFold3 folds crashed on a clean `pip install`. `biotite` was declared without an upper
  bound, so a fresh install resolved 1.7.1, which removed two internals the vendored AtomWorks
  featurizer uses; every `--model rf3` fold died at import before reaching a card. The
  requirement is now `biotite<1.7`. If you already have biotite 1.7 in an environment, `pip
  install -U tt-bio` will downgrade it. Affects every release that shipped RoseTTAFold3.

- `full_parity_gate.py --workers` no longer ssh-es a host to itself. The fleet short names
  `qb1` and `qb2` are recognised as their own boxes, and any host that is genuinely remote is
  probed once before the first fold: reachable, not this same machine, and the card node
  present. A bad worker name fails preflight in seconds instead of turning every device leg
  into an instant error.

- The release gates refuse to run on a Python environment that does not satisfy tt-bio's own
  declared dependencies, naming what is missing or out of bounds. Before, a gate host missing
  one package reported the model that needed it as a failure instead.

### Performance

- OpenFold3 folds 704 aa 1.34x faster (43.193 -> 32.230 s) and RoseTTAFold3 1.14x
  (45.332 -> 39.808 s). Both are bit-exact, so no prediction moves. The gain is at the sizes
  where the accurate softmax used to give up on splitting its work and run one unblocked pass;
  512, 576, 640, 768, 896 and 1024 aa already split and are unchanged. See
  `docs/openfold3-port.md`.

## [0.6.6] - 2026-08-22

### Added

- `tt-bio weights` lists every weight artifact with its status, on-disk size and resolved
  path; `--download [MODEL...]` prefetches, `--prune` reclaims superseded Hugging Face
  revisions and staging leftovers after printing the bytes and asking. `TT_BIO_CACHE` moves
  both halves of the cache (`~/.boltz` and the Hugging Face hub cache) in one setting, and
  every artifact takes a `TT_BIO_<ARTIFACT>` override; the older `PROTENIX_CKPT`, `OF3_CKPT`,
  `RF3_CKPT` and `OPENDDE_CKPT` still work. See `docs/weights.md`.

- `tt-bio predict --model rf3` folds with RoseTTAFold3 (AlphaFold3-family: MSA module,
  template embedder, 48-block Pairformer, atom diffusion, confidence head), on device.
  Proteins, RNA, DNA and ligands, plus non-canonical residues, covalent modifications and
  cyclic chains; MSA on by default through the same stage every other model uses. Writes
  an mmCIF/PDB with pLDDT in the B-factor column and an AlphaFold3-style
  `<name>_summary_confidences.json` (pTM, ipTM, chain-pair PAE/PDE, ranking score), and
  `--diffusion_samples N` ranks N samples by that score. Weights download from the
  Institute for Protein Design on first use, or set `RF3_CKPT`.
- `--partial_t N --partial_structure FILE` (rf3): start the diffusion rollout part-way down
  the noise schedule so it refines an existing structure instead of building from scratch.
  N is a schedule index, 0 for a normal fold and higher to stay closer to the input.
- `--early_stop_plddt X` (rf3): after the first trunk recycle, score mean pLDDT with the
  confidence head and abandon the target if it is below X. No structure is written and the
  entry in `results.json` carries `early_stopped: true` with the measured pLDDT, so a
  screening run can tell an abandoned target from a failed one.
- `tests/test_rf3_featurizer.py`: RF3 host-featurizer parity over ten capability classes
  from committed captures, with no device and no `rc-foundry` install.
- Both release gates honour a card grant. `TT_VISIBLE_DEVICES` is the set of cards a run may
  open: ask `--workers` for a card outside it and `full_parity_gate.py` refuses in preflight
  instead of taking a card another job on the box holds, and a leg needing more cards than the
  grant is skipped as `SKIPPED-CARD-GRANT` and listed under `COVERAGE REDUCED` so a narrowed
  gate cannot read as a green full one. Leaving the variable unset means the whole box and is
  the unchanged path a release run takes. Both gates also refuse to start when the 1-min
  loadavg is above 1.5x nproc (`--load-ceiling`, 0 disables).
- `tt-bio affinity DATA` predicts protein-ligand binding affinity without folding anything.
  Nesso-1 has no structure module, so it returns scalars rather than coordinates: an affinity
  value (mean of a two-member ensemble, and each member), a binary binder probability, and six
  distogram entropies. It takes the same YAML Boltz-2 affinity takes (`sequences` plus
  `properties: affinity`), and a directory is a screen that keeps the model resident across
  inputs, so a ligand series against one target pays the weight load and the kernel compile
  once. Output is one `<id>_affinity.json` per input plus an `affinity.csv` for the run.
  On DAVIS it reaches 0.662 mean within-target Pearson against measured Kd (0.175 for a
  molecular-weight-only control), matching the 0.636 the upstream implementation gets on an
  H200. One 512 aa prediction costs 8.3 s of model time on one Blackhole card and 33 s for
  the whole command, against 386 s for the same command through Boltz-2 affinity, which is
  what tt-bio shipped for this question before; against a GPU it is 7.9x off an H200 at that
  size, so choose it for what the answer costs on this hardware rather than expecting it to
  beat a GPU. The trunk runs bf16 by default (`--trunk fp32` switches
  back, and is the more faithful arm under ~150 tokens). Weights and the 413 MB `ccd.pkl`
  download on first use. See `docs/nesso1.md`.
- Nesso-1 joins every release gate leg: a correctness arm (`release_gate.py --model nesso1`,
  scoring its eleven output scalars against the torch reference and normalising by upstream's
  own featurization-draw spread), a perf cell in `affinities/s` on the same FKBP12+SB3 fixture
  the Boltz-2 affinity cell uses, a `full_parity_gate.py` leg, and the size-generality arm.

### Fixed

- A weight download killed mid-flight no longer poisons the cache. Re-downloads were gated on
  the destination merely existing, so a truncated multi-GB file was treated as present and
  reused forever, failing later with `PytorchStreamReader ... failed finding central
  directory`. Downloads now stage next to their destination, verify against the source's byte
  count and archive structure, and only then rename into place; archives are unpacked into a
  staging directory and a source archive that gets discarded after extraction is deleted only
  once the output verifies. Affected Boltz-2, Protenix-v2, the RF3 and RFD3 checkpoints, the
  CCD molecule library and the RFD3 weight split, where a partial extraction next to a deleted
  checkpoint was unrecoverable. Existing caches are adopted, not re-fetched.

- A device open now leases every card it holds, not just the one it computes on. ttnn brings up
  every card `TT_VISIBLE_DEVICES` makes visible, so an unpinned run on a four-card box held all
  four chips while leasing one, and the next job on that host was told three of them were free.
  It then either blocked on a lock or collided at the fd level. All of them are leased now, so
  such a run fails immediately, naming the card and the process holding it, instead of quietly
  sharing a chip. Pin `TT_VISIBLE_DEVICES` to the card you want and nothing changes, which is
  what the worker pool, the multi-card fan-out and every gate leg already do.
- `TT_BIO_LOGICAL_DEVICE_ID` past the end of `TT_VISIBLE_DEVICES` fails now, naming both values,
  instead of silently using the first visible card and leasing one the run never opens.
- Boltz-2 folds inputs that pad to 704 tokens again. 640 aa plus a 20-heavy-atom ligand, and
  641 to 704 aa on its own, both died about 6 s in on an on-device memory clash. Every size that
  folded before folds bit-identically, and the protein-plus-ligand ladder now passes at every
  64-aa rung from 256 to 1024 aa. See `docs/part-l1-budgets.md`.
- The parity gate's delegated legs (`boltzgen`, `opendde-abag`, `capacity`) run in the gate's own
  process and shell out from there, so they inherited an environment with no device restriction:
  boltzgen designed on card 0 whatever `--workers` said, and any fan-out from a delegated leg
  would have taken every card on the box. They are pinned now, and the pin is restored afterwards
  so one leg cannot leak it into the next.
- The `l1-budget` release-gate arm crashed before folding anything (`L1_BUDGET_PARTS` rows
  unpacked as 4-tuples after a DRAM field made them 5), and that arm's own tests were dead for
  the identical reason, so the arm had neither a working leg nor working tests. Also fixed three
  gate tests whose verdict depended on the host's loadavg instead of the logic under test.

- A partial `~/of3_ref_out.pkl` skips the OpenFold3 device tests that need the keys it
  lacks instead of failing them. The tests guarded on the golden's existence while
  depending on its contents, so on a host with a partial capture 11 of them died on
  `KeyError` and read as a regression in whatever branch was checked out.
- Histidine's ND1 carried no formal charge in the Protenix-v2 and OpenDDE featurizer. The
  PDB chemical component dictionary's ideal histidine is the protonated imidazolium, so ND1
  is +1, and the reference implementations read the CCD straight through. tt-bio did not, so
  any protein with a histidine in it, which is nearly every real target, folded from
  slightly the wrong input. Numbers move: on ubiquitin (76 residues, one histidine) the top
  structure shifts 0.07-0.38 A depending on seed and plDDT by about 0.0003, and the parity
  leg's all-atom RMSD against the official ByteDance Protenix reference improves from
  1.790 A to 1.774 A. Larger targets carry proportionally more histidines. Re-run any
  Protenix-v2 or OpenDDE prediction you need to compare against a new one. Boltz-2,
  ESMFold2, OpenFold3, BoltzGen and RFD3 use different featurizers and are unaffected.

  The charge table is now taken from the CCD over all 20 standard residues rather than from
  a golden feature dump: ARG NH2, LYS NZ and HIS ND1 are the only charged atoms, and nothing
  else was missing.

- The live progress view reports real progress for `--model rf3`. It announced the trunk phase
  once with no iteration count and never announced diffusion at all, so the bar sat empty and a
  normal rollout looked like a stall. rf3 now ticks per trunk recycle and per diffusion step like
  every other model. Reporting only: no prediction moves.

### Changed

- Protenix-v2, OpenDDE and OpenDDE-abag fold more accurately, and slightly faster (+8.6%, +3.6%,
  +4.3%). Their Pairformer softmax was losing about 2% of each row's normalisation; it is exact
  now. Four parity legs improved on their committed envelopes and none regressed, so predictions
  move a little: re-run anything you need to compare against a new result. ESMFold2 and OpenFold3
  are unchanged, their sites not having been measured yet. Each site is separately switchable
  with `TT_BIO_ACCURATE_SOFTMAX_AB`, a comma-separated list of `<model>.<site>` tokens where a
  bare token forces the exact chain on and a `-` prefix forces it off; `all` and `-all` cover
  every site that has no token of its own, so `TT_BIO_ACCURATE_SOFTMAX_AB=-all` puts all five
  sites that ship on (`protenix.trunk`, `protenix.confidence`, `opendde.trunk`,
  `opendde.confidence`, `opendde.refiner`) back on the old softmax. RoseTTAFold3 is not on this
  switch; its site is exact unconditionally.
  See [docs/implementation-parity.md](docs/implementation-parity.md).

### Performance

- RoseTTAFold3 folds 1024 aa 1.264x faster (52.468 -> 41.508 s per trunk recycle), and the
  768 -> 1024 aa scaling exponent drops from 3.63 to 2.82. Bit-exact, so no prediction moves.
  Boltz-2, Protenix-v2 and OpenDDE reach the same code only behind `BOLTZ2_FP32_SOFTMAX`, which
  is off by default, so they are unchanged. See `docs/size-generality.md`.

### Gates and documentation

- The performance page publishes two readings per row, whole fold and device only, and says what
  the NVIDIA cells actually time. Four H200 cells (Boltz-2, OpenFold3, Protenix-v2, OpenDDE)
  leave 0.24 to 6.31 s of featurisation and structure writing outside their timer, which made
  those ratios larger than a like-for-like comparison. The two readings agree within 0.2x on five
  of six rows; RoseTTAFold3 reads 3.556x whole fold and 9.388x device only, because half of that
  fold is host featurisation that runs on both sides. No published cell moved. See
  `site/data/perf-512aa.json`.
- `tt-bio predict --model rf3` runs from a plain `pip install tt-bio`. Four packages it imports
  at module load were undeclared, so it exited on `ModuleNotFoundError` before opening a card.
  RoseTTAFold3 now has a UX-gate leg and a perf-gate entry, which is what found it.
- A parity-gate workdir records the code it scored, and the gate refuses to resume one built from
  a different tree. The per-leg resume cache is keyed on the leg id alone, so a second release
  gate on the same machine used to replay the previous release's verdicts as its own.
- `packaging_smoke.py --fold` installs the wheel with `--force-reinstall`, so the guard cannot
  inherit a same-version `tt_bio` from the parent interpreter and silently test nothing.

### Known issues

- `TT_PROTENIX_CONF_DEVICE=1`, which keeps Protenix-v2's confidence head on the card, returns
  PAE and PDE that track the default path to 0.981 and 0.990 correlation, below the 0.99 a
  device path here has to clear; pLDDT is clean at 0.994. The flag ships off and the predicted
  structure never depends on it, so leave it off if you read PAE or PDE. 0.6.5 returns the same
  numbers: a pre-existing gap now measured, not a new one.
- `perf_regression.py`'s full-coverage assertion no longer misses a whole CLI verb. It exists
  so a model cannot ship with zero perf coverage, but it was keyed on the three model tuples
  that existed when it was written, so a new verb bringing its own tuple reopened exactly the
  hole it closed — `tt-bio affinity` would have shipped uncovered and the check would have
  stayed green. It now discovers every `*_MODELS` tuple in `tt_bio.main`.
- `full_parity_gate.py` pins every leg to its card. It pinned only the subprocess harnesses,
  so the legs that run in-process and shell out from there (boltzgen, opendde-abag, capacity,
  nesso1) inherited an environment with no device restriction and opened the whole mesh — the
  failure the function's own docstring describes, while asserting these legs were pinned.
- A partial `perf_regression.py --update-baseline` no longer restamps the machine-level date,
  version and note in `docs/perf_baselines.json`. Seeding one model's cell rewrote the note
  describing the whole block, so one reseed erased the only record of why another model's
  number had moved.

## [0.6.5] - 2026-08-20

### Fixed

- Predictions no longer depend on a target's position in a multi-target job. Two
  byte-identical inputs folded by one worker returned different affinity values
  (0.648724 / 0.722511 / 0.687149 for three copies of one CDK2 + ligand YAML); they now
  return the same value, bit-identical to folding either one on its own. Two causes.
  Ligand conformer generation left ETKDG's seed unset, so RDKit drew from a stream that
  advances on every embedding and the same SMILES got a different reference conformer
  each time it was parsed — that moved the structure prediction too, not just the
  affinity scalar, and it applies to BoltzGen as well. And the affinity checkpoint loads
  lazily inside the first affinity target of a run, which advanced the RNG the diffusion
  samples from, so target 1 sampled differently from every target after it.
  Ligand numbers move once with this fix: a pinned conformer is not the one the old
  stream happened to hand the first target, and the spread across conformers of one
  ligand is about 0.074 log10(IC50). Set `TT_BIO_ETKDG_SEED` to draw a different
  conformer. `scripts/boltz2_affinity_batch_position_repro.py` folds N targets in one
  process and fails if identical ones disagree, and `tt-bio` gates it as
  `scripts/release_gate.py --model batch-position`. CCD-bound ligands are unaffected.
  If you ran a multi-target SMILES-ligand job on 0.6.4 or earlier, re-run it: every
  target after the first one in that job got the wrong number.

- An affinity run's `results.json` reported only the structure leg's time, so the
  number was short by however long the affinity trunk ran. It now reports both.

### Changed

- RFD3 designs in 91.4 s on a p150a, down from 105.1 s, which is 3.33x an H200's
  27.5 s and inside the 4x bar the performance page holds every model to. The row is
  no longer hidden there. (`dff668d9`)

### Known issues

- Boltz-2 affinity on the trypsin parity leg differs about 2% from the CPU reference
  (2.552 against 2.606 log10(IC50)); the predicted pose is unaffected. The reference
  bound tightened this release because the conformer-seeding fix above removed noise
  that had been inflating it, so this is a pre-existing gap now visible, not a new one.
  See `docs/implementation-parity.md`.

## [0.6.4] - 2026-08-19

P300 Blackhole cards fold again. On a 110-core grid (every P300), a mid-size
protein+ligand target died in the triangle multiplication with "statically allocated
circular buffers clash with L1 buffers": the chunk-width budget behind that op was
measured on a 130-core card and admits widths that do not fit on a tighter one. The
trimul now catches the clash, which throws before anything runs, and retries one chunk
width narrower. Narrowing is bit-exact (the width only partitions an independent-channel
sum), and the failing width is remembered per shape, so a process pays one failed
compile and every later call starts narrow. Reported by Taylor Singletary in #11; his
grid sweep is what pinpointed the threshold.

### Added

- `scripts/release_gate.py --model l1-budget`, in the default arm set: a release leg for
  the class of defect #11 belonged to. It runs the trimul chunk-width budget for every
  part class in `L1_BUDGET_PARTS` and folds #11's own target across the grid ladder the
  running card can express, so a budget fitted on one card cannot ship unchecked on a
  card with fewer cores. `docs/part-l1-budgets.md` carries the measured per-part figures.
- OpenFold3 `--single_sequence` folds upstream's no-MSA mode through a one-row
  alignment, and upstream's own OpenFold3 inference suite now runs against this port
  with committed verdicts (`docs/openfold3-upstream-suite.md`).
- The release gate checks `RELEASE_GATE_MSA_DIR` before it opens a device and names the
  a3m files to seed. A dir that covered one target used to fail an unrelated-looking
  accuracy arm an hour into the run.

### Fixed

- Mid-size targets no longer die with an L1 circular-buffer clash on 110-core
  Blackhole grids (P300/P300C). The triangle multiplication's channel chunk narrows to
  the widest width that fits, falling back to DRAM residency at the floor, and outputs
  are bit-identical to grids that never clashed. (#11)
- A caught trimul clash now says so on stderr. tt-metal logs the clash at `critical`
  before raising, which reads like a fatal error even though the retry succeeds. (#11)
- `TT_VISIBLE_DEVICES` accepts PCI bus addresses (`0000:01:00.0`), the form ttnn's
  device open takes, resolving them to device indices; an unknown entry fails with a
  message naming the index form. (#11)
- Opening a whole P300 board pair no longer fails with "Physical chip id 0 not found
  in control plane chip mapping". The 1x1 mesh-graph descriptor a lone P300 chip needs
  is now applied only when exactly one chip is visible. (#11)
- ttnn-only models refuse `--accelerator cpu/gpu` on every path, and the CPU/GPU path
  no longer requires the ttnn wheel. (#10)
- `tt-bio predict` exits nonzero when a run loses targets instead of reporting success
  on an empty result set.

## [0.6.3] - 2026-08-17

Binding affinity runs on the card. Boltz-2's affinity model kept its 64-block trunk in
fp32 on the host CPU, which is why a single ligand took minutes and looked CPU-bound; it
now runs in fp32 on device, and FKBP12+SB3 at the default affinity protocol goes from
294 s to 206 s per ligand on one Blackhole p150a.

Large targets fit a 12 GiB card. Every structure model folds targets up to at least 1095
residues on a single Wormhole chip, OpenDDE included, by switching the pair track to
row-blocked execution above a size threshold smaller targets never reach. Their speed and
numerics are unchanged.

`--trace` no longer returns wrong structures. One process folding several same-size
Protenix-v2 or OpenDDE targets replayed the first target's conditioning for all of them.

### Added

- Targets up to at least 1095 residues fold on a single 12 GiB Wormhole card, on every
  structure model. The pair track row-blocks above a size threshold; below it nothing
  changes. See [`docs/large-targets.md`](docs/large-targets.md).

### Changed

- Boltz-2 binding affinity runs entirely on the card. The affinity model's 64-block
  trunk used to run in fp32 on the host CPU by default, which made a single ligand
  take minutes and look like a CPU bottleneck; it now runs in fp32 on device.
  A whole `tt-bio predict` on FKBP12+SB3 at the default affinity protocol drops from
  294 s to 206 s per ligand, 1.43x, on one Blackhole p150a. Three timed reps per arm,
  0.6.2 installed from its released PyPI wheel and 0.6.3 from this release's wheel, same
  card and same input; the arms do not overlap. All six committed affinity parity legs
  keep their committed verdicts. `BOLTZ2_AFFINITY_TRUNK_FP32_HOST` and
  `BOLTZ2_AFFINITY_TRUNK_FP32_DEVICE` are removed; the fp32 affinity trunk is no
  longer configurable, because a lower precision there shifts the predicted
  log10(IC50).
- RFdiffusion3 ships both fused bias kernels on by default (`881704d2`). The sparse
  attention bias is built in one pass instead of a poke walk (5.83x at the op,
  `703d12a1`) and the whole score+bias chain is one kernel (4.42x at the op,
  `923a9396`); both learned multiplicity batching, worth 6.26x at batch 2
  (`fa7246da`). Every step is bit-exact: the fold A/B legs land byte-identical
  designs at +12.36 %, +5.10 %, +6.83 % and +4.65 % on ms/step (`583961c4`,
  `ee4a8980`, `64a14e68`, `599d81ff`). The published throughput table in
  `docs/rfd3-design.md` was regenerated with them on (`5123065e`).
- Triangle attention runs the q-split at or below 1024 padded tokens and gates it
  off above (`063f89db`).
- Three paths ship off on purpose, each behind one environment variable.
  `BOLTZ2_TOKEN_DIT_SDPA=1` runs Boltz-2's token-DiT attention as a fused SDPA; it is
  faster and it is not bit-exact, because the exponentiated scores go through a bf16
  buffer. `TT_PROTENIX_CONF_DEVICE=1` keeps Protenix-v2's confidence head on the card;
  it correlates 0.98071 against a 0.99 floor, which is why it is not the default.
  `OPENDDE_DIFFUSION_FP32=1` lifts the bf16 pin on OpenDDE's diffusion for an A/B. Every
  other lever in this release is on by default.

### Fixed

- OpenDDE folded with a corrupt pair track, and no host-side check could see it. The
  pair-init bias was uploaded flat and reshaped at the end, which splits a tiled
  tensor's row axis whenever the token count is not a multiple of 32. The tensor reads
  back through `ttnn.to_torch` bit-exact and is wrong only as an operand of the next op,
  so the reference comparisons all passed while the fold's pair track was wrong.
  Introduced by `6c3f5eca` on 2026-08-08, the day after 0.6.2 was cut, and fixed by
  `1ea1e6f3` before this release: no released version ever shipped it.
- `--trace` with Protenix-v2 or OpenDDE silently returned wrong structures for every
  target after the first when one process folded several targets of the same size: the
  captured trace was keyed on shape alone and replayed the first target's conditioning.
  The trace is now re-captured when the conditioning changes. Regression gate:
  `scripts/trace_multitarget_parity.py` (two same-size targets, one process, trace on
  vs off, byte-identical CIFs required). Boltz-2 and BoltzGen were not affected: their
  predict path resets the trace cache between targets.
- Protenix-v2 crashed on targets between 385 and 506 residues: the h=1.5 normed pair
  tensor was held past its last use (`142e0109`). ESMFold2 hit the same class at large
  targets and now frees the pair-conditioning intermediates rather than row-tiling them
  (`08565983`). OpenDDE uploads `z_struct` as one allocation into an intact hole
  (`28a91107`) and frees the expander's row chunks after the loop (`d2ad024b`).
- A clean `pip install` was missing kernel sources: the wheel and sdist now ship every
  file under `tt_bio/kernels/` through recursive globs, so a new kernel directory cannot
  drop out again (`cb3ef828`, `baa6ad0a`).
- On a tt-metal built from source, tt-bio could not find the fabric mesh-graph
  descriptor (`ee73d9a4`) or the `generic_op` kernel sources (`e2fb610a`), so a lone
  Blackhole P300 chip would not open and the fused kernels would not build.
- A worker that died of an uncaught exception reported silence (`a5921d2d`), and a
  process holding a card could outlive whoever spawned it (`3bd84f04`, `26a8c085`).
- Protenix-v2 falls back to ttnn's own matmul planner when a tuned config clashes
  instead of failing the fold (`5a207fee`).

### Performance

- OpenFold3 at 512 residues: 51.19 -> 44.535 s on one Blackhole p150a, from running
  TriangleAttention's fp32-softmax tail height-sharded in L1 rather than
  DRAM-interleaved. Bit-exact — the same CIF digest (da9b4ed68f8c0405) and plDDT as the
  control arm, and it holds at 768 and 1024 aa (`abbf42ba`, `d0589dca`; four warm folds
  on main in `perf/of3x3/ab_512_postmerge_qb2c3.json`, spread 0.126 s).
- The trimul output tail's two projections and its gate are one kernel: -679.47 ms on
  the trimul body wall at 512 aa, byte-identical (`747e1b75`, re-verified on main
  `0febf057`).
- BoltzGen skips the template round trip when the input carries no template, worth 8.8 %
  (`4d07b39e`).
- ESMFold2's pair-FFN row blocking extends to 1024 residues: -3.858 s (1.0242x) at
  1024 aa, bit-exact (`ca9b6703`).
- End to end, from the release wheel against 0.6.2's released wheel on the same card and
  the same input: ESMFold2 at 512 residues single-sequence goes 233.5 -> 197.5 s
  (1.18x), and Boltz-2 affinity on FKBP12+SB3 goes 294 -> 206 s (1.43x). Both are three
  timed reps per arm on one Blackhole p150a with the arms interleaved.
- Boltz-2's diffusion hoists the per-step attention bias slices out of the step loop and
  memoises the AdaLN `s` terms, both on by default and both bit-exact: 24.822 -> 23.504 s
  at 512 residues (`bae4d627`, `6c07446f`, `6ce62967`).

## [0.6.2] - 2026-08-07

OpenFold3 lands: `tt-bio predict --model openfold3` folds proteins, RNA and DNA with the
OpenFold Consortium's AlphaFold3 reproduction, with per-chain MSAs and optional per-chain
templates, on the same scheduler, multi-card fan-out and MSA cache as Protenix-v2. Polymer
chains only — ligands, covalent bonds and `--write_pae` raise or are declined rather than
silently degrading. OpenFold3 is the one model whose weights tt-bio does not download: the
consortium's checkpoint is a public release you fetch yourself — point `OF3_CKPT` at it or
drop it at `~/.boltz/of3-p2-155k.pt`.

### Added

- **OpenFold3** — `tt-bio predict --model openfold3` folds proteins, RNA and DNA with the
  OpenFold Consortium's AlphaFold3 reproduction, with per-chain MSAs and optional per-chain
  templates. It rides the same scheduler, worker, multi-card fan-out and MSA cache as
  Protenix-v2. Parity-gated against the official CPU reference on seven legs (see
  `docs/implementation-parity.md`); the on-device diffusion sampler runs in fp32 by default
  to match the reference's own boundary. Polymer chains only — ligands, covalent bonds and
  `--write_pae` raise or are declined rather than silently degrading. This is the one model
  whose weights tt-bio does not download: set `OF3_CKPT`. (`9c10f08f9`)

### Fixed

- OpenFold3's vendored data pipeline imports `pydantic`, `pdbeccdutils`, `func_timeout`,
  `networkx` and `packaging` at module load, and none were declared, so `--model openfold3`
  failed on a clean `pip install tt-bio`. The vendored Apache-2.0 license text now ships in
  the wheel too.
- `tt-bio predict --model openfold3` showed no trunk or diffusion phase on the live
  progress view: the worker's progress adapter was never passed to the model. Wired
  through the trunk and sampler loops, mirroring Protenix-v2 (`df0ed79f`).

### Performance

- Protenix-v2 and OpenDDE at ~300 residues: the 298-aa GPU scaling gap is closed, 1.18x on
  the landed code (`ec28f3d2`; measured `62934f2f`). At that size the Pairformer trunk is
  74.5% of step time and scales as N² (`46c4fe29`); the SDPA chunk gate, the trimul
  `in0_block_w` and the transition chunk were re-tuned for it (`7349f407`, `ec50003d`).

### Gates and documentation

- The README, `predict --help`, the parity docs and the OpenFold3 port doc now match what the
  shipped model actually does, including template support (`a6403435`, `a35f7a3b`,
  `1327a080`, `2b0671ee`, `92d5b58a`, `50bc0bee`).
- `kabsch_rmsd` was labelled a CA RMSD but is computed over every atom name; the label is
  corrected at the source and on the OpenFold3 rows. The other models' rows follow in a
  deliberate pass — the affinity pocket legs use a different, genuinely CA-based scorer.
- OpenFold3 is enrolled in the accuracy, perf and UX release gates (`86932b20`, `1f236666`,
  `2a20f028`).
- `scripts/fetch_parity_fixtures.sh` verifies the fetched tarball by its hash field instead
  of the sidecar's recorded absolute path, which exists only on the machine that generated
  it (`f9d0afc1`).

### Release gate (Blackhole P150a on `tt-quietbox`)

Host suite: 225 passed / 23 skipped / 1 xfailed with 25 failures triaged one file per
process — 24 are environment or harness artefacts, not code regressions (qb1's boltzgen
transformers env gap, the parent-holds-device false-failure class, a stale July OF3 dev golden
that the P8+ tests skip without, and a 4e-6 PCC wobble on the confidence leg). The 25th was
real: `ec50003d`'s transition big-chunk gate admitted the Protenix-v2 N=512 pair shape
(W=512), which overflowed in-block L1 (`test_fold_512_no_oom`). Fixed as `e6678e21` (gate
tightened to W<=384, keeping the 298-aa fast path); the test passes on the fixed tree.
Packaging guard: 16/16 data files and 36/36 declared dependencies in the wheel and sdist.

**Accuracy gate** — every shipped fold architecture folded end-to-end with production sampling
(200 steps, 5 samples) and checked against a per-model ground-truth floor, not
self-consistency:

| model | RMSD (A) | TM | floor | result |
|---|---|---|---|---|
| boltz2 | 1.373 | 0.945 | <=3.0 / >=0.75 | PASS |
| esmfold2 | 4.462 | 0.563 | <=8.0 / >=0.40 | PASS |
| esmfold2-fast | 1.769 | 0.910 | <=4.5 / >=0.60 | PASS |
| protenix-v2 | 2.459 | 0.799 | <=6.0 / >=0.50 | PASS |
| opendde | 1.352 | 0.955 | <=6.0 / >=0.50 | PASS |
| openfold3 | 2.042 | 0.845 | <=3.5 / >=0.70 | PASS |

**Parity gate** (`scripts/full_parity_gate.py`, 30 legs): **24 PASS, 6 GAP, 0 DRIFT, 0 ERROR**,
every GAP reproducing a committed `GAP-evidenced` record. All seven OpenFold3 legs pass on the
release tree (ubq X=1.34 A within R=1.64/D=0.80; 8hel-msa and 9bk6 included). Four records are
newly evidenced this release: protenix-prot-msa, opendde-prot-prod and opendde-trpcage-nomsa
(`c97076c0`, root-caused to `ba6ede96`'s intended AttentionPairBias unfusing, device samples
inside each reference's own inter-seed spread and every numerator inside the committed noise
floor), and openfold3-7xi5-notmpl (`af8f886d`, fold accuracy verified against RCSB 7XI5
directly: all five device seeds at 0.589-0.609 A aligned CA-RMSD, the CPU reference's own
spread being 0.422-0.895 A).

**Performance gate**: every model within +/-15% of its committed baseline on both cards.
OpenFold3 vs its committed p300c baseline on `tt-quietbox2`: 2.191 structures/s vs 2.142
(+2.3%) — the 298-aa shared kernel-gate work does not regress it. Full p150a leg on
`tt-quietbox`: 15/15 PASS, worst |delta| 9.3% (boltzgen), everything else within +/-4%.
OpenFold3's p150a entry (0.990 structures/s) is a first seeded baseline, disclosed as such
(`cdebd298`) — it is compared against itself this release; the p300c comparison above is
this release's regression evidence for the model.

**UX gate**: PASS on the tag tree — every model's live progress advances through trunk
and diffusion, outputs parse, the CLI surface behaves. It earned its keep: OpenFold3's
first hardware UX leg caught the missing progress wiring (`df0ed79f`), re-run green after
the fix.

## [0.6.1] - 2026-08-07

Design gets one verb: `tt-bio design INPUT --model boltzgen|rfd3`, mirroring `tt-bio predict
--model ...`. `tt-bio gen` still works unchanged and is now a hidden deprecated alias for
`tt-bio design --model boltzgen`. ESMC-300M/600M single-sequence embedding is trace-captured,
and RFD3 multi-card design no longer strangles itself on host threads: four cards aggregated
0.68x of a single card before, 3.48x after. `--devices N` is honoured on the single-card embed
and RFD3 design paths, where it had been ignored, and `predict` now says so out loud when a run
finishes on fewer cards than you asked for instead of quietly returning a slower result.

This release also merges back the v0.6.0 release commits, which were tagged but never landed on
`main`. For five days `main` reported version 0.5.0, shipped no 0.6.0 changelog entry, and
carried a `tests/` file that aborted pytest collection so the host suite never ran.

### Changed

- **Unified design CLI** — `tt-bio design INPUT --model boltzgen|rfd3` is now the single design
  command, mirroring `tt-bio predict --model ...`. BoltzGen pipeline options (`--steps`,
  `--config STEP key=val`, `--num_designs`, `--budget`, `--devices`, `--out_dir`) moved onto the
  shared command as boltzgen-scoped flags (`boltzgen` is the default model, so existing
  single-model invocations need only swap the verb). The RFD3 checkpoint flag is renamed
  `--golden_dir` to `--checkpoint`, with the old spelling kept as a hidden deprecated alias for
  one release. The RFD3 engine modules moved from flat `tt_bio/rfd3*.py` files into a
  self-contained `tt_bio/rfd3/` package mirroring `tt_bio/boltzgen/`; `import tt_bio.rfd3`
  resolves to the package. Numerics are bit-identical: this is a CLI, layout and docs change
  only. BoltzGen user documentation moved from the README to `docs/boltzgen-design.md`, and the
  README now has one Design section covering both models. (`26cf293a`)
- `fa77e884` predict: warn loudly when a run completes on fewer cards than requested, rather
  than silently returning the slower result.
- `7bee7292` embed and `ef0265ef` rfd3: honour `--devices N` on the single-card path.

### Deprecated

- **`tt-bio gen`** — hidden from `--help` and prints a deprecation warning on stderr, then
  forwards every argument to BoltzGen unchanged. It still works; it will be removed in a future
  release. Use `tt-bio design INPUT --model boltzgen` instead (`gen run X --output out` becomes
  `design X --model boltzgen --out_dir out`).

### Fixed

- `280de387` tenstorrent: restore the bf16 construction defaults in fast mode. This was the root
  cause of ESMFold2 returning NaN confidence.
- `a657a636` boltz2: use one sample-chunk width for the whole diffusion trajectory and pad the
  short tail. Chunk width is not bit-inert, so a trajectory that changed width partway changed
  its own numbers.
- `46b641e3` protenix: `--trace` was a silent no-op for `--model protenix-v2`; the predict
  worker now forwards it.
- `c7313862` worker: self-terminate when the spawning dispatcher dies, instead of outliving it
  and holding a card.
- `69649a0e` stderr filter: kill the nanobind-filter grandchild together with its parent.
- `13107362` find_mmseqs: pair mmseqs with the `colabfold_search` actually in use.
- `176fc85b` embed: clean up the co-location nonce on failure, and stop accepting a missing
  shared file.

### Performance

- `3bb206e7`, `11cbb13a` ESMC-300M/600M: the single-sequence embed forward is captured as a ttnn
  trace, and the trace region is guaranteed at fleet load time. Measured 1.47x host-relative and
  bit-exact (`2584548d`).
- `75fe28b3` rfd3: cap host thread pools in the design fan-out. Four co-resident single-card
  designs aggregated 0.68x before the fix and 3.48x after, the same thread-oversubscription
  class that `--host_threads` addressed for folds in 0.6.0.
- `67786544`, `78e1ebec` embed: pin single-card visibility before the ttnn import, and raise the
  parallel npz writers to 32 threads.
- `719807ef`, `114f4b4d` embed: co-located workers hand back result paths instead of pushing
  base64 through the controller.

### Gates and documentation

- `cd891272` parity gate: pin the in-process harness legs to one card instead of the whole mesh.
- `81128ae7` parity gate: derive worker locality from the real hostname. `$HOSTNAME` is not
  exported to non-interactive shells, so every non-pc host classified itself as remote and
  tried to ssh to itself.
- `cca9e030` parity: make the esmfold2-trpcage leg runnable on Wormhole.
- `e50f285e` perf gate: seed the p300c baselines for opendde-abag and rfd3.
- `3d6239ee` parity docs: record the BoltzGen sampling bound and SaProt-1.3B near-pass.
- Restored from the unmerged v0.6.0 release commits: the README documentation for
  `--host_threads` and `--max_parallel_samples`, the re-synced RELEASING.md accuracy floors, the
  qb1 p150a opendde-abag perf baseline, and the fix that stopped a `tests/` script aborting
  pytest collection.

### Release gate (Blackhole P150a on `tt-quietbox`, card 0)

Host suite: 241 tests collect and run (on main before the v0.6.0 merge-back, collection aborted
at 220 and no test executed). Full run `7 failed, 212 passed, 22 skipped`; re-run one file per
process, six of the seven pass alone, the one-device-context-per-process false-failure class
from 0.6.0. The seventh, `test_confidence_device_resident_parity`, fails at PCC
0.9807124853748275, bit-identical at v0.6.0 and v0.5.0: pre-existing, not a regression in this
range. Packaging guard: 15/15 data files and 31/31 declared dependencies in the wheel and
sdist. UX gate: PASS on all 11 surfaces plus the CLI leg, including the deprecated `gen` alias
warning.

**Accuracy gate** — every shipped fold architecture folded end-to-end with production sampling
and checked against a per-model ground-truth floor, not self-consistency:

| model | RMSD (A) | TM | floor | result |
|---|---|---|---|---|
| boltz2 | 1.555 | 0.942 | <=3.0 / >=0.75 | PASS |
| esmfold2 | 1.834 | 0.906 | <=8.0 / >=0.40 | PASS |
| esmfold2-fast | 1.811 | 0.909 | <=4.5 / >=0.60 | PASS |
| protenix-v2 | 1.458 | 0.945 | <=6.0 / >=0.50 | PASS |
| opendde | 1.350 | 0.953 | <=6.0 / >=0.50 | PASS |

BoltzGen scRMSD 0.849 A at a 75% pass rate (floor <=2.0 A, >=50%); OpenDDE-abag DockQ 0.854
with fnat 0.922 (floor >=0.50); capacity leg peaked at 5.97 GiB against a 7.0 GiB budget,
writing all 50 samples. ESMC-300m/600m per-residue PCC 0.99961 / 0.99964, trace bit-exact.

**Parity gate** (`scripts/full_parity_gate.py`, 23 legs): **20 PASS, 3 GAP, 0 DRIFT**, every leg
reproducing its committed verdict. boltz2-prot-nomsa and boltz2-affinity-fkbp12-nomsa reproduce
their committed `GAP-evidenced` verdicts. The third, protenix-ubq-msa, is newly accepted
`GAP-evidenced` (`cf35cc15`): the in-range MSA row-chunking for large MSAs changes bf16
summation order on exactly the one leg whose MSA crosses the 0.25 GiB chunk budget, and the
observed 2.015 A sits inside both the chunking commit's own measured envelope (mean 0.738 /
max 3.98 A) and this target's committed noise floor (max 2.993 A). Root cause and evidence are in
`docs/implementation-parity.md`.

**Performance gate**: PASS on a quiet host, 13 of 14 models within the +/-15% threshold —
boltz2 +0.2%, esmfold2 -1.7%, esmfold2-fast -1.5%, protenix-v2 -8.6%, opendde +2.0%,
opendde-abag +0.9%, esmc-300m +0.3%, esmc-600m +0.2%, esmc-6b +1.7%, saprot-650m -0.3%,
boltzgen -0.4%, boltz2-affinity -3.7%, rfd3 +2.9%. `esmc-300m-single` is a **first seeded
tt-quietbox machine baseline** at 14.62 seq/s, not a compared number: its only prior baseline
was seeded on pc, and qb1's p150a reads ~30-36% slower on that leg from within-p150a machine
variance, which the machine-id baseline layer exists to absorb. A first run co-resident with a
CPU-only campaign read -10..-28% on every host-dispatch-heavy leg while the dispatch-light
control held +0.3%; the quiet re-run recovered all of them.


## [0.6.0] - 2026-08-01

ESMFold2 now folds at the paper's protocol by default: 10 recycling loops and 100 requested sampling
steps, of which 68 execute after the sigma-schedule clip. The old default under-recycled by 3.3x and
burned roughly twice the diffusion compute for no measured quality gain. Folds that draw several
samples per batch pick up `--max_parallel_samples` on Protenix-v2, OpenDDE and ESMFold2, where the
flag had been silently ignored, and single-card folds sharing a host can now be given a CPU budget
with the new `--host_threads` — four co-resident folds previously aggregated only 2.56x of a possible
4.00x because each sized its thread pools to every core. A fold error is no longer truncated before
its tail, which matters more than it sounds: a clipped out-of-memory message had hidden the real
evidence and produced a wrong "Wormhole cannot run this" verdict.

### Fixed

- `c145a1a6` worker: stop truncating away the tail of a fold error.
- `b62301f5` boltz2: width-based diffusion chunking. Upstream's `N % mps + 1` chunk-count formula
  produced a single 1000-sample chunk at N=1000 / mps=5 — about 0.9 GB of L1 forward buffers — and
  ran out of memory at both mps 5 and mps 3. It now matches `protenix.edm_sample`'s width<=mps
  semantics. `9f31125fc` refuses to run rather than derive a wrong sibling seed past two chunks.
- `364bdd46` predict: `--max_parallel_samples` was a silent no-op for protenix-v2, opendde and
  esmfold2.
- `f0fe2a72` esmfold2: thread the fold seed into the diffusion sampler's private RNG.
- `fce156d4` protenix/opendde: broadcast the sample-invariant diffusion biases instead of
  replicating them.
- `45036687` tests: a script dropped into `tests/` ran its scenarios at import and ended in
  `sys.exit()`, so pytest aborted collection with INTERNALERROR and the entire 209-test host suite
  silently never ran. Caught by this release's own gate.

### Changed

- `f8a00aed` esmfold2 / esmfold2-fast default to the paper protocol (10 recycling loops, 100
  requested sampling steps). Explicit flags are honoured verbatim; other models are unchanged.
- `228dc8e5` rfd3: raise the design batch clamp to the memory bound it stands for.

### Performance

RFdiffusion3 (43 commits, p19-p32): resident pair tables, a one-kernel head merge, build-once sparse
pair-bias construction, and an opt-in `RFD3_TUNE_MATMUL` calibration path. Two trace levers were
measured and recorded as negative rather than shipped (`5a32a203`), and `5945a944` fixes a traced
decoder handing out the buffer its own trace replays into. Measured effect at release: rfd3 runs at
0.1185 designs/s, +5.3% against its committed baseline.

### Documentation

- `54682d89` RELEASING.md's accuracy floor table had drifted from `release_gate.py:MODELS` — it
  listed ESMFold2 at 4.0 A / TM 0.65 while the code has enforced 8.0 A / TM 0.40 since `88c14f3b`.
  Re-synced and annotated with the source of truth.
- `6c63537b`, `2fcf0bad` document `--host_threads` and `--max_parallel_samples`, the two
  user-facing CLI flags that had shipped without ever reaching the options table.
- Profiling instrumentation (6 commits): measured Blackhole p150a roofline, `ttnn.graph` capture
  characterised as an instrument, real-model op counts (ESMFold2-Fast 1290 ops per diffusion step,
  whole-fold 43,291 ops), and a worked example showing ESMC-300M is DRAM-bandwidth-bound rather than
  matmul-bound.

### Release gate (Blackhole P150a on `tt-quietbox`, card 2)

Host suite, run one process per test file: **200 passed, 22 skipped, 1 failed**. The single failure,
`test_protenix_confidence.py::test_confidence_device_resident_parity`, is pre-existing: it produces
a bit-identical PCC of 0.9807124853748275 against its >0.99 bar at both this commit and v0.5.0, so it
predates this release. Packaging guard: 15/15 expected data files and 31/31 declared dependencies
present in the wheel and sdist. UX gate: PASS on all 11 surfaces.

**Accuracy gate** — every shipped fold architecture folded end-to-end with production sampling and
checked against a per-model ground-truth floor, not self-consistency:

| model | RMSD (A) | TM | floor | result |
|---|---|---|---|---|
| boltz2 | 1.555 | 0.942 | <=3.0 / >=0.75 | PASS |
| esmfold2 | 1.834 | 0.906 | <=8.0 / >=0.40 | PASS |
| esmfold2-fast | 1.811 | 0.909 | <=4.5 / >=0.60 | PASS |
| protenix-v2 | 1.458 | 0.945 | <=6.0 / >=0.50 | PASS |
| opendde | 1.350 | 0.953 | <=6.0 / >=0.50 | PASS |

BoltzGen scRMSD 1.300 A at a 100% pass rate (floor <=2.0 A, >=50%); OpenDDE-abag DockQ 0.856 with
fnat 0.922 (floor >=0.50); capacity leg peaked at 5.90 GiB against a 7.0 GiB budget on a
1095-token target, writing all 50 samples.

**Parity gate** (`scripts/full_parity_gate.py`, 23 legs): **21 PASS, 2 GAP, 0 DRIFT**. Both GAPs
reproduce a committed `GAP-evidenced` verdict rather than drifting — boltz2-prot-nomsa at
ratio 2.07 and boltz2-affinity-fkbp12-nomsa at 0.0033. Highlights: ESMC-300m/600m minimum
per-residue PCC 0.99918 / 0.99938, SaProt-35m/650m embedding PCC 0.99914 / 0.99964, ESMFold2 within
its noise floor on all four targets, Boltz-2 trp-cage envelope ratio 0.59 and HSA 0.71, Protenix-v2
7ROA 1.05 / ubiquitin 0.00 / HSA 0.48, both OpenDDE structure legs bit-identical at 0.0000, and the
RFD3 featurizer 43/43 keys bit-exact. Five of six Boltz-2 affinity legs pass and the sixth
(fkbp12-msa, 0.0028) improves on its recorded gap.

**Performance gate**: PASS on all 13 models within a +/-15% threshold — boltz2 +0.2%, esmfold2 +0.7%,
esmfold2-fast -0.9%, protenix-v2 -4.5%, opendde +0.9%, opendde-abag -0.3%, esmc-300m +0.3%,
esmc-600m +0.0%, esmc-6b +1.0%, saprot-650m -1.1%, boltzgen +0.0%, boltz2-affinity -4.9%,
rfd3 +5.3%. `opendde-abag` is a **first seeded baseline** at 1.208666 structures/s, not a compared
number: v0.5.0 reported seeding one but the JSON edit was never committed, leaving the model with no
baseline on any card for two releases.

Both the perf sweep and the host suite must be run **one model (or one test file) per process**.
tt-bio supports one device context per process, and iterating models inside a single long-lived
process produces false failures — it reported rfd3 at -42.5% and failed five host-suite tests that
each pass when run alone.

## [0.5.0] - 2026-07-27

Multi-sample folds are about 3x faster on Protenix-v2 and OpenDDE, which now draw every sample
from one batched device trajectory instead of looping one sample at a time. The parity reference
fixtures a clean checkout needs are published for the first time, so
`scripts/fetch_parity_fixtures.sh` works and the eight structure legs 0.4.0 had to skip are back
in the gate. RFdiffusion3 gains the parity, performance, and UX gate coverage it shipped without,
and a few percent of throughput.

**Release gate** (Blackhole P150a on `pc`, tt-bio 0.5.0): host suite 114 passed / 49 skipped;
packaging guard 15/15 data files and 31/31 declared dependencies present in the wheel and sdist;
UX gate PASS on every shipped surface, now including `tt-bio design`.

**Parity gate** (`scripts/full_parity_gate.py`, 22 legs): **20 PASS, 2 GAP**, both GAPs
reproducing a committed `GAP-evidenced` verdict rather than drifting. The eight envelope structure
legs score for the first time since they were externalized, because the fixture asset they read is
finally published: Boltz-2 trp-cage 0.074 Å against a 0.145 Å envelope (ratio 0.51), Boltz-2 HSA
0.92 vs 1.33 (0.69), Protenix-v2 7ROA 0.049 vs 0.042 (1.16), Protenix-v2 HSA 0.050 vs 0.052 (0.96).
ESMFold2 (4 targets), ESMC-300m/600m, SaProt-35m/650m, OpenDDE-abag, BoltzGen (scRMSD pass-rate
100%, median 0.68 Å) and the RFD3 featurizer (43/43 keys bit-exact) all reproduce their recorded
verdicts, as do five of six Boltz-2 affinity legs; the sixth improves on its recorded gap.

Three of those legs pass on the absolute floor rather than a measured envelope, because their
cached bf16 and fp32 references are identical, which collapses the envelope to zero: Protenix-v2
ubiquitin (0.037 Å against the 0.05 Å floor) and both OpenDDE structure legs. The envelope test
assumes a model with a torch CPU path to recompute in two dtypes, which a ttnn-only port does not
have. Those OpenDDE legs are covered by the R/D/X device-vs-reference diagnostic instead, the right
scorer for such a port, and `opendde-prot-prod` passes it (X 4.824 against a 1.499 floor).
Restoring a meaningful envelope for ttnn-only legs is follow-up work, not a coverage hole.

The two GAPs are Boltz-2 7ROA no-MSA structure (kabsch_rmsd ratio 2.04) and FKBP12+SB3 no-MSA
affinity. Both are recorded `GAP-evidenced` in `docs/implementation-parity.md`. The 7ROA one was
root-caused this release: the pinned envelope seed lands that target in an unusually chaotic
reverse-diffusion trajectory in the *reference*, whose own bf16-vs-fp32 spread swings 3.45 Å, 2.16 Å
and 0.81 Å across seeds 0, 1 and 2, and the leg passes cleanly at seeds 1 and 2. Two on-device fp32
levers were tried and neither moves it, so it is a property of the reference trajectory, not a port
defect.

**Perf gate** (`scripts/perf_regression.py`, trpcage 20 aa single-sequence, warm 2+5, ±15%):
boltz2 1.164 structures/s (-2.2%), esmfold2 1.586 (-7.0%), esmfold2-fast 2.167 (-5.4%),
protenix-v2 2.192 (-8.0%), opendde 1.891 (-1.6%), esmc-300m 25.65 seq/s (+53.3%), esmc-600m 20.99
(+0.3%), esmc-6b 4.429 (+39.7%), saprot-650m 239.2 (+7.4%), boltzgen 0.01708 designs/s (-0.8%),
rfd3 0.1178 designs/s (-3.6%), boltz2-affinity 0.00909 affinities/s (-4.2%, median of three
isolated runs). opendde-abag measures 1.789 structures/s and is seeding its first baseline. No
model regressed beyond the threshold and nothing OOMed.

### Added
- **Per-sample chain-pair ipTM and per-chain pTM for OpenDDE** — `pair_chains_iptm` and
  `chains_ptm` are now written for every sample in `all_runs`, like every other confidence
  scalar, instead of for the top-ranked sample only. For antibody-antigen work the
  antibody-vs-antigen chain-pair ipTM ranks candidates better than the global ipTM, and a
  winner-only value cannot rank the rest.
- **RFdiffusion3 gate coverage** — `tt-bio design` now has legs in the parity gate, the
  performance regression gate, and the UX gate. It shipped in 0.4.0 with none.

### Changed
- **Protenix-v2 / OpenDDE diffusion multiplicity batching** — `Protenix.fold` / `OpenDDE.fold`
  now draw `n_sample` samples from one batched device denoise trajectory instead of looping one
  sample at a time (mirrors `boltz2.AtomDiffusion.sample`'s `multiplicity` +
  `max_parallel_samples` pattern; `--max_parallel_samples` chunks batches too large to fit).
  Parity-verified against the established diffusion noise floor (batched-vs-unbatched drift
  within the seed-to-seed floor, not bit-exact, since diffusion is stochastic) on two hosts:
  Protenix X/floor 0.971-1.049, OpenDDE X/floor 0.995-1.003. Measured speedup at multiplicity 4:
  Protenix 3.19-3.57x, OpenDDE 3.15x.
- **RFdiffusion3 is a few percent faster** — the design matmuls that read a single tile of K now
  carry an explicit core-grid hint. It is bit-identical to the default and worth 5-15% depending
  on size. On one Blackhole p150a at 40 residues, batch 8 goes from 0.1216 to 0.1352 designs/sec
  and batch 1 from 0.0767 to 0.0807. The refreshed per-size table is in `docs/rfd3-design.md`.

### Fixed
- **OpenDDE re-ran a full offline MSA search on every multi-chain fold, and overwrote the MSAs
  the other models read.** The paired-MSA helper had no cache check, unlike the unpaired path, so
  each fold searched the whole database again. Worse, it wrote `{seq_hash}.a3m` into the shared
  `msa_dir`, silently replacing the files Boltz-2 and Protenix-v2 read for the same chains.
  Paired results now live in `msa_dir/paired/` and are only computed when absent, so a paired run
  can neither clobber the unpaired cache nor be served an unpaired file as if it were paired.
- **The parity reference fixtures were documented but never published.**
  `docs/implementation-parity.md` and `scripts/fetch_parity_fixtures.sh` both pointed at a
  `parity-fixtures-latest` release asset that did not exist, so on a clean checkout the eight
  envelope structure legs had no references and reported `BLOCKED-REF-REGEN-NEEDED` instead of a
  verdict. The asset is published and the fetch is verified end to end from a clean directory.
  Two fetch bugs went with it: a doubled `parity-fixtures-` prefix in the asset name, and a
  `grep -F` that treated the `$` end-anchor as a literal character.
- **Regenerating envelope references destroyed the fixtures' other provenance.** The envelope
  path and the legacy R/D/X path share one `meta.json` with incompatible schemas, and the
  envelope regenerator overwrote the whole file, wiping the `settings_tag` and upstream-reference
  provenance the R/D/X scorer needs. That is what made five legs flip between blocked and passing
  every time either gate was regenerated. The regenerator now nests its own bookkeeping under an
  `envelope` key and leaves harvested fields alone.

## [0.4.0] - 2026-07-26

First release shipping **RFdiffusion3** (`tt-bio design`) — an all-atom generative model that
designs new protein structures and the sequences that support them from a specification, instead
of folding a sequence you already have. Protein-binder design, motif scaffolding, and
nucleic-acid-binder design run end to end from a real input structure. The checkpoint downloads
itself from the Institute for Protein Design on first use, so no `rc-foundry` install is needed.
Multiple designs per specification share device forwards, and `--devices` fans a design set
across cards. See [`docs/rfd3-design.md`](docs/rfd3-design.md).

Also in this release: OpenDDE no longer runs its diffusion in fp32 (a >60x slowdown it inherited
from a Protenix-v2 default), opening a card that another process already holds now fails with a
clear error instead of colliding, and `transformers` moves to >= 5.5.0, clearing three dependabot
advisories.

**Release gate** (Blackhole P150a on `pc`, tt-bio 0.4.0): host suite 111 passed / 49 skipped;
packaging guard 15/15 data files and 31/31 declared dependencies present in the wheel and sdist;
UX gate PASS on every shipped surface (live-progress advancement, strict mmCIF/npz parse, and
results/manifest shape for boltz2, esmfold2, esmfold2-fast, protenix-v2, opendde, opendde-abag,
esmc-600m, saprot-650m, boltzgen, boltz2-affinity).

**Perf gate** (`scripts/perf_regression.py`, trpcage 20 aa single-sequence, 1 recycle / 10 steps /
1 sample, warm 2+5, ±15% threshold):

| model | metric | baseline | current | delta | result |
|---|---|---|---|---|---|
| boltz2 | structures/s | 1.19 | 1.15 | -3.4% | PASS |
| esmfold2 | structures/s | 1.705 | 1.601 | -6.1% | PASS |
| esmfold2-fast | structures/s | 2.29 | 2.141 | -6.5% | PASS |
| protenix-v2 | structures/s | 2.383 | 2.166 | -9.1% | PASS |
| opendde | structures/s | 1.922 | 1.785 | -7.1% | PASS |
| esmc-300m | seq/s | 16.74 | 25.26 | +50.9% | PASS |
| esmc-600m | seq/s | 20.92 | 20.78 | -0.7% | PASS |
| esmc-6b | seq/s | 3.171 | 4.363 | +37.6% | PASS |
| saprot-650m | seq/s | 222.7 | 237.9 | +6.8% | PASS |
| boltzgen | designs/s | 0.01723 | 0.01695 | -1.6% | PASS |
| boltz2-affinity | affinities/s | 0.009498 | 0.008148 | -14.2% | PASS |

No model regressed beyond the threshold. No OOM through the gate targets.

**Parity gate** (`scripts/full_parity_gate.py`, 21 legs): ESMC-300m/600m, SaProt-35m/650m,
ESMFold2 (4 targets, L20 to L129), OpenDDE-abag (global DockQ 0.853, fnat 0.932) and BoltzGen
(scRMSD pass-rate 100%, median 1.12 Å) all reproduce their recorded verdicts, as do five of the
six Boltz-2 affinity legs; the sixth, FKBP12+SB3 with MSA, improves on its recorded gap. The
FKBP12+SB3 no-MSA affinity scalar is now recorded `GAP-evidenced` under the integration-envelope
test on the same cross-backend bf16 floor already accepted for its MSA counterpart (see
`docs/implementation-parity.md`).

The eight envelope **structure** legs report `BLOCKED-REF-REGEN-NEEDED` rather than a verdict: the
CPU reference structures those legs compare against are not distributed with the repository, so
they cannot be reproduced from a clean checkout. Their recorded verdicts in
`docs/implementation-parity.md` are unchanged and remain the evidence of record; restoring the
reference set is tracked as follow-up work.

### Added
- **RFdiffusion3 (RFD3)** — `tt-bio design specs.json --from_pdb --out_dir ./designs`. Designs
  are specified with a contig mini-language (fixed regions taken verbatim from the input
  structure, designed regions of fixed or randomized length, chain breaks, indexed and unindexed
  motifs, per-atom fixing). Protein-binder design, motif scaffolding, and nucleic-acid-binder
  design accept a real PDB input; small-molecule-binder, enzyme, and symmetric-oligomer design
  run on device and are value-parity-verified against a captured reference, but the host
  featurizer does not build their input from a PDB yet and raises `NotImplementedError`.
  An independent ttnn reimplementation — no upstream RosettaCommons code is vendored, only the
  BSD-3-Clause checkpoint is fetched.
- **Batched multi-design generation for RFD3** — `--num_designs N` produces N designs per
  specification (noise seed `--seed + i`), sharing device forwards in batches of up to
  `--batch_size` (default 8, reduced automatically for larger atom counts). Batching is
  accuracy-free: the device forward is bit-identical across batch size, so a batched design
  reproduces its standalone run exactly (min trajectory PCC 1.000000, maxabs 0, at 200 timesteps
  and batch 8). Throughput depends on design size — on one Blackhole p150a at 200 timesteps,
  batch 8 is 1.59x batch 1 at 40 residues and 1.21x at 80 residues, and within a few percent of
  batch 1 above roughly 150 residues, where `--devices` is the parallelism that matters. The
  per-size numbers are in `docs/rfd3-design.md`.
- **`tt-bio design --devices 0,1,2,3`** — fans the (specification × `--num_designs`) jobs across
  the listed cards, one pinned subprocess per card, the same data-parallel pattern `tt-bio embed`
  and `tt-bio predict` use.
- **Physical-card lease at device open** — every device acquisition takes an exclusive `flock` on
  a per-card lock file for as long as the card is open. A second process opening the same card
  waits up to `TT_BIO_LEASE_TIMEOUT` (120 s) and then fails with `DeviceInUseError` naming the
  holder, instead of colliding at the fd level. The lock is released by the kernel on any process
  death, so a killed or orphaned job never leaves a phantom claim.

### Fixed
- **OpenDDE ran its diffusion in fp32, >60x slower than it needs to be.** OpenDDE reuses the
  Protenix-v2 diffusion stack, and Protenix-v2 defaults that stack to fp32 on device
  (`PROTENIX_DIFFUSION_FP32_DEVICE=1`, correct for Protenix-v2, where fp32 is what its own
  reference uses). OpenDDE silently inherited it, and fp32 on OpenDDE's atom-level tensors is
  catastrophically slow. OpenDDE now pins its own already-validated bf16 diffusion config
  explicitly instead of reading the env default. A real device fold of 1AHW is back to 539 s at
  DockQ 0.862, matching the 0.863 pre-regression baseline — accuracy-neutral, speed-only.
- **`--write_pae` was silently a no-op for `--model opendde` / `opendde-abag`.** The flag was
  parsed but never reached the OpenDDE prediction path, so no PAE was written. Now wired.
- **`--msa_db_path` was silently ignored by `opendde-abag`.** An offline paired-MSA run fell
  through to the network path and failed if the public ColabFold service was unreachable. The
  local paired-MSA database is now honored, so `opendde-abag` folds fully offline.

### Changed
- **`transformers` >= 5.5.0, `huggingface_hub` >= 1.5.0** (were `==4.57.6` and `<1.0`). Clears
  three dependabot advisories (Trainer, `config.json`, and LightGlue remote-code execution; the
  last is not reachable from tt-bio). The vendored ESMFold2 model runs on the stock transformers
  core, and the bump is CPU-level bit-exact on the same torch, weights, and seed.

## [0.3.4] - 2026-07-22

### Fixed
- **Missing package data broke every clean `pip install`** — the
  `[tool.setuptools.package-data]` table shipped only the two vendored
  ESM/ESMFold2 license files, so the 0.3.3 wheel and sdist omitted the 13
  runtime data files the package loads by path. A fresh `pip install tt-bio`
  crashed at featurization for protenix-v2 and opendde / opendde-abag folds
  (`FileNotFoundError: .../tt_bio/data/protein_ref_conformers.json`) and at
  `_configure` for every `tt-bio gen` design
  (`FileNotFoundError: .../boltzgen/resources/config/design.yaml`).
  boltz2 / esmfold2 / esmc / saprot were unaffected. Added the missing globs
  for `tt_bio.data` and the `tt_bio.boltzgen.resources` tree plus a
  `MANIFEST.in`, and added `scripts/packaging_smoke.py` to the release gate
  so a dropped data file fails the gate instead of shipping silently.
  Packaging-only; no model or behavior change.

## [0.3.3] - 2026-07-22

### Fixed
- **BoltzGen abort/resume robustness** — hard-killing `tt-bio gen` mid-download no
  longer poisons the artifact cache or leaks `.dl-*` staging directories. A sweep
  on startup reaps orphaned staging dirs left by a killed fetch, and a download-path
  patch keeps partial downloads out of the cache so a resumed run picks up cleanly.
- **BoltzGen design-spec-check perceived hang** — `tt-bio gen` now reports progress
  during the design-spec check (the gap after `mols.zip` is cached where the CLI
  previously showed no output), so a genuinely slow conformer-generation step reads
  as activity instead of a frozen process. Print-only; parsed molecule data is
  bit-identical.

### Changed
- **Repo docs reorganization** — internal engineering-journey notes moved out of the
  public repository into the private knowledge base, leaving the public docs focused
  on what a user needs to decide and use the package. Comment/docstring/doc-move
  only; no logic change.

## [0.3.2] - 2026-07-20

### Fixed
- **SaProt-1.3b config bug** — `CONFIGS["saprot-1.3b"]` carried a fabricated arch
  (hidden=2560 / n_heads=40 / n_layers=40 / intermediate=10240) that does not match
  the real `westlake-repl/SaProt_1.3B_AF2` checkpoint (1280 / 20 / 66 / 5120 — the
  650m width with double the layers). `load_state_dict(..., strict=False)` silently
  masked the mismatch, so the device ran with effectively untrained weights and the
  1.3b leg read as a parity failure. Config corrected; `Saprot.from_pretrained` now
  reads the checkpoint's `config.json` and refuses to build on an arch mismatch. With
  correct shapes saprot-1.3b reaches X_emb=0.99508 / X_logits=0.99895 (deterministic,
  qb1 card 1) — a near-pass; the per-residue embedding PCC lands just below the
  0.9987–0.9996 ESMC band (bf16 accumulation over 66 residual layers), so no clean
  PASS row is added to `docs/implementation-parity.md`. See `docs/saprot-parity.md`.
- **Perf-gate within-card-type false positives** — the perf-regression gate keyed
  baselines by card type only, so two machines with the same card type (pc vs qb1,
  both p150a) read as false ~30–36% regressions against each other. Added a machine-id
  layer under card type (`socket.gethostname()`), with backward-compatible fallback
  to the card-type block. `--update-baseline` now writes to the detected machine's
  block.

### Added
- **`tt-bio saprot --devices`** — multi-card data-parallel fanout for SaProt
  embeddings (one pinned worker per card, sequences sharded by length, results
  reassembled in input order), mirroring the ESMC `--devices` path. Row-independent:
  a sequence's output is identical to running it on one card.
- **esmc-300m and esmc-6b perf-gate baselines seeded** (esmc-300m 33.17 seq/s on
  p300c, esmc-6b 3.17 seq/s on p150a), activating the perf-regression legs specced
  in 0.3.1.
- **Release-gate perf + UX coverage for SaProt and Boltz-2 affinity** — both shipped
  in 0.3.1 with accuracy-leg coverage but no perf/UX gate legs; saprot-650m
  (222.69 seq/s, qb1 p150a) and boltz2-affinity (0.014319 affinities/s, p300c)
  baselines seeded.

### Removed
- **ProteinMPNN** — the `tt-bio design` inverse-folding port is dropped entirely.
  It ran CPU-only (dispatch-bound, no TT-card use), duplicated BoltzGen's
  inverse-fold capability, and reimplemented the mature upstream
  `dauparas/ProteinMPNN`. SaProt is untouched.

### Verify / benchmark hardening
- **Boltz-2 and Protenix-v2 ubiquitin flagship legs hardened 2+2 → 5+5 seeds**
  (seeds 0–4 both sides): R and D are now 10 pairwise distances each, so the parity
  verdict is a real statistical statement rather than a single-pair coincidence.
  Both PASS within floor on CA-RMSD and TM-score; CA-lDDT misses on Boltz-2 (a bf16
  narrower-basin residual on local structure, recorded as a borderline GAP) and
  passes on Protenix-v2.
- **TM-score and CA-lDDT added** alongside CA-RMSD for the two flagship stochastic
  legs — alignment-free metrics a pharma customer evaluating a binding interface
  actually feels.
- **Boltz-2 affinity leg widened 1 → 3 targets** (FKBP12, DHFR+MTX, trypsin+BAM)
  with ligand-pose RMSD and pocket-lDDT added alongside the scalar Δlog10(IC50).
  Scalar affinity PASSES on all three; ligand-pose RMSD passes on FKBP12 and
  trypsin, misses on DHFR; pocket-lDDT misses on all three (the consistent bf16
  narrower-basin residual on local interface geometry).
- **FKBP12 affinity GAP root-caused and fixed** — the boltz-2 worker is
  spawn-started (does not inherit the controller's RNG) and the affinity path calls
  `predict_affinity` without re-seeding, so the affinity diffusion's `torch.randn`
  draws ran from an unseeded global RNG; the tight affinity floor (R=0.010) surfaced
  this as a systematic GAP. Seeded the global RNG once before the boltz-2 structure
  step, matching the reference's single `seed_everything` → structure → affinity
  stream. `affinity_pred_value` now X=0.041 ± 0.024, within floor.
- **HSA (PDB 1AO6, L585) added** as the first L300–800 pharma-realistic large target
  on both flagship legs: Boltz-2 PASS (CA-RMSD X=1.47 ± 0.22 Å, within floor);
  Protenix-v2 GAP (X=1.03 ± 0.17 Å vs a tight GPU-bf16 reference floor R=0.70 Å — a
  tight-floor effect from bf16 numerical divergence between NVIDIA and Tenstorrent
  reduction orders, not a structural defect; both folds are correct HSA shapes). See
  `docs/implementation-parity.md`.

### Release gate (card p150a @ pc, tt-bio 0.3.1 baseline, warm 2 warmup + 5 timed)
Perf-regression gate: 11/11 models within ±15% of baseline, no regression. boltz2
1.188 structures/s (-0.2%), esmfold2 1.700 (-0.3%), esmfold2-fast 2.235 (-2.4%),
protenix-v2 2.376 (-0.3%), opendde 1.899 (-1.2%), esmc-300m 25.72 seq/s (+53.7%,
new p150a baseline seed), esmc-600m 21.06 (+0.7%), esmc-6b 4.413 (+39.2%, new
p150a baseline seed), saprot-650m 232.8 (+4.5%), boltzgen 0.01712 designs/s
(-0.6%), boltz2-affinity 0.008772 affinities/s (-7.6%, first pc p150a baseline).
Accuracy (`scripts/release_gate.py`) and UX (`scripts/ux_regression.py`) gates
green on the same commit lineage; no code changes since their last pass besides
the `TT_VISIBLE_DEVICES` default fix below, which does not touch either path.

## [0.3.1] - 2026-07-19

Adds **SaProt** structure-aware protein embeddings (`tt-bio saprot`, an ESM-2 encoder over a
fused amino-acid + Foldseek-3Di vocabulary). Purely additive: no existing model file changed.

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0, Blackhole P150a):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.541 Å | 0.939 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 1.774 Å | 0.915 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.725 Å | 0.909 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 1.417 Å | 0.936 | ≤6.0 Å / ≥0.50 | PASS |
| OpenDDE | 1.367 Å | 0.952 | ≤6.0 Å / ≥0.50 | PASS |

**BoltzGen designability** — n=4, `examples/binder.yaml`: scRMSD median 0.892 Å, 4/4 designs (100%) ≤2 Å (floor ≤2.0 Å / ≥50%) — PASS.

**ESMC embedding parity** (fused-RoPE shipped path vs reference esm, 76-residue sequence, PCC floor 0.99):

| model | per-res PCC | pooled | logits | argmax | result |
|---|---|---|---|---|---|
| esmc-300m | 0.99961 | 0.99993 | 0.99990 | 1.0000 | PASS |
| esmc-600m | 0.99964 | 0.99989 | 0.99996 | 1.0000 | PASS |

**SaProt embedding parity** (vs reference HF `EsmForMaskedLM` golden, PCC floor 0.99):

| model | embedding PCC | logits PCC | result |
|---|---|---|---|
| saprot-35m | 0.999138 | 0.999772 | PASS |
| saprot-650m | 0.999638 | 0.999927 | PASS |

**UX gate** (`scripts/ux_regression.py`, `examples/trpcage.yaml`): every shipped surface (Boltz-2,
ESMFold2, ESMFold2-fast, Protenix-v2, OpenDDE, ESMC-600m embed, BoltzGen) cleared live-progress
advancement, strict mmCIF/npz parse, and results/manifest shape — PASS.

**Perf gate** (`scripts/perf_regression.py`, Blackhole P150a, trpcage 20 aa single-sequence, warm 2+5, ±15% threshold):

| model | metric | baseline | current | delta | result |
|---|---|---|---|---|---|
| boltz2 | structures/s | 1.190 | 1.176 | -1.2% | PASS |
| esmfold2 | structures/s | 1.705 | 1.692 | -0.7% | PASS |
| esmfold2-fast | structures/s | 2.290 | 2.304 | +0.6% | PASS |
| protenix-v2 | structures/s | 2.383 | 2.329 | -2.3% | PASS |
| opendde | structures/s | 1.922 | 1.939 | +0.9% | PASS |
| esmc-600m | seq/s | 20.92 | 21.03 | +0.5% | PASS |
| boltzgen | designs/s | 0.01723 | 0.01745 | +1.3% | PASS |

No perf regression. No OOM observed through the gate targets.

### Added
- **SaProt** structure-aware protein embeddings (`tt-bio saprot`, `saprot-35m`/`saprot-650m`/`saprot-1.3b`).
- **esmc-300m** and **esmc-6b** legs in the perf-regression gate.

## [0.3.0] - 2026-07-17

First release shipping **OpenDDE** antibody-antigen co-folding (`--model opendde` / `opendde-abag`, built on the Protenix-v2 stack plus a structural-token expander), the **ESMC fused-RoPE** attention kernel (an accuracy-neutral speedup for the embed path), and opt-in **diffusion trace replay** for the Boltz-2, BoltzGen, and OpenDDE CLIs plus the Protenix-v2 Python API. Also lands the standing **perf-regression** and **UX-regression** harnesses as release-gate legs, plus the per-card performance baseline fix.

OpenDDE's antibody-antigen accuracy is weak on `9dsg`, a confirmed reference-level ceiling rather than a port bug; the device-vs-reference results for `9dsg` and `1ahw` are in `docs/implementation-parity.md`.

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0, Blackhole P150a):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.863 Å | 0.891 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 1.774 Å | 0.915 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.725 Å | 0.909 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 1.417 Å | 0.936 | ≤6.0 Å / ≥0.50 | PASS |

**BoltzGen designability** — n=4, `examples/binder.yaml`: scRMSD median 0.820 Å, 4/4 designs (100%) ≤2 Å (floor ≤2.0 Å / ≥50%) — PASS.

**ESMC embedding parity** (fused-RoPE shipped path vs reference esm, 76-residue sequence, PCC floor 0.99):

| model | per-res PCC | pooled | logits | argmax | result |
|---|---|---|---|---|---|
| esmc-300m | 0.99961 | 0.99993 | 0.99990 | 1.0000 | PASS |
| esmc-600m | 0.99964 | 0.99989 | 0.99996 | 1.0000 | PASS |

**UX gate** (`scripts/ux_regression.py`, `examples/trpcage.yaml`): every shipped surface (Boltz-2, ESMFold2, ESMFold2-fast, Protenix-v2, OpenDDE, ESMC-600m embed) cleared live-progress advancement, strict mmCIF/npz parse, and results/manifest shape — PASS.

**Perf gate** (`scripts/perf_regression.py`, Blackhole P150a, trpcage 20 aa single-sequence, 1 recycle / 10 steps / 1 sample, warm 2+5, ±15% threshold):

| model | metric | baseline | current | delta | result |
|---|---|---|---|---|---|
| boltz2 | structures/s | 1.186 | 1.190 | +0.3% | PASS |
| esmfold2 | structures/s | 1.665 | 1.705 | +2.4% | PASS |
| esmfold2-fast | structures/s | 2.271 | 2.290 | +0.8% | PASS |
| protenix-v2 | structures/s | 2.406 | 2.383 | -1.0% | PASS |
| opendde | structures/s | 1.920 | 1.922 | +0.1% | PASS |
| esmc-600m | seq/s | 21.09 | 20.92 | -0.8% | PASS |

No perf regression. No OOM observed through the gate targets.

### Added
- **OpenDDE** antibody-antigen co-folding (`opendde` / `opendde-abag`).
- **ESMC fused-RoPE** attention kernel for the embed path (accuracy-neutral speedup).
- Opt-in **diffusion trace replay** for the Boltz-2, BoltzGen, and OpenDDE CLIs and the Protenix-v2 Python API.
- **perf-regression** and **UX-regression** harnesses as standing release-gate legs.

### Fixed
- Perf gate compares against the correct per-card-type baseline (P300c vs P150a mismatch no longer reads as a false regression).

## [0.2.5] - 2026-07-11

Protenix-v2 accuracy fixes — the template embedder never ran in any real `predict` call
(`nt` always 0), and the trunk ran at 3 recycles instead of its spec 10; fixing both closes
a real delivered-RMSD gap that every PyPI install of 0.2.4 and earlier ships with. Also
includes the `embed --controller` persistent-worker dispatch and ESMC-6B multicard fanout
fix below (already hardware-gated at merge time, re-confirmed on this combined HEAD).

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.77 Å | 0.917 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 2.73 Å | 0.797 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.72 Å | 0.909 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 1.42 Å | 0.935 | ≤6.0 Å / ≥0.50 | PASS |

**Protenix-v2: 3.87 Å → 1.42 Å** — the template-embedder + recycling fixes below close the
gap to the other models; it's no longer the accuracy outlier. Boltz-2/ESMFold2/ESMFold2-fast
unchanged within seed-to-seed noise vs 0.2.4.

**BoltzGen designability** — n=4, `examples/binder.yaml`: scRMSD median 0.67 Å, 4/4 designs
(100%) ≤2 Å — no regression vs 0.2.4's n=8 measurement (0.84 Å median, 7/8 ≤2 Å).

No OOM: `examples/615.yaml` and `examples/1303.yaml` (Boltz-2 `--fast`) completed cleanly;
the full supported range to `examples/3233.yaml` (4-chain multimer + ligand) was already
verified OOM-free on this unchanged Boltz-2 code. No perf
regression: Boltz-2 `--fast` warm e2e at L=615 is **46.5 s**, vs the 43.4 s 0.2.4-era
baseline — within run-to-run/environment noise on the same unchanged code path.

### Fixed
- **Protenix-v2: template embedder never ran** — `nt` (template count) was always 0 in
  every real `predict` call, so the template-embedder pass was silently skipped.
- **Protenix-v2: `recycling_steps` default 3 → 10** — the trunk now runs at its spec
  recycle count (previously reused Boltz-2/ESMFold2's default of 3); the correct
  default once the template-embedder fix above made recycling actually informative.
  This makes Protenix-v2 slower per-fold than 0.2.4 (more recycles) — expected,
  not a regression; see the gate wall-clock above.
- ESMC-6B `--devices` fanout regression past 2 cards, root-caused to two independent
  host-side bottlenecks (both fixed, verified bit-exact, end-to-end scaling now
  monotonic to 4 cards):
  - **Redundant weight loading**: the N data-parallel workers now share one host-tiled
    copy of the 24 GB checkpoint via a `/dev/shm` cache (`esmc.load_esmc6b_shared` +
    `tenstorrent.weight_cache`) instead of each independently reading+tiling it.
    Per-worker load drops from ~10–16 s (∝N, bandwidth-contended) to ~2.2 s.
  - **Host CPU thread-pool oversubscription**: each shard subprocess's torch/OMP/BLAS
    pools defaulted to *all* host cores, so N co-resident shards oversubscribed the
    host (~21 loadavg on a 16-core host at N=4). `esmc._thread_cap_env` caps them to
    `cores // n_workers`, mirroring the existing `main._cap_worker_threads` fix for
    the fleet worker pool.
  - Net: esmc-6b/N=256 on qb2 goes from 0.66x@4-cards (regression) to **1.49x@4-cards**
    (monotonic 1.00x → 1.33x → 1.43x → 1.49x). Bit-exact vs single-card
    (`scripts/esmc6b_shared_cache_parity.py`, `scripts/esmc_multicard_parity.py`,
    max|Δ|=0); all other models and the single-card path are unchanged.

### Added

- `tt-bio embed --controller URL`: dispatch to a persistent `tt-bio controller`/`worker`
  pool instead of spawning per-call subprocesses. A worker's ESMC model stays resident
  across calls, so the weight reload that dominates `--devices` wall-clock for
  `esmc-6b` becomes a one-time cost per worker
  lifetime instead of a per-invocation tax (measured: esmc-6b N=48 50.0s cold -> 9.1s
  warm on 1 card, 261s cold -> 13.4s warm on 2 cards; bit-exact vs single-shot). Reuses
  the existing predict/design scheduler/lease machinery (`tt_bio/distributed.py`,
  `tt_bio/worker.py`) — no new dispatch mechanism. `--devices` (per-call subprocess
  fanout) is unchanged and still the right choice for one-off invocations with no
  standing controller.

### Measured
- Re-measured `esmc-300m`/`esmc-600m` `--devices` wall-clock scaling on qb2 post
  thread-cap fix (N=48/256/4096): the original
  table's `esmc-600m/N=256` 3-card 0.62x cliff does not reproduce (now a 0.87x dip,
  within run-to-run noise) — no regression for either model at any previously-fine
  config. New finding: both models scale far more modestly on qb2 (~1.1x@4cards for
  N=4096) than the original table's qb1 numbers (~2x), most likely because `embed
  --devices` pays an extra per-shard mesh-topology setup cost on qb2 that `esmc-6b`'s
  large weight load absorbs but these smaller models don't — also surfaced that
  `embed --devices` with >1 device currently TT_FATALs out-of-the-box on qb2 unless
  `TT_MESH_GRAPH_DESC_PATH` is set manually (the `predict` path already handles this
  P300-board-misdetection quirk automatically; `embed`'s fanout path doesn't yet).
  Parity re-verified bit-exact for both models.

## [0.2.4] - 2026-07-10

Device-resident trunk for `tt-bio gen` (BoltzGen) — no structure-model code changed for
Boltz-2/ESMFold2/Protenix-v2 (the new `TokenDistanceRecycle`/`TrunkModule` params default to
off/`None`, purely additive).

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.43 Å | 0.944 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 2.76 Å | 0.798 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.74 Å | 0.907 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 3.87 Å | 0.706 | ≤6.0 Å / ≥0.50 | PASS |

No regression vs 0.2.3 (within TT diffusion's seed-to-seed variance band).

**BoltzGen designability** — n=8 fixed-length-100 designs, `examples/binder.yaml`: scRMSD
median 0.84 Å (resident) vs 0.91 Å (host), 7/8 designs ≤2 Å strict pass (comparable to host's
8/8) — no regression. Wall-clock (design + refold + confidence + analysis + filtering) **697 s
→ 479 s, ~31% faster**.

### Added
- **BoltzGen device-resident trunk** — `TokenDistanceRecycle` (mirrors `TemplateRecycle`) keeps
  the per-iteration token-distance injection fully on-device, collapsing 4 host↔device
  crossings/iteration to 2 (only the template sub-module still round-trips). `Boltz.__init__`
  takes `use_resident_trunk: bool = True`; set `false` to fall back to the original host path.

### Changed
- Promoted Protenix-v2's diffusion denoiser-unit and `AttentionPairBias(has_s=True)` ad-hoc
  checks to proper pytest cases (test-coverage only, no functional change).

## [0.2.3] - 2026-07-09

Multi-card fanout parity for `predict`, a designability (scRMSD) verify script for `tt-bio gen`,
and `tt-bio embed` input/UX polish. No structure-model code changed vs 0.2.2 (`tt_bio/boltz2.py`,
`protenix.py`, `esmfold2.py`, `tenstorrent.py` are byte-identical) — only `esmc.py` and the CLI
(`main.py`) changed, so the release gate below is a confirmation run, not a re-verification.

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.60 Å | 0.931 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 2.28 Å | 0.832 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.74 Å | 0.907 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 3.87 Å | 0.706 | ≤6.0 Å / ≥0.50 | PASS |

Full test suite: 71 passed, 46 skipped (missing optional reference checkpoints/packages, same
gap as prior releases), 0 failed. No OOM: `examples/615.yaml` and `examples/1303.yaml`
(Boltz-2 `--fast`) completed cleanly; the full supported range up to `examples/3233.yaml`
(4-chain multimer + ligand) was already verified OOM-free on this same unchanged model code.
No perf regression: Boltz-2 `--fast` warm e2e at L=615 is
**43.4 s**, matching the 0.2.2-era baseline exactly (same code path since before 0.2.2).

### Added
- **`tt-bio predict --devices`** — alias for `--device_ids` (comma-separated card ids), matching `tt-bio embed`'s flag name; `--device_ids` still works for back-compat.
- **BoltzGen designability (scRMSD) verify script** — `scripts/boltzgen_designability.py` harvests the self-consistency RMSD `tt-bio gen` already computes and summarizes/gates on it; see `docs/boltzgen-designability.md`.
- **`tt-bio embed --devices` wall-clock scaling measured** — real ~2x @ 4 cards for `esmc-600m` on large batches, but flat/worse for small batches and for `esmc-6b` beyond 2 cards (concurrent weight-load contention); README softened to match. Performance-only finding, no change to the (already bit-exact) sharding correctness.

### Changed
- **`tt-bio embed` input handling** — `DATA` now also accepts a YAML `{id: sequence}` mapping or a bare sequence string (previously FASTA file/directory only), writes a `manifest.json` (model/pool/shapes/dtype + which output file holds each sequence) alongside the embeddings, and reports bad input as a one-line error instead of a raw traceback.

## [0.2.2] - 2026-07-09

Turns MSA on by default for Boltz-2 / Protenix-v2 (the fix for the misleading no-MSA
accuracy result) and ships the ESMC multi-card embedding fanout. No model numerics changed
vs 0.2.1 — the MSA compute path was already hardware-gated; this only flips its default and
adds a local-DB→online fallback with a privacy notice, plus a `--single_sequence` opt-out.
Ground-truth gate on the default path (`examples/prot.yaml`): Boltz-2 CA-RMSD 2.49 Å / TM
0.78, Protenix-v2 3.47 Å / TM 0.75.

### Added
- **Multi-card fanout for `tt-bio embed`** — `--devices 0,1,2,3` (CLI) / `devices=[...]` (`tt_bio.esmc.embed`) shards a sequence set across several TT cards, one pinned worker per card, and reassembles the embeddings in input order. Data-parallel and lossless: each shard's output is bit-exact to the single-card path (verified on-hardware, Δ=0 per-residue/pooled/logits).
- **`--single_sequence` flag** for `predict` — deliberately fold Boltz-2/Protenix-v2 without an MSA (skips both the local-DB lookup and the online fallback), for batch-screening orphan sequences.

### Changed
- **Boltz-2 and Protenix-v2 use an MSA by default** — these MSA-dependent models no longer silently fold single-sequence. With no MSA flags, `predict` uses a local ColabFold DB (`~/.boltz/msa_db`) if present, else falls back to the online ColabFold server and prints a one-line notice naming the server the sequences are sent to (they leave the machine). Pass `--msa_db_path` for a private offline DB, or `--single_sequence` to skip the MSA. ESMFold2 / ESMFold2-Fast are unchanged (single-sequence by design). Ground-truth gate on the default path (`examples/prot.yaml`): Boltz-2 CA-RMSD 2.49 Å / TM 0.78, Protenix-v2 3.47 Å / TM 0.75.

## [0.2.1] - 2026-07-09

Adds the ESMC embeddings capability merged since 0.2.0 (already hardware-gated at merge
time) and fixes packaging/docs metadata that was stale since 0.2.0. No model code changed
for existing capabilities — the 0.2.0 accuracy/perf/OOM gate still holds.

### Added
- **ESMC protein-language-model embeddings** — `tt-bio embed` CLI + Python API
  (`tt_bio.esmc.embed`): per-residue and pooled embeddings from ESMC-300M/600M/6B, no
  folding head or MSA required. Parity vs reference ESMC: per-residue/pooled PCC
  0.9995-0.9999 across variants (normal and `--fast`).
- Automatic batching + length-bucketing for `tt-bio embed` on ESMC-300M/600M (~18.5x warm
  throughput vs unbatched); exact row-independence (masked batched output bit-identical to
  running each sequence alone), PCC 0.9996+.

### Fixed
- `pyproject.toml` `description` was still "Boltz-2 implementation..." — now lists every
  shipped capability (Boltz-2, ESMFold2, Protenix-v2, BoltzGen, ESMC).
- `pyproject.toml` had no `readme` field, so the PyPI project page rendered with an empty
  long description — now points at `README.md`.
- README: `pip install tt-bio` (PyPI) is now the primary install path (the wheel has been
  on PyPI since 0.2.0); git/source moved to a secondary section. Intro paragraph now
  mentions ESMC embeddings. The dense Boltz-2/ESMFold2/Protenix-v2 feature-support
  paragraph is now a compact table.

## [0.2.0] - 2026-07-09

Release gate verified on Blackhole (p150a): Protenix-v2 e2e real-weight parity (seed0-vs-reference
Kabsch RMSD 8.7 Å, within the sampler's own seed-to-seed variance band); Protenix component parity
14/14, Boltz-2 13/13, ESMFold2 plddt/distogram parity, host suite green; no OOM across the supported
size range.

### Added
- **Protenix-v2 denoise ttnn trace** — opt-in `fold(trace=True)` (with
  `get_device(trace_region_size=1 << 30)`): captures and replays the dispatch-bound
  denoise stream. Lossless (bit-exact vs untraced) and ~22% faster warm diffusion at L256,
  a larger end-to-end win as `diffusion_samples` grows.

### Changed
- Trace/device toggles are now normal function arguments (`fold(trace=...)`,
  `get_device(trace_region_size=...)`) instead of environment variables.

### Fixed
- Input validation hardening: unique chain ids past 26 chains, reject inputs that share a
  name stem, keep blank-id FASTA chains, reject empty polymer sequences, and validate
  explicit `--device_ids` against the cards actually present.
- `tt_bio.__version__` now reports the installed `tt-bio` version (previously read the wrong
  package and could be undefined).
- README/docs consistency pass (flags, examples, model list).

## [0.1] - initial
- Boltz-2, ESMFold2, Protenix-v2 structure prediction and BoltzGen binder design on
  Tenstorrent Blackhole / Wormhole, single- and multi-card. Installed from source.
