# SC26 booth demo

A protein folding live on four Tenstorrent Blackhole chips, drawn as point clouds and molecular
surfaces. Runs fully offline on one QuietBox 2.

Served at http://127.0.0.1:8626/ with its frame stream at `/stream` on the same origin.
The stream format is in [PROTOCOL.md](PROTOCOL.md); booth staff read [BOOTH.md](BOOTH.md).

| path | what |
|------|------|
| `engine/` | fold workers, one per chip, and the stream server |
| `gallery/` | curated proteins and their recorded trajectories |
| `web/render/` | WebGL renderer: points, cartoon, camera |
| `web/app/` | the kiosk app, attract loop and interaction |
| `hardware/` | per-chip telemetry |
| `ops/` | boot-to-demo service, watchdogs, recovery |

The model hooks the demo uses are flag-gated and off by default; nothing here changes a fold's
result.

## Models on screen

The left edge lists every model tt-bio runs on Tenstorrent hardware, grouped by what it does
(structure, design, embeddings, affinity). The list is tt-bio's own model registry
(`PREDICT_MODELS`, `DESIGN_MODELS`, `EMBED_MODELS`, `SAPROT_MODELS` and `AFFINITY_MODELS` in
`tt_bio/main.py`), one entry per model family, without Protenix-v2, whose weights' licence is
unresolved. Boltz-2 is lit: it is the model the booth runs.

Every fold on the stage is Boltz-2. The four chips fold the attract proteins and visitors' names
live, each chip with its own resident copy, and between them the screen plays larger Boltz-2
complexes recorded on this box. The attract proteins read MSAs searched ahead of time
(`engine/msa/`); a name folds from its sequence alone, so the booth never needs the network.

The engine's `--models` option is the switch (default `boltz2`). Give it several, for example
`--models boltz2,esmfold2`, and the chips take turns, recordings of every listed model play, and
each fold and chip row names its model again.

## The claim

The title is "Unprecedented Throughput per Dollar", with "The first unified software stack for bio
models optimized from silicon to serving" under it. The footer gives the measured basis: Boltz-2
(tt-bio's defaults: 3 recycles, 200 sampling steps, as on screen) on human serum albumin residues
1-300 from the sequence alone, the four chips kept busy through the booth engine, finished 72 folds
in 180 s, 0.40 a second, median 9.79 s a fold, with the chips at a median 1350 MHz
(`engine/throughput.py`; result in `claim/tt-quietbox2-boltz2-hsa300.json`, 4 Oct 2026). The QR
code goes to https://tt-bio.com.

## Time on screen

Every number on screen that counts seconds counts real seconds, in step with what you see.

* **A fold on the stage has already finished on its chip.** Its time is a measured fact and is
  shown still from the first frame: "Folded live on chip 4 in 5.62 s". Nothing on the stage counts.
  What moves is the sampler's own step counter ("Diffusion step 7 of 200"), and the line under it
  says how the steps are paced: "Replayed 6× slower than the chip ran it", or "at the chip's own
  pace" when the diffusion took six to nine seconds and plays in real time.
  One exception, stated here: the first time a chip meets a new size it compiles inside one
  sampler step, which can take 50 times as long as the others. A step longer than five times the
  fold's median step is replayed at the median, so nobody watches still noise for seconds; the
  fold's measured time still includes it.
* **A recorded gallery fold shows no time.** It reads "Recorded on this box" with its size and
  clock. The recorder wrote all 200 sampler steps to disk, which made its diffusion 16 to 35 s
  against about 3 s live, so its seconds would undersell the chip.
* **Your own fold is the one running clock.** From the moment a chip takes it, the counter shows
  wall-clock seconds and stops when the fold lands. It is then replaced by the chip's own measured
  time, which can differ from the counter by a few tenths of a second of network and page latency.
* **The chip rows have no seconds.** Each row names what the chip is folding, which part of the
  model is running (trunk, diffusion, confidence) and a bar for how far along it is.
  The hardware view behind Tab follows the same rule: a chip that is folding shows the protein and
  its length, and seconds only for a fold that has finished ("last: 56 residues in 0.6 s").
