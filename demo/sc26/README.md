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
unresolved. OpenFold3 and OpenBind-0 lead the structure models. The dot marks the model the chips
run, read from the engine's `--models`, so it moves with the engine and never needs editing here.

Every fold on the stage is OpenFold3 (preview2, Apache-2.0). The chips fold the same eleven
proteins the stage shows, from insulin (51 residues) to a T cell receptor on HLA (833 residues,
five chains), each chip with its own resident copy, and visitors' names when `?visitors=1` turns
typing back on (it is off for now). Each protein's live fold takes its recording's place on the
stage; the recordings, made on this box through the same chip worker, fill the stage until then.
`gallery/build.py` writes the rotation (`engine/attract.json`) from the gallery's picks, so the two
never disagree. A fold over 400 residues takes 35 to 94 s and can only give its chip to a visitor
between two trunk recycles, up to 14 s apart, so at most all chips but one are on one: one chip
always turns over every 5 to 13 s, and a visitor takes that chip first. The proteins read MSAs
searched ahead of time (`engine/msa/`, written by `engine/msa_search.py`); a name folds from its
sequence alone, so the booth never needs the network. A fold reports its stages as the chip finishes them: input preparation, each trunk
recycle, each of the 200 diffusion steps, the confidence head (the worker synchronises the device
before it stamps a stage, because ttnn returns before the chip is done).

The engine's `--models` option is the switch (default `openfold3`). Give it several, for example
`--models openfold3,boltz2`, and the chips take turns, recordings of every listed model play, and
each fold and chip row names its model again.

## The claim

The title is "Biology Models at Unprecedented Throughput per Dollar", with "One open stack for every
model, inference and training, from a single card to a Galaxy supercluster" under it, each one line
at 1080p and 4K (Moritz, 5 Oct 2026). The type scale steps about 1.25 from a 40 px title, and the
frame keeps the same 72 px margin on all four sides (measured on the rendered page, both sizes). The
measurement behind the throughput claim: Boltz-2
(tt-bio's defaults: 3 recycles, 200 sampling steps, as on screen) on human serum albumin residues
1-300 from the sequence alone, the four chips kept busy through the booth engine, finished 72 folds
in 180 s, 0.40 a second, median 9.79 s a fold, with the chips at a median 1350 MHz
(`engine/throughput.py`; result in `claim/tt-quietbox2-boltz2-hsa300.json`, 4 Oct 2026). The
screen prints neither the basis nor the clock; both stay in the engine's logs.

Two QR codes sit at the bottom left, the same size (37 modules with the quiet zone, 2 frame pixels
a module): "Learn more" goes to https://github.com/moritztng/tt-bio and "Try it" to
https://japanfold.aiand.com.

## Time on screen

Every number on screen that counts seconds counts real seconds, in step with what you see.

* **A fold on the stage has already finished on its chip.** Its time is a measured fact and is
  shown still from the first frame: "Folded live on chip 4, just now, in 5.62 s". Nothing on the stage counts.
  What moves is the sampler's own step counter ("Diffusion step 7 of 200"), and the line under it
  says how the steps are paced: "Replayed 6× slower than the chip ran it", or "at the chip's own
  pace" when the diffusion took six to nine seconds and plays in real time.
  One exception, stated here: the first time a chip meets a new size it compiles inside one
  sampler step, which can take 50 times as long as the others. A step longer than five times the
  fold's median step is replayed at the median, so nobody watches still noise for seconds; the
  fold's measured time still includes it.
* **Live or recorded is one rule.** The stage shows each protein's newest live fold from the chips;
  a recording only stands in for a protein no chip has folded since the page started (the first
  minutes after a restart, or a chip out of service), and a recording never displaces a live fold.
  The live label says how long ago the chip finished it ("Folded live on chip 2, 3 min ago, in"), and
  a recording says why it is one ("A recording: the chips have not folded this one yet").
* **A recorded gallery fold shows the time it took when it was recorded**: "Recorded on this box
  in 34.81 s". The recorder is the booth's chip worker with a booth worker's CPU share, so these are
  booth times; the AICLK during each is in `gallery/manifest.json`.
* **Your own fold is the one running clock.** From the moment a chip takes it, the counter shows
  wall-clock seconds and stops when the fold lands. It is then replaced by the chip's own measured
  time, which can differ from the counter by a few tenths of a second of network and page latency.
* **The chip rows show only what the chips report.** Each row names the protein, the stage the chip
  last reported with the chip's own counter in it ("trunk 2/4", "diffusion 143/200"), and the
  seconds since the chip took the fold, counted from the chip's own timestamps. The bar is the
  sampler's steps and moves only when the chip reports one. Under it is the measured time of the
  fold the chip finished last ("Last: Haemoglobin, 574 amino acids in 37.04 s", measured on chip 1 on 5 Oct). Nothing is
  forecast: no row guesses how long a fold will take. A chip that has sent nothing for 30 s (twice
  the longest real gap, one trunk recycle at 833 residues) says "no word for 31 s" instead of
  counting; at 120 s the engine stops the fold and the row says so, then "Resetting its board".
  A row also says "Warming up, compiling for Haemoglobin", "Ready", "Out of service" (a chip taken
  out of the demo with the engine's `--out-of-service`) or "Not in the demo".
  The hardware view behind Tab follows the same rule: a chip that is folding shows the protein and
  its length, and seconds only for a fold that has finished ("last: 56 residues in 0.6 s").
