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
unresolved. The model of the fold on the stage is lit.

Only two of them are on the stage, because only two have real sampler states here: ESMFold2 folds
live on the four chips, and Boltz-2 plays folds recorded on this box. The attract loop never shows
the same model twice in a row while a fold by the other is waiting, so both appear within half a
minute. Every chip row, here and behind Tab, names the model it is running.

## Time on screen

Every number on screen that counts seconds counts real seconds, in step with what you see.

* **A fold on the stage has already finished on its chip.** Its time is a measured fact and is
  shown still from the first frame: "Folded live on chip 4 in 5.62 s", or "Folded on this box in"
  for a recorded gallery fold. Nothing on the stage counts. What moves is the sampler's own step
  counter ("Diffusion step 7 of 14"), and the line under it says how the steps are paced: "Replayed
  6× slower than the chip ran it", or "at the chip's own pace" when the diffusion took six to nine
  seconds and plays in real time. A diffusion that took longer than nine seconds (a large Boltz-2
  complex) plays over nine seconds and says so: "Replayed 4× faster than the chip ran it".
  One exception, stated here: the first time a chip meets a new size it compiles inside one
  sampler step, which can take 50 times as long as the others. A step longer than five times the
  fold's median step is replayed at the median, so nobody watches still noise for seconds; the
  fold's measured time still includes it.
* **Your own fold is the one running clock.** From the moment a chip takes it, the counter shows
  wall-clock seconds and stops when the fold lands. It is then replaced by the chip's own measured
  time, which can differ from the counter by a few tenths of a second of network and page latency.
* **The chip rows have no seconds.** Each row names what the chip is folding, which part of the
  model is running (language model, trunk, diffusion, confidence) and a bar for how far along it is.
  The hardware view behind Tab follows the same rule: a chip that is folding shows the protein and
  its length, and seconds only for a fold that has finished ("last: 56 residues in 0.6 s").
