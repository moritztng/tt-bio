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

## Time on screen

Every number on screen that counts seconds counts real seconds, in step with what you see.

* **A fold on the stage has already finished on its chip.** Its time is a measured fact and is
  shown still from the first frame: "Folded live on chip 4 in 5.62 s", or "Folded on this box in"
  for a recorded gallery fold. Nothing on the stage counts. What moves is the sampler's own step
  counter ("Diffusion step 7 of 14"), and the line under it says how the steps are paced: "Replayed
  6× slower than the chip ran it", or "at the chip's own pace" when the diffusion took longer than
  six seconds and plays in real time.
* **Your own fold is the one running clock.** From the moment a chip takes it, the counter shows
  wall-clock seconds and stops when the fold lands. It is then replaced by the chip's own measured
  time, which can differ from the counter by a few tenths of a second of network and page latency.
* **The chip rows have no seconds.** Each row names what the chip is folding, which part of the
  model is running (language model, trunk, diffusion, confidence) and a bar for how far along it is.
