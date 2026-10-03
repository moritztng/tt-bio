# SC26 stream protocol, version 1

The engine (`demo/sc26/engine/server.py`) serves everything from one local origin,
`http://127.0.0.1:8626/`. The app's static files are served from `demo/sc26/web/`. The stream is a
WebSocket at `ws://127.0.0.1:8626/stream`. Nothing goes over the network.

Every message is one JSON object with a `type`. The server sends the same messages to every
connected browser, so two screens show the same thing.

## Coordinates

Coordinates are little-endian float32, base64-encoded, `x y z` per atom in atom order, in
Angstrom. Decode with:

```js
const xyz = new Float32Array(Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer);
```

They are the sampler's raw numbers, never smoothed or interpolated. Each frame also carries a rigid
transform `R` (row-major 3x3) and `T` (3) for display: `display = R * raw + T`. The diffusion
sampler rotates the structure randomly at every step, and the transform undoes that by aligning
each frame onto the previous one, so the picture holds still while the structure forms. It is a
camera, not an edit. Apply it to `xyz` and to `x0`.

The final frame's `xyz` is the scored structure bit for bit, and so is `fold_done.xyz`.

## Server to browser

### `hello`
Sent once on connect. Carries the `status` fields below plus `protocol: 1`.

### `status`
Once a second.
```json
{"type":"status","queue":0,"replays":12,"t_wall":1790985482.4,
 "chips":[{"chip":0,"state":"busy","job":"a17","aiclk_mhz":1350,"folds":212,"restarts":0,
           "last_fold":{"n_res":118,"seconds":2.1,"aiclk_mhz":{"min":1331,"median":1350,"max":1350,"n":9}}}]}
```
`state` is one of `starting`, `warming` (loading weights, compiling), `ready`, `busy`,
`stalled` (the watchdog is stopping it), `recovering` (restarting), `stopped`. `aiclk_mhz` is
read from the chip's sysfs clock when the message is built; `null` means the chip did not answer.

### `chip`
A chip changed state. `{"type":"chip","chip":2,"state":"recovering","aiclk_mhz":800,"rc":0}`.

### `fold_start`
A fold began. Everything the renderer needs to lay out atoms before the first frame.
```json
{"type":"fold_start","id":"a17","chip":0,"kind":"attract","source":"live","model":"esmfold2",
 "sequence":"MTYKLILNG...","n_res":56,"n_atoms":436,"steps":20,"loops":3,"seed":0,
 "rg_expected":10.9,
 "atoms":{"element":["N","C","C","O",...],"name":["N","CA","C","O",...],"residue":[0,0,0,0,...]}}
```
* `kind`: `attract` (the box folding its own list), `visitor` (somebody typed it), `replay`.
* `source`: `live` (a chip is computing it now) or `replay` (a recording). The screen must show
  which; the audience will ask.
* `atoms.residue` is the 0-based residue index of each atom. `rg_expected` is a radius of gyration
  estimate in Angstrom from the length alone (2.2 * n^0.38), for framing the camera before the
  real structure exists. Replays can do better: their last frame is known in advance.

### `stage`
Where the fold is. `{"type":"stage","id":"a17","chip":0,"stage":"trunk","step":1,"total":3,"t":0.41}`.
For ESMFold2 the stages run in order: `lm` (the ESMC-6B language model reads the sequence),
`trunk` (the folding trunk, `total` recycles), `diffusion` (the structure sampler, one `frame`
per step), `confidence` (pLDDT). `t` is seconds since the fold started.

### `frame`
One real state of the diffusion sampler.
```json
{"type":"frame","id":"a17","chip":0,"step":3,"of":14,"t":0.52,
 "xyz":"<base64 f32>","x0":"<base64 f32>","R":[1,0,0,0,1,0,0,0,1],"T":[0,0,0]}
```
* `step` runs from `-1` (pure noise, before the first step) to `of - 1` (the final structure), so
  a fold sends `of + 1` frames.
* `xyz` is the sampler's current state: noise that condenses into the protein.
* `x0` is the network's prediction of the finished structure at this step: already
  protein-shaped early on, sharpening as noise falls. `null` on step `-1`. Both are real. Pick
  one and say which in the UI copy if it matters; `xyz` is the more literal "emerging from noise".
* On the live path frames arrive faster than anyone can watch (a 200-residue fold's 15 frames
  arrive in about 0.25 s). Use `t` for honest timing labels, and pace the animation yourself.

### `fold_done`
```json
{"type":"fold_done","id":"a17","chip":0,"kind":"attract","source":"live","model":"esmfold2",
 "n_res":56,"seconds":0.86,"stages":{"lm":0.16,"trunk":0.46,"diffusion":0.17,"confidence":0.07},
 "aiclk_mhz":{"min":1325,"median":1350,"max":1350,"n":4},
 "xyz":"<base64 f32>","plddt":[0.91,0.93,...],"ptm":0.71}
```
* `seconds` is wall time on the chip from input to confidence, with the chip's AICLK sampled during
  the fold (`aiclk_mhz`, MHz, every 0.2 s). Any number on screen should come from here.
* `plddt` is per residue, 0 to 1. Confidence exists only at the end; frames have none.
* Apply the last frame's `R`/`T` to `xyz` to land exactly where the animation ended.

### `fold_error`
`{"type":"fold_error","id":"a17","chip":0,"reason":"preempted"}`. Reasons: `preempted` (an attract
fold gave its chip to a visitor), `chip_lost` (the worker died; a visitor's fold is retried once on
another chip and gets a new `fold_start` with the same `id`), `stopped` (watchdog or shutdown), or
an error message for bad input. Drop the fold's frames and move on; never show the reason text to
the audience.

### `queued` and `rejected`
Answers to a browser's `fold` request, sent only to that browser.
`{"type":"queued","id":"v31","position":1,"n_res":118}` or
`{"type":"rejected","reason":"length","min":10,"max":400}` / `{"reason":"letters","allowed":"ACDE..."}`.

## Browser to server

* `{"type":"fold","sequence":"MTYKLILNG..."}`: fold this. 10 to 400 of the 20 standard amino-acid
  letters; whitespace is ignored and case does not matter. A visitor's fold jumps the queue. If every
  chip is busy and the visitor is still waiting 1 s later, one attract fold is dropped (`fold_error`
  `preempted`) at its next sampler step and the visitor's fold takes the chip. Measured with two
  chips: a 76-residue visitor fold's first frame arrived 1.86 s after the request, done at 2.07 s.
* `{"type":"status"}`: send a `status` now.

`POST /fold` with `{"sequence": ...}` and `GET /status` do the same over plain HTTP.
`GET /telemetry` returns the per-chip clock, power, temperature and fold ledger, in the format
`hardware/README.md` gives for the `chips` message. The chip lanes poll it twice a second.

## Replay

A recording is the live fold's messages, one JSON object per line (`*.jsonl`), exactly as they were
streamed. The server plays recordings from `demo/sc26/gallery/trajectories/` and
`demo/sc26/engine/recordings/` through the identical protocol, with the recorded timing (gaps
capped at 2 s), `source: "replay"`, `chip: null` and `recorded: "<file stem>"`. It does so
whenever no chip is ready or busy, so the screen is never blank. To develop the frontend with no
chip at all:

    python3 demo/sc26/engine/server.py --replay-only

Every finished live fold is saved in this format under `demo/sc26/engine/runs/recorded/`, so the
gallery is grown by running the box.
