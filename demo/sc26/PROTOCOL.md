# SC26 stream protocol, version 2

The engine (`demo/sc26/engine/server.py`) serves everything from one origin,
`http://127.0.0.1:8626/`. The app's static files are served from `demo/sc26/web/`. The stream is a
WebSocket at `ws://127.0.0.1:8626/stream`. The page may be remote, over a thin link: see "Slow
links" at the end.

Every stream message is one small JSON object with a `type`. The stream carries no coordinates.
The server sends the same messages to every connected browser, so two screens show the same thing.
A finished fold's coordinates are pulled once, by the page that wants them, with `GET /fold/<id>`.

Version 1 sent every sampler state as base64 float32 `xyz` and `x0` to every page as it was
computed: 214 KB per frame and 43 MB per fold at 833 residues, 21 Mbit/s on average with three
chips folding. Version 2 sends 8.2 MB per fold at 833 residues to a page that pulls every state,
and 1.2 MB to a page that pulls every 8th.

## Coordinates

Angstrom, `x y z` per atom in atom order, little-endian.

* The **final structure**, the one that is scored, is float32, bit for bit what the model output.
* **Every other sampler state** is int16. The chip worker (`engine/trajectory.py`) superposes each
  state onto the final structure, with the rotation fitted on the state's own `x0` (the network's
  denoised estimate, which shares the state's random frame but already has the protein's shape;
  `x0` itself never leaves the box). It then quantises the state around its own centroid
  `origin[i]` with a step `scale[i]` of its own extent / 32767. Decoded: `xyz = q * scale[i] + origin[i]`.
  The error is at most half a step: 0.15 A on the pure-noise state of an 833-residue fold, whose
  cloud spans thousands of Angstrom; 0.03 A at step 20; 0.01 A at step 60; under 0.002 A from step
  100 on; 0 on the final structure. Rendered at 1920x1080 next to the float32 path, the two are
  indistinguishable (`state/bth-stream.md`, FORMAT).

The diffusion sampler rotates the structure randomly at every step (120-145 degrees between
consecutive raw states, measured), so the superposition is a camera, not an edit: without it the
cloud spins. `demo/sc26/science/rotation.py` measures what is left.

## Server to browser

### `hello`
Sent once on connect. Carries the `status` fields below plus `protocol: 2` and `folds`: the
`fold_done` summary of every fold the page can pull now, recordings included, so a page that
loads fills its stage at once.

### `status`
Every 2 s.
```json
{"type":"status","queue":0,"replays":11,"t_wall":1790985482.4,
 "chips":[{"chip":0,"state":"busy","job":"a17","aiclk_mhz":1350,"folds":212,"restarts":0,
           "doing":{"id":"a17","kind":"attract","name":"Haemoglobin","n_res":574,"t_wall":1790985461.0,
                    "plan":[["taken",0],["start",2.0],["trunk",3.77],["diffusion",18.88],["confidence",32.15],["done",34.82]]},
           "last_fold":{"name":"Lysozyme","n_res":130,"seconds":8.31,"t_wall":1790985460.9,
                        "aiclk_mhz":{"min":1331,"median":1350,"max":1350,"n":9}}},
          {"chip":2,"state":"out_of_service"}]}
```
`doing` is the fold the chip holds, with the engine's time when the chip took it, so a page that
connects mid-fold can name it and count its seconds against the message's own `t_wall`, on one
clock. `plan` is how long this fold is expected to take on this box, the chip seconds at which each
event of it is expected: `taken` (the chip took the job), `start` (`fold_start`, the input is
prepared), the first `stage` event of each stage, and `done` (`fold_done.seconds`). It is the median of
the last five folds of this length, or interpolated between the nearest lengths, measured from the
gallery recordings and every live fold since the engine started (engine/stages.py). The chip bars
draw each fold against it; `null` until the model has finished a fold. A warming chip carries `warming`: `{"name","n_res","stage"}` of the warm-up fold it last
reported. A chip named in the engine's `--out-of-service` is listed with that state only.
`state` is one of `starting`, `warming` (loading weights, compiling), `ready`, `busy`, `out_of_service`,
`stalled` (the watchdog is stopping it), `recovering` (restarting), `resetting` (its board is
being reset with `tt-smi -r`; both chips of the board show it, and the clock reads `null` or 800
until it is back), `stopped`. `aiclk_mhz` is
read from the chip's sysfs clock when the message is built; `null` means the chip did not answer.

### `chip`
A chip changed state. `{"type":"chip","chip":2,"state":"recovering","aiclk_mhz":800,"rc":0}`.
When the worker takes a job the state is `busy` with the job and its `doing` (as in `status`, with the
`plan`): that moment is zero on the chip's clock for this fold, the `t` of every event that follows.
During warm-up the worker repeats `warming` at most every 5 s from inside each warm-up fold, with
what it is compiling for: `{"type":"chip","chip":0,"state":"warming","name":"Haemoglobin","n_res":574,"stage":"trunk"}`.
While its board is resetting the state stays `resetting` and `worker` carries what the worker
itself reported: `{"type":"chip","chip":3,"state":"resetting","worker":"recovering"}`.

### `reset`
A board reset finished. `{"type":"reset","chips":[2,3],"rc":0,"seconds":44.0}`. `rc` 0 means both
chips answered afterwards; the lanes then go `recovering`, `warming`, `ready` as the workers come
back. A worker the watchdog stalled resets its board at once, since the chip is still inside a
device call; a worker that exits uncleanly by itself resets it on the second such exit in a row.
Anything else leaves them `recovering` and the engine retries later (at most one reset per
board every 10 minutes). For the operator log; nothing on screen needs it.

### `fold_start`
A fold began.
```json
{"type":"fold_start","id":"a17","chip":0,"kind":"attract","source":"live","model":"esmfold2",
 "sequence":"MTYKLILNG...","n_res":56,"n_atoms":436,"steps":20,"loops":3,"seed":0,
 "rg_expected":10.9}
```
The worker's `fold_start` also carries `atoms`; the engine keeps it for `GET /fold` and sends the
rest.
* `kind`: `attract` (the box folding its own list), `visitor` (somebody typed it), `replay`.
* `source`: `live` (a chip is computing it now) or `replay` (a recording). The screen must show
  which; the audience will ask.
* In the `GET /fold` metadata, `atoms.residue` is the 0-based residue index of each atom (a ligand is one residue; OpenFold3
  also sends `atoms.chain`, the chain id of each atom). `rg_expected` is a radius of gyration
  estimate in Angstrom from the length alone (2.2 * n^0.38), for framing the camera before the
  real structure exists. Replays can do better: their last frame is known in advance.

### `stage`
Where the fold is. `{"type":"stage","id":"a17","chip":0,"stage":"trunk","step":1,"total":3,"t":0.41}`.
For ESMFold2 the stages run in order: `lm` (the ESMC-6B language model reads the sequence),
`trunk` (the folding trunk, `total` recycles), `diffusion` (the structure sampler, one `frame`
per step), `confidence` (pLDDT). `t` is seconds since the fold started.
For OpenFold3, the booth's model, there is no `lm`: `trunk` steps 1 to `total` are the recycles
(4 by default, the first pass and 3 recycles), then `diffusion` steps 1 to 200, then `confidence`.
The worker prepares the input (features, MSA, templates) before the first stage; that time is
`stages.prep` in `fold_done`. Every stage event is stamped after the chip finished the work
before it (the worker synchronises the device first), so the gaps between events are the chip's
time per stage.

### `frame`
The sampler took a step. `{"type":"frame","id":"a17","chip":0,"step":3,"of":200,"t":0.52}`.
`step` runs from `-1` (pure noise, before the first step) to `of - 1` (the final structure).
No coordinates: they come with the finished fold.

`stage` and `frame` are paced on the wire: at most one of each per fold every 0.25 s, enough
for a lane that redraws a few times a second, while a sampler reports up to 80 steps a second. A
new stage and a stage's last step always go out. The `step` in every message is the chip's own.

### `fold_done`
A summary. The coordinates stay on the engine for `GET /fold/<id>`.
```json
{"type":"fold_done","id":"a17","chip":0,"kind":"attract","source":"live","model":"openfold3",
 "name":"Lysozyme","sequence":"KVFGR...","n_res":130,"n_atoms":1001,"n_frames":201,
 "seconds":8.31,"stages":{"prep":0.4,"trunk":3.1,"diffusion":4.2,"confidence":0.6},
 "aiclk_mhz":{"min":1331,"median":1350,"max":1350,"n":40},"plddt_mean":0.87,"ptm":0.71,"t_wall":1790985460.9}
```
* `seconds` is wall time on the chip from input to confidence, with the chip's AICLK sampled during
  the fold (`aiclk_mhz`, MHz, every 0.2 s). Any number on screen should come from here.
* `n_frames` is how many sampler states the engine holds. A `fold_done` without it (a test worker
  that sends no coordinates) has nothing to pull.

### `fold_error`
`{"type":"fold_error","id":"a17","chip":0,"reason":"preempted"}`. Reasons: `preempted` (an attract
fold gave its chip to a visitor), `chip_lost` (the worker died; a visitor's fold is retried once on
another chip and gets a new `fold_start` with the same `id`), `stopped` (watchdog or shutdown),
`out_of_memory` (the chip's DRAM is full; the worker restarts itself with empty DRAM, and a
visitor's fold is retried once like `chip_lost`; the raw message is in `detail`), or
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

## `GET /fold/<id>?every=k`

One finished fold, binary, `application/octet-stream`:

    u32       length L of the metadata
    L bytes   metadata, gzipped JSON, then zero bytes up to a multiple of 4
    n*12      the final structure, float32 x y z per atom
    m*n*6     m packed states, int16 x y z per atom, in sampler order

`n` is the metadata's `n_atoms`. The metadata is `fold_start` (with `atoms`) and `fold_done`
(with `plddt`) merged, plus `every`, `step` and `t` for the m packed states and then the final one,
and `origin` and `scale` for the packed states. `every=k` sends every k-th state from the
first (pure noise) and always the final structure, so a fold of 201 states is 201 states at k=1 and
26 at k=8. Every state sent is a real one with its real `step`; nothing is interpolated. A fold
the engine no longer holds (it keeps the newest 64, one per protein and source) answers 404.

## Replay

A recording is the live fold's messages, one JSON object per line (`*.jsonl`), as the chip worker
sent them: `fold_done` carries `frames` (`step`, `t`, `origin`, `scale`, `q16` base64) and the
final `xyz`, as `engine/trajectory.py` packs them. Protocol-1 recordings are converted when they
are loaded (`python3 demo/sc26/engine/trajectory.py convert <files>` rewrites them in place). The
server plays recordings from `demo/sc26/gallery/trajectories/` and `demo/sc26/engine/recordings/`
through the identical protocol, with the recorded timing (gaps capped at 2 s), `source: "replay"`,
`chip: null` and `recorded: "<file stem>"`, and holds every one for `GET /fold` from startup. It
plays them all the time, every `--replay-gap` seconds (3) with no chip live and every
`--replay-gap-live` (20) while chips fold, so the gallery reaches the screen between live folds and
the screen is never blank. To develop the frontend with no chip at all:

    python3 demo/sc26/engine/server.py --replay-only

Every finished live fold is saved in this format under `demo/sc26/engine/runs/recorded/`, so the
gallery is grown by running the box.

## Slow links

Measured through 3 Mbit/s with 40 ms each way and 1 % loss (`ops/thin_link.sh`):

* The stream is a few KB a second. A browser that has more than 64 KB unsent is skipped, not
  queued for, until it drains, and one that takes nothing for 30 s is closed. A page that missed
  messages catches up from the next `status` (every 2 s).
* The page pulls one fold at a time, and only what changes its stage: a protein it has not got,
  then a live fold of one it holds only as a recording, then a refresh; a newer fold of a protein
  replaces an older one still waiting. It picks `every` so a pull takes about 6 s at the rate its
  last pulls achieved, at most 8. The words on screen then say how many of the fold's diffusion
  steps it shows.
* A socket silent for 6 s (the engine sends `status` every 2 s) is replaced by a new one, with no
  reload. What the page holds keeps playing meanwhile, and a chip lane does not call a chip quiet
  for time the page itself was cut off.
