# SC26 renderer

Draws a protein from the fold stream as points of light, a molecular surface, or a ribbon, and
moves between them smoothly. WebGL2 with no dependencies: everything it needs is in this directory,
so it runs offline.

```js
import { Renderer } from './src/renderer.js';
import { topologyFrom, frameFrom } from './src/protocol.js';

const r = new Renderer(canvas);          // scale and MSAA pick themselves from the canvas size
r.setTopology(topologyFrom(msg.topology));
r.loadReplay(frames);                    // a recorded trajectory, or r.push(frame) per live frame
r.setMode('auto');                       // 'auto' | 'points' | 'surface' | 'glass' | 'ribbon'
r.play();
requestAnimationFrame(function f(t) { r.render(dt); requestAnimationFrame(f); });
```

A frame is `{coords: Float32Array (xyz per atom), time: seconds, progress: 0..1, final}`. The
topology needs per-atom name, element and residue index, and per-residue name and chain.
`src/protocol.js` adapts the stream to these; it is the only file that knows the wire format.

## What it shows

- **Auto.** The fold drives the look. While the structure is noise every atom is a point, with
  depth of field so distant noise reads as soft bokeh. As it condenses, a surface grows out of the
  points (small droplets that fuse into a skin), turns to glass with the points glowing inside, and
  settles into an opaque surface.
- **Colour** is position along the chain (indigo at the start, rose at the end), so you can watch
  the two ends of one string find each other. `setScheme('water')` colours by hydrophobicity and
  `'confidence'` by the model's per-residue confidence when the stream carries it.
- **Camera.** Framed once, from the final structure (or, live, from the size a protein of that
  length folds to), and never refitted, so the noise spills off screen and the protein arrives.
  The orbit is 3 degrees a second around the structure's shortest axis.

## Honesty

Diffusion samplers rotate their working frame at every step, so frames are rigidly superposed
before display (rotation and translation only). In a replay everything is superposed onto the
final structure and the final frame is drawn from its own coordinates, untouched. Between two
real frames the display interpolates linearly for smoothness; it never invents a frame beyond
the last one received.

## Performance on the booth box

Measured on qb2's integrated Radeon (Ryzen 7 9700X, Mesa radeonsi) in Firefox, 397-residue
protein with its surface re-meshing: 60 fps at 1920x1080 (full resolution, 4x MSAA) and 60 fps
at 3840x2160 (0.75 render scale, 2x MSAA), when no other program is using the GPU. Re-meshing the
surface takes 15 to 30 ms in a worker and does not block drawing. Details in
`state/sc26-render.md` on the fleet host.

## Development

`index.html` is a harness, not the booth app. `tools/make_fixture.py` builds a development
trajectory from any mmCIF/PDB by adding synthetic diffusion noise; such files are marked
`source: synthetic` and must never be shown at the booth. `tools/look.sh` opens the harness on a
headless Wayland output driven by the box's own GPU and takes stills, recordings or frame-rate
measurements; `tools/bench.sh` runs a matrix of them.
