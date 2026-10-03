# SC26 renderer

Draws a fold from the stream the way a structural biologist would recognise it: every atom as a
plain point at the sampler's real coordinates while it folds, then a cartoon of the scored
structure coloured by pLDDT. WebGL2 with no dependencies: everything it needs is in this
directory, so it runs offline.

```js
import { Renderer } from './src/renderer.js';
import { topologyFrom, frameFrom } from './src/protocol.js';

const r = new Renderer(canvas, { ease: 0.12, final: 'cartoon' });   // ease 0: real states only
r.setTopology(topologyFrom(msg.topology));
r.loadReplay(frames);                    // a recorded trajectory, or r.push(frame) per live frame
r.play();
requestAnimationFrame(function f(t) { r.render(dt); requestAnimationFrame(f); });
```

A frame is `{coords: Float32Array (xyz per atom), x0?, step?, time: seconds, progress: 0..1, final}`. The
topology needs per-atom name, element and residue index, and per-residue name and chain.
`src/protocol.js` adapts the stream to these; it is the only file that knows the wire format.

## What it shows

- **While folding**, every atom is a small matte grey point at the sampler's coordinates for the
  step on screen, depth-cued by size and fog. Each real state is held; the only motion that is not
  a sampler state is a 0.12 s linear blend between two consecutive states, which `ease: 0` turns
  off. `r.step` is the sampler's own step index for the state on screen.
- **The final structure** is a cartoon (helix, strand, coil) built from the scored coordinates,
  secondary structure assigned with the DSSP hydrogen-bond rules (`src/cartoon.js`), coloured on
  the AlphaFold pLDDT scale with its four standard colours. On qb2's recordings the assignment
  agrees with PyMOL's `dss` on 85 % (GFP) and 93 % (protein G B1) of residues
  (`../../science/pymol_compare.py`). Ligands are balls in element colours. `final: 'surface'`
  draws a Gaussian molecular surface instead (van der Waals radii, probe radius 0).
- **Light** is a matte material, one key light, sky/ground ambient, occlusion baked into the
  cartoon, and depth cueing. No bloom, glow, glass, rim light or depth of field.
- **Camera.** Framed once, from the final structure (or, live, from the size a protein of that
  length folds to), looking down its shortest axis, and never refitted, so the noise spills off
  screen and the protein arrives. It does not move until the fold is done; then it turns at
  5 degrees a second.

## Honesty

Diffusion samplers rotate their working frame at every step (120 to 145 degrees between
consecutive raw frames on qb2's recordings), so every frame is rigidly superposed onto one fixed
reference, the final structure, with the rotation fitted on the frame's own `x0` and applied to
its points (rotation and translation only). The final frame is drawn from its own coordinates,
untouched. `../../science/rotation.py` measures the rotation left between consecutive frames.

## Performance on the booth box

Measured on qb2's integrated Radeon (Ryzen 7 9700X, Mesa radeonsi) in Firefox, 397-residue
protein: 60 fps at 1920x1080 (full resolution, 4x MSAA) and 60 fps at 3840x2160 (0.75 render
scale, 2x MSAA), when no other program is using the GPU (measured with the earlier bloom pipeline,
which cost more than this one). The cartoon is built once per fold, 50 to 80 ms for 238 residues.

## Development

`index.html` is a harness, not the booth app. `tools/make_fixture.py` builds a development
trajectory from any mmCIF/PDB by adding synthetic diffusion noise; such files are marked
`source: synthetic` and must never be shown at the booth. `tools/look.sh` opens the harness on a
headless Wayland output driven by the box's own GPU and takes stills, recordings or frame-rate
measurements; `tools/bench.sh` runs a matrix of them.
