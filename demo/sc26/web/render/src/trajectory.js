// Frames in, one displayed state out.
//
// Diffusion samplers apply a random rotation to their working frame at every step, so raw dumps
// tumble (measured on qb2's recordings: 120-145 deg between consecutive steps, science/rotation.py).
// Every frame is therefore rigidly moved before display, onto ONE fixed reference: the final
// structure when it is known (replay, and every fold the app shows), else the fold's first x0
// (live, until the final lands, then everything is re-superposed onto the final). The rotation is
// fitted on the frame's x0, the network's denoised estimate at that step, which lives in the same
// frame as the step's xyz and already has the protein's shape, so the fit never chases noise; it is
// then applied to xyz. Rotation + translation only: shape and every interatomic distance are
// untouched, and the final frame is not moved at all, so the coordinates on screen at the end are
// the scored structure's own floats.
//
// A state is decoded and moved the first time it is shown, not when the fold is loaded: a large
// complex's trajectory is tens of MB, and preparing all of it at once stalled the page at every
// fold change. Each state costs one decode and one fit, about a millisecond at 6,800 atoms.
//
// Playback holds each real state and steps to the next. `ease` (seconds, default 0.12) is the
// only motion that is not a sampler state: a linear blend between two CONSECUTIVE real states over
// the last `ease` seconds before the later one is reached. `ease: 0` turns it off and the display
// is only ever a real state.

import { decodeCoords } from './protocol.js';

// Optimal rotation by Horn's quaternion method: largest eigenvector of a symmetric 4x4.
function jacobiTopEigen(N) {
  const a = N.map(r => r.slice()), v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 4; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 4; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

export function centroid(x) {
  const n = x.length / 3; let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += x[3 * i]; cy += x[3 * i + 1]; cz += x[3 * i + 2]; }
  return [cx / n, cy / n, cz / n];
}

// The rigid transform that best superposes `mov` onto `ref`: {R (row-major 3x3), cm, cr}, applied
// as x' = R (x - cm) + cr.
export function fit(mov, ref) {
  const n = mov.length / 3, cm = centroid(mov), cr = centroid(ref);
  let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
  for (let i = 0; i < n; i++) {
    const mx = mov[3 * i] - cm[0], my = mov[3 * i + 1] - cm[1], mz = mov[3 * i + 2] - cm[2];
    const rx = ref[3 * i] - cr[0], ry = ref[3 * i + 1] - cr[1], rz = ref[3 * i + 2] - cr[2];
    Sxx += mx * rx; Sxy += mx * ry; Sxz += mx * rz;
    Syx += my * rx; Syy += my * ry; Syz += my * rz;
    Szx += mz * rx; Szy += mz * ry; Szz += mz * rz;
  }
  const N = [
    [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
  ];
  const [w, x, y, z] = jacobiTopEigen(N);
  const R = [
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  ];
  return { R, cm, cr };
}

// Returns a new array: `x` moved by the transform `f`.
export function applyFit({ R, cm, cr }, x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length / 3; i++) {
    const mx = x[3 * i] - cm[0], my = x[3 * i + 1] - cm[1], mz = x[3 * i + 2] - cm[2];
    out[3 * i] = R[0] * mx + R[1] * my + R[2] * mz + cr[0];
    out[3 * i + 1] = R[3] * mx + R[4] * my + R[5] * mz + cr[1];
    out[3 * i + 2] = R[6] * mx + R[7] * my + R[8] * mz + cr[2];
  }
  return out;
}

// `mov` rigidly superposed onto `ref`.
export const superpose = (mov, ref) => applyFit(fit(mov, ref), mov);

// A frame decoded and moved onto `ref`, the rotation fitted on its x0 when it has one.
const onto = (f, ref) => {
  const x = decodeCoords(f.coords);
  return { ...f, coords: applyFit(fit(f.x0 ? decodeCoords(f.x0) : x, ref), x), x0: null };
};

export function rmsd(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / (a.length / 3));
}

// Real frames on a clock. `time` is seconds on the stream's own clock (the play schedule in the
// app, the chip's timestamps in the harness, arrival time live). A frame may carry `x0` (its
// step's denoised estimate) and `step` (the sampler's own step index, -1 for the starting noise).
export class Timeline {
  constructor({ ease = 0.12 } = {}) { this.frames = []; this.final = null; this.delay = 0; this.ease = ease; this.ref = null; }

  // Every frame known up front: each is moved onto the final when first shown (state()), the final
  // is left alone.
  load(frames) {
    const n = frames.length;
    this.final = { ...frames[n - 1], coords: decodeCoords(frames[n - 1].coords) };
    this.frames = [...frames.slice(0, n - 1), this.final];
    this.moved = new Uint8Array(n);
    this.moved[n - 1] = 1;
  }

  // frames[i], moved onto the final the first time it is asked for
  state(i) {
    if (this.moved && !this.moved[i]) { this.frames[i] = onto(this.frames[i], this.final.coords); this.moved[i] = 1; }
    return this.frames[i];
  }

  // Live: onto the fold's first x0 until the final arrives, then everything onto the final.
  push(frame) {
    this.raw = [...(this.raw ?? []), frame];
    if (frame.final) return this.load(this.raw);
    this.ref ??= frame.x0 ? decodeCoords(frame.x0) : null;
    const prev = this.frames[this.frames.length - 1];
    this.frames.push(this.ref ? onto(frame, this.ref) : frame);
    if (prev) {
      const dt = frame.time - prev.time;
      this.delay = this.delay ? 0.8 * this.delay + 0.2 * dt : dt;
    }
  }

  get duration() { return this.frames.length ? this.frames[this.frames.length - 1].time - this.frames[0].time : 0; }

  // -> {a, b, alpha, index, progress}: display = a + (b - a) * alpha, where a is the last real
  // state reached (frames[index]) and b the next; alpha > 0 only inside the ease window.
  at(t) {
    const F = this.frames;
    if (!F.length) return null;
    const prog = (i) => F[i].progress ?? i / Math.max(1, F.length - 1);
    let hi = F.length - 1;
    if (t <= F[0].time) { const a = this.state(0); return { a, b: a, alpha: 0, index: 0, progress: prog(0) }; }
    if (t >= F[hi].time) return { a: F[hi], b: F[hi], alpha: 0, index: hi, progress: prog(hi) };
    let lo = 0;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (F[m].time <= t) lo = m; else hi = m; }
    const e = Math.min(this.ease, F[hi].time - F[lo].time);
    const alpha = e > 0 ? Math.max(0, (t - (F[hi].time - e)) / e) : 0;
    const a = this.state(lo);
    return { a, b: alpha > 0 ? this.state(hi) : a, alpha, index: lo, progress: prog(lo) };
  }
}
