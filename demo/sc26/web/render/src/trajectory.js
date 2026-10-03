// Frames in, one displayed state out.
//
// Diffusion samplers apply a random rotation to their working frame at every step, so raw dumps
// tumble (measured on OpenDDE: ~140 deg between consecutive late steps). Every frame is therefore
// rigidly superposed before display: onto the final structure when it is known (replay), else onto
// the previous displayed frame (live). Superposition is rotation + translation only, so shape and
// every interatomic distance are untouched. In replay the final frame is not moved at all: the
// coordinates on screen at the end are the scored structure's own floats.
//
// Between two real frames the display interpolates linearly, for smoothness at 60 fps. Only real
// frames are ever endpoints; when playback reaches a frame, the display is exactly that frame.

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

// Returns a new array: `mov` rigidly superposed onto `ref`.
export function superpose(mov, ref) {
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
  const out = new Float32Array(mov.length);
  for (let i = 0; i < n; i++) {
    const mx = mov[3 * i] - cm[0], my = mov[3 * i + 1] - cm[1], mz = mov[3 * i + 2] - cm[2];
    out[3 * i] = R[0] * mx + R[1] * my + R[2] * mz + cr[0];
    out[3 * i + 1] = R[3] * mx + R[4] * my + R[5] * mz + cr[1];
    out[3 * i + 2] = R[6] * mx + R[7] * my + R[8] * mz + cr[2];
  }
  return out;
}

export function rmsd(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / (a.length / 3));
}

// Frames on a clock. `time` is seconds on the stream's own clock (the chip's timestamps in a
// replay, arrival time live). The display runs `delay` seconds behind the newest frame so it always
// has two real endpoints to interpolate between.
export class Timeline {
  constructor() { this.frames = []; this.final = null; this.delay = 0; }

  // Replay: every frame known up front; align everything to the final, leave the final alone.
  load(frames) {
    const fin = frames[frames.length - 1].coords;
    this.frames = frames.map((f, i) => ({ ...f, coords: i === frames.length - 1 ? fin : superpose(f.coords, fin) }));
    this.final = this.frames[this.frames.length - 1];
  }

  // Live: align to what is on screen now.
  push(frame) {
    const prev = this.frames[this.frames.length - 1];
    const coords = prev ? superpose(frame.coords, prev.coords) : frame.coords;
    this.frames.push({ ...frame, coords });
    if (frame.final) this.final = this.frames[this.frames.length - 1];
    if (this.frames.length > 1) {
      const dt = frame.time - prev.time;
      this.delay = this.delay ? 0.8 * this.delay + 0.2 * dt : dt;
    }
  }

  get duration() { return this.frames.length ? this.frames[this.frames.length - 1].time - this.frames[0].time : 0; }

  // -> {a, b, alpha, progress}: display = a + (b - a) * alpha
  at(t) {
    const F = this.frames;
    if (!F.length) return null;
    if (t <= F[0].time) return { a: F[0], b: F[0], alpha: 0, progress: F[0].progress ?? 0 };
    let lo = 0, hi = F.length - 1;
    if (t >= F[hi].time) return { a: F[hi], b: F[hi], alpha: 0, progress: F[hi].progress ?? 1 };
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (F[m].time <= t) lo = m; else hi = m; }
    const alpha = (t - F[lo].time) / (F[hi].time - F[lo].time);
    const pa = F[lo].progress ?? lo / (F.length - 1), pb = F[hi].progress ?? hi / (F.length - 1);
    return { a: F[lo], b: F[hi], alpha, progress: pa + (pb - pa) * alpha };
  }
}
