// The stage: one fold at a time, very large, drawn by sc26-render's Renderer.
//
// A fold's real sampler states arrive faster than anyone can watch (an ESMFold2 fold's whole
// diffusion takes a fraction of a second), so the stage plays them back over CONDENSE seconds and
// says how much slower that is. Every real state is shown, in order, and held for the time the chip
// took to produce it, scaled by one factor for the whole fold: a fold shown in real time keeps the
// chip's own pace step by step, and "N× slower" is true of every step. The only motion that is not a sampler state is the renderer's short blend between two
// consecutive states (`?ease=<s>`, default 0.12 s; `?ease=0` shows real states only). `step` and
// `of` are the sampler's own step counter for the state on screen. The last frame is the scored
// structure as the chip produced it.

import { Renderer } from '../render/src/renderer.js';

export const CONDENSE = 6.0;   // seconds the real states are spread over

export class Stage {
  constructor(canvas, opt = {}) {
    this.canvas = canvas;
    this.r = new Renderer(canvas, { fill: 0.82, ...opt });
    this.fold = null;
    this.t = 0;            // seconds since this fold took the stage
    this.fitted = false;
  }

  resize() {
    this.canvas.width = Math.round(innerWidth * devicePixelRatio);
    this.canvas.height = Math.round(innerHeight * devicePixelRatio);
    this.r.resize();
  }

  setOffset(x, y) { this.r.opt.offset = [x, y]; }

  // Put a fold on the stage, starting from its first (noise) state.
  // Never faster than the chip ran it: a fold whose diffusion took longer than CONDENSE plays at
  // the chip's own pace.
  show(fold, { condense = CONDENSE } = {}) {
    this.fold = fold;
    this.condense = Math.max(condense, fold.diffusionSeconds > 0 ? fold.diffusionSeconds : 0);
    this.times = playTimes(fold.tReal, this.condense);
    const n = fold.coords.length;
    const frames = fold.coords.map((c, i) => ({ coords: c, x0: fold.x0?.[i] ?? null, step: fold.steps?.[i] ?? i - 1,
      time: this.times[i], progress: n > 1 ? i / (n - 1) : 1, final: i === n - 1 }));
    // how much wider than the final structure each state's cloud is, by radius of gyration
    const rg = gyration(fold.coords[n - 1]) || 1;
    this.spreads = new Map(frames.map(f => [f.step, gyration(f.coords) / rg]));
    this.r.setTopology(fold.topo);
    this.r.loadReplay(frames);
    this.r.speed = 1;
    this.r.pause();
    this.r.seek(0);
    this.t = 0;
    this.holdNoise = 0.3;   // a beat of pure noise before it moves
  }

  // How many times slower than the chip the condensation plays.
  get slowdown() {
    const f = this.fold;
    if (!f || !(f.diffusionSeconds > 0)) return null;
    return this.condense / f.diffusionSeconds;
  }

  // The real chip time of the state on screen now, for the counting number.
  get realTime() {
    const f = this.fold;
    if (!f) return 0;
    const T = this.times, n = T.length, t = this.playT;
    if (n < 2 || t >= T[n - 1]) return f.tReal[n - 1] ?? 0;
    let i = 0;
    while (T[i + 1] <= t) i++;
    const x = (t - T[i]) / (T[i + 1] - T[i] || 1);
    return f.tReal[i] + (f.tReal[i + 1] - f.tReal[i]) * x;
  }

  // The sampler step on screen: -1 is the starting noise, of - 1 the scored structure.
  get step() { return this.r.step ?? -1; }
  get of() { return this.fold?.of ?? 0; }
  get spread() { return this.spreads?.get(this.step) ?? 1; }

  get playT() { return Math.max(0, this.t - this.holdNoise); }
  get landed() { return this.fold && this.playT >= this.condense; }

  frame(dt) {
    if (this.fold) {
      this.t += dt;
      if (this.t >= this.holdNoise && !this.r.playing) this.r.play();
    }
    this.r.render(dt);
  }
}

// When each real state is on screen: the chip's own timestamp for it, stretched to `total` seconds.
// (Re-timing states by how much the cloud shrinks made a "real time" counter run at 3.4x through the
// noise and 0.2x through the folding.) Equal steps if the timestamps are missing.
function playTimes(tReal, total) {
  const n = tReal?.length ?? 0;
  if (n < 2) return [0];
  const span = tReal[n - 1] - tReal[0];
  return tReal.map((t, i) => total * (span > 0 ? (t - tReal[0]) / span : i / (n - 1)));
}

function gyration(x) {
  const m = x.length / 3;
  let cx = 0, cy = 0, cz = 0, s = 0;
  for (let i = 0; i < m; i++) { cx += x[3 * i]; cy += x[3 * i + 1]; cz += x[3 * i + 2]; }
  cx /= m; cy /= m; cz /= m;
  for (let i = 0; i < m; i++) s += (x[3 * i] - cx) ** 2 + (x[3 * i + 1] - cy) ** 2 + (x[3 * i + 2] - cz) ** 2;
  return Math.sqrt(s / m);
}

// Which finished fold takes the stage next. Holds a pool of recent folds and never repeats a
// protein while another one is available that has not been on stage in the last few slots.
//
// The server streams the gallery's recordings between live folds. If the pool is still empty (the
// page has just loaded, or the server is down), the stage shows the last few visitors' folds (as
// "A visitor's name", never the name), and before anything has arrived, the one recording bundled
// with the app. The stage is never empty.
export class Director {
  constructor({ pool = 24, avoid = 4 } = {}) {
    this.pool = []; this.max = pool; this.avoid = avoid;
    this.history = [];   // names, most recent last
    this.visitors = [];  // last few visitors' folds, shown only when the pool is empty
    this.fallback = null;
  }

  key(f) { return f.name ?? f.sequence; }

  add(f) {
    if (f.kind === 'visitor') {
      this.visitors = [...this.visitors.filter(v => v.sequence !== f.sequence), f].slice(-3);
      return;
    }
    // keep one copy per protein: the newest
    this.pool = this.pool.filter(p => this.key(p) !== this.key(f));
    this.pool.push(f);
    if (this.pool.length > this.max) this.pool.shift();
  }

  next() {
    if (!this.pool.length) {
      if (!this.visitors.length) return this.fallback;
      const v = this.visitors.shift();     // oldest first, then round to the back
      this.visitors.push(v);
      return v;
    }
    const distinct = new Set(this.pool.map(p => this.key(p))).size;
    const recent = new Set(this.history.slice(-Math.min(this.avoid, distinct - 1)));
    let cands = this.pool.filter(p => !recent.has(this.key(p)));
    if (!cands.length) cands = this.pool;
    // least recently shown first; among never-shown, live before replay, newest first
    const last = (p) => { const i = this.history.lastIndexOf(this.key(p)); return i < 0 ? -1 : i; };
    cands.sort((a, b) => last(a) - last(b) || (a.source === 'live' ? -1 : 0) - (b.source === 'live' ? -1 : 0)
      || b.received - a.received);
    const f = cands[0];
    this.history.push(this.key(f));
    if (this.history.length > 64) this.history.shift();
    return f;
  }
}
