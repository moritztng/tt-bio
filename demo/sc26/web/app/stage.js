// The stage: one fold at a time, very large, drawn by sc26-render's Renderer.
//
// A fold's real sampler states arrive faster than anyone can watch (an ESMFold2 fold's whole
// diffusion takes a fraction of a second), so the stage plays them back over CONDENSE seconds,
// one equal slice per real step, and says how much slower that is. Between two real states the
// renderer interpolates linearly for smooth motion; the last frame is the scored structure as the
// chip produced it.

import { Renderer } from '../render/src/renderer.js';

export const CONDENSE = 6.0;   // seconds the real states are spread over
const FADE = 0.6;

export class Stage {
  constructor(canvas, opt = {}) {
    this.canvas = canvas;
    this.r = new Renderer(canvas, { fill: 0.62, ...opt });
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
  show(fold, { condense = CONDENSE } = {}) {
    this.fold = fold;
    this.condense = condense;
    const n = fold.coords.length;
    const frames = fold.coords.map((c, i) => ({ coords: c, time: n > 1 ? i * condense / (n - 1) : 0,
      progress: n > 1 ? i / (n - 1) : 1, final: i === n - 1 }));
    this.r.setTopology(fold.topo);
    this.r.loadReplay(frames);
    this.r.setMode('auto');
    this.r.speed = 1;
    this.r.pause();
    this.r.seek(0);
    this.t = 0;
    this.holdNoise = 0.9;   // a beat of pure noise before it moves
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
    const n = f.tReal.length, x = Math.min(1, Math.max(0, this.playT / this.condense)) * (n - 1);
    const i = Math.floor(x), a = f.tReal[i], b = f.tReal[Math.min(n - 1, i + 1)];
    return a + (b - a) * (x - i);
  }

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

// Which finished fold takes the stage next. Holds a pool of recent folds and never repeats a
// protein while another one is available that has not been on stage in the last few slots.
export class Director {
  constructor({ pool = 24, avoid = 4 } = {}) {
    this.pool = []; this.max = pool; this.avoid = avoid;
    this.history = [];   // names, most recent last
  }

  key(f) { return f.name ?? f.sequence; }

  add(f) {
    if (f.kind === 'visitor') return;      // a visitor's fold is theirs, not attract content
    // keep one copy per protein: the newest
    this.pool = this.pool.filter(p => this.key(p) !== this.key(f));
    this.pool.push(f);
    if (this.pool.length > this.max) this.pool.shift();
  }

  next() {
    if (!this.pool.length) return null;
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
