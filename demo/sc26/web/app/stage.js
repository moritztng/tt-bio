// The stage: one fold at a time, very large, drawn by sc26-render's Renderer.
//
// A fold's real sampler states arrive faster than anyone can watch (an ESMFold2 fold's whole
// diffusion takes a fraction of a second), so the stage plays them back over CONDENSE seconds and
// says how much slower that is. A recorded fold whose diffusion took longer than CAP seconds on its
// chip (a large Boltz-2 complex) plays over CAP seconds and says how much faster, so one complex
// does not hold the stage for a minute. Every real state is shown, in order, and held for the time
// the chip took to produce it, scaled by one factor for the whole fold: a fold shown in real time
// keeps the chip's own pace step by step, and "N× slower" or "N× faster" is true of every step. The
// only motion that is not a sampler state is the renderer's short blend between two consecutive
// states (`?ease=<s>`, default 0.12 s; `?ease=0` shows real states only). `step` and `of` are the
// sampler's own step counter for the state on screen. The last frame is the scored structure as the
// chip produced it.

import { Renderer } from '../render/src/renderer.js';

export const CONDENSE = 6.0;   // seconds the real states are spread over
export const CAP = 9.0;        // longest a diffusion replay plays

export class Stage {
  constructor(canvas, opt = {}) {
    this.canvas = canvas;
    this.r = new Renderer(canvas, { fill: 0.72, ...opt });
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

  // Put a fold on the stage, starting from its first (noise) state. A fold whose diffusion took
  // between CONDENSE and CAP seconds plays at the chip's own pace; a longer one plays over CAP.
  show(fold, { condense = CONDENSE, cap = CAP } = {}) {
    this.fold = fold;
    this.span = paced(fold.tReal).span;
    this.condense = Math.min(Math.max(condense, this.span), Math.max(cap, condense));
    this.times = playTimes(fold.tReal, this.condense);
    const n = fold.coords.length;
    const frames = fold.coords.map((c, i) => ({ coords: c, x0: fold.x0?.[i] ?? null, step: fold.steps?.[i] ?? i - 1,
      time: this.times[i], progress: n > 1 ? i / (n - 1) : 1, final: i === n - 1 }));
    this.r.setTopology(fold.topo);
    this.r.loadReplay(frames);
    this.r.speed = 1;
    this.r.pause();
    this.r.seek(0);
    this.t = 0;
    this.holdNoise = 0.3;   // a beat of pure noise before it moves
  }

  // How many times slower than the chip the replay plays (below 1: faster).
  get slowdown() {
    const f = this.fold;
    if (!f || !(this.span > 0)) return null;
    return this.condense / this.span;
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
  // before the replay starts the renderer can still report the last fold's final step
  get step() { return this.t < this.holdNoise ? -1 : this.r.step ?? -1; }
  get of() { return this.fold?.of ?? 0; }

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

// A step that took more than OUTLIER times the fold's median step is a one-off stall, not sampling:
// the first time a chip meets a new size it compiles inside a step (0.63 s against 12.5 ms per
// step for one 70-residue name). It is replayed at the median step, so a visitor does not watch
// still noise for seconds, and "N× slower" stays true of every step shown. The measured fold time
// on screen still includes it.
const OUTLIER = 5;

function paced(tReal) {
  const n = tReal?.length ?? 0;
  if (n < 2) return { gaps: [], span: 0 };
  const raw = tReal.slice(1).map((t, i) => t - tReal[i]);
  const med = [...raw].sort((a, b) => a - b)[raw.length >> 1];
  const gaps = raw.map(g => (g > OUTLIER * med ? med : g));
  const span = gaps.reduce((a, g) => a + g, 0);
  return { gaps, span: Number.isFinite(span) ? span : 0 };
}

// When each real state is on screen: the chip's own time for each step, stretched to `total`
// seconds. (Re-timing states by how much the cloud shrinks made a "real time" counter run at 3.4x
// through the noise and 0.2x through the folding.) Equal steps if the timestamps are missing.
function playTimes(tReal, total) {
  const { gaps, span } = paced(tReal);
  if (!gaps.length) return [0];
  let acc = 0;
  return [0, ...gaps.map((g, i) => total * (span > 0 ? (acc += g) / span : (i + 1) / gaps.length))];
}

// How often a protein takes the stage, by its size in residues. The booth is for the large
// complexes: an 800-residue complex comes up five times as often as anything under 150 residues.
export const weight = (f) => Math.min(1.5, Math.max(0.3, (f.nres ?? 0) / 500));

// Which finished fold takes the stage next. Holds a pool of recent folds and never repeats a
// protein while another one is available that has not been on stage in the last few slots. It
// never shows the same model twice in a row while a fold by another model is waiting, and never
// three times while one is in the pool at all, so a visitor who watches for half a minute sees
// more than one model. Among the rest it is stride scheduling: each protein's next turn is
// 1/weight slots after its last, and a protein that has just arrived joins at the current turn,
// so it is on stage soon.
//
// The server streams the gallery's recordings between live folds. If the pool is still empty (the
// page has just loaded, or the server is down), the stage shows the last few visitors' folds (as
// "A visitor's name", never the name), and before anything has arrived, the one recording bundled
// with the app. The stage is never empty.
export class Director {
  constructor({ pool = 24, avoid = 4 } = {}) {
    this.pool = []; this.max = pool; this.avoid = avoid;
    this.history = [];   // names, most recent last
    this.turn = new Map(); this.now = 0;   // stride scheduling: each protein's next turn
    this.visitors = [];  // last few visitors' folds, shown only when the pool is empty
    this.fallback = null;
  }

  key(f) { return f.name ?? f.sequence; }

  add(f) {
    if (f.kind === 'visitor') {
      this.visitors = [...this.visitors.filter(v => v.sequence !== f.sequence), f].slice(-3);
      return;
    }
    // keep one copy per protein: the newest live fold, or the newest recording until a chip has folded
    // it. A recording never displaces a live fold, so live or recorded is never a matter of which
    // copy happened to arrive last.
    const had = this.pool.find(p => this.key(p) === this.key(f));
    if (had?.source === 'live' && f.source !== 'live') return;
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
    const other = cands.filter(p => p.model !== this.lastModel);
    if (other.length) cands = other;
    else if (this.run >= 2) {   // two in a row already: the other model, even a protein seen lately
      const any = this.pool.filter(p => p.model !== this.lastModel);
      if (any.length) cands = any;
    }
    // earliest turn first; on a tie, live before replay, newest first
    const turn = (p) => Math.max(this.now, this.turn.get(this.key(p)) ?? this.now);
    cands.sort((a, b) => turn(a) - turn(b) || (a.source === 'live' ? -1 : 0) - (b.source === 'live' ? -1 : 0)
      || b.received - a.received);
    const f = cands[0];
    this.now = turn(f);
    this.turn.set(this.key(f), this.now + 1 / weight(f));
    this.history.push(this.key(f));
    this.run = f.model === this.lastModel ? (this.run ?? 0) + 1 : 1;
    this.lastModel = f.model;
    if (this.history.length > 64) this.history.shift();
    return f;
  }
}
