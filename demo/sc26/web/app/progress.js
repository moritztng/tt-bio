// How far along a chip's fold is, in the chip's own seconds.
//
// A bar is the fold's elapsed time over its expected total, so a stage takes the share of the bar
// it takes of the clock. The expectation is measured on this box, never written down by hand: every
// live fold_done reports its seconds per stage, and the next fold of the same model and length
// expects those (the same chip's last fold if there is one, else any chip's). A length the box has
// not folded yet is interpolated between the nearest lengths it has, never extrapolated. Until a
// model has finished one live fold, its bars stay empty and only the stage word moves.
//
// Inside the running stage, the stage's own step counter (recycle k of K, sampler step k of K)
// says how much of it is left; between two events the bar runs on with the wall clock, for at most
// a second past the last event, so a stalled chip stops its bar instead of pretending.

const KEY = 'sc26.timing.v1';
const RUN_ON = 1.0;

export class Timing {
  constructor({ persist = true } = {}) {
    this.persist = persist;   // off while rehearsing (?as=), so a relabelled model's times are never kept
    this.seen = {};           // model -> { n_res -> { chip -> {seconds, stages} } }
    if (persist) try { this.seen = JSON.parse(localStorage.getItem(KEY)) ?? {}; } catch { /* a fresh profile */ }
  }

  // a finished live fold: {model, n_res, chip, seconds, stages}
  learn(m) {
    if (m.source !== 'live' || m.chip == null || !m.stages || !(m.seconds > 0)) return;
    ((this.seen[m.model] ??= {})[m.n_res] ??= {})[m.chip] = { seconds: m.seconds, stages: m.stages };
    if (this.persist) try { localStorage.setItem(KEY, JSON.stringify(this.seen)); } catch { /* full or private */ }
  }

  // {seconds, stages} expected for this model, length and chip, or null if nothing is measured yet
  expect(model, n, chip) {
    const by = this.seen[model];
    if (!by) return null;
    const pick = (c) => c[chip] ?? Object.values(c)[0];
    if (by[n]) return pick(by[n]);
    const ns = Object.keys(by).map(Number).sort((a, b) => a - b);
    const hi = ns.find(x => x > n), lo = [...ns].reverse().find(x => x < n);
    if (lo == null || hi == null) return pick(by[lo ?? hi]);
    const a = pick(by[lo]), b = pick(by[hi]), w = (n - lo) / (hi - lo);
    const mix = (x, y) => x + w * (y - x);
    const stages = {};
    for (const s of Object.keys(a.stages)) stages[s] = mix(a.stages[s], b.stages[s] ?? a.stages[s]);
    return { seconds: mix(a.seconds, b.seconds), stages };
  }

  // 0..1 for a running job (stream.js jobs[id]), or null when there is nothing honest to draw
  progress(job, now) {
    if (job.stage === 'done') return 1;
    const e = this.expect(job.model, job.n, job.chip);
    if (!e || job.tChip == null) return null;
    const order = Object.keys(e.stages);
    const i = order.indexOf(job.stage);
    if (job.stage && i < 0) return job.p ?? null;
    // seconds left when the last event arrived: the rest of this stage by its own step counter, then
    // every later stage, then whatever the fold's total holds beyond its stages. Before the first
    // stage event, the whole fold is ahead.
    const frac = job.total > 0 ? Math.min(1, job.step / job.total) : 0;
    const later = order.slice(i + 1).reduce((s, k) => s + e.stages[k], 0);
    const outside = Math.max(0, e.seconds - order.reduce((s, k) => s + e.stages[k], 0));
    const total = i < 0 ? e.seconds : job.tChip + e.stages[job.stage] * (1 - frac) + later + outside;
    const elapsed = job.tChip + Math.min(RUN_ON, now - job.tAt);
    job.p = Math.max(job.p ?? 0, Math.min(0.995, elapsed / total));
    return job.p;
  }
}
