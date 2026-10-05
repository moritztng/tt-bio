// How far along a chip's fold is, in the chip's own seconds.
//
// A bar is the fold's elapsed time over its expected total, so a stage takes the share of the bar
// it takes of the clock. The expectation is measured on this box, never written down by hand: every
// live fold_done reports its seconds per stage, and a fold expects the median of the last folds of
// the same model and length. A length the box has not folded yet is interpolated between the
// nearest lengths it has, never extrapolated. Until a model has finished one live fold, its bars
// stay empty and only the stage word moves.
//
// Inside the running stage the chip's own step counter (recycle k of K, sampler step k of K) and
// the pace it has kept so far say how much of the stage is left, trusted more the further the stage
// has run. Between two events the bar runs on with the wall clock for at most a second, so a
// stalled chip stops its bar instead of pretending.

const KEY = 'sc26.timing.v2';
const KEEP = 8;          // folds remembered per model and length
const RUN_ON = 1.0;

const median = (xs) => { const s = [...xs].sort((a, b) => a - b), k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };

export class Timing {
  constructor({ persist = true } = {}) {
    this.persist = persist;   // off while rehearsing (?as=), so a relabelled model's times are never kept
    this.seen = {};           // model -> { n_res -> [{seconds, stages}, ...] }, newest last
    if (persist) try { this.seen = JSON.parse(localStorage.getItem(KEY)) ?? {}; } catch { /* a fresh profile */ }
  }

  // a finished live fold: {model, n_res, chip, seconds, stages}
  learn(m) {
    if (m.source !== 'live' || m.chip == null || !m.stages || !(m.seconds > 0)) return;
    const list = ((this.seen[m.model] ??= {})[m.n_res] ??= []);
    list.push({ seconds: m.seconds, stages: m.stages });
    if (list.length > KEEP) list.shift();
    if (this.persist) try { localStorage.setItem(KEY, JSON.stringify(this.seen)); } catch { /* full or private */ }
  }

  // {seconds, stages} expected for this model and length, or null if nothing is measured yet
  expect(model, n) {
    const by = this.seen[model];
    if (!by) return null;
    const at = (k) => {
      const list = by[k], stages = {};
      for (const s of Object.keys(list[list.length - 1].stages)) stages[s] = median(list.map(x => x.stages[s] ?? 0));
      return { seconds: median(list.map(x => x.seconds)), stages };
    };
    if (by[n]) return at(n);
    const ns = Object.keys(by).map(Number).sort((a, b) => a - b);
    const hi = ns.find(x => x > n), lo = [...ns].reverse().find(x => x < n);
    if (lo == null || hi == null) return at(lo ?? hi);
    const a = at(lo), b = at(hi), w = (n - lo) / (hi - lo), mix = (x, y) => x + w * (y - x);
    const stages = {};
    for (const s of Object.keys(a.stages)) stages[s] = mix(a.stages[s], b.stages[s] ?? a.stages[s]);
    return { seconds: mix(a.seconds, b.seconds), stages };
  }

  // 0..1 for a running job (stream.js jobs[id]), or null when there is nothing honest to draw
  progress(job, now) {
    if (job.stage === 'done') return 1;
    const e = this.expect(job.model, job.n);
    if (!e || job.tChip == null) return null;
    const order = Object.keys(e.stages);
    const i = order.indexOf(job.stage);
    if (job.stage && i < 0) return job.p ?? null;
    let total = e.seconds;   // before the first stage event the whole fold is ahead
    if (i >= 0) {
      // the rest of this stage, then every later stage, then whatever the total holds beyond them
      const f = job.total > 0 ? Math.min(1, job.step / job.total) : 0;
      const ran = job.tChip - (job.at?.[job.stage] ?? job.tChip);
      const byPlan = e.stages[job.stage] * (1 - f), byPace = f > 0 ? ran * (1 - f) / f : byPlan;
      const later = order.slice(i + 1).reduce((s, k) => s + e.stages[k], 0);
      const outside = Math.max(0, e.seconds - order.reduce((s, k) => s + e.stages[k], 0));
      total = job.tChip + f * byPace + (1 - f) * byPlan + later + outside;
    }
    const elapsed = job.tChip + Math.min(RUN_ON, now - job.tAt);
    job.p = Math.max(job.p ?? 0, Math.min(0.995, elapsed / total));
    return job.p;
  }
}
