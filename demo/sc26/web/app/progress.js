// A chip's bar is the whole fold: input, trunk, diffusion and confidence, each as wide as its share
// of real time on this box. The engine sends each fold's plan with the job (PROTOCOL.md `plan`, measured
// here by length, engine/stages.py): the chip seconds at which each of its events is expected, from the
// chip taking the job to fold_done. A stage's steps (trunk recycle k of K, sampler step k of K) split it evenly.
//
// The bar moves on the chip's own events. Each one puts it at the plan's time for that event, and
// between two events it runs on toward the next at the pace the chip has kept so far, slowing as it
// nears it and never reaching it: the bar never shows a step the chip has not reported. A chip that
// falls behind its plan leaves its bar waiting short of the next step; a quiet chip's bar stands still.

// Where an event leaves the fold, in plan seconds: [now, the next event]. `k` of `K` steps of stage
// `key` are done. A stage the plan does not have moves nothing.
export function place(plan, key, k = 0, K = 0) {
  const i = plan.findIndex(([s]) => s === key);
  if (i < 0) return null;
  const a = plan[i][1], b = plan[i + 1]?.[1] ?? a;
  if (!(K > 0)) return [a, b];
  const at = (s) => a + (b - a) * Math.min(1, s / K);
  return [at(k), at(k + 1)];
}

// The bar, 0..1, for a job whose last event put it at `pos` with `next` the plan time of the event it
// waits for, `tChip` the chip's seconds at that event and `quiet` the seconds since it arrived.
// The chip's pace against its plan so far, chip seconds per plan second, settled by a tenth of the fold
// so that one early event cannot swing it.
const pace = (plan, pos, tChip) => { const T = plan[plan.length - 1][1]; return (tChip + 0.1 * T) / (pos + 0.1 * T); };

export function barAt({ plan, pos, next, tChip }, quiet) {
  if (!plan || pos == null) return null;
  const T = plan[plan.length - 1][1], gap = next - pos;
  if (!(T > 0)) return null;
  if (!(gap > 0)) return Math.min(1, pos / T);
  const x = quiet / (gap * pace(plan, pos, tChip));
  // straight on at the planned speed, then bending away from the next step it has not reached
  const f = x < 0.8 ? x : 1 - 0.2 * Math.exp(-(x - 0.8) / 0.2);
  return (pos + f * gap) / T;
}

// Seconds of silence after which the lane says the chip has gone quiet: three times the step it is
// on, and never under 10 s. A fold this page joined mid-way has no step yet: 30 s, a trunk recycle at
// 833 residues twice over.
export function quietAfter({ plan, pos, next, tChip }) {
  return !plan || pos == null ? 30 : Math.max(10, 3 * (next - pos) * pace(plan, pos, tChip));
}
