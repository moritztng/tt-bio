"""How long each stage of a fold takes on this box, by model and length, measured here.

A fold's schedule is the chip time at which each of its stages began, in the order the chip ran
them, plus its total: [["taken", 0], ["start", 4.97], ["trunk", 10.38], ["diffusion", 67.40],
["confidence", 87.78], ["done", 93.90]] for TCR-HLA (833 residues) on OpenFold3. "taken" is the
chip taking the job, "start" is fold_start, the end of input preparation; every other mark is the
`t` of the first stage event of that name, and "done" is fold_done's seconds. The page draws a fold's bar against the schedule
the engine expects for it (PROTOCOL.md, `plan`), so each stage's share of the bar is its share of
real time on this box.

Schedules come from the gallery recordings at startup (folded on this box, gallery/record.py)
and from every live fold after that. A length folded before gets the median of its last five;
any other is interpolated between the two nearest lengths on a log-log scale, which is how fold
time grows with length, and extrapolated the same way past either end.
"""
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import median

KEEP = 5


class Stages:
    def __init__(self, store=None):
        self.seen = defaultdict(lambda: deque(maxlen=KEEP))   # (model, n_res) -> recent schedules
        for f in sorted(Path(store).glob("*.json")) if store else []:
            try:
                d = json.loads(f.read_text())
                marks = [["start", d["stages"]["prep"]]]
                for stage, _, _, t in d.get("stage_events", []):
                    marks.append([stage, t])
                self.add(d["model"], d["n_res"], marks + [["done", d["seconds"]]])
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def add(self, model, n_res, marks):
        """One finished fold's marks, in the order they happened; a stage counts from its first event."""
        sched, names = [], set()
        for name, t in marks:
            if name not in names and t is not None:
                names.add(name)
                sched.append((name, float(t)))
        if len(sched) > 2 and sched[-1][0] == "done":
            self.seen[(model, n_res)].append(sched)

    def _median(self, runs):
        order = [name for name, _ in runs[-1]]
        return {name: median(t for r in runs for k, t in r if k == name) for name in order}, order

    def plan(self, model, n_res):
        sizes = sorted(n for m, n in self.seen if m == model)
        if not sizes:
            return None
        if n_res in sizes:
            at, order = self._median(self.seen[(model, n_res)])
        else:
            lo, hi = [n for n in sizes if n < n_res], [n for n in sizes if n > n_res]
            near = lo[-1:] + hi[:1] if lo and hi else lo[-2:] or hi[:2]   # the two nearest that bracket it
            a, b = near[0], near[-1]
            (ta, order), (tb, _) = self._median(self.seen[(model, a)]), self._median(self.seen[(model, b)])
            w = 0.0 if a == b else math.log(n_res / a) / math.log(b / a)
            at = {k: math.exp((1 - w) * math.log(max(ta[k], 1e-3)) + w * math.log(max(tb[k], 1e-3)))
                  for k in order if k in tb}
            order = [k for k in order if k in at]
        out, prev = [["taken", 0.0]], 0.0   # the chip took the job: its clock starts
        for k in order:
            prev = max(prev + 1e-3, at[k])   # a schedule never runs backwards
            out.append([k, round(prev, 3)])
        return out
