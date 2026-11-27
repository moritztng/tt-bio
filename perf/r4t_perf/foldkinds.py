#!/usr/bin/env python3
"""Census of trunk_step's fold sources for D42(2): for every recorded device weight, the chain of torch ops
between it and the upstream leaves (grad_fn names), grouped by that signature with counts and elements, and
how many upstream leaves feed more than one device copy. Printed once, at the first Folds.backward call.

    foldkinds.py -- <trunk_step.py args>
"""
import collections, json, os, runpy, sys

argv = sys.argv[sys.argv.index("--") + 1:]
step = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t_step", "trunk_step.py")

from tt_bio.train import folds as F

_real, _done = F.Folds.backward, []


def sig(t):
    names, seen, stack = set(), set(), [t.grad_fn]
    while stack:
        f = stack.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        n = type(f).__name__
        if n != "AccumulateGrad":
            names.add(n.replace("Backward0", "").replace("Backward1", "").replace("Backward", ""))
        stack += [g for g, _ in f.next_functions]
    return "+".join(sorted(names)) or "leaf"


def backward(self, grads):
    if not _done:
        _done.append(1)
        by, leaf_uses = collections.defaultdict(lambda: [0, 0]), collections.Counter()
        ids = {id(v): k for k, v in self.leaves.items()}
        for h, s in self.sources.values():
            k = sig(s); by[k][0] += 1; by[k][1] += s.numel()
            stack, seen = [s.grad_fn], set()
            if s.grad_fn is None and id(s) in ids:
                leaf_uses[ids[id(s)]] += 1
            while stack:
                f = stack.pop()
                if f is None or f in seen:
                    continue
                seen.add(f)
                if type(f).__name__ == "AccumulateGrad" and id(f.variable) in ids:
                    leaf_uses[ids[id(f.variable)]] += 1
                stack += [g for g, _ in f.next_functions]
        rows = sorted(([k, n, e] for k, (n, e) in by.items()), key=lambda r: -r[2])
        print(json.dumps(dict(foldkinds=1, sources=len(self.sources), elems=sum(r[2] for r in rows),
                              multi_copy_leaves=sum(1 for v in leaf_uses.values() if v > 1),
                              kinds=rows)), flush=True)
    return _real(self, grads)


F.Folds.backward = backward
sys.argv = [step] + argv
runpy.run_path(step, run_name="__main__")
