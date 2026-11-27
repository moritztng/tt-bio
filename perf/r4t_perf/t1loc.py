#!/usr/bin/env python3
"""Locate lever T1's digest move (D42(2)): at the first Folds.backward, read every device grad back both ways
(A: plain, B: card transpose as uint32 bits + a transposed view for TBackward-topped 2-D sources), check the
host values are equal, then run torch.autograd.grad through the folds both ways and compare every upstream
leaf bitwise. Production backward then runs unchanged (A).    t1loc.py -- <trunk_step.py args>"""
import json, os, runpy, sys
import torch, ttnn
from tt_bio.train import folds as F

argv = sys.argv[sys.argv.index("--") + 1:]
step = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t_step", "trunk_step.py")
_real, _done = F.Folds.backward, []


def backward(self, grads):
    grads = list(grads)
    if not _done:
        _done.append(1)
        srcs, ga, gb, nflip, vdiff = [], [], [], 0, 0
        for h, g in grads:
            s = self.source(h)
            a = ttnn.to_torch(g).to(s.dtype).reshape(s.shape)
            if type(s.grad_fn).__name__ == "TBackward0" and s.dim() == 2 and all(d == 1 for d in tuple(g.shape)[:-2]):
                u = ttnn.transpose(ttnn.bitcast(g, ttnn.uint32), -2, -1)
                b = ttnn.to_torch(ttnn.bitcast(u, ttnn.float32)).reshape(s.shape[1], s.shape[0]).t()
                nflip += 1
                vdiff += int((a != b).sum())
            else:
                b = a
            srcs.append(s); ga.append(a); gb.append(b)
        names = list(self.leaves); L = [self.leaves[n] for n in names]
        ra = torch.autograd.grad(srcs, L, ga, retain_graph=True, allow_unused=True)
        rb = torch.autograd.grad(srcs, L, gb, retain_graph=True, allow_unused=True)
        bad = []
        for n, x, y in zip(names, ra, rb):
            if x is None or y is None:
                continue
            if not torch.equal(x, y):
                d = (x - y).abs()
                bad.append([n, int((x != y).sum()), x.numel(), float(d.max()), float(x.abs().max())])
        print(json.dumps(dict(t1loc=1, n=len(srcs), flipped=nflip, value_diffs=vdiff, leaves=len(names),
                              differing_leaves=len(bad), bad=bad[:40])), flush=True)
    return _real(self, grads)


F.Folds.backward = backward
sys.argv = [step] + argv
runpy.run_path(step, run_name="__main__")
