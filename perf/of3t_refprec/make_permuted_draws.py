"""Negative control for the precision comparison: the same draws, reassigned to other samples.

PROTOCOL 4a makes the draws inputs to the update rule, so a run handed a different assignment is
evaluating a different function and its gradient is not comparable to the reference's. The draw
SET is unchanged as a multiset -- only the sample axis of the 48-sample diffusion noise is
permuted -- so nothing about the magnitude or the distribution of the input moves. If the headline
does not move by orders of magnitude here, the headline is not reading the gradient.
"""
import torch

src = "/home/ttuser/of3t_refprec/bundle_ref/draws_recycles0.pt"
d = torch.load(src, map_location="cpu", weights_only=False)
r = d["torch_randn"]
idx = [i for i, t in enumerate(r) if tuple(t.shape) == (1, 48, 422, 3)]
assert len(idx) == 1, idx
i = idx[0]
g = torch.Generator().manual_seed(4242)
perm = torch.randperm(48, generator=g)
assert not torch.equal(perm, torch.arange(48))
r[i] = r[i][:, perm].contiguous()
out = "/home/ttuser/of3t_refprec/bundle_ref/draws_recycles0_PERMUTED.pt"
torch.save(d, out)
print("wrote", out, "perm[:8]", perm[:8].tolist())
