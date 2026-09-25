"""The rollout draws of the full step: sampled once by the float64 reference, replayed everywhere else.

`bundle_min.DrawRecorder` records `torch.randn` and `random.random`. The full step's rollout draws
through two more doors: upstream's EDM noise is `torch.randn_like` (diffusion_module.py), and the
two stacks ask for the same numbers in different shapes, upstream with its batch and sample axes
([1, 1, n_atom, 3], [1, 1, 4], [1, 1, 3]) and our `_gen_rollout` without them ([n_atom, 3], [4],
[3]). So this recorder routes `randn_like` through `randn` and compares draws by element count,
reshaping a recorded draw to the shape asked. A draw whose element count differs, a call with no
recorded draw left, and a recorded draw nobody consumed are each one mismatch.
"""
from __future__ import annotations

import torch


class Draws:
    def __init__(self, replay=None):
        self.replay = replay
        self.recorded: list[torch.Tensor] = []
        self.mismatch: list[str] = []
        self.i = 0

    def __enter__(self):
        self._randn, self._randn_like = torch.randn, torch.randn_like
        self.recorded, self.mismatch, self.i = [], [], 0
        orig = self._randn

        def randn(*args, **kwargs):
            out = orig(*args, **kwargs)          # always draw: the stream advances as it would
            if self.replay is not None:
                if self.i >= len(self.replay):
                    self.mismatch.append(f"call {self.i}: no recorded draw left")
                elif self.replay[self.i].numel() != out.numel():
                    self.mismatch.append(f"call {self.i}: recorded {tuple(self.replay[self.i].shape)}"
                                         f" != asked {tuple(out.shape)}")
                else:
                    out = self.replay[self.i].reshape(out.shape).to(out.dtype)
                self.i += 1
            self.recorded.append(out.detach().cpu().clone())
            return out

        def randn_like(x, *args, dtype=None, device=None, **kwargs):
            return randn(x.shape, dtype=dtype or x.dtype, device=device or x.device)

        torch.randn, torch.randn_like = randn, randn_like
        return self

    def __exit__(self, *exc):
        torch.randn, torch.randn_like = self._randn, self._randn_like
        if self.replay is not None and exc[0] is None and self.i != len(self.replay):
            self.mismatch.append(f"{len(self.replay) - self.i} recorded draws left unconsumed")
        return False
