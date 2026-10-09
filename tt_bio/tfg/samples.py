"""The rigid passes of guidance on several host cores: one child process per group of samples.

The rigid solvers act on each sample alone and their turns are batch-invariant (``rigid.small_rotation``),
so a group's output equals the rows of the batched output exactly. Only the rigid passes go to the
children: the per-step potentials stay batched in the fold's process, where batching amortises the op
overhead better than a process per group would.

The children are plain ``python -m tt_bio.tfg.samples`` processes talking pickle over their pipes,
so nothing of the parent (its ``__main__``, the open device) is imported or inherited.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import torch


class SampleWorkers:
    """``workers`` one-thread children, each guiding a contiguous group of the samples with its own Guidance.

    Started with the Guidance, so their start-up (an interpreter and the tt_bio import, ~3 s) runs while the
    trunk does; a step splits whatever samples it gets."""

    def __init__(self, feats, schedule, config, workers):
        self.workers = workers
        self.procs = []
        root = str(Path(__file__).resolve().parents[2])                 # the tt_bio this process runs
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for _ in range(workers):
            p = subprocess.Popen([sys.executable, "-m", "tt_bio.tfg.samples"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, env=env)
            pickle.dump((feats, schedule, config), p.stdin)
            p.stdin.flush()
            self.procs.append(p)

    def call(self, method, x, *args):
        """Guidance.<method>(x[group], *args) in every child, rows concatenated back in order."""
        rows = [r for r in torch.arange(x.shape[0]).tensor_split(self.workers) if len(r)]
        for p, r in zip(self.procs, rows):
            pickle.dump((method, x[r], args), p.stdin)
            p.stdin.flush()
        return torch.cat([self._recv(p) for p in self.procs[:len(rows)]])

    @staticmethod
    def _recv(p):
        out = pickle.load(p.stdout)
        if isinstance(out, BaseException):
            raise out
        return out

    def close(self):
        for p in self.procs:
            try:
                pickle.dump(None, p.stdin)
                p.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            p.wait()
        self.procs = []


def _serve():
    torch.set_num_threads(1)
    from tt_bio.tfg.guidance import Guidance
    src, dst = sys.stdin.buffer, sys.stdout.buffer
    feats, schedule, config = pickle.load(src)
    g = Guidance(feats, schedule=schedule, config=config)
    while True:
        msg = pickle.load(src)
        if msg is None:
            return
        method, x, args = msg
        try:
            out = getattr(g, method)(x, *args)
        except Exception as exc:  # noqa: BLE001 -- handed to the parent, which raises it
            out = exc
        pickle.dump(out, dst)
        dst.flush()


if __name__ == "__main__":
    _serve()
