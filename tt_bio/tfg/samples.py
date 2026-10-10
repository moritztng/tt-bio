"""Guidance for several samples on several host cores: one child process per group of samples.

Guidance is separable over samples (every term and solver acts on each sample alone, and the rigid
turns are batch-invariant, see ``rigid.small_rotation``), so a group's output equals the rows of
the batched output exactly. Batching amortises Python and op overhead, so this only pays when the
fold's host share is wider than the 1-2 threads a batched step uses well.

The children are plain ``python -m tt_bio.tfg.samples`` processes talking pickle over their pipes,
so nothing of the parent (its ``__main__``, the open device) is imported or inherited.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import torch


def _rows(v, rows, m):
    """The group's rows of a per-sample value (a tensor with leading dim m), else the value itself."""
    return v[rows] if torch.is_tensor(v) and v.ndim and v.shape[0] == m else v


class SampleWorkers:
    """``workers`` one-thread children, each guiding a contiguous group of the samples with its own Guidance.

    Started with the Guidance and fed from a thread, so their start-up (an interpreter and the tt_bio import,
    ~3 s) runs while the trunk does; a step splits whatever samples it gets."""

    def __init__(self, feats, schedule, config, workers):
        self.workers = workers
        root = str(Path(__file__).resolve().parents[2])                 # the tt_bio this process runs
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.procs = [subprocess.Popen([sys.executable, "-m", "tt_bio.tfg.samples"], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, env=env) for _ in range(workers)]
        # A child reads its features only once its imports are done, and the write blocks on the pipe until then:
        # hand them over from a thread so the fold goes on to the trunk instead of waiting out every start-up.
        self._sent = threading.Thread(target=self._send, args=((feats, schedule, config),), daemon=True)
        self._sent.start()

    def _send(self, msg):
        for p in self.procs:
            pickle.dump(msg, p.stdin)
            p.stdin.flush()

    def step(self, x_noisy, x0, **kw):
        self._sent.join()
        m = x_noisy.shape[0]
        rows = [r for r in torch.arange(m).tensor_split(self.workers) if len(r)]
        for p, r in zip(self.procs, rows):
            pickle.dump((x_noisy[r], x0[r], {k: _rows(v, r, m) for k, v in kw.items()}), p.stdin)
            p.stdin.flush()
        return torch.cat([self._recv(p) for p in self.procs[:len(rows)]])

    @staticmethod
    def _recv(p):
        out = pickle.load(p.stdout)
        if isinstance(out, BaseException):
            raise out
        return out

    def close(self):
        """Stop every child; a thread reaps them, so the fold does not wait out their interpreter teardown."""
        self._sent.join()
        for p in self.procs:
            try:
                pickle.dump(None, p.stdin)
                p.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        procs, self.procs = self.procs, []
        threading.Thread(target=lambda: [p.wait() for p in procs], daemon=True).start()


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
        x_noisy, x0, kw = msg
        try:
            out = g._step(x_noisy, x0, **kw)
        except Exception as exc:  # noqa: BLE001 -- handed to the parent, which raises it
            out = exc
        pickle.dump(out, dst)
        dst.flush()


if __name__ == "__main__":
    _serve()
