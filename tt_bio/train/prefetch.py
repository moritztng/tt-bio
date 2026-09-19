"""Build the next step's micro-batches while the card is still working on this one.

``abb3_run``'s loop assembles its micro-batches, then times ``step.step()``. Only the second
half was ever instrumented, so the first was free in the history and not free in the wall clock:
2.232 s a step on the first 5-day leg, 17.7 % of its cadence, with the card idle for all of it.

The seam is :meth:`SabdabFvs.host` / :meth:`SabdabFvs.upload`. The host half is ``torch.load``
plus padding and stacking and touches no device; the upload half is the ``ttnn.from_torch`` calls
and the two on-card feature expansions, and must stay on the thread that owns the card, because
one process owns one device context and ttnn is not re-entrant across threads. So the worker
thread runs ahead building host tensors and the training thread does nothing but upload them.

**Depth is bounded and small on purpose.** Each queued step holds its whole micro-batch set in
host memory. Depth 1 covers one step of device work, which is the entire opportunity, and deeper
queues buy nothing. The host half used to build the 132-channel pair one-hot as well, 138 MB a
micro-batch, so depth was also a memory decision; :mod:`tt_bio.train.abb3_features_device` now
expands that on the card from 8 KB of index and the memory argument is gone with it.
"""

from __future__ import annotations

import queue
import threading

__all__ = ["host_stream"]

_END = object()


def host_stream(plan, pick, dataset, depth: int = 0):
    """Yield ``(batch, host_micros)`` for every batch in ``plan``.

    ``pick(batch)`` returns this rank's micro-batch index lists. ``depth`` 0 is the serial
    build, in line, exactly as the loop did it before; ``depth`` n runs it on one worker thread
    with at most n steps queued ahead.
    """
    if depth <= 0:
        for b in plan:
            yield b, [dataset.host(ix) for ix in pick(b)]
        return

    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        try:
            for b in plan:
                q.put((b, [dataset.host(ix) for ix in pick(b)]))
        except BaseException as exc:          # noqa: BLE001 - re-raised on the training thread
            q.put(exc)
        else:
            q.put(_END)

    t = threading.Thread(target=work, name="abb3-host-prefetch", daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                # Raised HERE, on the training thread, so a broken loader stops the run instead
                # of stalling it behind a queue that will never fill again.
                raise item
            yield item
    finally:
        # A caller that breaks early (max_seconds, a tripwire) leaves the worker blocked on a
        # full queue. Draining lets it see the daemon flag and exit with the process.
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
