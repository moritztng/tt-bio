"""One background thread for host work the chip should not wait for.

A fold has a few places where the chip sits idle while the host builds or converts a tensor:
the trunk's input features, the trunk pair the confidence head reads, each confidence sample's
logits. The work itself does not change; it moves to this thread so the main thread can keep
enqueueing device work, and the result is collected where it used to be computed. Torch ops
and ttnn's host untilize (`Tensor.to_torch`) release the GIL, so the lane really runs beside a
main thread that is dispatching or blocked on a device read. ttnn's host tilize (the tensor
constructor) does not, so uploads stay on the main thread.

One worker, so jobs run in submission order. Set TT_BIO_HOST_LANE=0 to run every job inline
at submit time, which is the same arithmetic in the old order.
"""
from concurrent.futures import Future, ThreadPoolExecutor

from .envflags import env_flag

_POOL = None


def enabled():
    return env_flag("TT_BIO_HOST_LANE", True)


def submit(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` on the lane; returns a Future. Inline when the lane is off."""
    global _POOL
    if not enabled():
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:   # surfaced by .result(), as the lane would
            fut.set_exception(exc)
        return fut
    if _POOL is None:
        _POOL = ThreadPoolExecutor(1, thread_name_prefix="tt-bio-host-lane")
    return _POOL.submit(fn, *args, **kwargs)


def read(t):
    """``ttnn.from_device(t)`` without the GIL. Both are ``Tensor::cpu``, but the ``ttnn.from_device``
    binding keeps the GIL while it waits for the chip to reach the read, and a confidence sample's
    read waits the sample's whole device time. Through it the lane ran none of its post-processing
    until the last sample was read (measured: 4.4 s of lane work queued at the end of a c730 fold)."""
    return t.cpu()


def to_torch(t, then=None):
    """A device tensor's bytes, split the way ``ttnn.to_torch`` does it internally: the device read
    happens now, on the calling thread (it is ordered on the command queue), and the host untilize
    runs on the lane, followed by ``then`` if given. Returns a Future of ``then(ttnn.to_torch(t))``."""
    import torch
    h = read(t)
    return submit(lambda: (then or (lambda x: x))(torch.Tensor(h.to_torch())))
