"""Time each rfd4 trunk call from launch to readback.

The engine drives the trunk as `launch` (upload and enqueue) then `collect` (readback), and
`run_bucketed` is the two back to back, so a harness that wraps `run_bucketed` alone records
nothing. With one lane a span is the trunk call's wall time; with interleaved lanes the spans
overlap, so their sum can exceed the trunk's share of a design.
"""
import time


def time_trunk(trunk, sink: list) -> None:
    """Append one launch-to-readback span, in seconds, to `sink` per trunk call."""
    launch, collect = trunk.launch, trunk.collect
    started = {}

    def timed_launch(*args, **kwargs):
        t = time.perf_counter()
        pending = launch(*args, **kwargs)
        started[id(pending)] = t
        return pending

    def timed_collect(pending):
        t = started.pop(id(pending))
        out = collect(pending)
        sink.append(time.perf_counter() - t)
        return out

    trunk.launch, trunk.collect = timed_launch, timed_collect
