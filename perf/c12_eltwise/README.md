# c12_eltwise — is a hand-written eltwise chain actually adjacent in the graph the device runs?

The question this directory answers, and the reason a grep cannot: fusing two ops into one deletes
the producer's output write and the consumer's read of it, but only if the producer's result has
**exactly one consumer**. On a pre-norm residual trunk the sum of `add_(z, update)` is read both by
the next sub-layer's norm and by the next in-place add, so it has two consumers and there is
nothing to fuse. A source-level count of adjacent `multiply`/`add` lines cannot see that.

    run_trace.sh [card]        one Boltz-2 512 aa fold, recording the executed graph
    trace_graph.py             the tracer: wraps 44 ttnn ops, records (buffer address, version)
    analyze_pairs.py           reduces a trace into fusable pairs, priced in bytes then seconds

Three things the tracer gets right on purpose:

**Dataflow is keyed on (buffer address, version), never on tensor id.** ttnn reuses freed
addresses, so an address alone aliases unrelated tensors. A version counter incremented on every
write to that address gives clean SSA numbering, and a read resolves to the most recent definition.

**Every op that can read a buffer is wrapped**, not just the ones being fused. One unwrapped reader
would make its producer look single-consumer when it is not, which is the error direction that
invents a fusion that then fails.

**The run is untimed.** Graph shape and def-use edges do not depend on the clock or the card, so
this takes no benchlock and pins no AICLK. Seconds come from `c10-fold-census`'s measured per-key
rates: keys it measured at or above 97 % of the 442.9 GB/s DRAM roof convert deleted bytes at the
roof, every other key at its own measured rate.

Usage:

    perf/c12_eltwise/run_trace.sh 2
    perf/c12_eltwise/analyze_pairs.py --trace perf/c12_eltwise/runs/trace_512_c2.json \
        --census <c10-fold-census sweep2 budget.json> --out perf/c12_eltwise/runs/pairs_512.json

Traces are large and are gitignored; commit the reduced `pairs_*.json` instead.
