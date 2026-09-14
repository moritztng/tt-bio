# Predicted before the capture ran

`FUSION_PAIRS_DIFFUSION.md` reads 322.1 GB/fold of single-use DRAM round trip across the two
diffusion layer shapes, off `perf/b2x_difflayer/graph_512_all.json.gz`. This row re-takes the same
ranking from a buffer-keyed, call-site-tagged trace. The prediction is written here first.

## PREDICTED: below 322.1 GB/fold, and by a large fixed amount, not a tolerance

The graph capture's commit is `65eac7c9a`, 2026-09-11 15:04 UTC. `fc7fed56f`, 2026-09-11 19:43 UTC,
flipped `BOLTZ2_TOKEN_DIT_SDPA` from False to True. The capture therefore records the token DiT's
UNFUSED attention: `transpose -> matmul -> add_ -> multiply_ -> softmax -> matmul`. On today's tip
that whole chain is one `ttnn.transformer.scaled_dot_product_attention`, and the 8.39 MB probability
matrix is never allocated. The table's #1 row, `softmax -> matmul` at 80.53 GB/fold, is predicted to
be ABSENT from a default-arm trace.

Second flip in the same commit: `TT_BIO_ATOM_AXIS_BUCKET` False -> True, which buckets the atom
window count. If it changes the atom layer's shape from the captured `1x224x32x128`, the atom
column moves too, in an unpredicted direction.

So two arms, and they predict different things:

  CONTROL  `BOLTZ2_TOKEN_DIT_SDPA=0 TT_BIO_ATOM_AXIS_BUCKET=0`, the capture's own configuration.
           Predicted to AGREE with the published per-layer figures within **20 %** on the
           single-use round-trip total, and to reproduce the top three producer->consumer pairs of
           each layer by op code. 20 %, not 5 %: the two instruments disagree structurally.
           `itemize` charges a DRAM buffer to the op that allocated it and needs a hand-written
           view-alias exclusion to avoid inventing traffic (that rule alone moved the published
           total 425.9 -> 322.1, 24 %). The trace keys on the device address plus a generation
           counter, so an alias is the same buffer by construction and needs no rule. Where they
           differ, the trace is the one with the identity.

  DEFAULT  no overrides, today's `main`. Predicted **235-250 GB/fold**: 322.1 minus the 80.53 the
           token-DiT SDPA already deleted, plus or minus whatever the atom bucket moves.

## Direction of the residual disagreement, if the control misses

Predicted sign, if the control arm and the published table disagree: the **trace reads lower**. Both
instruments are counting the same physical allocations, but `itemize`'s view exclusion is a set of
op names (`NO_TRAFFIC`), so any aliasing op not in that set is still counted as a producer. The
trace cannot make that error; it can only make the opposite one, of missing an allocation the
wrapper never saw, and the wrapper sits on every `ttnn` operation in the module tree.

## Not a performance number

The tracer wraps every `ttnn` operation and walks the Python stack on each recorded call. That is a
large per-op host cost. Every figure this row produces is a BYTE count. No wall clock from a traced
fold means anything and none is quoted.
