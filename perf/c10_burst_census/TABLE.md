# Bounded identity-control cycle table

These are 16 small smoke invocations at sampled 1350 MHz, with node-3 co-tenancy and graph/profiler instrumentation. They are not a model census or a roof. All outputs equal the independent float64 references exactly.

| Control | Calls | Raw program spans, cycles | Fenced envelope, cycles | Unclassified gaps, cycles | Matrix shape FLOPs | Modeled compulsory bytes | Roof / utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| reblock_on | 4 | 321495 | 129475896 | 129154401 | not a contraction | 2097152 | unmeasured |
| reblock_off | 4 | 320931 | 582077952 | 581757021 | not a contraction | 2097152 | unmeasured |
| matmul_off | 4 | 12490 | 713340508 | 713328018 | 2097152 | 98304 | unmeasured |
| matmul_on | 4 | 12480 | 109325518 | 109313038 | 2097152 | 98304 | unmeasured |

Physical reads and issued arithmetic remain unknown. Gaps are not CPU work. The four control envelopes are disjoint; no parent/child sum or invocation extrapolation is used. Zero model invocations were captured. Matmul uses HiFi4 with FP32 accumulation; reblock's source descriptor uses HiFi2. Neither is a throughput roof.

STOP: CoreRange getter mismatch prevents per-core runtime snapshots. Repair and revalidate the observer before any current-default model windows or matched roofs. No whole-fold floor, binding constraint, headroom or campaign ceiling follows from this table.
