# Bounded identity-control cycle table

These are 16 small smoke invocations at sampled 1350 MHz, with node-3 co-tenancy and graph/profiler instrumentation. They are not a model census or a roof. All outputs equal the independent float64 references exactly.

| Control | Calls | Raw program spans, cycles | Fenced envelope, cycles | Unclassified gaps, cycles | Matrix shape FLOPs | Modeled compulsory bytes | Roof / utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| reblock_on | 4 | 321733 | 129245496 | 128923763 | not a contraction | 2097152 | unmeasured |
| reblock_off | 4 | 321299 | 103379070 | 103057771 | not a contraction | 2097152 | unmeasured |
| matmul_off | 4 | 12695 | 80131259 | 80118564 | 2097152 | 98304 | unmeasured |
| matmul_on | 4 | 12393 | 110312752 | 110300359 | 2097152 | 98304 | unmeasured |

Physical reads and issued arithmetic remain unknown. Gaps are not CPU work. The four control envelopes are disjoint; no parent/child sum or invocation extrapolation is used. Zero model invocations were captured. Matmul uses HiFi4 with FP32 accumulation; reblock's source descriptor uses HiFi2. Neither is a throughput roof.

Verdict: GO. Live bounded identity, float64 output, multiplicity and address rebinding controls pass; model census remains required. No whole-fold floor, binding constraint, headroom or campaign ceiling follows from this table.
