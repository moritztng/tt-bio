# perf/wh-esmfold2 — ESMFold2 on the Wormhole Galaxy

Instruments for making ESMFold2 as fast as it goes on `UF-EV-A13-GWH02` (8x9 = 72 cores,
1,466,080 B unreserved L1 per core, 218.5 GB/s measured DRAM roof) without costing Blackhole
anything. Plan, gate census and the ranked levers: `~/.coworker/state/wh-perf-esmfold2.md`.

The five scripts are the ones that produced the Blackhole 512 aa decomposition, recovered from
`origin/wk/esmfold2-512aa-deep-perf:perf/esm512/` so the two architectures are measured by the same
instrument rather than by two harnesses that agree by luck. Three changes were needed:

- `roofs.py`, `screen.py`, `op_parity.py` hardcoded `BlackholeComputeKernelConfig`, which throws on
  a wormhole_b0 part. They now dispatch on `tenstorrent.is_wormhole()`, the same way tt-bio does.
- `decomp.py` and `fold_ab512.py` never passed `fast` to `build_fold`, so they could only fold in
  normal precision. Wormhole forces esmfold2 into `--fast` (the ESMC-6B LM is ~12.8 GB against a
  ~12 GB chip), so without the flag the run OOMs. A `--fast` arm compares only to another one.
- `roofs.py` and `screen.py` gained `--fast`, which sets the module flag before any module is
  built, and `roofs.py` gained the `bfloat8_b` matmul roof — `--fast` executes bf8, so bf8 is the
  roof it runs against, and the Wormhole compute roof had never been measured at all.

| script | what it produces |
|---|---|
| `roofs.py` | DRAM + matmul roofs (bf16 and bf8, three fidelities, both grids) and per-body op walls |
| `decomp.py` | one instrumented fold, inclusive and exclusive time per named component |
| `screen.py` | op-level A/B of a lever at the production shape, the screen that decides GO/NO-GO |
| `op_parity.py` | which op in a rewrite loses bit-exactness, differing-element count and max abs |
| `fold_ab512.py` | the fold-level A/B, arms interleaved in one process, CIF sha256 + plDDT per arm |

Every timed run goes under `benchlock.sh`. On this host prod is the steady state, not a co-tenant:
the default `BENCHLOCK_FOREIGN_RE` matches the 26 live JapanFold workers, so narrow it to our own
harnesses and raise `BENCHLOCK_MAXLOAD`. Exact settings and the reasoning are in the state doc.
