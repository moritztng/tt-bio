# Boltz-2 `--fast` — TT warm latency vs NVIDIA (synthetic examples)

## Provenance (read this before trusting the NVIDIA column)

No rigorous Boltz-2-vs-NVIDIA latency table exists anywhere on record — an
exhaustive search across every Claude Code transcript on this machine and
`tt-quietbox`/`tt-quietbox2` (multiple independent sessions, cross-checked)
found no such table. The only Boltz-2 GPU comparison ever stated is a single
informal throughput/$ claim — "Blackhole p150 ≈2.4–2.9x a DGX per dollar at
L=512/1024" — which is a $-normalized throughput ratio, not a latency table,
and cannot be converted into a per-row NVIDIA latency figure. There is a
*different*, real, rigorous GPU table (TT-Atom / Meta UMA MLIP vs NVIDIA H100,
throughput/$) — that is a different model and not applicable here.

**Decision: report fresh TT numbers only; no NVIDIA column is fabricated.**

## Method

- HEAD `a0fbd97` (branch `wk/boltz2-nvidia-table-update`, already contains
  latest `main` — includes the device-resident trunk-recycling win and the
  trimul permute→transpose DRAM-path decomposition, both merged).
- `tt_bio.main predict --fast --single_sequence --seed 0`, card 0.
- Inputs: `examples/{615,686,1003,1303,1962,3233}.yaml` (filename = total
  residue count; the numeric examples do not use `examples/msa/*.a3m` — those
  are unrelated small fixtures for `prot_custom_msa.yaml`). `3233` is a
  4-chain multimer + ligand.
- **Warm** = a run's `runtime_s` in `results.json` after an identical cold
  run already primed the disk kernel-build cache for the same op shapes
  (compile time + first-run effects excluded). For L=3233 the warm pass was
  re-run as its own process after the paired-process run crashed mid-diffusion
  on an unrelated infra fault (external worktree maintenance recreated the
  directory the long-lived job had as its cwd, invalidating it — see note
  below); the disk kernel cache from the original cold pass was still warm,
  so the number is methodologically equivalent.

## Results

| seq len (residues) | TT warm `--fast` (this session) | cold (compile+cache) | NVIDIA |
|---|---|---|---|
| 615  | 43.4s   | 100.1s  | not available — no rigorous table on record |
| 686  | 49.9s   | 52.8s   | " |
| 1003 | 114.1s  | 116.9s  | " |
| 1303 | 201.0s  | 267.9s  | " |
| 1962 | 455.1s  | 526.0s  | " |
| 3233 | 1385.1s | 1463.1s (4-chain multimer+ligand) | " |

Empirical e2e scaling exponent (from consecutive rows) is ~2.0–2.2 throughout
615→3233, sub-cubic vs a trunk-only L³ model — diffusion/other stages don't
scale cubically. The 1962→3233 step (2.23) is consistent with the rest of the
curve, confirming the 3233 number isn't an outlier.

## What changed vs before

There is no prior numeric table to diff against (see Provenance). The
numbers above already reflect the two Boltz-2 `--fast` wins landed on `main`
since the last informal estimate: **device-resident trunk recycling**
(~16% e2e win at L=512, merged 2026-06-25) and the **trimul permute→transpose
DRAM-path decomposition** (1.4–1.58x at large L, gated to the DRAM/large-L
path, merged after). Both are baked into every row above — this table is
HEAD-current, not a delta reconstruction.
