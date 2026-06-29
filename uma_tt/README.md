# Meta UMA on Tenstorrent (ttnn) — experimental port

Forward energy+forces inference of Meta UMA (fairchem `eSCNMDBackbone` / eSEN, MoLE
pre-merged to plain Linear at inference) on Blackhole P150 via ttnn. See `REPORT.md` for
the full results & verdict.

## TL;DR results (random weights — facebook/UMA download is HF-gated 403)
- Profile: ~85-90% of compute is dense GEMM/BMM; Wigner-D construction + radius graph <1% (host).
- SO2_Convolution (dominant module) ported fully device-resident: PCC so2_1 1.00000, so2_2 0.99990.
- End-to-end forward through TT: node-embedding PCC 0.99999-1.0, energy relerr <1%.
- Forces (autograd-free, FD on TT energy): PCC 0.999 (H2O, C2H6).
- Throughput: 3.5 Medges/s per card (~18 TFLOP/s eff, ~45x vs CPU on the dominant module);
  4-card fan-out 14.1 Medges/s, perfect linear scaling.

## Environment (isolated from production ~/tt-bio/env)
Uses `~/.uma_run/env` (copy of the prod venv + fairchem-core 2.21, ase, pymatgen; numpy pinned
<2 for ttnn). Always set:
    export TT_MESH_GRAPH_DESC_PATH=$VENV/lib/python3.12/site-packages/ttnn/tt_metal/fabric/\
mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
(otherwise ttnn.open_device aborts: "Custom fabric mesh graph descriptor path must be specified").

## Files
- `ref_harness.py`   — instantiate eSCNMDBackbone (random, seed 0) + energy head; build
                       AtomicData (ASE+pymatgen edges); CPU energy+autograd forces; save golden.
- `profile_ref.py`   — torch.profiler op-time breakdown (matmul vs geometric) on Cu supercells.
- `capture_so2.py`   — capture golden I/O + weights of the two SO2_Convolution modules.
- `tt_smoke.py`      — ttnn GEMM PCC + perf smoke test.
- `tt_so2.py`        — ttnn SO2_Convolution (device GEMMs + host glue), PCC vs golden.
- `tt_so2_resident.py` — FULLY device-resident SO2 core (block-GEMM per-m, tile-aligned slices);
                       PCC validate + throughput bench (`--bench`), one card per process.
- `tt_e2e.py`        — wire TT SO2 convs into the real model; end-to-end energy/node PCC vs golden.
- `tt_forces_fd.py`  — autograd-free finite-difference forces from the TT energy surface.

## Run
    cd ~/.uma_run            # scripts import ref_harness; golden lives here
    P=~/.uma_run/env/bin/python
    TT_VISIBLE_DEVICES=0 $P ref_harness.py
    TT_VISIBLE_DEVICES=0 $P profile_ref.py
    TT_VISIBLE_DEVICES=0 $P capture_so2.py
    TT_VISIBLE_DEVICES=0 $P tt_so2_resident.py --device 0 --bench
    TT_VISIBLE_DEVICES=0 $P tt_e2e.py
    TT_VISIBLE_DEVICES=0 $P tt_forces_fd.py
