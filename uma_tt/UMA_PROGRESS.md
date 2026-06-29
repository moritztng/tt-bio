UMA DONE
# UMA on Tenstorrent — autonomous build progress (qb2, branch exp/uma-tt)

## ==== FINAL STATUS: DONE (outcome a) ====
UMA forward energy+forces ported to Tenstorrent ttnn (Blackhole P150). Dominant module
(SO2_Convolution) fully device-resident, PCC 1.0/0.9999; end-to-end forward node-embedding
PCC 0.99999-1.0, energy <1%; autograd-free FD forces PCC 0.999. Throughput 3.5 Medges/s/card,
14.1 Medges/s on 4 cards (perfect linear scaling), ~45x vs CPU on the dominant module.
Random weights (facebook/UMA HF-gated 403; validated apples-to-apples). Committed a888f4c to
exp/uma-tt (uma_tt/). Full verdict in ~/.uma_run/REPORT.md and uma_tt/REPORT.md.
To do real-weights validation later: get HF access to facebook/UMA, drop checkpoint, reuse
ref_harness/tt_e2e (swap random state_dict for the loaded one).


This file is the source of truth across loop restarts. Each iteration: read it,
do the first unchecked step, update it. When fully done, first line -> `UMA DONE`.

## Goal
Meta UMA (fairchem) energy+forces forward inference on TT via ttnn, exp/uma-tt,
throughput-first, PCC>=0.98 vs PyTorch reference. See ~/.uma_run/uma_prompt.txt.

## Steps
- [x] 0. Isolated env DONE. ~/.uma_run/env is a copy of ~/tt-bio/env. ttnn imports OK.
       fairchem-core 2.21.0 + ase 3.29 installed into COPY ONLY. numpy pinned 1.26.4
       (ttnn needs <2; fairchem declares >=2 but runs fine on 1.26.4). torch 2.8.0+cu128
       in copy (CPU inference fine). ORIGINAL env verified UNCHANGED (numpy1.26.4/torch2.12).
       Worktree ~/tt-bio-uma on exp/uma-tt confirmed.
- [x] 1. Research DONE. See architecture map below. Weight access: token (moritztng) can
       READ facebook/UMA metadata but DOWNLOAD is 403 (manual approval, not granted) ->
       using RANDOM weights path (prompt-blessed). MoLE pre-merges to plain Linear at
       inference, so plain eSCNMDBackbone == MoLE-at-inference for the port.
- [ ] 2. Reference harness: eSCNMDBackbone (random wts) + energy head on CPU; energy+forces
       on a molecule + periodic system. Save inputs+outputs as ground truth. <- IN PROGRESS
- [ ] 3. Port matmul-heavy modules to ttnn incrementally; PCC>=0.98 per module.
- [ ] 4. Forces (autograd-free if feasible). PCC>=0.98.
- [ ] 5. Perf: device-resident, program cache, trace, batching, 4 cards.
- [ ] 6. Validate e2e PCC, MAE, size sweep, no OOM.
- [ ] 7. Commit; verdict to REPORT.md; first line -> UMA DONE.

## Architecture map (fairchem v2.21, model: UMA = eSEN / eSCN-MD backbone + MoLE)
Package: ~/.uma_run/env/.../fairchem/core/models/uma/
Model class: `eSCNMDBackbone` (escn_md.py). MoLE wrapper: escn_moe.py (no-op stubs in
base MOLEInterface -> plain backbone uses nn.Linear == pre-merged MoLE at inference).
UMA-s defaults: sphere_channels=128, lmax=2, mmax=2, num_layers=2(real uma-s≈6), hidden=128,
edge_channels=128, num_distance_basis=512, cutoff=5.0, max_neighbors=300, norm=rms_norm_sh,
act=gate, ff=grid, direct_forces=True default. sph_feature_size=(lmax+1)^2=9 for lmax2.

Forward (escn_md.py:672):
  1. csd_embedding (charge/spin/dataset embeddings -> mix_csd Linear)  [cheap]
  2. _generate_graph (otf radius graph)  [HOST geometric]
  3. _get_rotmat_and_wigner: per-edge Wigner-D rotation matrices from edge dir vectors
     (quaternion or Euler+Jd). wigner: [n_edges, |L|^2, |L|^2]-ish.  [HOST geometric, irregular]
  4. atom embedding (nn.Embedding) + edge degree embedding
  5. distance_expansion GaussianSmearing -> x_edge = [dist_basis(512)+src_emb+tgt_emb]
  6. num_layers x eSCNMD_Block:
       norm_1 (rms_norm_sh) -> Edgewise -> +res -> norm_2 -> Atomwise(grid) -> +res
     Edgewise (escn_md_block.py:41): node->edge via Wigner permute (BATCHED MATMUL over edges:
       wigner @ x_node), SO2_Convolution_1 (MATMUL), gate/s2 act, SO2_Convolution_2 (MATMUL),
       edge->node via wigner_inv permute+scatter.  <- HEAVY: SO2 conv matmuls + Wigner rotations
     Atomwise grid (GridAtomwise): to_grid (SH->grid, matmul w/ SO3_grid), grid_mlp (3x Linear
       + SiLU), from_grid.  <- HEAVY: grid_mlp matmuls + grid projection matmuls
  7. final norm -> node_embedding [N, |L|^2, sphere_channels]
Energy head (outputs.py compute_energy): take L=0 scalar [N,C] -> energy_block (Linear MLP)
  -> per-node energy -> reduce to system energy. Forces: direct head OR autograd -dE/dpos.

HEAVY OPS (TT targets, all matmul): SO2_Convolution (so2_layers.py, fc_m0 + per-m fc),
SO3_Linear (so3_layers.py), GridAtomwise.grid_mlp, to_grid/from_grid (SO3_Grid matmul),
distance_expansion, energy head. IRREGULAR/HOST: Wigner-D construction, radius graph, scatter.
The Wigner ROTATION application (wigner @ x) is a batched dense matmul -> TT-friendly; the
Wigner-D *construction* is irregular -> keep on host. MEASURE shares in step 2.

## Measurements (REPORT.md has the table)
(pending step 2 profile)

## Progress log
- [x] 2 DONE. ref_harness.py runs eSCNMDBackbone(random) + energy head, autograd forces, CPU.
      Golden saved to ~/.uma_run/golden/ (model_random.pt, ref_results.pkl, so2_io.pt).
      profile_ref.py -> Cu-256: matmul (mm+bmm+addmm)=~1.2s dominates; Wigner+graph <0.5%.
- [x] 3 PARTIAL. tt_so2.py ports SO2_Convolution (the dominant module) to ttnn.
      so2_conv_1 PCC 0.99999 (gate 0.99998), so2_conv_2 PCC 0.99990. Micro-GEMM 60x vs CPU/card.
      ttnn device open needs TT_MESH_GRAPH_DESC_PATH=<env>/ttnn/tt_metal/fabric/
      mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto  (else CUSTOM cluster TT_FATAL).
      Key layout insight: flatten [E,9,C]->[E,1152]; m-blocks are TILE-ALIGNED channel slices
      (m0=[0:384],m1=[384:896],m2=[896:1152], all mult of 32) -> fully device-resident feasible.

## Progress log (cont.)
- [x] 3 DONE. tt_so2_resident.py: fully device-resident SO2 core (block-GEMM per-m, tile-aligned
      slices, radial+concat on device). PCC so2_1=1.0, so2_2=0.9999. Bench: 3.5 Medges/s 1-card,
      14.1 Medges/s 4-card (perfect linear). ~18 TFLOP/s eff, ~45x vs CPU/layer.
- [x] 5 DONE (throughput+multicard, see REPORT). [x] e2e forward: tt_e2e.py wires TT convs into
      real model -> node_embedding PCC 0.99999-1.0, energy relerr <1% on H2O/CH4/C2H6/Cu.
- [~] 4 forces: autograd CANNOT flow through ttnn.to_torch (graph break) -> forces via autograd
      are wrong. Correct design = autograd-free (FD validates energy surface; analytic = transpose
      matmuls on TT). NEXT: tt_forces_fd.py finite-diff forces on TT energy vs ref autograd forces.

## Current blocker / NEXT action
NEXT: (a) tt_forces_fd.py: central-difference forces from TT energy on H2O+C2H6, compare PCC/cos
to ref autograd forces (proves TT energy surface -> correct forces, autograd-free). (b) finalize
REPORT verdict, commit code to exp/uma-tt, set UMA DONE. Use ~/.uma_run/env/bin/python; always
export TT_MESH_GRAPH_DESC_PATH (p150 textproto). Cards 0-3 free, no production running.
