# The seven TRAIN branches composed, and the accuracy gates read on the composition

Every device-side accuracy invariant this campaign owns had been measured branch by branch, on the
branch that introduced it, and never on the tree that would actually land. This directory is that
tree read once.

Composed from `origin/main` at `a4319c696`, merging in order: `train-orchestrator`,
`train-b2-abb3-port`, `train-g-perf-citations`, `train-f-no-global-grad`, `train-b3-train`,
`train-h-fape-24mb`, `train-j-multihost`. Head `eeb0cc99e`. One conflict, in `docs/training.md`
only, no code file: see the state doc.

Card: qb1 card 0, p150a, board id `00000403319140fd`, bus `0000:01:00.0`. AICLK sampled DURING each
run, reported per gate in the state doc.

## Files

| file | what it is |
|---|---|
| `reference_gate.txt` | composed tree, host float64 |
| `reference_gate_b2ctl.txt` | `wk/train-b2-abb3-port` head alone, same machine |
| `loss_gate_train-m-compose.txt` | composed tree vs upstream `ABodyBuilder3/src` |
| `loss_gate_train-m-ctl-b2.txt` | B2 head alone, same upstream |
| `model_gate_compose.txt` | composed tree, card 0, `--sigma 0.05 --tokens 64 --batch 2` |
| `model_gate_b2ctl.txt` | B2 head alone, card 0, same args |
| `model_gate_ca7a.txt` | B2 at `ca7a7573e`, card 0, same args. That commit wrote `perf/abb3_port/model_gate_qb1c3.txt` |
| `step_gate_compose_m2.txt` | composed tree, `--micro 2 --accumulate 32`, the args the recorded run used |
| `step_gate_compose.txt` | composed tree at the SHIPPED defaults: OOM |
| `step_gate_b2ctl.txt` | B2 head alone at the shipped defaults: the same OOM, so not a composition defect |
| `op_gradcheck_compose.txt` | composed tree, 27 cases, card 0 |

## The controls are the point

A number from the composed tree on its own proves nothing, because it has no baseline that shares a
card. Every gate here was run twice, composed and B2-alone, on the same card in the same hour. Where
the two agree digit for digit, composition changed nothing, whatever the branch's recorded file says.
Where the composed tree and the recorded file disagree but the two live runs agree, the recorded file
is what moved.
