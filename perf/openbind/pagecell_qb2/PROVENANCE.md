# OpenBind-0 on the perf page fixture — feasibility fold (NOT a page cell)

Tree under test: a clean merge of the merge-ready land branch into main, made only to answer
"can OpenBind fold this page's own fixture at all". OpenBind is not on `origin/main`.

    merge parents : origin/wk/flight-land-openbind c45a627c  +  origin/main aaf05c9b
    merge result  : 549b17fd (detached, no conflicts, 0 unmerged paths)
    fixture       : perf/size512/fixtures/cdk2x2_512.yaml (512 tokens, 4116 atoms)
    harness       : perf/of3_4xpd/xmodel_ab.py --model openbind --size 512 --repeat 1
    host/card     : tt-quietbox2 card 2 (card 3 was leased and live to worker:rfd3-b8-to-4x-p3)
    ttnn          : 0.68.0   grid 11x10   recycles 3   sampling 200   samples 1

The merge commit is detached and will be pruned; the two parents above reproduce the tree exactly.

## Result

    cold  43.069 s   plddt 0.559887   cif f5ae1145ccb4842b
    warm  38.076 s   plddt 0.559887   cif f5ae1145ccb4842b

Published OpenFold3 p150a cell on the same fixture: 38.254 s, plddt 0.547851
(perf/of3_4xpd/xmodel_qb2c3/openfold3_B1.json 38.261, _B2.json 38.212).

## Why this is not the published cell

One warm fold, no A/A repeat, no benchlock hold, loadavg 3.45 rising to 4.84 against benchlock's
2.0 gate. It answers feasibility and nothing else. The cell needs Stage B2 of
state/perf-page-newmodels-catchup.md: six warm folds across two processes, alternated with a
same-day OpenFold3 control, in one quiet benchlock hold.
