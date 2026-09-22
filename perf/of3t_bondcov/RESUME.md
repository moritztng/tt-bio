# Where this row is, if it is picked up again

Done and committed: the mask measurement (both crops), the collation fix, the coverage table's
DEMONSTRATED half with of3t-gradients' seven terms re-derived from their artifact, and the state-doc
renderer.

The one thing that has to run on the host is the gradient arm. It takes about an hour on pc:

    bash perf/of3t_bondcov/gradrun.sh 12 4g5j_c1

Its inputs live outside the repo and survive between launches:

    /home/moritz/.coworker/scratch/of3t-bondcov/of3pkg043                 openfold3 0.4.3
    /home/moritz/.coworker/scratch/of3t-bondcov/datasets                  the 4g5j+4byh corpus
    /home/moritz/.coworker/scratch/of3t-bondcov/batch_step003.pt          rank template
    /home/moritz/.coworker/scratch/of3t-bondcov/training_yamls            upstream stage configs
    /home/moritz/.boltz/of3-p2-155k.pt                                    checkpoint
    /home/moritz/of3-upstream-venv                                        torch 2.13.0+cpu

Rebuild the corpus if the scratch is gone:

    python scripts/of3_port/build_of3_subset.py \
        --target-dir /home/moritz/.coworker/scratch/of3t-bondcov/datasets \
        --split train --ids 4g5j,4byh
    # corpus-digest e20b564af303d16ecf155cbc283ae773f7a5df2c3e02d655d3c80d8bdf505975

Then the state doc, which is rendered and not written by hand:

    python perf/of3t_bondcov/make_evidence.py --report perf/of3t_bondcov/bond_gradient_4g5j_c1.json \
        --out perf/of3t_bondcov/evidence/bond.json
    python perf/of3t_bondcov/render_state.py \
        --mask perf/of3t_bondcov/bond_mask_4g5j_4byh.json \
        --mask2 perf/of3t_bondcov/bond_mask_4g5j_4byh_crop256.json \
        --grad perf/of3t_bondcov/bond_gradient_4g5j_c1.json \
        --baseline perf/of3t_auxheads/bond_coverage_finetune1.json \
        --evidence perf/of3t_bondcov/evidence \
        --template perf/of3t_bondcov/state_template.md \
        --out /home/moritz/.coworker/state/of3t-bondcov.md
