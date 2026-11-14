"""An added loss term, in a real `losses.csv` written by BindCraft 2s own recorder.

`TrajectoryRecorder.__call__` records every scalar in `prediction.metrics`, and `af2.py` puts each
loss terms weighted value there, so a term added through `bindcraft2.loss_terms` gets a column in
`losses.csv` with no further wiring. This drives the recorder over the same small gradient design
loop `perf/fdx_loss/design_run.py` uses, so the file is written by BindCraft 2 and not by us.

    PYTHONPATH=<bindcraft2>:<repo>/perf/fdx_loss python3 losses_csv.py --steps 3 --out /tmp/traj
"""
import argparse
import shutil
import sys

import jax

import design_run
from bindcraft.loss import build_design_losses
from bindcraft.protein import Protein
from bindcraft.sequence_optimization import LogitSequenceOptimizer
from bindcraft.target_schedule import losses_for_active_states
from bindcraft.trajectory import transfer_binder_sequences
from bindcraft.trajectory_output import TrajectoryRecorder

import effect_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=design_run.AF2_DATA_DIR)
    arguments = parser.parse_args()
    design_run.AF2_DATA_DIR = arguments.data_dir
    shutil.rmtree(arguments.out, ignore_errors=True)

    model_key, binder_key = jax.random.split(jax.random.PRNGKey(arguments.seed))
    from bindcraft.af2 import AlphaFoldDesignModel
    model = AlphaFoldDesignModel(presets="model_1_ptm", data_dir=arguments.data_dir, key=model_key,
                                 num_recycle=1, length_bucket_size=design_run.LENGTH_BUCKET, dropout=False)
    protein_states = {design_run.TARGET_STATE: {
        design_run.BINDER_CHAIN: Protein.empty(design_run.BINDER_LENGTH, binder_key),
        design_run.TARGET_CHAIN: Protein.from_fasta(design_run.TARGET_SEQUENCE)}}
    optimizer = LogitSequenceOptimizer(iterations=arguments.steps)
    recorder = TrajectoryRecorder(arguments.out, keep_frames=False)
    recorder.design_stage = "gradient"
    recorder.sequence_parameters = optimizer.sequence_parameters

    with effect_terms.with_aromatic_binder():
        losses = losses_for_active_states(
            build_design_losses({}, {design_run.TARGET_STATE: 1.0}, design_run.LENGTH_BUCKET, arguments.seed),
            protein_states)
        print("terms " + ",".join(f"{n}={e.weight:+.3f}" for n, e in sorted(losses.items())), file=sys.stderr)
        for step in range(arguments.steps):
            softmax_weight, one_hot_weight, temperature, logit_scale = optimizer.sequence_parameters()
            predictions, chain_gradients, _ = model.sequence_gradients(
                protein_states, losses, model="model_1_ptm", softmax_weight=softmax_weight,
                one_hot_weight=one_hot_weight, temperature=temperature, logit_scale=logit_scale)
            recorder(step + 1, predictions)
            updated = optimizer.update_sequence(protein_states, {name: [gradient] for name, gradient in chain_gradients.items()})
            protein_states = transfer_binder_sequences(protein_states, updated)
    recorder.write_csv(f"{arguments.out}/losses.csv")
    print(open(f"{arguments.out}/losses.csv").read())


if __name__ == "__main__":
    main()
