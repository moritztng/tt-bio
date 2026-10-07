"""The custom-loss seam on a Tenstorrent chip: the same design loop, trunk on card.

`perf/fdx_loss/design_run.py` proves the seam on BindCraft 2s own JAX trunk. This runs the
identical loop with tt-bios Evoformer on a chip, through `bindcraft2.predictor(card=N)`, with a
term added by `bindcraft2.loss_terms`. It samples the cards own AICLK while the loop runs, because
a fold time without the clock it was measured at is not a measurement.

    PYTHONPATH=<bindcraft2>:<repo>/perf/fdx_loss python3 card_run.py --card 0 --steps 4 \\
        --loss-attr with_aromatic_binder --json out.json
"""
import argparse
import contextlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import jax

import design_run
from bindcraft.loss import build_design_losses
from bindcraft.protein import ATOM_INDEX, Protein
from bindcraft.prediction import collect_shared_chains
from bindcraft.sequence_optimization import LogitSequenceOptimizer
from bindcraft.target_schedule import losses_for_active_states
from bindcraft.trajectory import transfer_binder_sequences


class Clock(threading.Thread):
    """AICLK off the cards own sysfs node, sampled during the loop and not before it."""

    def __init__(self, card: int, period: float = 2.0):
        super().__init__(daemon=True)
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        self.period, self.samples, self.stop = period, [], threading.Event()

    def run(self) -> None:
        while not self.stop.wait(self.period):
            try:
                self.samples.append(int(self.path.read_text().split()[0]))
            except OSError as error:
                self.samples.append(-1)
                print(f"aiclk read failed: {error}", file=sys.stderr)

    def summary(self) -> dict:
        good = [s for s in self.samples if s > 0]
        if not good:
            return {"n": 0, "path": str(self.path)}
        return {"n": len(good), "min": min(good), "max": max(good),
                "median": int(statistics.median(good)),
                "under_1200": sum(1 for s in good if s < 1200), "path": str(self.path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-module", default="effect_terms")
    parser.add_argument("--loss-attr", default="")
    parser.add_argument("--data-dir", default=design_run.AF2_DATA_DIR)
    parser.add_argument("--json", dest="json_path", default="")
    arguments = parser.parse_args()

    from tt_bio import bindcraft2
    hook = design_run.loss_hook(arguments.loss_module, arguments.loss_attr)
    clock = Clock(arguments.card)
    model_key, binder_key = jax.random.split(jax.random.PRNGKey(arguments.seed))
    protein_states = {design_run.TARGET_STATE: {
        design_run.BINDER_CHAIN: Protein.empty(design_run.BINDER_LENGTH, binder_key),
        design_run.TARGET_CHAIN: Protein.from_fasta(design_run.TARGET_SEQUENCE)}}
    optimizer = LogitSequenceOptimizer(iterations=arguments.steps)
    trace, design_loss, prediction = [], None, None

    with bindcraft2.predictor(card=arguments.card) as build:
        model = build(presets="model_1_ptm", data_dir=arguments.data_dir, key=model_key,
                      num_recycle=1, length_bucket_size=design_run.LENGTH_BUCKET, dropout=False)
        with hook:
            losses = losses_for_active_states(
                build_design_losses({}, {design_run.TARGET_STATE: 1.0},
                                    design_run.LENGTH_BUCKET, arguments.seed), protein_states)
            print("terms " + ",".join(f"{n}={e.weight:+.3f}" for n, e in sorted(losses.items())),
                  file=sys.stderr, flush=True)
            clock.start()
            for step in range(arguments.steps):
                started = time.perf_counter()
                softmax_weight, one_hot_weight, temperature, logit_scale = optimizer.sequence_parameters()
                predictions, chain_gradients, design_loss = model.sequence_gradients(
                    protein_states, losses, model="model_1_ptm", softmax_weight=softmax_weight,
                    one_hot_weight=one_hot_weight, temperature=temperature, logit_scale=logit_scale)
                prediction = predictions[design_run.TARGET_STATE]
                updated = optimizer.update_sequence(
                    protein_states, {name: [gradient] for name, gradient in chain_gradients.items()})
                protein_states = transfer_binder_sequences(protein_states, updated)
                aromatic = design_run.binder_aromatic(protein_states)
                trace.append({"step": step + 1, "loss": float(design_loss),
                              "binder_aromatic": aromatic,
                              "seconds": time.perf_counter() - started})
                print(f"step {step + 1}/{arguments.steps} loss={float(design_loss):.6f} "
                      f"binder_aromatic={aromatic:.6f} {trace[-1]['seconds']:.1f}s",
                      file=sys.stderr, flush=True)
            clock.stop.set()
    binder = prediction.protein_complex[design_run.BINDER_CHAIN]
    result = {"card": arguments.card, "steps": arguments.steps, "seed": arguments.seed,
              "loss_attr": arguments.loss_attr, "aiclk": clock.summary(),
              "final_loss": float(design_loss), "trace": trace,
              "weights": {name: float(entry.weight) for name, entry in sorted(losses.items())},
              "terms": {name: float(prediction.metrics[name]) for name in losses
                        if name in prediction.metrics},
              "coord_sha256": design_run.canonical_digest(binder.atoms[:, ATOM_INDEX["CA"]]),
              "seq_sha256": design_run.canonical_digest(
                  collect_shared_chains(protein_states)[1][design_run.BINDER_CHAIN].sequence)}
    print("AICLK " + json.dumps(result["aiclk"]))
    print(f"FINAL_LOSS {result['final_loss']:.10f}")
    print("TERMS " + ",".join(f"{n}={v:.6f}" for n, v in sorted(result["terms"].items())))
    print(f"COORD_SHA256 {result['coord_sha256']}")
    print(f"SEQ_SHA256 {result['seq_sha256']}")
    if arguments.json_path:
        Path(arguments.json_path).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
