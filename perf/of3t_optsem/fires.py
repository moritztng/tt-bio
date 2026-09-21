#!/usr/bin/env python3
"""Does a zero-participation step occur under the shipped sampler? The arithmetic.

Every input is READ from upstream's own files rather than typed in here, so the answer moves
if upstream moves:

  * dataset sampling weights and the per-dataset loss-weight overrides -- their
    `examples/training_yamls/initial_training.yml`
  * which four terms count as confidence -- `LossConfig.confidence_loss_names`,
    `projects/of3_all_atom/config/dataset_config_components.py`
  * the default loss weights a dataset inherits when it overrides nothing -- `LossWeights`,
    same file
  * the global batch -- `pl_trainer_args.devices` x `num_nodes` x `data_module_args.batch_size`
    x `settings.manual_optimization.accumulate_grad_batches`

The sampler is `OF3DistributedSampler` (`core/data/framework/stochastic_sampler_dataset.py`).
`get_dataset_indices` draws `epoch_len` dataset indices from one `torch.multinomial(...,
replacement=True)` over the yaml weights, and `__iter__` then hands each rank a stride of that
one global list. So the samples inside one optimizer step are iid draws from the normalised
dataset weights, and the participation count `_sync_and_average_grads` divides by is the SUM
across all ranks. A zero-participation step therefore needs EVERY sample of the global batch
to be a zero-confidence one.

    python3 fires.py <upstream-checkout> [out.json]
"""
import ast
import json
import re
import sys

import yaml


def loss_config(components_src):
    """`confidence_loss_names` and the `LossWeights` defaults, from their pydantic classes."""
    tree = ast.parse(components_src)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in ("LossWeights",
                                                                   "LossConfig"):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or stmt.value is None:
                continue
            try:
                value = ast.literal_eval(stmt.value)
            except ValueError:
                continue     # `loss_weights: LossWeights = LossWeights()`, a nested model
            out.setdefault(node.name, {})[stmt.target.id] = value
    return out["LossConfig"]["confidence_loss_names"], out["LossWeights"]


def accumulate_grad_batches(model_config_src):
    block = re.search(r'"manual_optimization":\s*\{(.*?)\}', model_config_src, re.S)
    return int(re.search(r'"accumulate_grad_batches":\s*(\d+)', block.group(1)).group(1))


def analyse(root):
    cfg = yaml.safe_load(open(f"{root}/examples/training_yamls/initial_training.yml"))
    conf_names, defaults = loss_config(
        open(f"{root}/openfold3/projects/of3_all_atom/config/"
             "dataset_config_components.py").read())
    accum = accumulate_grad_batches(
        open(f"{root}/openfold3/projects/of3_all_atom/config/model_config.py").read())

    datasets = {}
    for name, spec in cfg["dataset_configs"]["train"].items():
        overrides = (spec.get("config", {}).get("loss", {}) or {}).get("loss_weights", {}) or {}
        weights = {**defaults, **overrides}
        total_conf = sum(float(weights[n]) for n in conf_names)
        datasets[name] = {"sampling_weight": float(spec["weight"]),
                          "confidence_weight_sum": total_conf,
                          # `_get_sample_disabled_param_names`, runner.py:363-385: a sample
                          # whose confidence weights sum to zero disables the confidence head.
                          "disables_confidence_head": total_conf <= 0.0}

    w_total = sum(d["sampling_weight"] for d in datasets.values())
    p_zero = sum(d["sampling_weight"] for d in datasets.values()
                 if d["disables_confidence_head"]) / w_total

    trainer, dm = cfg["pl_trainer_args"], cfg["data_module_args"]
    batch = int(trainer["devices"]) * int(trainer["num_nodes"]) * int(dm["batch_size"]) * accum

    # `set_loss_weights` (core/data/pipelines/featurization/loss_weights.py:43-51) ALSO zeroes
    # the four confidence terms when a sample's resolution is None or outside
    # [min_resolution, max_resolution]. That fraction lives in the weighted-pdb training cache,
    # which is not in this checkout, so it is bounded rather than asserted: solve for the share
    # f of weighted-pdb samples that would have to be resolution-excluded before an all-zero
    # global batch reaches probability 1e-06.
    w_conf = sum(d["sampling_weight"] for d in datasets.values()
                 if not d["disables_confidence_head"])
    p_needed = 1e-06 ** (1.0 / batch)
    f_needed = (p_needed * w_total - (p_zero * w_total)) / w_conf

    scales = {str(b): p_zero ** b for b in (1, 2, 4, 8, 16, 32, 64, 128, batch)}
    return {
        "source": root,
        "confidence_loss_names": conf_names,
        "default_loss_weights": defaults,
        "datasets": datasets,
        "p_sample_disables_confidence_head": p_zero,
        "global_batch_per_optimizer_step": batch,
        "global_batch_terms": {"devices": trainer["devices"], "num_nodes": trainer["num_nodes"],
                               "batch_size": dm["batch_size"],
                               "accumulate_grad_batches": accum},
        "p_zero_participation_step_by_global_batch": scales,
        "p_zero_participation_step_at_shipped_scale": p_zero ** batch,
        "resolution_exclusion_share_of_weighted_pdb_needed_for_1e-06": f_needed,
    }


if __name__ == "__main__":
    result = analyse(sys.argv[1])
    print(json.dumps(result, indent=1))
    if len(sys.argv) > 2:
        json.dump(result, open(sys.argv[2], "w"), indent=1)
