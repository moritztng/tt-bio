"""Upstream OpenFold3 on CPU, keeping the full confidence matrices it computes.

Runs aqlaboratory openfold3's own `run_openfold predict` in-process and wraps its
`get_confidence_scores` to save, per sample, the token-token PAE and PDE (upstream's
probs_to_expected_error) and the contact probabilities (upstream's
compute_global_predicted_distance_error, bin ends <= 8 Å). Run with the upstream venv:

    ~/of3-upstream-venv/bin/python perf/fdx_confidence/of3_upstream_ref.py \
        --query-json q.json --out-dir /tmp/fdxconf/of3_9bk6 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from of3_ref_fixture import RUNNER_TEMPLATE  # noqa: E402

import openfold3.projects.of3_all_atom.runner as runner  # noqa: E402
from openfold3.core.metrics.confidence import compute_global_predicted_distance_error  # noqa: E402
from openfold3.run_openfold import cli  # noqa: E402

def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--ckpt", default=str(Path.home() / ".boltz/of3-p2-155k.pt"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    _orig = runner.get_confidence_scores


    def _keep(batch, outputs, config, compute_per_sample=False):
        scores = _orig(batch=batch, outputs=outputs, config=config,
                       compute_per_sample=compute_per_sample)
        d = config.confidence.distogram
        _, contacts = compute_global_predicted_distance_error(
            pde=scores["pde"], logits=outputs["distogram_logits"].float(),
            bin_min=d.bin_min, bin_max=d.bin_max, no_bins=d.no_bins)
        np.savez(a.out_dir / f"upstream_seed{a.seed}.npz",
                 pae=scores["pae"].float().cpu().numpy(), pde=scores["pde"].float().cpu().numpy(),
                 contact_probs=contacts.float().cpu().numpy(),
                 distogram_logits=outputs["distogram_logits"].float().cpu().numpy(),
                 **{k: scores[k].float().cpu().numpy() for k in ("sample_ranking_score", "ptm", "iptm") if k in scores},
                 coords=outputs["atom_positions_predicted"].float().cpu().numpy())
        return scores


    runner.get_confidence_scores = _keep
    ry = a.out_dir / f"runner_seed{a.seed}.yml"
    ry.write_text(RUNNER_TEMPLATE.format(seed=a.seed, use_templates="false", template_settings="")
                  + "data_module_args:\n  num_workers: 0\n")
    sys.argv = ["run_openfold", "predict", "--query-json", a.query_json,
                "--inference-ckpt-path", a.ckpt, "--num-diffusion-samples", str(a.samples),
                "--use-msa-server", "False", "--use-templates", "False",
                "--output-dir", str(a.out_dir / f"seed{a.seed}"), "--runner-yaml", str(ry)]
    torch.set_num_threads(6)
    sys.exit(cli())


if __name__ == "__main__":
    main()
