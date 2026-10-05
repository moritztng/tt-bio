"""Upstream Boltz-2 (boltz 2.2.1) on CPU, keeping the full PAE, PDE and the distogram.

`boltz predict --write_full_pae --write_full_pde` writes the matrices; the distogram is not an
upstream output, so Boltz2.forward is wrapped to save the trunk's `pdistogram`. 9bk6 folds
single-sequence (msa: empty), 3 recycles, 200 steps, one sample, the settings the device leg
passes tt-bio.

    /tmp/fdxconf/venv-boltz/bin/python perf/fdx_confidence/boltz2_upstream_ref.py \
        --out /tmp/fdxconf/boltz_9bk6 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / "examples/9bk6.yaml").read_text())
    for s in doc["sequences"]:
        s["protein"]["msa"] = "empty"
    inp = a.out / "9bk6.yaml"
    inp.write_text(yaml.safe_dump(doc))

    import torch
    from boltz.model.models import boltz2

    torch.set_num_threads(6)
    fwd = boltz2.Boltz2.forward

    def keep(self, *args, **kw):
        out = fwd(self, *args, **kw)
        np.save(a.out / f"pdistogram_seed{a.seed}.npy",
                out["pdistogram"][0, :, :, 0].float().cpu().numpy())
        return out

    boltz2.Boltz2.forward = keep
    from boltz.main import cli
    sys.argv = ["boltz", "predict", str(inp), "--out_dir", str(a.out / f"seed{a.seed}"),
                "--accelerator", "cpu", "--seed", str(a.seed), "--recycling_steps", "3",
                "--sampling_steps", "200", "--diffusion_samples", "1", "--write_full_pae",
                "--write_full_pde", "--override", "--num_workers", "0",
                "--cache", str(Path.home() / ".boltz")]
    cli()


if __name__ == "__main__":
    main()
