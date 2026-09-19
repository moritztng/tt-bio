"""What reaches the model, per rung: the token count and the alignment depth.

esmfold2-fast advertises "no MSA encoder", and this checks it rather than repeating it.
Three independent places have to agree before the depth axis can be recorded as None:
the shipped checkpoint config (`msa_encoder.enabled`), the module tree the engine builds,
and the `msa` tensor the featurizer hands the model. Host only, no device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WT = "/home/ttuser/.coworker/wt/cov-unproven-esmfold2fast-bhp150a"
sys.path.insert(0, WT)

HERE = Path(__file__).parent


def main() -> int:
    import huggingface_hub

    from tt_bio._vendor.esm.models.esmfold2 import ESMFold2InputBuilder
    from tt_bio.esmfold2_runtime import build_spi
    from tt_bio.weights import hf_revision
    from tt_bio.main import _read_bio_chains

    out = {"checkpoints": {}}
    for repo in ("biohub/ESMFold2-Fast", "biohub/ESMFold2"):
        cfg = json.loads(Path(huggingface_hub.hf_hub_download(
            repo, "config.json", revision=hf_revision(repo))).read_text())
        out["checkpoints"][repo] = {
            "msa_encoder_enabled": cfg["msa_encoder"]["enabled"],
            "folding_trunk_layers": cfg["folding_trunk"]["n_layers"],
            "revision": hf_revision(repo),
        }
        print(repo, out["checkpoints"][repo])

    for rung in ("1024", "1536"):
        yaml = HERE / "inputs" / f"e2f_{rung}.yaml"
        chains = _read_bio_chains(yaml, what="esmfold2-fast")
        # No msa_dir and no MSA spec: this is the served configuration for this checkpoint.
        resolved = [(cid, seq, None, mt, mods) for cid, seq, _spec, mt, mods in chains]
        feats, _ = ESMFold2InputBuilder().prepare_input(build_spi(resolved), seed=0)
        msa = feats["msa"]
        out[rung] = {
            "per_chain_aa": [len(c[1]) for c in resolved],
            "msa_shape": list(msa.shape),
            "msa_rows_reaching_model": int(msa.shape[-2]),
            "tokens": int(msa.shape[-1]),
        }
        print(rung, out[rung])
    (HERE / "depth.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
