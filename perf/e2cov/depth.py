"""The alignment depth that actually REACHES the model, per rung.

Not the `>` record count of the a3m: `MSA.from_a3m` caps at --max_msa_seqs (8192 is
esmfold2's shipped default) and `construct_paired_msa` then merges the chains by taxonomy
and caps again, so the only honest number is the first dimension of the `msa` tensor the
featurizer hands the model. Runs the engine's own resolve_msa -> build_spi ->
ESMFold2InputBuilder.prepare_input on the host, with no device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WT = "/home/ttuser/.coworker/wt/cov-unproven-esmfold2-bhp150a"
sys.path.insert(0, WT)

HERE = Path(__file__).parent
MAX_MSA = 8192


def main() -> int:
    from tt_bio._vendor.esm.models.esmfold2 import ESMFold2InputBuilder
    from tt_bio.esmfold2_runtime import build_spi, resolve_msa
    from tt_bio.main import _read_bio_chains

    out = {}
    for rung in ("1024", "1536"):
        yaml = HERE / "inputs" / f"e2_{rung}.yaml"
        chains = _read_bio_chains(yaml, what="esmfold2")
        resolved = [(cid, seq, resolve_msa(spec, seq, HERE / "msa", max_sequences=MAX_MSA),
                     mt, mods) for cid, seq, spec, mt, mods in chains]
        per_chain = [(c[0], len(c[1]), c[2].depth if c[2] is not None else 0) for c in resolved]
        feats, _ = ESMFold2InputBuilder().prepare_input(build_spi(resolved), seed=0)
        msa = feats["msa"]
        out[rung] = {
            "per_chain": [{"id": i, "aa": n, "a3m_rows_after_cap": d} for i, n, d in per_chain],
            "msa_shape": list(msa.shape),
            "msa_rows_reaching_model": int(msa.shape[-2]),
            "tokens": int(msa.shape[-1]),
            "max_msa_seqs": MAX_MSA,
        }
        print(rung, out[rung])
    (HERE / "depth.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
