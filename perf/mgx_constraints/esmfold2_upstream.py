#!/usr/bin/env python3
"""Upstream ESMFold2 on CPU: the vendored torch model left unpatched, fed by the ttnn ESMC-6B.

    TT_VISIBLE_DEVICES=<card> python perf/mgx_constraints/esmfold2_upstream.py \
        <esmfold2|esmfold2-fast> <out_dir> <input.yaml>...

A bond or ring closure never reaches the language model; it enters ESMFold2 through
token_bonds in the pair init, which this runs in upstream's own torch. The LM is the ttnn
ESMC-6B (validated separately, tests/test_esmc.py) because no installed transformers ships
ESMCModel, the same split scripts/esmfold2_e2e_parity.py uses. Folds each input the way the
worker does (same chain reader, same _read_bio_bonds), 5 samples, and writes
<out_dir>/<model>/ref/structures/<stem>[_model_N].cif, the layout score.py reads.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model  # noqa: E402
from tt_bio.esmfold2_runtime import _ESMCAdapter, fold_complex  # noqa: E402
from tt_bio.main import (RECYCLING_STEPS, SAMPLING_STEPS, _read_bio_bonds,  # noqa: E402
                         _read_bio_chains, _write_structure)
from tt_bio.weights import ESMFOLD2_FAST_REPO, ESMFOLD2_REPO, hf_revision  # noqa: E402

model_id, out = sys.argv[1], Path(sys.argv[2])
repo = ESMFOLD2_FAST_REPO if model_id == "esmfold2-fast" else ESMFOLD2_REPO
torch.set_num_threads(8)
model = ESMFold2Model.from_pretrained(repo, load_esmc=False, revision=hf_revision(repo)).eval()
model._esmc = _ESMCAdapter("biohub/ESMC-6B")
d = out / model_id / "ref" / "structures"
d.mkdir(parents=True, exist_ok=True)
for f in map(Path, sys.argv[3:]):
    chains = _read_bio_chains(f, what=model_id)
    chains = [(c, s, None, mt, mo) for c, s, _sp, mt, mo in chains]
    with torch.no_grad():
        ranked = fold_complex(model, chains, num_loops=RECYCLING_STEPS.get(model_id, 3),
                              num_sampling_steps=SAMPLING_STEPS[model_id],
                              num_diffusion_samples=5, seed=0, return_all=True,
                              bonds=_read_bio_bonds(f, chains))
    for r, s in enumerate(ranked):
        _write_structure(s.complex, d / (f"{f.stem}.cif" if r == 0 else f"{f.stem}_model_{r}.cif"),
                         "cif")
    print(f.stem, "done", flush=True)
