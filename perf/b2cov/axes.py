"""Both axes AT THE MODEL for boltz2, CPU only, no device open.

boltz2 does not write n_tokens or msa_depth into results.json (worker.py writes a different
metrics block for it than for rf3), so the axes have to come from the production featuriser
itself rather than from the fixture name. This runs `tt_bio.main.prepare_features` -- the same
parse -> resolve MSA -> tokenize -> featurize the worker runs -- at the served flags
(use_msa=True, max_msa=None, single_sequence=False) and reports what came out.
"""
import sys
from pathlib import Path
from tt_bio import main as M
from tt_bio.data import const
from tt_bio.data.featurizer import Boltz2Featurizer
from tt_bio.data.mol import load_canonicals
from tt_bio.data.tokenize import Boltz2Tokenizer

mol_dir = Path("/home/cust-team/.boltz") / "mols"
ccd = load_canonicals(mol_dir)
msa_dir = Path("/tmp/b2cov_axes_msa"); msa_dir.mkdir(parents=True, exist_ok=True)
tok, feat = Boltz2Tokenizer(), Boltz2Featurizer()
for p in sys.argv[1:]:
    feats, _ = M.prepare_features(
        Path(p), ccd, mol_dir, msa_dir, tok, feat,
        use_msa=True, msa_url=None, msa_strategy="greedy", msa_user=None,
        msa_pass=None, api_key=None, max_msa=None, single_sequence=False,
    )
    a3m = Path(__import__("yaml").safe_load(Path(p).read_text())
               ["sequences"][0]["protein"]["msa"])
    print(Path(p).name,
          "n_tokens", int(feats["token_pad_mask"].shape[-1]),
          "msa_shape", tuple(feats["msa"].shape),
          "msa_depth_at_model", int(feats["msa"].shape[0]),
          "atoms", int(feats["atom_pad_mask"].sum().item()),
          "a3m_records", sum(1 for l in a3m.open() if l.startswith(">")),
          "const.max_msa_seqs", const.max_msa_seqs, flush=True)
