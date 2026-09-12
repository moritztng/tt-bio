"""What depth does the MSA track actually run at, and what does the 1024 bucket pad it to?

CPU only: seeds the MSA cache exactly like the perf harness, featurizes through the shipped
`prepare_features`, and reports the real depth against the single 1024-row device bucket. No
device is opened.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TT_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import torch                                                          # noqa: E402
torch.set_grad_enabled(False)
from tt_baseline import seed_msa_cache                                # noqa: E402
from tt_bio import main as tb_main                                    # noqa: E402
from tt_bio.data import const                                         # noqa: E402
from tt_bio.data.tokenize import Boltz2Tokenizer            # noqa: E402
from tt_bio.data.mol import load_canonicals                  # noqa: E402
from tt_bio.data.featurizer import Boltz2Featurizer          # noqa: E402

MSA_BUCKET = 1024


def probe(yaml_path: Path, a3m: Path, cache: Path, msa_dir: Path):
    msa_dir.mkdir(parents=True, exist_ok=True)
    n_msa = seed_msa_cache(yaml_path, a3m, msa_dir)
    ccd = load_canonicals(cache / "mols")
    feats, _ = tb_main.prepare_features(
        yaml_path, ccd, cache / "mols", msa_dir, Boltz2Tokenizer(), Boltz2Featurizer(),
        use_msa=True, msa_url=None, msa_strategy="greedy", msa_user=None, msa_pass=None,
        api_key=None, max_msa=8192,
    )
    depth = int(feats["msa"].shape[1])
    tokens = int(feats["token_pad_mask"].shape[-1])
    padded = depth + (-depth) % MSA_BUCKET
    return {"fixture": yaml_path.name, "a3m_seqs": n_msa, "tokens": tokens,
            "msa_depth": depth, "padded_1024": padded,
            "overcompute_x": round(padded / max(depth, 1), 3),
            "ladder_32": depth + (-depth) % 32}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--cache", default=str(Path.home() / ".boltz"))
    ap.add_argument("--msa-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for f in a.fixtures:
        y = Path(f)
        rows.append(probe(y, y.with_suffix(".a3m"), Path(a.cache), Path(a.msa_dir)))
        print(json.dumps(rows[-1]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
