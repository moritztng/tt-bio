"""Template features from .a3m (hmmsearch) and .hhr hits equal Mol-1's own, kalign 3.3.5 realignment included.

The digests (tests/data/mol1_io/template_digests.json) are sha256 of Mol-1 e6e64866's template_* features for
upstream's two template examples, made by `scripts/mol1_io/feature_goldens.py --use_template` with kalign 3.3.5 and
the release mmCIFs (the arrays are 15 MB, the digests are what is kept). Needs kalign on PATH (or under
$MOL1_DATA/tools/kalign/bin), $MOL1_DATA/{common,mmcif} and the upstream clone; skipped otherwise. CPU only.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "mol1_io"))
import compare_features as cf  # noqa: E402

DATA = Path(os.environ.get("MOL1_DATA", "/home/moritz/mol1-upstream/io-data"))
DIGESTS = json.loads((cf.DATA / "template_digests.json").read_text())


def _jobs():
    text = (cf.DATA / "ref_template.json").read_text().replace("@MOL1@", cf.MOL1)
    return {j["name"]: j for j in json.loads(text)}


def _digest(v):
    v = np.asarray(v)
    return hashlib.sha256(np.ascontiguousarray(v.astype(np.int64 if v.dtype.kind in "iub" else np.float32))
                          .tobytes()).hexdigest()


@pytest.mark.parametrize("job", sorted(DIGESTS))
def test_template_features_equal_upstream(tmp_path, monkeypatch, job):
    monkeypatch.setenv("PATH", f"{DATA / 'tools/kalign/bin'}{os.pathsep}{os.environ['PATH']}")
    if not shutil.which("kalign") or not (DATA / "mmcif").is_dir() or not (DATA / "common").is_dir():
        pytest.skip("needs kalign and $MOL1_DATA/{common,mmcif}")
    jobs = _jobs()
    paths = [v for e in jobs[job]["sequences"] for b in e.values() if isinstance(b, dict)
             for k, v in b.items() if k.endswith("Path")]
    if not all(Path(p).exists() for p in paths):
        pytest.skip("the upstream clone ($MOL1_UPSTREAM) is not here")
    monkeypatch.setenv("MOL_ROOT_DIR", str(DATA))
    monkeypatch.setitem(cf.MODES, "template", dict(use_template=True))
    ours = cf.tt_features(jobs[job], "template", tmp_path)
    want = DIGESTS[job]
    for k, shape in want["shapes"].items():
        if k in ("template_atom_positions", "template_atom_mask"):
            continue                       # cf. test_mol1_features.NOT_BUILT: the embedder reads the derived keys
        assert list(ours[k].shape) == shape, k
        assert _digest(ours[k]) == want[k], k
    assert ours["template_aatype"][0].min() >= 0 and ours["template_pseudo_beta_mask"].any()  # templates were found
