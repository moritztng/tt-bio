"""Mol-1's token axis on hardware: a ragged length is bucketed to 32, masked, and inert.

Mol-1 runs through the shared Protenix trunk, so it inherits `protenix.bucketed_pairformer`;
this checks it on Mol-1's own widths (c_z 448 PairMixer blocks, 320 Pairformer blocks, a 448
confidence head) with the census probe tests/test_token_axis_bucketing_hw.py uses. 98 tokens
is deliberately not a multiple of 32. Needs the checkpoint: MOL1_CKPT, else skipped (printed).
"""
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perf", "bucketing_audit"))
CKPT = os.environ.get("MOL1_CKPT", "")
SEQ = "NLYIQWLKDGGPSSGRPPPS" * 4 + "NLYIQWLKDGGPSSGRPP"          # 98 aa, ragged
BUCKET_ON = {"TT_BIO_TOKEN_BUCKET": "1"}
# Both switches that close a ragged axis, as in test_token_axis_bucketing_hw._RAGGED_ARM.
RAGGED = {"TT_BIO_TOKEN_BUCKET": "0", "TT_BIO_SDPA_RAGGED_PAD": "0"}

pytestmark = [pytest.mark.device,
              pytest.mark.skipif(not os.path.exists(CKPT), reason="MOL1_CKPT not set or missing")]


def _fold(env_extra, d, tag):
    import token_axis_probe as P
    cdir = os.path.join(d, "census-" + tag)
    os.makedirs(cdir)
    out = os.path.join(d, tag + ".pt")
    tap = os.path.join(d, tag + ".tap")
    env = dict(os.environ, TOKEN_AXIS_CENSUS_DIR=cdir, TT_BIO_TRUNK_TAP=tap, **env_extra)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO, "perf", "bucketing_audit", "censusenv"),
                                         REPO, env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, os.path.join(REPO, "perf", "mol1_smoke", "fold.py"),
                        CKPT, SEQ, out, "2", "4"], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0, f"mol1 fold failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    s = P.merge(cdir)
    assert s["ragged_total"] + s["aligned_total"] > 0, "the probe saw no calls; a pass means nothing"
    print(P.render(s))
    return s, [l for l in open(tap).read().splitlines() if "trunk_exit" in l]


def _ragged(s):
    return [f"{r['site']} x{r['masked_ragged']} {r['shapes']}" for r in s["rows"] if r["masked_ragged"]]


def test_mol1_ragged_length_is_bucketed_and_the_pad_is_inert():
    with tempfile.TemporaryDirectory() as d:
        s0, t0 = _fold(dict(BUCKET_ON, TT_BIO_TOKEN_PAD_POISON="0"), d, "p0")
        assert not _ragged(s0), "Mol-1 presented a ragged key axis to the fused SDPA: " + "; ".join(_ragged(s0))
        _, t1 = _fold(dict(BUCKET_ON, TT_BIO_TOKEN_PAD_POISON="1000"), d, "p1000")
        # The trunk's fingerprints, not the structure: the trunk is deterministic, the sampler's
        # output is not bit-stable run to run (test_token_axis_bucketing_hw's A/A note).
        assert t0 and t0 == t1, f"padding is not inert: poison 0 {t0} vs poison 1000 {t1}"


def test_mol1_ragged_axis_is_there_to_be_closed():
    """Negative control: with both switches off the same input DOES reach a ragged axis, so the
    census above is alive and its zero is evidence."""
    with tempfile.TemporaryDirectory() as d:
        s, _ = _fold(RAGGED, d, "ragged")
        assert _ragged(s), "bucket off and no ragged axis seen: the census is not measuring Mol-1"
