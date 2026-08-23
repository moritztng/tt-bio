"""On-hardware half of the token-axis guard: the primitive taxonomy, and a real fold's counters.

`tests/test_token_axis_bucketing.py` checks the declarations. This file checks the two facts those
declarations rest on, on a real card, so a ttnn bump or a model change that breaks either one is a
test failure instead of a silent 72x.

  1. `ttnn.softmax(dim=-1)` masks its own ragged tile tail. Half the audit's IMMUNE rows are immune
     *because of this*, so it is pinned here against the wheel rather than trusted.
  2. A model declared BUCKETED presents no ragged key axis to the fused SDPA. Runs a real, tiny job
     under `tests/token_axis_probe.py` and reads the counters.

Check 2 needs weights and a card, so it runs only for the models whose weights are already cached
and only when asked. Skips are printed, never silent: a skipped model is recorded UNCENSUSED in
`tt_bio/token_axis.py`, not assumed to pass.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python3 -m pytest tests/test_token_axis_bucketing_hw.py
    TOKEN_AXIS_HW_MODELS=esmc-300m,saprot-35m ... -k census      # opt in to the fold-level check
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio import token_axis as TA  # noqa: E402

# One tiny job per model, cheapest input that still runs its token axis at a RAGGED length. The
# 98-aa sequence is deliberately not a multiple of 32: an aligned input would let a model with no
# bucket at all pass.
_SEQ = "NLYIQWLKDGGPSSGRPPPS" * 4 + "NLYIQWLKDGGPSSGRPP"          # 98 aa
_PREDICT = ["predict", "{fasta}", "--single_sequence", "--model"]
JOBS = {
    "esmc-300m": ["embed", "{fasta}", "--model", "esmc-300m"],
    "esmc-600m": ["embed", "{fasta}", "--model", "esmc-600m"],
    "saprot-35m": ["saprot", "{fasta}", "--model", "saprot-35m"],
    "saprot-650m": ["saprot", "{fasta}", "--model", "saprot-650m"],
    "esmfold2": _PREDICT + ["esmfold2"],
    # Declared EXPOSED today, so these skip on status -- and start running the moment Phase 2
    # flips one of them to BUCKETED. That is the point: the claim gets checked, not the intent.
    "protenix-v2": _PREDICT + ["protenix-v2"],
    "opendde": _PREDICT + ["opendde"],
}


def _device_available():
    return bool(os.environ.get("TT_VISIBLE_DEVICES")) and os.path.exists("/dev/tenstorrent")


@pytest.mark.skipif(not _device_available(),
                    reason="needs a TT card and TT_VISIBLE_DEVICES pinned to it")
def test_ttnn_softmax_masks_its_ragged_tail():
    """The measured fact half the IMMUNE rows depend on, pinned to the installed ttnn.

    Logical [1,1,32,33] over a padded 64, every entry -5.0. If the 31 padded columns entered the
    reduction at exp(0) they would swamp 33 real columns at exp(-5), so the two answers are two
    orders of magnitude apart and no threshold tuning is involved:
        masked   1/33            = 0.030303
        unmasked exp(-5)/(33*exp(-5)+31) = 0.000216
    """
    import torch
    import ttnn

    from tt_bio.tenstorrent import get_device
    dev = get_device()
    W = 33
    x = ttnn.from_torch(torch.full((1, 1, 32, W), -5.0), dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    assert int(x.padded_shape[-1]) == 64, "the input is not physically padded, so this proves nothing"
    got = float(ttnn.to_torch(ttnn.softmax(x, dim=-1))[0, 0, 0, 0])
    masked, unmasked = 1.0 / W, 0.000216
    print(f"ttnn.softmax at logical W={W} padded 64: {got:.6f} "
          f"(masked {masked:.6f}, unmasked {unmasked:.6f})")
    assert abs(got - masked) < 1e-3, (
        f"ttnn.softmax no longer masks its ragged tail: got {got:.6f}, masked {masked:.6f}, "
        f"unmasked {unmasked:.6f}. Every IMMUNE row in tt_bio/token_axis.py rests on this.")


def _census(model, argv):
    """Run one tiny job under the probe and return the merged counters."""
    import token_axis_probe as P
    with tempfile.TemporaryDirectory() as d:
        fasta = os.path.join(d, "q.fasta")
        with open(fasta, "w") as fh:
            fh.write(">q|protein\n" + _SEQ + "\n")
        cdir = os.path.join(d, "census")
        os.makedirs(cdir)
        env = dict(os.environ)
        env["TOKEN_AXIS_CENSUS_DIR"] = cdir
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(REPO, "perf", "bucketing_audit", "censusenv"), REPO,
             env.get("PYTHONPATH", "")])
        cmd = [sys.executable, "-m", "tt_bio.main"] + [
            a.format(fasta=fasta) for a in argv] + ["--out_dir", os.path.join(d, "out")]
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
        assert r.returncode == 0, f"{model} job failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
        s = P.merge(cdir)
        assert s["ragged_total"] + s["aligned_total"] > 0, (
            f"the probe recorded no calls at all for {model} -- it did not reach the model process, "
            "so a pass here would mean nothing")
        print(P.render(s))
        return s


@pytest.mark.skipif(not _device_available(),
                    reason="needs a TT card and TT_VISIBLE_DEVICES pinned to it")
@pytest.mark.parametrize("model", sorted(JOBS))
def test_no_bucketed_model_reaches_a_ragged_fused_sdpa(model):
    """A BUCKETED model runs a 98-aa (ragged) input without one ragged fused-SDPA key axis.

    The counter that matters is `masked_ragged`: a key axis short of its physical tile extent under
    an additive bias the caller sized to the LOGICAL length. That is the 71-76x defect, and it is
    the one thing bucketing exists to prevent.
    """
    want = os.environ.get("TOKEN_AXIS_HW_MODELS")
    if want and model not in [m.strip() for m in want.split(",")]:
        pytest.skip(f"{model} not in TOKEN_AXIS_HW_MODELS")
    status = TA.TOKEN_AXIS[model][0]
    if status != TA.BUCKETED:
        pytest.skip(f"{model} is declared {status}, not {TA.BUCKETED}")
    s = _census(model, JOBS[model])
    bad = [r for r in s["rows"] if r["masked_ragged"]]
    assert not bad, (
        f"{model} is declared BUCKETED but presented a ragged key axis to the fused SDPA at: "
        + "; ".join(f"{r['site']} x{r['masked_ragged']} {r['shapes']}" for r in bad))


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-s", "-q"]))
