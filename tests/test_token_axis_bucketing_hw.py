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
# rfd3 designs 148 residues against 150 fixed ones: 298 tokens, so ragged. The target structure
# comes out of an in-repo payload, so the test needs nothing from the machine but the checkpoint.
_RFD3 = ["design", "{spec}", "--model", "rfd3", "--from_pdb", "--num_timesteps", "2",
         "--num_designs", "1"]
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
    "rfd3": _RFD3,
}
_RFD3_TARGET = os.path.join(REPO, "perf", "wh-correctness", "results", "payloads",
                            "des_rfd3_binder.json")


# The softmax probe runs in a CHILD, like every other test in this file. pytest itself must never
# open the card: an in-process `get_device()` here, even with `cleanup()` afterwards, left the chip
# in a state that WEDGED the next subprocess to open it (0% CPU, futex_do_wait, needing tt-smi -r 0)
# -- twice, on the rfd3 job that runs in 11 s standalone.
_SOFTMAX_CHILD = r"""
import os, sys, json
sys.path.insert(0, os.environ["REPO"])
import torch, ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
W = 33
x = ttnn.from_torch(torch.full((1, 1, 32, W), -5.0), dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT, device=dev)
out = {"padded": int(x.padded_shape[-1]),
       "got": float(ttnn.to_torch(ttnn.softmax(x, dim=-1))[0, 0, 0, 0])}
print("RESULT " + json.dumps(out))
"""


@pytest.mark.device
def test_ttnn_softmax_masks_its_ragged_tail():
    """The measured fact half the IMMUNE rows depend on, pinned to the installed ttnn.

    Logical [1,1,32,33] over a padded 64, every entry -5.0. If the 31 padded columns entered the
    reduction at exp(0) they would swamp 33 real columns at exp(-5), so the two answers are two
    orders of magnitude apart and no threshold tuning is involved:
        masked   1/33            = 0.030303
        unmasked exp(-5)/(33*exp(-5)+31) = 0.000216
    """
    env = dict(os.environ, REPO=REPO)
    r = subprocess.run([sys.executable, "-c", _SOFTMAX_CHILD], env=env, cwd=REPO,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"probe failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))
    res = json.loads(line[len("RESULT "):])
    assert res["padded"] == 64, "the input is not physically padded, so this proves nothing"
    got, masked, unmasked = res["got"], 1.0 / 33, 0.000216
    print(f"ttnn.softmax at logical W=33 padded 64: {got:.6f} "
          f"(masked {masked:.6f}, unmasked {unmasked:.6f})")
    assert abs(got - masked) < 1e-3, (
        f"ttnn.softmax no longer masks its ragged tail: got {got:.6f}, masked {masked:.6f}, "
        f"unmasked {unmasked:.6f}. Every IMMUNE row in tt_bio/token_axis.py rests on this.")


def _census(model, argv, env_extra=None, seq=None, tap=None):
    """Run one tiny job under the probe and return the merged counters."""
    import token_axis_probe as P
    with tempfile.TemporaryDirectory() as d:
        fasta = os.path.join(d, "q.fasta")
        with open(fasta, "w") as fh:
            fh.write(">q|protein\n" + (seq or _SEQ) + "\n")
        spec = os.path.join(d, "spec.json")
        if "{spec}" in " ".join(argv):
            with open(_RFD3_TARGET) as fh:
                pdb = json.load(fh)["structure"]
            with open(os.path.join(d, "target.pdb"), "w") as fh:
                fh.write(pdb)
            with open(spec, "w") as fh:
                json.dump({"m298": {"input": os.path.join(d, "target.pdb"),
                                    "contig": "A1-150,148", "length": "148"}}, fh)
        cdir = os.path.join(d, "census")
        os.makedirs(cdir)
        env = dict(os.environ)
        env["TOKEN_AXIS_CENSUS_DIR"] = cdir
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(REPO, "perf", "bucketing_audit", "censusenv"), REPO,
             env.get("PYTHONPATH", "")])
        env.update(env_extra or {})
        if tap:
            env["TT_BIO_TRUNK_TAP"] = tap
        cmd = [sys.executable, "-m", "tt_bio.main"] + [
            a.format(fasta=fasta, spec=spec) for a in argv] + ["--out_dir", os.path.join(d, "out")]
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
        assert r.returncode == 0, f"{model} job failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
        s = P.merge(cdir)
        assert s["ragged_total"] + s["aligned_total"] > 0, (
            f"the probe recorded no calls at all for {model} -- it did not reach the model process, "
            "so a pass here would mean nothing")
        print(P.render(s))
        return s


@pytest.mark.device
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


_ALIGNED_SEQ = ("NLYIQWLKDGGPSSGRPPPS" * 7)[:128]      # 128 tokens: already a tile multiple
_BUCKET_OFF = {"TT_BIO_PROTENIX_TOKEN_BUCKET": "0"}


@pytest.mark.device
@pytest.mark.parametrize("model", ["protenix-v2", "opendde"])
def test_protenix_ragged_sdpa_is_there_to_be_closed(model):
    """The negative control: with the bucket OFF these two DO present a ragged key axis.

    Both are declared BUCKETED and default ON, so the census test above already proves the fix
    holds. What that test cannot prove is that the counter it reads is alive: a probe that missed
    the model process, or a census that stopped classifying, passes it just as cleanly. Turning the
    only thing that closes those axes back off has to bring the ragged calls back, or the guard is
    measuring nothing. Three axes had to be closed to reach zero and the census found each one
    AFTER the previous was fixed: the trunk (1208 calls at N=98), the confidence head's Pairformer
    at the real N (8), and -- OpenDDE only -- the structural-token refiner at Ns=181 (8 more).
    """
    s = _census(model, JOBS[model], env_extra=_BUCKET_OFF)
    assert any(r["masked_ragged"] for r in s["rows"]), (
        f"{model} with the token bucket OFF presented NO ragged key axis at all. Either the "
        "census no longer sees this model's calls or the ragged axes closed somewhere else -- "
        "either way the BUCKETED census above is no longer evidence of anything.")


@pytest.mark.device
def test_protenix_token_bucket_is_a_noop_at_an_aligned_length():
    """The A/A control: at N=128 the bucket has nothing to pad, so the trunk must not move.

    The "on" arm is the shipped default and "off" is the flag forced to 0 -- written that way round
    deliberately, because an arm spelled `{}` was the OFF arm before the default flipped and would
    now make this an ON-vs-ON tautology that can never fail.

    Compares TT_BIO_TRUNK_TAP fingerprints rather than the output structure, because the STRUCTURE
    is not a usable control here -- protenix-v2 folds the same 98-aa input to two different CIFs in
    two consecutive runs, well below the 256-aa nondeterminism this model was known for. The trunk
    itself is deterministic, which is what makes this comparison meaningful.
    """
    with tempfile.TemporaryDirectory() as d:
        taps = {}
        for arm, extra in (("off", _BUCKET_OFF), ("on", {})):
            t = os.path.join(d, f"tap_{arm}.txt")
            _census("protenix-v2", JOBS["protenix-v2"], env_extra=extra,
                    seq=_ALIGNED_SEQ, tap=t)
            taps[arm] = [l for l in open(t).read().splitlines() if "trunk_exit" in l]
            print(arm + ": " + " ".join(taps[arm]))
        assert taps["off"] and taps["off"] == taps["on"], (
            "the token bucket moved the trunk at an already-aligned N=128:\n"
            f"  off {taps['off']}\n  on  {taps['on']}")


@pytest.mark.device
@pytest.mark.parametrize("name", ["esmc-300m", "saprot-35m"])
def test_bucket_lives_at_the_op_boundary_not_in_the_caller(name):
    """A DIRECT `Model.forward` at a ragged L must give the bucketed answer, exactly.

    esmc and saprot bucket in `_batch_tokens`, so the shipped CLI never presents a ragged token
    axis -- but `Model.forward` is public and used to take whatever length it was handed. Two
    assertions, and the first is the change's own A/A control:

      * at an already-aligned L, `forward` and the unbucketed `_dispatch` are bit-identical, so the
        shipped path did not move;
      * at a ragged L, `forward` now equals what the caller would get by padding to the bucket
        itself and slicing back -- bit-exact, not merely close.
    """
    import torch

    if name.startswith("esmc"):
        from tt_bio import esmc as M
        model = M.load_esmc(name, trace=False)
        ids = M.tokenize("NLYIQWLKDGGPSSGRPPPS" * 4 + "NLYIQWLKDGGPSSGRPP")[0]
        call = lambda t, *m: model.forward(t, *m)
        raw = lambda t, *m: model._dispatch(t, *m)
        nmask = 2
    else:
        from tt_bio import saprot as M
        model = M.load_saprot(name)
        ids = M.tokenize("NLYIQWLKDGGPSSGRPPPS" * 4 + "NLYIQWLKDGGPSSGRPP", "")[0]
        call = lambda t, *m: model.forward(t, *m)
        raw = lambda t, *m: model._dispatch(t, *m)
        nmask = 3
    L = int(ids.numel())
    assert L % M.BUCKET, f"the input has to be RAGGED for this to test anything (L={L})"

    # A/A: an aligned length takes the untouched path.
    aligned = torch.nn.functional.pad(ids, (0, M.BUCKET - L % M.BUCKET),
                                      value=M.PAD_TOKEN if nmask == 2 else M.PAD).unsqueeze(0)
    a = call(aligned, *([None] * nmask))
    b = raw(aligned, *([None] * nmask))
    for x, y, lbl in zip(a, b, ("logits", "emb")):
        assert torch.equal(x, y), f"{name}: {lbl} moved at an already-aligned L={aligned.shape[1]}"

    # Ragged: forward must equal the caller-side bucket, sliced back.
    got = call(ids.unsqueeze(0), *([None] * nmask))
    padded = M.bucket_token_axis(ids.unsqueeze(0), *([None] * nmask),
                                 **({} if nmask == 2 else {"pad_token": M.PAD}))
    want = raw(*padded[:nmask + 1])
    for x, y, lbl in zip(got, want, ("logits", "emb")):
        assert torch.equal(x, y[:, :L]), (
            f"{name}: {lbl} at a ragged L={L} does not match the bucketed answer; "
            f"maxabs {(x.float() - y[:, :L].float()).abs().max().item():g}")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-s", "-q"]))
