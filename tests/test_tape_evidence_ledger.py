"""What the one tape is verified by, and what it is not verified by any more.

Three gradient implementations existed for Protenix v2: `tt_bio/autograd.py` (hallgrad), the
seven taped module twins in `perf/ptxft/tape_block.py` (ptxft), and `tt_bio/train/`, which turned
out to be a consumer of the first rather than a third tape. Commit 81eaa6a6a retired the twin,
correctly -- a per-model fork of the pairformer block is the thing UNIFIED-NEVER-PER-MODEL exists
to prevent, and the block it forked is the one the shipped model runs.

But nine checks were deleted with it rather than re-pointed at the production path, and they were
not small: the pair track against the production `PairformerLayer` on real weights, the single
track holding 52.9 % of every block's parameters, four confidence heads, the DiT block against
ByteDance's own module, the whole diffusion module at one sampled timestep, a memorisation run
from random initialisation, and a held-out fine-tune result. A consolidation that drops a
verification quietly is worse than the three implementations it replaced, so this file makes the
drop loud.

It is a ledger, not a check of the gradients themselves -- those need a card. What it asserts is
that the ledger is honest: every claim still standing names an artifact that exists, every retired
one names a commit the harness can actually be read back from, and the number outstanding is
pinned so that re-establishing one, or losing another, cannot happen without editing this file.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The commit that retired the twin. Every RETIRED harness is readable at its parent.
RETIRED_AT = "81eaa6a6a"

LIVE, REPLACED, RETIRED = "live", "replaced", "retired"

# claim -> (status, artifact, note)
#   live      artifact must exist and is the check itself
#   replaced  artifact must exist and covers the claim by other means
#   retired   artifact is the deleted path, recoverable at RETIRED_AT^, and nothing covers it
LEDGER = {
    "eight ProtenixLoss terms, value and gradient, against upstream's nn.Modules in float64": (
        LIVE, "perf/ptxft/losscheck.py",
        "needs the upstream protenix package, so it cannot run in CI; the golden pins in "
        "tests/test_training_losses_regression.py are the CI-runnable half"),
    "the float64 references every device gradient is scored against": (
        LIVE, "tests/test_autograd_reference_gate.py",
        "26 references, worst 1.26e-09 against a 2e-6 bar, card-free"),
    "per-op device gradients against those references": (
        LIVE, "perf/hallgrad/gradcheck.py", "16 cases, needs a card"),
    "the shipped call sites' gradients, with their own kernel config and core grid": (
        LIVE, "perf/train_a1_defork/gradcheck_dispatch.py", "10 cases, needs a card"),
    "end to end: distogram loss to sequence-logit gradient, plus a directional difference": (
        LIVE, "perf/hallgrad/e2e_distogram.py", "the only surviving whole-chain gradient check"),
    "a gradient-driven loop that moves an objective on device": (
        LIVE, "perf/hallgrad/halloop.py",
        "hallucination from a trained checkpoint, which is a weaker signal than memorisation "
        "from random init -- see the overfit row"),
    "data parallelism measured, and masters bit-identical across ranks": (
        REPLACED, "perf/train_d_dp/scale.py",
        "supersedes perf/ptxft/dpscale.py and improves on it: arms interleaved, clocks and "
        "cotenants recorded per arm. The bit-identical assertion also runs host-side in "
        "tests/test_training_dp_launcher.py"),
    "an interrupted run resumes with both Adam moments and the step, or fails loudly": (
        REPLACED, "tests/test_training_checkpoint_resume.py",
        "covers the checkpoint round-trip half of perf/ptxft/train_distogram.py, host-side "
        "and exact; the held-out result it was part of is retired separately"),

    # --- outstanding: no harness scores these on the production path today ---
    "the taped pair track against the production PairformerLayer on real weights": (
        RETIRED, "perf/ptxft/block_parity.py", "ptx-fastpath owns the replacement"),
    "the single track: attention_pair_bias and single_transition, 52.9 % of block parameters": (
        RETIRED, "perf/ptxft/single_track.py", "ptx-fastpath owns the replacement"),
    "the four confidence heads against float64 on real checkpoint weights": (
        RETIRED, "perf/ptxft/confhead.py", "unowned"),
    "one DiT block against ByteDance's own module": (
        RETIRED, "perf/ptxft/ditcheck.py", "ptx-diffusion owns the replacement"),
    "the diffusion module differentiated end to end at one sampled timestep": (
        RETIRED, "perf/ptxft/denoiser.py", "ptx-diffusion owns the replacement"),
    "memorisation of a tiny set from RANDOM initialisation": (
        RETIRED, "perf/ptxft/overfit.py",
        "the strongest single correctness signal a training stack has, and the one a "
        "fine-tune from a good checkpoint cannot give. ptx-objective owns the replacement, "
        "on the composed objective rather than the distogram alone"),
    "a held-out improvement from a LoRA fine-tune of the trunk": (
        RETIRED, "perf/ptxft/train_distogram.py", "ptx-objective owns the replacement"),
    "how many blocks can be taped at what token count inside 34.23 GB": (
        RETIRED, "perf/ptxft/memgate.py", "ptx-crop owns the replacement"),
}

OUTSTANDING = 8   # pinned: re-establishing one, or losing another, must edit this file


def _statuses(want):
    return [(c, a, n) for c, (s, a, n) in LEDGER.items() if s == want]


@pytest.mark.parametrize("claim,path,note", _statuses(LIVE) + _statuses(REPLACED),
                         ids=lambda v: v[:48] if isinstance(v, str) else v)
def test_a_standing_claim_names_an_artifact_that_exists(claim, path, note):
    assert (REPO / path).is_file(), (
        f"the ledger says {claim!r} is covered by {path}, and it is not there. Either the "
        f"harness moved and this line is stale, or the coverage is gone and the claim belongs "
        f"in the retired block with an owner.")


@pytest.mark.parametrize("claim,path,note", _statuses(RETIRED),
                         ids=lambda v: v[:48] if isinstance(v, str) else v)
def test_a_retired_harness_is_actually_recoverable(claim, path, note):
    """'Recoverable from git' is a fact about a commit, not a hope. Check the object reads."""
    r = subprocess.run(["git", "cat-file", "-e", f"{RETIRED_AT}^:{path}"],
                       cwd=REPO, capture_output=True)
    assert r.returncode == 0, (
        f"{path} is listed as retired and recoverable at {RETIRED_AT}^, but git cannot read it "
        f"there: {r.stderr.decode().strip()}. A retired check nobody can read back is simply a "
        f"deleted one.")


@pytest.mark.parametrize("claim,path,note", _statuses(RETIRED),
                         ids=lambda v: v[:48] if isinstance(v, str) else v)
def test_a_retired_harness_is_not_secretly_back(claim, path, note):
    assert not (REPO / path).exists(), (
        f"{path} exists again but the ledger still calls it retired. If it now runs against the "
        f"production path, move it to LIVE and drop OUTSTANDING by one; the count is the whole "
        f"point of the pin.")


def test_the_outstanding_count_is_what_the_ledger_says():
    n = len(_statuses(RETIRED))
    assert n == OUTSTANDING, (
        f"{n} claims are outstanding, the pin says {OUTSTANDING}. Re-establishing a check on the "
        f"production path is progress and should lower this; losing one should raise it. Neither "
        f"should be able to happen without this line changing.")


def test_every_claim_has_one_of_the_three_statuses():
    bad = {c: s for c, (s, _, _) in LEDGER.items() if s not in (LIVE, REPLACED, RETIRED)}
    assert not bad, bad


def test_the_twin_itself_is_gone_and_stays_gone():
    """The fork is what UNIFIED forbids. Its retirement is the part that must not regress."""
    assert not (REPO / "perf/ptxft/tape_block.py").exists(), (
        "perf/ptxft/tape_block.py is back. It is a second implementation of the pairformer "
        "block, a fork of the one the shipped model runs, and retiring it is the whole reason "
        "the checks above are outstanding. Differentiate the production block instead.")
