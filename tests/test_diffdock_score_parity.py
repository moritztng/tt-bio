"""The assembled score network against upstream's own (tr, rot, tor).

Every module below this one is graded on its own; this is the grade that says the assembly
is right -- both embedders, both towers, the cross graph, eleven convolutions and the three
heads, with the released v1.1 checkpoint, on a real PoseBusters complex at several steps of
a real rollout.

The inputs are not re-derived here. ``scripts/diffdock_port/score_model_grade.py
--save-fixture`` writes the complex graph, its ESM-2 block, the per-step inputs and
upstream's captured outputs into one file, so this test needs neither the reference
environment nor the PoseBusters tree -- only the released checkpoint. Both paths are
skipped when absent, with the command that rebuilds them.

Bar: per-output PCC >= 0.999 (D-dfd-orch-8). It is not tighter because it cannot be: eleven
convolutions compound fp32 rounding, and the torsion output additionally rides upstream's
torus table, which is a 10,000-draw unseeded Monte Carlo estimate with a measured 0.8 %
median spread between upstream processes. Grading tor to 1e-6 would be grading against one
of upstream's samples.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

FIXTURE = Path("/home/moritz/scratch/dfd/fixtures/7A9E_R4W_score_steps.pt")
CKPT = Path("/home/moritz/scratch/dfd/ckpt/extracted/score_model")
PCC_BAR = 0.999
OUTPUTS = ("tr", "rot", "tor")


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def replay():
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; rebuild with scripts/diffdock_port/"
                    "score_model_grade.py --complex 7A9E_R4W --save-fixture")
    if not (CKPT / "best_ema_inference_epoch_model.pt").exists():
        pytest.skip(f"missing the released score checkpoint under {CKPT}")

    from tt_bio.diffdock import config as C
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released

    fx = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    sd = torch.load(CKPT / "best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, C.load(str(CKPT), old_model=False))
    net = HostScoreNetwork(model, lm_embeddings=fx["emb"])
    return fx, net.score_fn(fx["graph"])


@pytest.mark.parametrize("step", (0, 9, 18))
def test_score_net_pcc_vs_cpu_golden(replay, step):
    """Upstream's own outputs for 10 poses at one step of its 19-step rollout."""
    fx, score = replay
    if step not in fx["steps"]:
        pytest.skip(f"the fixture carries steps {sorted(fx['steps'])}")
    with torch.no_grad():
        tr, rot, tor = score(fx["steps"][step])
    want = fx["want"][step]
    got = {"tr": tr.reshape(want["tr"].shape), "rot": rot.reshape(want["rot"].shape),
           "tor": tor.reshape(want["tor"].shape) if want["tor"].numel() else tor}
    for name in OUTPUTS:
        if not want[name].numel():
            continue                     # a rigid ligand has no torsion output
        pcc = _pcc(got[name], want[name])
        assert pcc >= PCC_BAR, f"step {step} {name}: PCC {pcc:.6f} below {PCC_BAR}"


def test_the_recorded_grade_covers_every_output(replay):
    """The fixture is not vacuous: every step carries all ten poses and a real torsion set."""
    fx, _ = replay
    for step, want in fx["want"].items():
        assert want["tr"].shape == want["rot"].shape, step
        assert want["tr"].shape[0] == 10, (step, want["tr"].shape)
        assert want["tor"].numel() > 0, step
