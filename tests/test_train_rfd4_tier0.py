"""`tt-bio train --model rfd4`: the Tier-0 contract over RFD4's own step.

The step body is `train.rfd4_run` (a winner among k candidates, only its last recycle taped),
so what these tests pin is the shell around it: flag legality decided with no device and no
tape import, the argv the step loop is handed, resume from the run directory, and a flag the
step does not honour refused by name instead of ignored.
"""

import json
import sys

import pytest
from click.testing import CliRunner


def test_rfd4_is_offered_and_dry_runs_without_a_tape():
    """`--model rfd4 --dry-run` answers on a laptop: no ttnn, no tape, no device."""
    from tt_bio.train.cli import TRAINABLE, train

    assert "rfd4" in TRAINABLE
    before = set(sys.modules)
    res = CliRunner().invoke(train, ["--model", "rfd4", "--dry-run"])
    assert res.exit_code == 0, res.output
    new = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not new & {"ttnn", "torch"}, f"the dry run imported {new & {'ttnn', 'torch'}}"
    assert "rfd4" in res.output or "256" in res.output, res.output


def test_rfd4_refuses_a_flag_its_step_does_not_honour():
    """`--lr` on an RFD4 run is refused, by name, before anything opens.

    Upstream's rule runs at 1e-4 and the step does not read `--lr`; accepting it would write a
    status file that disagreed with the run. The same `--lr` on openfold3 is honoured, which is
    why the refusal is keyed on the flag the user TYPED rather than on its default.
    """
    from tt_bio.train.cli import train

    res = CliRunner().invoke(train, ["--model", "rfd4", "--lr", "1e-3"])
    assert res.exit_code == 2 and "--lr is not honoured" in res.output, res.output
    res = CliRunner().invoke(train, ["--model", "rfd4", "--train", "adapters"])
    assert res.exit_code == 2 and "trains the trunk's own weights" in res.output, res.output
    # The untyped default must not trip it: a plain dry run has lr=3e-4 in ctx.params.
    assert CliRunner().invoke(train, ["--model", "rfd4", "--dry-run"]).exit_code == 0


def test_rfd4_batch_directory_refuses_by_name(tmp_path):
    """No DATA, and an empty DATA directory, say what an RFD4 batch is and where it comes from."""
    from tt_bio.train.rfd4_tier0 import batches

    with pytest.raises(Exception, match="needs DATA"):
        batches(None)
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(Exception, match="loss_parity_reference"):
        batches(tmp_path)
    f = tmp_path / "b0.npz"
    f.write_bytes(b"")
    assert batches(tmp_path) == [f]


def test_rfd4_tier0_hands_the_step_loop_its_argv(tmp_path, monkeypatch):
    """The flags a user types become the step loop's argv, and a second run resumes."""
    from tt_bio.train import rfd4_run, rfd4_tier0

    seen = {}

    def fake_main(argv, on_step=None):
        seen["argv"] = list(argv)
        if on_step is not None:
            on_step({"step": 1, "loss": 0.5, "step_s": 2.0})
        return 0

    monkeypatch.setattr(rfd4_run, "main", fake_main)
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        (data / f"b{i}.npz").write_bytes(b"")
    out = tmp_path / "run"
    out.mkdir()

    class Status:
        rows = []

        def step(self, row):
            Status.rows.append(row)

    files = rfd4_tier0.batches(data)
    rfd4_tier0.train_rfd4(files=files, out_dir=out, steps=None, chip_ids=(0,),
                          checkpoint_every=100, tokens=None, status=Status())
    argv = seen["argv"]
    assert argv[argv.index("--steps") + 1] == "3", "no --steps means one pass over DATA"
    assert argv[argv.index("--save-every") + 1] == "100"
    assert "--resume" not in argv and "--dp-cards" not in argv
    assert Status.rows == [{"step": 1, "loss": 0.5, "step_s": 2.0}], "the row reaches the status writer unchanged; rfd4_run adds `s`"

    # A checkpoint in the run directory is resumed from, and multiple chips go data-parallel.
    # A DP run checkpoints under rank0/, so that is where its resume comes from.
    (out / "adapter-00000001.safetensors").write_bytes(b"")
    rfd4_tier0.train_rfd4(files=files, out_dir=out, steps=5, chip_ids=(0,),
                          checkpoint_every=0, tokens=None, status=Status())
    assert seen["argv"][seen["argv"].index("--resume") + 1].endswith(
        "adapter-00000001.safetensors")
    (out / "rank0").mkdir()
    (out / "rank0" / "adapter-00000009.safetensors").write_bytes(b"")
    rfd4_tier0.train_rfd4(files=files, out_dir=out, steps=10, chip_ids=(0, 2),
                          checkpoint_every=0, tokens=None, status=Status())
    argv = seen["argv"]
    assert argv[argv.index("--resume") + 1].endswith("rank0/adapter-00000009.safetensors")
    assert argv[argv.index("--dp-cards") + 1] == "0,2"
    assert argv[argv.index("--steps") + 1] == "10"
    assert "--save-every" not in argv


def test_rfd4_dp_summary_reads_rank0s_last_step_row(tmp_path):
    """A data-parallel run's status comes off rank 0's log, and skips non-step rows."""
    from tt_bio.train.rfd4_tier0 import last_row

    log = tmp_path / "rank0.log"
    log.write_text("\n".join([
        json.dumps({"step": 1, "loss": 9.0, "step_s": 3.0}),
        json.dumps({"step": 2, "loss": 7.0, "step_s": 3.1}),
        json.dumps({"saved": "/tmp/adapter-000002.safetensors"}),
        "not json at all",
    ]) + "\n")
    assert last_row(log)["step"] == 2 and last_row(log)["loss"] == 7.0
    assert last_row(tmp_path / "missing.log") == {}
