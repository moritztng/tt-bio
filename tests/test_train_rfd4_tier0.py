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
    res = CliRunner().invoke(train, ["--model", "rfd4", "--lr", "1e-3", "--dry-run"])
    assert res.exit_code == 2 and "--lr is not honoured" in res.output, res.output
    res = CliRunner().invoke(train, ["--model", "rfd4", "--train", "adapters"])
    assert res.exit_code == 2 and "trains the trunk's own weights" in res.output, res.output
    # The untyped default must not trip it: a plain dry run has lr=3e-4 in ctx.params.
    assert CliRunner().invoke(train, ["--model", "rfd4", "--dry-run"]).exit_code == 0


def test_rfd4_help_says_data_is_required():
    """`--help` must agree with `batches(None)`: DATA is required for RFD4."""
    from tt_bio.train.cli import train

    out = CliRunner().invoke(train, ["--help"]).output
    assert "without DATA an RFD4 run" not in out, out
    assert "DATA for RFD4 is required" in out, out


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
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    rfd4_tier0.train_rfd4(files=files, out_dir=out, steps=None, chip_ids=(0,),
                          checkpoint_every=100, tokens=None, status=Status())
    argv = seen["argv"]
    assert argv[argv.index("--steps") + 1] == "3", "no --steps means one pass over DATA"
    assert argv[argv.index("--save-every") + 1] == "100"
    assert argv[argv.index("--optimizer") + 1] == "muon"
    assert argv[argv.index("--exempt") + 1] == "default"
    assert "--resume" not in argv and "--dp-cards" not in argv
    assert Status.rows == [{"step": 1, "loss": 0.5, "step_s": 2.0}], "the row reaches the status writer unchanged; rfd4_run adds `s`"
    import os
    assert os.environ.get("TT_VISIBLE_DEVICES") == "0"
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "28")
    rfd4_tier0.train_rfd4(files=files, out_dir=out, steps=2, chip_ids=(0,),
                          checkpoint_every=0, tokens=None, status=Status())
    assert os.environ.get("TT_VISIBLE_DEVICES") == "28", "preserve ambient pin for default --chips 1"
    rfd4_tier0.train_rfd4(files=files, out_dir=out, steps=2, chip_ids=(31,),
                          checkpoint_every=0, tokens=None, status=Status())
    assert os.environ.get("TT_VISIBLE_DEVICES") == "31", "honour explicit single chip id"

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


def _make_fake_sysfs(root, pci_id: str, count: int):
    for i in range(count):
        d = root / f"tenstorrent!{i}" / "device"
        d.mkdir(parents=True)
        (d / "device").write_text(pci_id + "\n")
    return str(root)


def test_host_topology_reads_sysfs_without_importing_ttnn(tmp_path):
    """Pure-sysfs detection distinguishes BH, BH Galaxy, WH, and CPU-only hosts."""
    from tt_bio.train.rfd4_tier0 import host_topology

    assert host_topology(str(tmp_path / "absent")) == (None, 0)
    assert host_topology(_make_fake_sysfs(tmp_path / "bh1", "0xb140", 1)) == ("blackhole", 1)
    assert host_topology(_make_fake_sysfs(tmp_path / "bh32", "0xb140", 32)) == ("blackhole", 32)
    assert host_topology(_make_fake_sysfs(tmp_path / "wh32", "0x401e", 32)) == ("wormhole_b0", 32)


def test_rfd4_resident_default_opt_out_and_fallbacks(tmp_path, monkeypatch):
    """--resident defaults on on Blackhole with Muon, opts out cleanly, and falls back on AdamW/WH."""
    from tt_bio.train import rfd4_run, rfd4_tier0

    seen = []
    monkeypatch.setattr(rfd4_run, "main", lambda argv, on_step=None: (seen.append(list(argv)), 0)[1])
    data = tmp_path / "data"
    data.mkdir()
    f0 = data / "b0.npz"
    f0.write_bytes(b"")
    out = tmp_path / "run"
    out.mkdir()

    class _Status:
        def step(self, row):
            pass

    # Default on Blackhole -> passes --resident and --optimizer muon
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0,),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          topology=("blackhole", 1))
    assert "--resident" in seen[-1]
    assert seen[-1][seen[-1].index("--optimizer") + 1] == "muon"

    # Explicit opt-out (--no-resident) -> omits --resident
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0,),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          resident=False, topology=("blackhole", 1))
    assert "--resident" not in seen[-1]
    assert seen[-1][seen[-1].index("--optimizer") + 1] == "muon"

    # Automatic fallback when --optimizer adamw is requested -> omits --resident
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0,),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          optimizer="adamw", topology=("blackhole", 1))
    assert "--resident" not in seen[-1]
    assert seen[-1][seen[-1].index("--optimizer") + 1] == "adamw"

    # Non-Blackhole hosts (Wormhole, CPU-only) -> default does not pass --resident
    for topo in (("wormhole_b0", 32), (None, 0)):
        rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0,),
                              checkpoint_every=0, tokens=None, status=_Status(),
                              topology=topo)
        assert "--resident" not in seen[-1], topo


@pytest.mark.parametrize("shape,n_chips", [("1x4", 4), ("2x4", 8), ("4x4", 16), ("8x4", 32)])
def test_rfd4_galaxy_spmd_mesh_dispatch_and_single_process_resume(tmp_path, monkeypatch, shape, n_chips):
    """Galaxy SPMD --mesh passes --mesh + --resident, no --dp-cards, and resumes from out_dir."""
    from tt_bio.train import rfd4_run, rfd4_tier0

    captured = {}

    def fake_main(argv, on_step=None):
        captured["argv"] = list(argv)
        captured["on_step"] = on_step
        if on_step is not None:
            on_step({"step": 1, "loss": 1.2, "step_s": 2.5})
        return 0

    monkeypatch.setattr(rfd4_run, "main", fake_main)
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    data = tmp_path / "data"
    data.mkdir()
    f0 = data / "b0.npz"
    f0.write_bytes(b"")
    out = tmp_path / "run"
    out.mkdir()
    (out / "adapter-00000004.safetensors").write_bytes(b"")

    rows = []

    class _Status:
        def step(self, row):
            rows.append(row)

    # Auto-selected on a 32-chip BH Galaxy when chip count matches a Galaxy mesh
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=5, chip_ids=tuple(range(n_chips)),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          topology=("blackhole", 32))
    argv = captured["argv"]
    assert argv[argv.index("--mesh") + 1] == shape
    assert "--resident" in argv
    assert "--dp-cards" not in argv
    assert captured["on_step"] is not None
    assert rows == [{"step": 1, "loss": 1.2, "step_s": 2.5}]
    assert argv[argv.index("--resume") + 1].endswith("adapter-00000004.safetensors")
    import os
    assert "TT_VISIBLE_DEVICES" not in os.environ, "SPMD --mesh must not pin to one chip"

    # Explicit --mesh overrides chip_ids=(0,) on non-Galaxy topology
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=5, chip_ids=(0,),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          mesh=shape, topology=("blackhole", 1))
    assert captured["argv"][captured["argv"].index("--mesh") + 1] == shape
    assert "--resident" in captured["argv"] and "--dp-cards" not in captured["argv"]


def test_rfd4_dp_cards_fallback_on_non_galaxy_boxes_and_non_mesh_widths(tmp_path, monkeypatch):
    """QuietBox multi-chip, non-mesh chip counts, and --no-resident fall back to --dp-cards."""
    from tt_bio.train import rfd4_run, rfd4_tier0

    seen = []
    monkeypatch.setattr(rfd4_run, "main", lambda argv, on_step=None: (seen.append((list(argv), on_step)), 0)[1])
    data = tmp_path / "data"
    data.mkdir()
    f0 = data / "b0.npz"
    f0.write_bytes(b"")
    out = tmp_path / "run"
    out.mkdir()

    class _Status:
        def step(self, row):
            pass

    # 4-chip Blackhole QuietBox (not a 32-chip Galaxy) -> --dp-cards + --resident, NO --mesh
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0, 1, 2, 3),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          topology=("blackhole", 4))
    argv, on_step = seen[-1]
    assert argv[argv.index("--dp-cards") + 1] == "0,1,2,3"
    assert "--resident" in argv and "--mesh" not in argv and on_step is None

    # 2 chips on a BH Galaxy (not a 1x4/2x4/4x4/8x4 shape) -> --dp-cards + --resident
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0, 2),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          topology=("blackhole", 32))
    argv, on_step = seen[-1]
    assert argv[argv.index("--dp-cards") + 1] == "0,2"
    assert "--resident" in argv and "--mesh" not in argv and on_step is None

    # 4 chips on a BH Galaxy with --no-resident -> --dp-cards, NO --mesh, NO --resident
    rfd4_tier0.train_rfd4(files=[f0], out_dir=out, steps=2, chip_ids=(0, 1, 2, 3),
                          checkpoint_every=0, tokens=None, status=_Status(),
                          resident=False, topology=("blackhole", 32))
    argv, on_step = seen[-1]
    assert argv[argv.index("--dp-cards") + 1] == "0,1,2,3"
    assert "--resident" not in argv and "--mesh" not in argv and on_step is None


def test_rfd4_cli_mesh_dry_run_and_help_layering():
    """`--mesh 8x4 --dry-run` plans 32 chips; expert flags stay out of --help and in --help-all."""
    from tt_bio.train.cli import train

    res = CliRunner().invoke(train, ["--model", "rfd4", "--mesh", "8x4", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "plan: fits at 256 tokens on 32 chip(s)" in res.output, res.output
    assert "2.58 s/step" in res.output and "17.85 GB" in res.output, res.output

    # Auto-fallback with --optimizer adamw succeeds cleanly on --dry-run
    res_adam = CliRunner().invoke(train, ["--model", "rfd4", "--optimizer", "adamw", "--dry-run"])
    assert res_adam.exit_code == 0, res_adam.output

    short = CliRunner().invoke(train, ["--help"]).output
    full = CliRunner().invoke(train, ["--help-all"]).output
    for flag in ("--resident", "--optimizer", "--mesh"):
        assert flag not in short, (flag, short)
        assert flag in full, (flag, full)


@pytest.mark.parametrize("args,expect", [
    (["--model", "rfd4", "--mesh", "3x4", "--dry-run"], "Galaxy SPMD meshes are"),
    (["--model", "rfd4", "--mesh", "2x2", "--dry-run"], "Galaxy SPMD meshes are"),
    (["--model", "rfd4", "--mesh", "bogus", "--dry-run"], "Galaxy SPMD meshes are"),
    (["--model", "rfd4", "--mesh", "2x4", "--chips", "4", "--dry-run"], "requires 8 chips, got 4"),
    (["--model", "rfd4", "--mesh", "1x4", "--chips", "0,1", "--dry-run"], "requires 4 chips, got 2"),
    (["--model", "rfd4", "--mesh", "1x4", "--no-resident", "--dry-run"], "cannot be combined with --no-resident"),
    (["--model", "rfd4", "--mesh", "1x4", "--optimizer", "adamw", "--dry-run"], "cannot be combined with --optimizer adamw"),
    (["--model", "rfd4", "--resident", "--optimizer", "adamw", "--dry-run"], "cannot be combined with --optimizer adamw"),
    (["--model", "rfd4", "--mesh", "8x4", "--global-batch", "16", "--dry-run"], "does not divide over 32 chips"),
    (["--model", "rfd4", "--chips", "0", "--dry-run"], "Invalid value for --chips"),
    (["--model", "rfd4", "--chips", "0,0", "--dry-run"], "repeats a chip"),
    (["--model", "rfd4", "--optimizer", "sgd", "--dry-run"], "Invalid value for '--optimizer'"),
    (["--model", "rfd4", "--not-a-real-flag", "--dry-run"], "No such option"),
    (["--model", "openfold3", "--mesh", "1x4", "--dry-run"], "only supported for --model rfd4"),
    (["--model", "openfold3", "--resident", "--dry-run"], "only supported for --model rfd4"),
    (["--model", "openfold3", "--optimizer", "muon", "--dry-run"], "only supported for --model rfd4"),
])
def test_rfd4_negative_invalid_mesh_chips_and_broken_flags_refuse_loudly(args, expect):
    """Every invalid mesh, chip count mismatch, contradictory or broken flag fails loudly."""
    from tt_bio.train.cli import train

    res = CliRunner().invoke(train, args)
    assert res.exit_code == 2, (args, res.exit_code, res.output)
    assert expect in res.output, (args, res.output)


def test_rfd4_negative_wormhole_refuses_explicit_resident_and_mesh():
    """Explicit --resident or --mesh refuses loudly on a Wormhole host."""
    import click
    from tt_bio.train.rfd4_tier0 import resolve_execution

    with pytest.raises(click.UsageError, match="Blackhole only"):
        resolve_execution(chip_ids=(0,), resident=True, topology=("wormhole_b0", 32))
    with pytest.raises(click.UsageError, match="Blackhole Galaxy only"):
        resolve_execution(chip_ids=(0,), mesh="1x4", topology=("wormhole_b0", 32))

