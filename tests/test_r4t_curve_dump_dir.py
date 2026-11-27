"""`read_dump_dir` is the only way a TT curve leg reaches its input tensors, so it is shown able
to refuse before it is trusted to accept: a file dumped for another plan, a gap in the run, and
an empty directory. Built on real dumps written by `curve_upstream.py --dump-dir`."""
from pathlib import Path

import numpy as np
import pytest

from tt_bio.train.curve import CurveContractError, read_dump_dir

AA = Path.home() / "r4t/accuracy/curve/dump_aa"      # 4 steps of the 8-step A/A plan
G3 = Path.home() / "r4t/accuracy/curve/dump_g3_3"    # 3 steps of the graded 2,000-step plan

pytestmark = pytest.mark.skipif(not (AA / "plan.json").exists() or not (G3 / "plan.json").exists(),
                                reason="dumps live under ~/r4t/accuracy/curve (pc)")


def copy_dump(src: Path, dst: Path, names=None) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "plan.json").write_bytes((src / "plan.json").read_bytes())
    for f in sorted(src.glob("step*.npz")):
        if names is None or f.name in names:
            (dst / f.name).write_bytes(f.read_bytes())
    return dst


def test_a_matching_dump_reads_back_in_step_order():
    plan, files = read_dump_dir(AA)
    assert [f.name for f in files] == [f"step{i:05d}.npz" for i in range(len(files))]
    for f in files:                       # every file names the plan it was dumped for
        assert str(np.load(f, allow_pickle=False)["plan_digest"]) == plan.digest()
    # the depth each step trains at is the plan's own seeded draw, not the file's opinion
    for i, f in enumerate(files):
        assert int(np.load(f, allow_pickle=False)["depth"]) == plan.step(i).n_recycle


def test_a_file_from_another_plan_is_refused(tmp_path):
    d = copy_dump(AA, tmp_path / "mixed")
    (d / "step00001.npz").write_bytes((G3 / "step00001.npz").read_bytes())
    with pytest.raises(CurveContractError, match="was dumped for plan"):
        read_dump_dir(d)


def test_a_dump_pointed_at_a_foreign_plan_is_refused(tmp_path):
    d = copy_dump(AA, tmp_path / "swapped")
    (d / "plan.json").write_bytes((G3 / "plan.json").read_bytes())
    with pytest.raises(CurveContractError, match="was dumped for plan"):
        read_dump_dir(d)


def test_a_gap_in_the_steps_is_refused(tmp_path):
    d = copy_dump(AA, tmp_path / "gap", names={"step00000.npz", "step00002.npz", "step00003.npz"})
    with pytest.raises(CurveContractError, match="gapless"):
        read_dump_dir(d)


def test_an_empty_directory_is_refused(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "plan.json").write_bytes((AA / "plan.json").read_bytes())
    with pytest.raises(CurveContractError, match="no step"):
        read_dump_dir(d)
