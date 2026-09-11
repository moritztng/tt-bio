"""The host OpenMPI check warns, names what it saw, and changes nothing.

moritztng/tt-bio#12: a host OpenMPI 4.1.6 exporting OMPI_MCA_pml=ucx aborted tt-metal's
bundled MPI in MPI_Init_thread, before any Python ran, and the abort went to a worker's
/dev/null. The detection is the cheap half of that fix; scrubbing the variables is not our
call, so these tests pin both.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio.runtime import MPI_UNSET_HINT, conflicting_mpi_env, mpi_env_warning  # noqa: E402


def test_neutral_environment_is_silent():
    assert conflicting_mpi_env({}) == []
    assert conflicting_mpi_env({"PATH": "/usr/bin", "LD_LIBRARY_PATH": ""}) == []


def test_ompi_mca_variables_are_named_with_their_values():
    found = conflicting_mpi_env({"OMPI_MCA_pml": "ucx", "OMPI_MCA_plm": "^rsh"})
    assert found == ["OMPI_MCA_plm=^rsh, OMPI_MCA_pml=ucx"]
    warning = mpi_env_warning(found)
    assert "OMPI_MCA_pml=ucx" in warning
    assert MPI_UNSET_HINT in warning
    assert "Nothing has been changed for you." in warning


def test_opal_prefix_counts_on_its_own():
    assert conflicting_mpi_env({"OPAL_PREFIX": "/home/u/opt/mpi-stack"}) == [
        "OPAL_PREFIX=/home/u/opt/mpi-stack"]


def test_foreign_libmpi_on_the_library_path_is_found(tmp_path):
    (tmp_path / "libmpi.so.40").write_bytes(b"")
    found = conflicting_mpi_env({"LD_LIBRARY_PATH": f"/nonexistent{os.pathsep}{tmp_path}"})
    assert found == [f"libmpi.so.40 on LD_LIBRARY_PATH at {tmp_path}"]


def test_the_bundled_libmpi_is_not_a_conflict(tmp_path, monkeypatch):
    """ttnn's own libs directory on the path is the build we want, not a foreign one."""
    import tt_bio.runtime as runtime

    libs = tmp_path / "ttnn" / "libs"
    libs.mkdir(parents=True)
    (libs / "libmpi.so.40.40.7").write_bytes(b"")
    monkeypatch.setattr(runtime, "_bundled_mpi_root", lambda: tmp_path / "ttnn")
    assert runtime.conflicting_mpi_env({"LD_LIBRARY_PATH": str(libs)}) == []


def test_the_check_never_scrubs(monkeypatch):
    monkeypatch.setenv("OMPI_MCA_pml", "ucx")
    monkeypatch.setenv("OPAL_PREFIX", "/opt/mpi-stack")
    assert conflicting_mpi_env()
    assert os.environ["OMPI_MCA_pml"] == "ucx"
    assert os.environ["OPAL_PREFIX"] == "/opt/mpi-stack"
