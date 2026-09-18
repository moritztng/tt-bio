"""A gate arm whose scorer is not installed must fail at startup, not after the fold.

Device-free and torch-free. On 2026-09-18 the opendde-abag arm folded 1ahw_abag successfully --
"1 ok, 0 failed", 348.5 s of device time -- and then died importing DockQ, which
scripts/opendde_dockq.py documents as "an eval-time requirement into the run venv", i.e. something
nothing in the install path provides. The gate rendered it "GATE FAIL - opendde-abag missed parse
or the DockQ floor", so a red arm that measured nothing sat on the critical path of a merge
decision. This is the same class _preflight_msa_cache was written for, in a different costume.

Run: python3 tests/test_release_gate_eval_scorer_preflight.py, or via pytest.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg


def test_every_declared_scorer_names_a_module_a_pin_and_its_user():
    assert rg._EVAL_SCORERS, "the table is the whole check; an empty one silently passes"
    for arm, entry in rg._EVAL_SCORERS.items():
        mod, pin, user = entry[0], entry[1], entry[2]
        assert "." in mod or mod.isidentifier(), mod
        assert "==" in pin, f"{arm}: pin the version the arm was validated against, got {pin!r}"
        assert user.endswith(".py"), user
        assert len(entry) > 3, (f"{arm}: name the interpreter that imports {mod} at scoring "
                               f"time (None = the gate's own); checking the wrong one is the "
                               f"bug this field exists for")


def test_an_unselected_arm_is_never_checked():
    """Deselecting the arm with --models is the documented escape hatch, so it must work."""
    rg._preflight_eval_scorers(["boltz2", "esmfold2"])


def test_a_missing_scorer_exits_before_any_device_work():
    saved = dict(rg._EVAL_SCORERS)
    try:
        rg._EVAL_SCORERS["probe-arm"] = ("no_such_module_xyz", "nope==1.0",
                                         "scripts/probe.py", None)
        try:
            rg._preflight_eval_scorers(["probe-arm"])
        except SystemExit as exc:
            m = str(exc)
            # It must name the arm, the module, the pin and where to install it -- a bare
            # ImportError is what sent the last reader looking for a model defect.
            for want in ("probe-arm", "no_such_module_xyz", "nope==1.0", "measures nothing"):
                assert want in m, (want, m)
            assert sys.executable in m, "say WHICH venv to install into"
        else:
            raise AssertionError("a missing scorer must stop the gate")
    finally:
        rg._EVAL_SCORERS.clear()
        rg._EVAL_SCORERS.update(saved)


def test_a_present_scorer_is_silent():
    saved = dict(rg._EVAL_SCORERS)
    try:
        rg._EVAL_SCORERS["probe-arm"] = ("json", "stdlib==1.0", "scripts/probe.py", None)
        rg._preflight_eval_scorers(["probe-arm"])
    finally:
        rg._EVAL_SCORERS.clear()
        rg._EVAL_SCORERS.update(saved)


def test_the_delegated_interpreter_is_the_one_checked():
    """opendde-abag imports DockQ in OPENDDE_DOCKQ_PYTHON, not here.

    The negative control is the point: a module this interpreter HAS and the delegate does not
    must still fail. Before the fix the preflight imported in-process, so it passed exactly the
    configuration it cannot score and failed the supported one.
    """
    saved = dict(rg._EVAL_SCORERS)
    try:
        # /bin/false is an interpreter that never imports anything: stands in for a venv
        # without the scorer, without needing a second venv on the box.
        rg._EVAL_SCORERS["probe-arm"] = ("json", "stdlib==1.0", "scripts/probe.py", "/bin/false")
        assert __import__("json"), "this interpreter HAS json -- that is the control"
        try:
            rg._preflight_eval_scorers(["probe-arm"])
        except SystemExit as exc:
            assert "/bin/false" in str(exc), "name the interpreter that was checked"
            assert "probe-arm" in str(exc)
        else:
            raise AssertionError("a scorer absent from the DELEGATE must stop the gate even "
                                 "when the gate's own venv has it")
    finally:
        rg._EVAL_SCORERS.clear()
        rg._EVAL_SCORERS.update(saved)


def test_a_delegate_that_does_have_the_module_is_silent():
    saved = dict(rg._EVAL_SCORERS)
    try:
        rg._EVAL_SCORERS["probe-arm"] = ("json", "stdlib==1.0", "scripts/probe.py",
                                         sys.executable)
        rg._preflight_eval_scorers(["probe-arm"])
    finally:
        rg._EVAL_SCORERS.clear()
        rg._EVAL_SCORERS.update(saved)


def test_the_opendde_abag_entry_points_at_the_python_that_scores_it():
    """Wired to the real global, so moving OPENDDE_DOCKQ_PYTHON cannot silently unwire it."""
    assert rg._EVAL_SCORERS["opendde-abag"][3] == rg.OPENDDE_DOCKQ_PYTHON


def test_it_runs_before_the_msa_preflight_and_before_any_fold():
    """Ordering is the point: both preflights must precede the first device call in main()."""
    src = open(os.path.join(REPO, "scripts", "release_gate.py")).read()
    i_eval = src.index("_preflight_eval_scorers(models)")
    i_msa = src.index("_preflight_msa_cache(models)")
    i_rows = src.index("rows = [run_model(")
    assert i_eval < i_msa < i_rows, (i_eval, i_msa, i_rows)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
