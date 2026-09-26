"""`fullstep.py --help` has to exit 0, because for a while it could not.

Two rows added a `--loss-shape` to the same parser in the same week with different vocabularies
(`model|harness` from of3t-p10samples, `per-root|model` from of3t-p10host). The two lines sit
nine apart, so the merge was textually clean and neither branch had a conflict to resolve.
`argparse` then raised `conflicting option string: --loss-shape` while BUILDING the parser, so
the merged harness could not run at all -- not one argv, not `--help`. Every measurement the
campaign takes with this file went through a tree where that was true for hours.

A parser is the cheapest thing in the file to check and the only one a merge can break without
touching a line either side wrote. `--help` builds every argument and exits before the import
of ttnn matters, so this needs no card and costs ~1 s.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "perf" / "of3t_stepfloor" / "fullstep.py"


def test_the_step_harness_can_build_its_parser():
    r = subprocess.run([sys.executable, str(HARNESS), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"`fullstep.py --help` exited {r.returncode}:\n{r.stderr}"
    assert "--loss-shape" in r.stdout


def test_no_option_string_is_declared_twice():
    """The failure mode by name: one `ap.add_argument` per option string in the file."""
    import re
    src = HARNESS.read_text()
    names = re.findall(r"ap\.add_argument\(\s*\"(--[a-z0-9-]+)\"", src)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"declared more than once: {dupes}"
