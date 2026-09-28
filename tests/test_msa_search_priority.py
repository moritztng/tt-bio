"""An offline MSA search runs at the lowest CPU priority, so it only takes what folds leave.

Host load has taken the same fold from 158 s to 262 s. The search runs the real
``compute_msa_offline`` in its own process with a stand-in ``colabfold_search`` on PATH that
records the nice value it ran at.
"""
import os
import subprocess
import sys
import textwrap

STUB = """#!/bin/sh
# colabfold_search FASTA DB OUT ...: note the nice value; write an a3m per query.
cut -d' ' -f19 /proc/$$/stat > "$LOG"
out="$3"; mkdir -p "$out"
grep '^>' "$1" | cut -c2- | while read -r n; do printf '>q\\nMK\\n' > "$out/$n.a3m"; done
"""

SEARCH = textwrap.dedent("""
    import sys
    from pathlib import Path
    from tt_bio import main as m
    m.compute_msa_offline({"s0": "MKTAYIAKQR"}, "s0", Path(sys.argv[1]), sys.argv[2], pair=False)
""")


def test_a_search_runs_at_the_lowest_cpu_priority(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    stub = bin_ / "colabfold_search"
    stub.write_text(STUB)
    stub.chmod(0o755)
    (tmp_path / "db").mkdir()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "LOG": str(tmp_path / "log")}
    r = subprocess.run([sys.executable, "-c", SEARCH, str(tmp_path / "msa"), str(tmp_path / "db")],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "log").read_text().split() == ["19"]
    assert (tmp_path / "msa" / "s0.a3m").exists()
    assert os.nice(0) != 19           # the caller's own priority is untouched
