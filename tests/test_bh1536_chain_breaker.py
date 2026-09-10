"""A wedged card must not empty the ladder's queue into it.

chain2 lost eight rungs in under five minutes that way — the whole esmfold2 ladder and every
1300-token rung — because a wedged chip fails each one in about 20 s and the chain walked on.
Nothing false was published (they all read WEDGED, which `measured.py` does not count as a
measurement) but the queue was gone and the chain had exited.

Drives the REAL `chain.sh` with a stub `PY`, so what is under test is the shipped script.
Host-only: no device, no model, and `run_rung.py` is never actually invoked.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

CHAIN = Path(__file__).resolve().parents[1] / "perf" / "bh1536" / "chain.sh"

#: The stub has to dispatch on WHICH script chain.sh hands it. chain.sh asks `measured.py`
#: first, where exit 0 means "already measured, skip this rung"; a stub returning one code for
#: both made every rung skip, so the first version of this test measured skipping and not the
#: breaker at all.
STUB = """\
#!/bin/bash
case "$1" in
  *measured.py) exit 1 ;;
  *run_rung.py) {body} ;;
  *) exit 0 ;;
esac
"""


def _stub(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "stub.sh"
    p.write_text(STUB.format(body=body))
    p.chmod(0o755)
    return p


def _run(stub: Path, *specs: str):
    env = {**os.environ, "PY": str(stub)}
    r = subprocess.run(["bash", str(CHAIN), *specs], capture_output=True, text=True,
                       env=env, timeout=300)
    started = r.stdout.count("\n=== ") + r.stdout.startswith("=== ")
    return r, started


@pytest.mark.skipif(not CHAIN.is_file(), reason="chain.sh not in this checkout")
def test_three_rungs_that_measure_nothing_stop_the_chain(tmp_path):
    r, started = _run(_stub(tmp_path, "exit 75"), *[f"m{i}:1" for i in range(6)])
    assert started == 3, f"walked {started} rungs into a dead card, not 3\n{r.stdout}"
    assert r.returncode == 75, f"exit {r.returncode}"
    assert "STOPPING: 3 rungs in a row" in r.stdout, r.stdout
    # The point of stopping: the rest of the list is still un-walked and a relaunch gets it.
    assert "m5" not in r.stdout.split("STOPPING")[0]


@pytest.mark.skipif(not CHAIN.is_file(), reason="chain.sh not in this checkout")
def test_a_healthy_chain_is_never_cut(tmp_path):
    """The negative control. A breaker that fires on a working card destroys the ladder instead
    of protecting it."""
    r, started = _run(_stub(tmp_path, "exit 0"), *[f"m{i}:1" for i in range(6)])
    assert started == 6, f"only {started} of 6 rungs ran\n{r.stdout}"
    assert r.returncode == 0, f"exit {r.returncode}"
    assert "STOPPING" not in r.stdout


@pytest.mark.skipif(not CHAIN.is_file(), reason="chain.sh not in this checkout")
def test_one_good_rung_resets_the_counter(tmp_path):
    """Consecutive, not cumulative: a card that recovers mid-list must not be given up on. Two
    dead, one good, two dead is five rungs and no stop."""
    counter = tmp_path / "n"
    body = textwrap.dedent(f"""\
        n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
             [ "$n" -eq 3 ] && exit 0 || exit 75""")
    r, started = _run(_stub(tmp_path, body), *[f"m{i}:1" for i in range(5)])
    assert started == 5, f"a mid-list recovery still tripped the breaker\n{r.stdout}"
    assert r.returncode == 0, f"exit {r.returncode}"
