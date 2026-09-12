"""What moving SHARD_CAP_MEASURED_LINK 2.299 -> 2.801 does to the campaign's best nameable route.

The workstream asks for this number and `ceiling_v3.py` lives on `wk/b2z2-orchestrator`, not here.
Rather than restate its arithmetic -- which is how a campaign ends up with two ceilings that
disagree -- this file RUNS that file, twice, with the one constant patched. It is checked out from
git at run time into this directory so its `ROOT = parents[2]` still resolves to the worktree root
and it still reads the cell off `site/data/perf-512aa.json`, then removed again. If the orchestrator
edits `ceiling_v3.py`, this picks the edit up; nothing is vendored.

    python3 perf/b2z2_bshardtime/ceiling_delta.py
"""

import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SRC = "origin/wk/b2z2-orchestrator:perf/b2z2_orch/ceiling_v3.py"
OLD_CAP, NEW_CAP = 2.299, 2.801          # measured link, b_shard off / on
OLD_FREE, NEW_FREE = 2.990, 4.845        # free link

src = subprocess.run(["git", "show", SRC], cwd=HERE.parents[1], capture_output=True, text=True)
if src.returncode:
    raise SystemExit(f"cannot read {SRC}: {src.stderr.strip()}\n"
                     "run `git fetch origin wk/b2z2-orchestrator` first")


def run(cap, free):
    """Exec ceiling_v3 with the two cap constants patched, and return its own JSON."""
    body = re.sub(r"^SHARD_CAP_MEASURED_LINK = [\d.]+", f"SHARD_CAP_MEASURED_LINK = {cap}",
                  src.stdout, count=1, flags=re.M)
    body = re.sub(r"^SHARD_CAP_FREE_LINK = [\d.]+", f"SHARD_CAP_FREE_LINK = {free}",
                  body, count=1, flags=re.M)
    assert f"SHARD_CAP_MEASURED_LINK = {cap}" in body, "the constant did not patch"
    f = HERE / "_ceiling_v3_run.py"
    f.write_text(body)
    try:
        with tempfile.NamedTemporaryFile(suffix=".json") as tf:
            r = subprocess.run([sys.executable, str(f), "--json", tf.name],
                               capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stdout + r.stderr)
            return json.loads(pathlib.Path(tf.name).read_text())
    finally:
        f.unlink(missing_ok=True)


before, after = run(OLD_CAP, OLD_FREE), run(NEW_CAP, NEW_FREE)
cell = before["cell_s"]
print(f"published cell {cell:.3f} s, 2x target {cell/2:.3f} s\n")
print(f"{'route':<34} {'shard':>10} {'+b_shard':>10} {'move':>9}")
for key, label in (("cap_fold", "both shards at their caps"),
                   ("cap_fold_calibrated", "the same, WH->BH calibrated")):
    b, a = before[key], after[key]
    print(f"{label:<34} {b:>9.4f}x {a:>9.4f}x {a/b:>8.4f}x")
    print(f"{'':34} {cell/b:>8.2f} s {cell/a:>8.2f} s {cell/b - cell/a:>7.2f} s")
print()
print(f"the block cap itself             {OLD_CAP:>9.3f}x {NEW_CAP:>9.3f}x "
      f"{NEW_CAP/OLD_CAP:>8.4f}x")
print(f"WH->BH shard calibration          {before['wh_to_bh_shard_calibration']:.4f}x "
      "(unchanged: it is read off the N=2 route, which this does not touch)")
print(f"\n2x needs {cell/2:.3f} s; the calibrated route with b_shard lands at "
      f"{cell/after['cap_fold_calibrated']:.2f} s. Still not 2x.")
out = {"cell_s": cell, "cap_block": {"shard": OLD_CAP, "bshard": NEW_CAP},
       "cap_block_free": {"shard": OLD_FREE, "bshard": NEW_FREE},
       "fold": {k: {"shard": before[k], "bshard": after[k], "move": after[k] / before[k]}
                for k in ("cap_fold", "cap_fold_calibrated")},
       "seconds": {k: {"shard": cell / before[k], "bshard": cell / after[k]}
                   for k in ("cap_fold", "cap_fold_calibrated")},
       "wh_to_bh_shard_calibration": before["wh_to_bh_shard_calibration"],
       "source": SRC}
(HERE / "ceiling_delta.json").write_text(json.dumps(out, indent=1))
print(f"wrote {HERE / 'ceiling_delta.json'}")
