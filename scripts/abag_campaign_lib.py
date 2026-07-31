"""Shared machinery for the AbAg fold-campaign drivers (`abag_*_fold.py`).

Every driver runs the same loop: for each (target, model) not already `ok` in
its progress.jsonl, invoke `python -m tt_bio.main predict` as a subprocess with
a hard timeout, then read the target's results.json entry. This module keeps
that loop in one place; each driver supplies its own targets, yaml dir, output
layout, and per-record extras (DockQ scoring, PAE paths).
"""
import json
import os
import signal
import subprocess
import sys
import time

FOLD_TIMEOUT_S = 1200  # a real hang was observed (opendde-abag's paired-MSA
# network call stuck at 0% CPU under concurrent same-host load); kill the whole
# process group (multiprocessing forks don't die from a plain subprocess
# timeout on the direct child) and record it, so the campaign self-heals
# instead of needing a manual kill + tt-smi reset.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR_PREFIX = {"opendde-abag": "opendde", "boltz2": "boltz2", "protenix-v2": "protenix"}
DOCKQ_PYTHON = os.path.expanduser("~/.opendde_dockq_venv/bin/python3")
CONFIDENCE_KEYS = ("confidence_score", "ptm", "iptm", "protein_iptm", "complex_plddt", "runtime_s")


def done_pairs(progress):
    """(target, model) pairs already recorded `ok` in the progress file."""
    seen = set()
    if os.path.exists(progress):
        for line in open(progress):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    seen.add((r["target"], r["model"]))
            except Exception:
                pass
    return seen


def sample_cifs(struct_dir, tid):
    """The winner CIF plus any `<tid>_model_<i>.cif` sample CIFs, in order."""
    winner = f"{struct_dir}/{tid}.cif"
    files = [winner] if os.path.exists(winner) else []
    i = 1
    while os.path.exists(f"{struct_dir}/{tid}_model_{i}.cif"):
        files.append(f"{struct_dir}/{tid}_model_{i}.cif")
        i += 1
    return files


def dockq(cif, native):
    r = subprocess.run([DOCKQ_PYTHON, "scripts/opendde_dockq.py", cif, native],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode not in (0, 2):
        return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
    try:
        d = json.loads(r.stdout)
        return {"global_dockq": d["global_dockq"], "n_interfaces": d["n_interfaces"],
                "interfaces": d["interfaces"]}
    except Exception as e:
        return {"error": str(e), "stdout": r.stdout[-300:]}


def run_predict(tid, yaml, model, out_dir, msa_dir, device, extra_args=(),
                timeout=FOLD_TIMEOUT_S, python=None):
    """Fold one target id `tid` (5 samples, seed 42). Returns
    (status, payload, wall_s): status is "ok" / "timed_out" / "fold_failed";
    payload is the parsed results.json entry on "ok", the stderr tail otherwise.
    """
    t0 = time.time()
    proc = subprocess.Popen(
        [python or sys.executable, "-m", "tt_bio.main", "predict", yaml,
         "--model", model, "--out_dir", out_dir, "--diffusion_samples", "5",
         "--msa_dir", msa_dir, "--seed", "42", "--override", *extra_args],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "TT_VISIBLE_DEVICES": str(device), "PYTHONPATH": ROOT},
        start_new_session=True,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        returncode = -9
    wall_s = time.time() - t0
    rjson = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{tid}/results.json"
    if timed_out:
        return "timed_out", f"killed after {timeout}s (process group); tail: {(out or '')[-1000:]}", wall_s
    if returncode != 0 or not os.path.exists(rjson):
        return "fold_failed", (out or "")[-2000:], wall_s
    results = json.load(open(rjson))
    entry = results[0] if isinstance(results, list) else results
    if entry.get("status") == "failed":
        # results.json parses fine but records an internal failure (e.g. a stale cwd
        # from a torn-down worktree) -- catch this or it silently records status=ok
        # with every confidence field null (hit once for decoy_22psab_9obnag/opendde-abag).
        return "fold_failed", str(entry.get("error", "")), wall_s
    return "ok", entry, wall_s
