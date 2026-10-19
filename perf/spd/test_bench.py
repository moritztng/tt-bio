"""CPU tests for bench.py's --model plumbing: no weights, no device (`--dry` stops after the CLI config).

    PYTHONPATH=. python -m pytest perf/spd/test_bench.py -q
"""
import importlib.util, json, math, os, subprocess, sys
from pathlib import Path

import pytest

BENCH = Path(__file__).with_name("bench.py")
REPO = BENCH.parents[2]


# bench.py imports the engine (torch) before it reads its arguments; an interpreter without it can only run the
# harness-independent cases.
needs_engine = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="bench.py needs tt_bio's torch")


def dry(tmp_path, *extra, data=None):
    d = data or tmp_path / "data"
    (d / "inputs").mkdir(parents=True, exist_ok=True); (d / "msa").mkdir(exist_ok=True)
    (d / "inputs" / "t.yaml").write_text("version: 1\nsequences:\n- protein: {id: A, sequence: MKTAYIAKQR}\n")
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(BENCH), "--out", str(out), "--chip", "0", "--inputs", "t",
                        "--data", str(d), "--dry", *extra], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, PYTHONPATH=str(REPO)))
    recs = [json.loads(l) for l in (out / "bench.jsonl").read_text().splitlines()] if (out / "bench.jsonl").exists() else []
    return p, recs, d


@needs_engine
def test_protenix_argv_unchanged(tmp_path):
    """The Protenix-v2 CLI line is the one every BOARD record before --model was taken with."""
    p, recs, d = dry(tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert recs[-1]["cli"] == ["predict", str(d / "inputs/t.yaml"), "--model", "protenix-v2", "--diffusion_samples",
                               "5", "--recycling_steps", "10", "--accelerator", "tenstorrent", "--output_format",
                               "cif", "--msa_dir", str(d / "msa"), "--msa_cache_only", "--out_dir",
                               str(tmp_path / "out/cli")]
    assert recs[-1]["cfg"]["recycling_steps"] == 10 and recs[-1]["cfg"]["diffusion_samples"] == 5


@needs_engine
@pytest.mark.parametrize("model,recycles", [("opendde", 10), ("openfold3", 3)])
def test_engine_default_recycles(tmp_path, model, recycles):
    """Other models fold with the engine's own recycles unless told, as a JapanFold job does."""
    p, recs, _ = dry(tmp_path, "--model", model)
    assert p.returncode == 0, p.stderr[-2000:]
    cfg = recs[-1]["cfg"]
    assert "--recycling_steps" not in recs[-1]["cli"]
    assert (cfg["model"], cfg["recycling_steps"], cfg["diffusion_samples"], cfg["msa_cache_only"]) == \
        (model, recycles, 5, True)


@needs_engine
def test_boltz2_config(tmp_path):
    p, recs, _ = dry(tmp_path, "--model", "boltz2", "--recycles", "4")
    assert p.returncode == 0, p.stderr[-2000:]
    pa = recs[-1]["cfg"]["conf_kwargs"]["predict_args"]
    assert (pa["recycling_steps"], pa["diffusion_samples"], pa["sampling_steps"]) == (4, 5, 200)


@needs_engine
def test_fast_flag_reaches_config(tmp_path):
    p, recs, _ = dry(tmp_path, "--model", "openfold3", "--arm", "f:fast")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "--fast" in recs[-1]["cli"] and recs[-1]["cfg"]["fast"] is True


@needs_engine
@pytest.mark.parametrize("model", ["af2ig", "bindcraft2"])
def test_lever_arm_refused_where_no_levers(tmp_path, model):
    """A model that reads no tenstorrent.LEVERS refuses an L= arm: it would time the default and call it a lever."""
    p, recs, _ = dry(tmp_path, "--model", model, "--arm", "x:L=fast-lofi")
    assert p.returncode != 0 and f"{model} reads no tenstorrent.LEVERS" in p.stderr and not recs


@needs_engine
def test_bindcraft2_settings(tmp_path):
    """One rep = one trajectory of BindCraft 2's shipped PD-L1 campaign: cold + warm trajectories, one length."""
    bc2 = Path(os.environ.get("SPD_BC2", "~/bcx_shipped/bc2")).expanduser()
    if not (bc2 / "examples/pdl1.json").exists():
        pytest.skip(f"no BindCraft 2 checkout at {bc2}")
    try:
        import jax  # noqa: F401
    except ImportError:
        pytest.skip("bindcraft needs jax")
    p, recs, _ = dry(tmp_path, "--model", "bindcraft2", "--warm", "2", "--seed", "7", "--inputs", "90",
                     "--bc2", str(bc2))
    assert p.returncode == 0, p.stderr[-2000:]
    s = recs[-1]["settings"]
    assert (s["max_trajectories"], s["campaign_seed"], s["binder_lengths"]) == (3, 7, [90])
    assert recs[-1]["model"] == "bindcraft2" and recs[-1]["binder_length"] == 90


# A stand-in BindCraft 2 checkout: its campaign runs three trajectories of ITERS[n] design iterations, writing one
# losses.csv row per iteration and one trajectory-table row each, the files both harnesses read.
ITERS = {1: 140, 2: 140, 3: 52}
FAKE_BC2 = {
    "examples/pdl1.json": "{}",
    "bindcraft/__init__.py": "",
    "bindcraft/settings.py": """
import json
def read_settings(path, overrides):
    return dict(overrides)
def value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v
def parse_setting_overrides(items):
    return {k: value(v) for k, v in (i.split("=", 1) for i in items)}
""",
    "bindcraft/preflight.py": "def cleaned_campaign_settings(s):\n    return s\n",
    "bindcraft/model_weights.py": "def model_weights():\n    return None, None\n",
    "bindcraft/campaign_output.py": """
import csv, glob, os
TRAJECTORY_STAGE = "1_Trajectories"
def stage_table(project, stage):
    return os.path.join(project, stage + ".csv")
def accepted_table(project):
    return os.path.join(project, "accepted.csv")
def trajectory_metric_rows(project):
    return [list(csv.DictReader(open(p))) for p in sorted(glob.glob(os.path.join(project, TRAJECTORY_STAGE, "*", "*_losses.csv")))]
""",
    "bindcraft/campaign.py": f"""
import csv, os, time
from bindcraft.campaign_output import TRAJECTORY_STAGE, stage_table
ITERS = {ITERS!r}
def run_trajectory(project, n):
    d = os.path.join(project, TRAJECTORY_STAGE, f"t{{n}}"); os.makedirs(d)
    with open(os.path.join(d, f"t{{n}}_losses.csv"), "w") as f:
        f.write("iteration,loss\\n" + "".join(f"{{i}},0.5\\n" for i in range(ITERS[n])))
    time.sleep(0.05)
def redesign_and_validate_binders(project, n):
    new = not os.path.exists(stage_table(project, TRAJECTORY_STAGE))
    with open(stage_table(project, TRAJECTORY_STAGE), "a") as f:
        f.write("trajectory,design,terminated\\n" * new + f"{{n}},t{{n}},{{'mutate' if n == 3 else ''}}\\n")
def run_campaign(settings, project, **kw):
    for n in range(1, settings["max_trajectories"] + 1):
        run_trajectory(project, n)
        redesign_and_validate_binders(project, n)
""",
}
GATE_KEYS = ("kind", "design_s", "iterations", "terminated", "s_per_iter", "loadavg")


def fake_bc2(tmp_path):
    root = tmp_path / "bc2"
    for rel, text in FAKE_BC2.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def check_reps(recs):
    """What workstreams/_bc4_donecheck.py reads off each line: per-trajectory iterations, design_s/iterations finite."""
    reps = [r for r in recs if r.get("ev") == "rep"]
    assert [r["kind"] for r in reps] == ["cold", "warm", "warm"]
    for r, n in zip(reps, ITERS.values()):
        assert all(k in r for k in GATE_KEYS), sorted(set(GATE_KEYS) - set(r))
        assert r["iterations"] == n  # this trajectory's rows, not the campaign's running total
        assert math.isfinite(r["design_s"] / r["iterations"]) and r["s_per_iter"] == r["design_s"] / r["iterations"]
        assert len(r["loadavg"]) == 2
    assert [r["terminated"] for r in reps] == [None, None, "mutate"]


def test_bc2_gpu_rep_record(tmp_path):
    """The GPU harness on a stand-in BindCraft 2: no jax, no GPU."""
    bc2, out, bin_ = fake_bc2(tmp_path), tmp_path / "gpu", tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "jax.py").write_text("__version__ = 'fake'\ndef devices():\n    return ['fake:0']\n")  # version + devices only
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 'Fake GPU, 1000.00 W, 0'\n"); (bin_ / "nvidia-smi").chmod(0o755)
    p = subprocess.run([sys.executable, str(REPO / "perf/kitcmp/bc2_gpu.py"), "--bc2", str(bc2), "--out", str(out),
                        "--warm", "2", "--set", "use_cueq=true"], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", PYTHONPATH=str(bin_)))
    assert p.returncode == 0, p.stderr[-2000:]
    recs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
    check_reps(recs)
    assert recs[0]["cpu"] and recs[0]["settings"]["use_cueq"] is True and recs[0]["settings"]["auto_multi_gpu"] is False


@needs_engine
def test_bindcraft2_rep_record(tmp_path):
    """bench.py --model bindcraft2 on a stand-in BindCraft 2, tt_bio.bindcraft2 replaced so no card opens."""
    bc2, out = fake_bc2(tmp_path), tmp_path / "tt"
    run = f"""
import contextlib, runpy, sys, types
import bindcraft.campaign as campaign
fake = types.ModuleType("tt_bio.bindcraft2")
fake.campaign_predictor = lambda **kw: contextlib.nullcontext()
fake.run_campaign = lambda settings, project, trajectories_per_card, **kw: campaign.run_campaign(settings, project)
sys.modules["tt_bio.bindcraft2"] = fake
sys.argv = ["bench.py", "--out", {str(out)!r}, "--chip", "0", "--model", "bindcraft2", "--warm", "2", "--bc2", {str(bc2)!r}]
runpy.run_path({str(BENCH)!r}, run_name="__main__")
"""
    p = subprocess.run([sys.executable, "-c", run], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, PYTHONPATH=f"{bc2}:{REPO}"))
    assert p.returncode == 0, p.stderr[-2000:]
    recs = [json.loads(l) for l in (out / "bench.jsonl").read_text().splitlines()]
    check_reps(recs)
    assert all(r["model"] == "bindcraft2" and r["cpu"] and r["campaign_seed"] == 101 and r["binder_length"] == 80
               for r in recs if r.get("ev") == "rep")
