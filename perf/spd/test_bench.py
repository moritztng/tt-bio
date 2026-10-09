"""CPU tests for bench.py's --model plumbing: no weights, no device (`--dry` stops after the CLI config).

    PYTHONPATH=. python -m pytest perf/spd/test_bench.py -q
"""
import json, os, subprocess, sys
from pathlib import Path

import pytest

BENCH = Path(__file__).with_name("bench.py")
REPO = BENCH.parents[2]


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


def test_protenix_argv_unchanged(tmp_path):
    """The Protenix-v2 CLI line is the one every BOARD record before --model was taken with."""
    p, recs, d = dry(tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert recs[-1]["cli"] == ["predict", str(d / "inputs/t.yaml"), "--model", "protenix-v2", "--diffusion_samples",
                               "5", "--recycling_steps", "10", "--accelerator", "tenstorrent", "--output_format",
                               "cif", "--msa_dir", str(d / "msa"), "--msa_cache_only", "--out_dir",
                               str(tmp_path / "out/cli")]
    assert recs[-1]["cfg"]["recycling_steps"] == 10 and recs[-1]["cfg"]["diffusion_samples"] == 5


@pytest.mark.parametrize("model,recycles", [("opendde", 10), ("openfold3", 3)])
def test_engine_default_recycles(tmp_path, model, recycles):
    """Other models fold with the engine's own recycles unless told, as a JapanFold job does."""
    p, recs, _ = dry(tmp_path, "--model", model)
    assert p.returncode == 0, p.stderr[-2000:]
    cfg = recs[-1]["cfg"]
    assert "--recycling_steps" not in recs[-1]["cli"]
    assert (cfg["model"], cfg["recycling_steps"], cfg["diffusion_samples"], cfg["msa_cache_only"]) == \
        (model, recycles, 5, True)


def test_boltz2_config(tmp_path):
    p, recs, _ = dry(tmp_path, "--model", "boltz2", "--recycles", "4")
    assert p.returncode == 0, p.stderr[-2000:]
    pa = recs[-1]["cfg"]["conf_kwargs"]["predict_args"]
    assert (pa["recycling_steps"], pa["diffusion_samples"], pa["sampling_steps"]) == (4, 5, 200)


def test_fast_flag_reaches_config(tmp_path):
    p, recs, _ = dry(tmp_path, "--model", "openfold3", "--arm", "f:fast")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "--fast" in recs[-1]["cli"] and recs[-1]["cfg"]["fast"] is True


@pytest.mark.parametrize("model", ["opendde", "openfold3", "boltz2", "bindcraft2"])
def test_lever_arm_refused_off_protenix(tmp_path, model):
    """No other model reads tenstorrent.LEVERS: an L= arm there would time the default and call it a lever."""
    p, recs, _ = dry(tmp_path, "--model", model, "--arm", "x:L=fast-lofi")
    assert p.returncode != 0 and "L= arms build Protenix-v2 only" in p.stderr and not recs


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
