"""D56: the RFD4 step moved from perf/r4t_step/trunk_step.py into tt_bio.train.rfd4_run.

Host-only. The device A/A (master digests before = after on the r4t-step 3x3 protocol) is the real
proof; these pin what can break on a laptop: the perf caller still reaches the package entry point,
and the batch reader the step uses is the one the upstream harness writes for.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_the_perf_caller_runs_the_package_step():
    out = subprocess.run([sys.executable, str(ROOT / "perf/r4t_step/trunk_step.py"), "--help"],
                         capture_output=True, text=True, timeout=120, cwd="/")
    assert out.returncode == 0, out.stderr
    for flag in ("--batch", "--curve-dump", "--dp-cards", "--cand-levers", "--attn-fp32"):
        assert flag in out.stdout
    src = (ROOT / "perf/r4t_step/trunk_step.py").read_text()
    assert "from tt_bio.train.rfd4_run import main" in src and "argparse" not in src


def test_the_step_module_imports_without_ttnn():
    code = ("import sys; import tt_bio.train.rfd4_run; "
            "assert 'ttnn' not in sys.modules and 'torch' not in sys.modules")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr


def test_load_batch_reads_a_dumped_batch(tmp_path):
    from tt_bio.train.rfd4_data import load_batch
    f = tmp_path / "batch00.npz"
    np.savez(f, **{"in/padding_mask": np.ones((4, 8), bool), "in/x_t/bb_ca": np.zeros((4, 8, 3), np.float32),
                   "depth": 2, "seed": 7, "winners": np.array([1]), "rng_state": np.zeros(5, np.uint8),
                   "out/bb_ca": np.ones(3), "loss/total": np.array([0.5, 0.4])})
    batch, meta, rec, losses = load_batch(f)
    assert batch["strict_feats"] is True and tuple(batch["x_t"]["bb_ca"].shape) == (4, 8, 3)
    assert meta["depth"] == 2 and meta["seed"] == 7 and list(rec) == ["bb_ca"] and list(losses) == ["total"]
    sys.path.insert(0, str(ROOT / "perf/r4t"))
    import loss_parity_reference
    assert loss_parity_reference.load_batch is load_batch
