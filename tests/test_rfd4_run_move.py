"""D56: the RFD4 step moved from perf/r4t_step/trunk_step.py into tt_bio.train.rfd4_run.

Host-only. The device A/A (master digests before = after on the r4t-step 3x3 protocol) is the real
proof; these pin what can break on a laptop: the perf caller still reaches the package entry point,
and the batch reader the step uses is the one the upstream harness writes for.
"""
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_the_perf_caller_runs_the_package_step():
    out = subprocess.run([sys.executable, str(ROOT / "perf/r4t_step/trunk_step.py"), "--help"],
                         capture_output=True, text=True, timeout=120, cwd="/")
    assert out.returncode == 0, out.stderr
    for flag in ("--batch", "--curve-dump", "--dp-cards", "--cand-levers", "--win-pair",
                 "--attn-fp32", "--optimizer", "--keep", "--pin-spike"):
        assert flag in out.stdout
    assert "serve" in out.stdout
    src = (ROOT / "perf/r4t_step/trunk_step.py").read_text()
    assert "from tt_bio.train.rfd4_run import main" in src and "argparse" not in src
    run_src = (ROOT / "tt_bio/train/rfd4_run.py").read_text()
    assert "ax.sum_scalars" in run_src, "DP G2 rank-sum guard dropped during merge"
    assert "ag.suspended()" in run_src and "pair_taped(" in run_src, "P2+K14 dropped during merge"


def test_step_loop_calls_opt_step_once():
    """Guard against duplicate opt.step() merge regressions (doubles single-chip opt_s and
    crashes DP with UnreducedGradients on step 1)."""
    tree = ast.parse((ROOT / "tt_bio/train/rfd4_run.py").read_text())
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    step_loops = [n for n in ast.walk(main_fn)
                  if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                  and n.target.id == "step"]
    assert len(step_loops) == 1
    def _has_opt_step(node):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "step" and isinstance(c.func.value, ast.Name)
                   and c.func.value.id == "opt" for c in ast.walk(node))
    stmts = [s for s in step_loops[0].body if _has_opt_step(s)]
    assert len(stmts) == 1, f"expected 1 opt.step statement in step loop, found {len(stmts)}"


def test_checkpoint_due_saves_final_step_and_avoids_double_save():
    from tt_bio.train import rfd4_run

    assert rfd4_run._checkpoint_due(1, 3, save_every=100, save_at=0) == (False, False)
    assert rfd4_run._checkpoint_due(3, 3, save_every=100, save_at=0) == (True, True)
    assert rfd4_run._checkpoint_due(2, 4, save_every=2, save_at=0) == (True, False)
    assert rfd4_run._checkpoint_due(4, 4, save_every=2, save_at=0) == (True, True)
    assert rfd4_run._checkpoint_due(3, 3, save_every=100, save_at=3) == (False, False)
    assert rfd4_run._checkpoint_due(3, 3, save_every=0, save_at=0) == (False, False)


def test_upstream_exempt_documents_ss_embedding():
    from tt_bio.train import rfd4_run

    key = "glue.init_repr_factory.feat_creators.7.ss_embedding.weight"
    assert key in rfd4_run.UPSTREAM_EXEMPT and "padding_idx=0" in rfd4_run.UPSTREAM_EXEMPT[key]


def test_aiclk_sampler_resolves_open_device_node():
    """On a Galaxy, TT_VISIBLE_DEVICES != /dev/tenstorrent/<minor>, so reading
    tenstorrent!{card}/tt_aiclk watches an idle chip (800 MHz). Must use provenance.open_nodes."""
    src = (ROOT / "tt_bio/train/rfd4_run.py").read_text()
    assert "open_nodes" in src and "tenstorrent!{clk_node}/tt_aiclk" in src


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
