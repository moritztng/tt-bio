#!/usr/bin/env python3
"""TT baseline leg of the GPU-vs-Tenstorrent Protenix-v2 / OpenDDE head-to-head.

Measures per-fold latency on one Blackhole card, in-process, at production
config: examples/prot.yaml (117 aa, single protein chain, MSA on), 10 recycling
steps / 200 diffusion sampling steps / 1 sample / seed 0. The model is loaded
once; one cold fold absorbs first-kernel compile and the one-time MSA fetch
(both reported separately, never in the warm numbers); then N warm folds give
min/median/max. Every timed fold hits the MSA cache ({seq_hash}.a3m in
--msa-dir), so MSA search is never inside a timed region.

Usage (on a TT host, card pinned via TT_VISIBLE_DEVICES):

    TT_VISIBLE_DEVICES=1 python3 scripts/gpu_vs_tt/tt_baseline.py \
        --model protenix-v2 --repeat 3 --out tt_protenix.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROT = REPO_ROOT / "examples" / "prot.yaml"

# Production config, identical to what the GPU leg must run (fairness contract).
RECYCLING_STEPS = 10
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _ttnn_version() -> str:
    try:
        from importlib.metadata import version
        return version("ttnn")
    except Exception:
        return "unknown"


def _card_info() -> dict:
    """Card type + AICLK from tt-smi, best-effort (no device open)."""
    info: dict = {}
    tt_smi = None
    for c in (Path(sys.executable).parent / "tt-smi",
              Path.home() / ".local" / "bin" / "tt-smi",
              Path("/usr/local/bin/tt-smi")):
        if c.is_file() and os.access(c, os.X_OK):
            tt_smi = str(c)
            break
    if tt_smi is None:
        return info
    try:
        out = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=20)
        devs = json.loads(out.stdout).get("device_info", [])
        visible = int((os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"))
        idx = min(visible, len(devs) - 1) if devs else 0
        d = devs[idx]
        info["card_type"] = d.get("board_info", {}).get("board_type")
        tel = d.get("telemetry", {}) or {}
        if tel.get("aiclk"):
            info["aiclk_mhz"] = tel["aiclk"]
    except Exception:
        pass
    return info


def measure(model: str, repeat: int, msa_dir: Path, out_path: Path) -> dict:
    import torch  # noqa: F401
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name, cleanup
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    _noop = lambda *a, **k: None
    _E.set_progress(_noop)

    get_device()  # open the chip once (lease enforced here)
    hw = arch_name()

    work = Path(tempfile.mkdtemp(prefix=f"ttbase-{model}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        model=model, fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    t_load = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t_load
    state.bind_run("ttbase", cfg)
    state.pfn = _noop

    job_cfg = dict(cfg)

    def one_fold():
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(PROT, job_cfg)
        return time.perf_counter() - t0, metrics

    # Cold fold: first-kernel compile + one-time MSA fetch (cache miss). Never
    # counted in the warm numbers.
    cold_s, cold_metrics = one_fold()
    msa_hits = sorted(p.name for p in msa_dir.glob("*.a3m"))

    times = []
    for _ in range(repeat):
        t, _m = one_fold()
        times.append(t)

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    result = dict(
        model=model, side="tenstorrent",
        hardware=hw, machine=socket.gethostname(),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES", "0"),
        **_card_info(),
        tt_bio_git=_git_sha(), ttnn_version=_ttnn_version(),
        input="prot.yaml (117 aa, single protein, MSA on)",
        msa_cache=msa_hits,
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED,
        load_s=round(load_s, 2), cold_s=round(cold_s, 2),
        warm_times_s=[round(t, 3) for t in times],
        warm_min_s=round(times_sorted[0], 3), warm_median_s=round(median, 3),
        warm_max_s=round(times_sorted[-1], 3),
        cold_metrics=cold_metrics,
        date=time.strftime("%Y-%m-%d"),
    )
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{model}] warm median {median:.2f}s "
          f"(min {times_sorted[0]:.2f} / max {times_sorted[-1]:.2f}), "
          f"cold {cold_s:.1f}s, load {load_s:.0f}s", file=sys.stderr)
    state.reset()
    cleanup()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--repeat", type=int, default=3, help="timed warm folds (default 3)")
    ap.add_argument("--msa-dir", type=Path,
                    default=Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser())
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    measure(args.model, args.repeat, args.msa_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
