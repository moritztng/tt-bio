#!/usr/bin/env python3
"""TT baseline leg of the GPU-vs-Tenstorrent Protenix-v2 / OpenDDE head-to-head.

Measures per-fold latency on one Blackhole card, in-process, at production
config: a single-chain target YAML (MSA on), 10 recycling steps / 200 diffusion
sampling steps / 1 sample / seed 0. The model is loaded once; one cold fold
absorbs first-kernel compile (reported separately, never in the warm numbers);
then N warm folds give min/median/max. The committed alignment is seeded into
the MSA cache as {seq_hash}.a3m before any fold, so MSA search never runs at all
-- not in the timed region and not in the cold fold.

Two targets share this harness, which is what makes the scaling comparison
possible: examples/prot.yaml (117 aa) and examples/prot300.yaml (CDK2, 298 aa).
Both use a 35-sequence alignment, so token count is the only variable.

Usage (on a TT host, card pinned via TT_VISIBLE_DEVICES):

    TT_VISIBLE_DEVICES=1 python3 scripts/gpu_vs_tt/tt_baseline.py \
        --model protenix-v2 --repeat 3 --out tt_protenix.json

    TT_VISIBLE_DEVICES=1 python3 scripts/gpu_vs_tt/tt_baseline.py \
        --model protenix-v2 --repeat 3 --target examples/prot300.yaml \
        --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --label "298 aa" \
        --out tt_protenix_300.json
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
FIXTURES = Path(__file__).resolve().parent / "fixtures"

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


def seed_msa_cache(target: Path, a3m: Path, msa_dir: Path) -> int:
    """Install ``a3m`` as the cached alignment for ``target``'s chain, so the timed
    folds hit the cache and no MSA search ever runs. Same contract the 117-aa run
    used ({sha256(seq)[:16]}.a3m in msa_dir); asserts the a3m's query row is the
    target's own sequence, which is what makes "identical bytes both sides" a
    checked fact rather than a claim.
    """
    import hashlib

    from tt_bio.main import _read_bio_chains

    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    text = a3m.read_text()
    rows = text.split("\n")
    assert rows[1] == seq, f"{a3m} query row does not match {target}'s sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / f"{h}.a3m").write_text(text)
    return text.count(">")


def build_fold(model: str, msa_dir: Path, target: Path, a3m: Path,
               samples: int = DIFFUSION_SAMPLES):
    """Open the card, load the model, seed the MSA cache; return ``(one_fold, meta)``.

    Split out of ``measure`` so the multi-card fan-out driver (``tt_concurrency.py``)
    folds through exactly the same path as the single-card latency baseline. Aggregate
    throughput and per-fold latency have to come from the same fold or the box-level
    number cannot be checked against the per-card one.
    """
    import torch  # noqa: F401
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
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
        diffusion_samples=samples, seed=SEED, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    _ensure_local_artifacts(cfg)
    n_msa = seed_msa_cache(target, a3m, msa_dir)

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
        metrics, _best, _feats = state.predict_one(target, job_cfg)
        return time.perf_counter() - t0, metrics

    return one_fold, dict(hardware=hw, load_s=round(load_s, 2), n_msa=n_msa,
                          msa_dir=str(msa_dir), diffusion_samples=samples,
                          job_cfg=job_cfg, struct_dir=str(struct_dir),
                          **_card_info()), state


def measure(model: str, repeat: int, msa_dir: Path, out_path: Path,
            target: Path, a3m: Path, label: str) -> dict:
    one_fold, meta, state = build_fold(model, msa_dir, target, a3m)
    from tt_bio.tenstorrent import cleanup
    hw, load_s, n_msa = meta["hardware"], meta["load_s"], meta["n_msa"]

    # Cold fold: first-kernel compile. Never counted in the warm numbers. The MSA
    # cache was seeded above, so no fold here -- cold or warm -- runs a search.
    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"
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
        input=f"{target.name} ({label}, single protein, MSA on)",
        target=str(target), n_msa=n_msa,
        n_tokens=cold_metrics.get("n_tokens"), n_residues=cold_metrics.get("n_residues"),
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
    ap.add_argument("--target", type=Path, default=REPO_ROOT / "examples" / "prot.yaml",
                    help="single-chain target YAML (default: the 117-aa prot.yaml)")
    ap.add_argument("--msa-a3m", type=Path, default=FIXTURES / "prot117.a3m",
                    help="committed alignment, seeded into the MSA cache before folding")
    ap.add_argument("--label", default="117 aa",
                    help="size label recorded in the result JSON")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    measure(args.model, args.repeat, args.msa_dir, args.out,
            args.target, args.msa_a3m, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
