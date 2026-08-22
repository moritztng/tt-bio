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
               samples: int = DIFFUSION_SAMPLES, hoist: bool = False,
               instrument: bool = False, fast: bool = False, trace: bool = False):
    """Open the card, load the model, seed the MSA cache; return ``(one_fold, meta)``.

    Split out of ``measure`` so the multi-card fan-out driver (``tt_concurrency.py``)
    folds through exactly the same path as the single-card latency baseline. Aggregate
    throughput and per-fold latency have to come from the same fold or the box-level
    number cannot be checked against the per-card one.

    ``hoist`` changes the timed region to match the GPU harness exactly: featurization
    (chain read, MSA resolve, build_complex_features) runs ONCE up front and the CIF
    write is suppressed, so a timed fold is ``model.fold`` only -- the same boundary as
    the GPU leg's ``runner.predict`` + ``cuda.synchronize()``. The first call to
    ``one_fold`` is still the full ``predict_one`` (cold fold: validates the MSA cache
    and warms every kernel); only the timed calls switch to the hoisted region. The raw
    ``predict_one`` boundary stays the default; the committed numbers used it.

    ``instrument`` (raw mode only) wraps the three phases of ``_predict_protenix_one``
    with timers -- build_complex_features, model.fold, _write_protenix_structure -- and
    appends one ``{feat, fold, write, total}`` entry per fold to ``meta["phase_times"]``,
    so the host/device split is measured on the same folds the raw number comes from.
    Monkeypatches module attributes inside this worker process only; production code is
    untouched. protenix-v2 only (the disputed model).
    """
    import torch  # noqa: F401
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    if (hoist or instrument) and model != "protenix-v2":
        raise ValueError("hoist/instrument are protenix-v2 only (the disputed model)")

    _noop = lambda *a, **k: None
    _E.set_progress(_noop)

    # --trace replays a captured ttnn trace of the per-step diffusion device graph, and
    # opendde.py:481 refuses to fold with trace=True unless the device carries a region.
    get_device(trace_region_size=(1 << 30) if trace else 0)  # open the chip once (lease enforced here)
    hw = arch_name()

    work = Path(tempfile.mkdtemp(prefix=f"ttbase-{model}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        model=model, fast=fast, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=samples, seed=SEED, trace=trace,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    _ensure_local_artifacts(cfg)
    # a3m=None means the target carries its own alignment (a yaml with per-chain msa:
    # dirs). seed_msa_cache asserts a monomer, so a heterodimer can only come in that
    # way; the per-chain specs are resolved below by _resolve_a3m_text.
    n_msa = seed_msa_cache(target, a3m, msa_dir) if a3m is not None else None

    state = _WorkerState("tenstorrent")
    t_load = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t_load
    state.bind_run("ttbase", cfg)
    state.pfn = _noop

    job_cfg = dict(cfg)

    phase_times: list[dict] = []
    if instrument:
        import tt_bio.main as _main
        import tt_bio.protenix_data as _pd

        _orig_feat, _orig_write = _pd.build_complex_features, _main._write_protenix_structure
        _cur: dict[str, float] = {}

        def _timed_feat(*a, **k):
            t = time.perf_counter()
            try:
                return _orig_feat(*a, **k)
            finally:
                _cur["feat"] = _cur.get("feat", 0.0) + time.perf_counter() - t

        def _timed_write(*a, **k):
            t = time.perf_counter()
            try:
                return _orig_write(*a, **k)
            finally:
                _cur["write"] = _cur.get("write", 0.0) + time.perf_counter() - t

        _orig_model_fold = state.model.fold

        def _timed_fold(*a, **k):
            t = time.perf_counter()
            try:
                return _orig_model_fold(*a, **k)
            finally:
                _cur["fold"] = _cur.get("fold", 0.0) + time.perf_counter() - t

        _pd.build_complex_features = _timed_feat
        _main._write_protenix_structure = _timed_write
        state.model.fold = _timed_fold

    hoisted = None
    if hoist:
        from tt_bio.main import (_read_bio_chains, _read_bio_constraints,
                                 _resolve_a3m_text)
        from tt_bio.protenix_data import build_complex_features

        chains = _read_bio_chains(target)
        bonds = _read_bio_constraints(target)
        chain_specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir)
                        if mt == "protein" else None, mt)
                       for _cid, cseq, spec, mt in chains]
        t_feat = time.perf_counter()
        feats_h = build_complex_features(
            chain_specs, mol_dir=cfg.get("mol_dir"),
            chain_ids=[cid for cid, _s, _sp, _mt in chains], bonds=bonds)
        feat_once_s = time.perf_counter() - t_feat
        n_res = sum(len(cseq) for _c, cseq, _s, mt in chains if mt != "ligand")

        def hoisted():
            t0 = time.perf_counter()
            with torch.no_grad(), state._maybe_ref_bf16():
                # Fresh top-level mapping per fold. The GPU leg had to do this because
                # protenix's forward deletes the MSA keys out of the feature dict in
                # place; if this fold does the same, reusing one dict would quietly
                # fold folds 2..n without an MSA and report a fast, wrong number. The
                # pLDDT check in tt_concurrency is what actually proves it did not.
                _coords, conf = state.model.fold(
                    dict(feats_h), n_step=SAMPLING_STEPS, n_sample=samples, seed=SEED,
                    progress_fn=_noop, return_confidence=True,
                    n_cycles=RECYCLING_STEPS,
                    max_parallel_samples=cfg.get("max_parallel_samples"), trace=False)
            confs = conf if isinstance(conf, list) else [conf]
            metrics = {"plddt": confs[0]["plddt"], "msa": True,
                       "n_tokens": int(feats_h["restype"].shape[0]),
                       "n_residues": n_res, "samples": samples}
            return time.perf_counter() - t0, metrics

    cold_done = {"v": False}

    def one_fold():
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        if hoist and cold_done["v"]:
            return hoisted()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(target, job_cfg)
        total = time.perf_counter() - t0
        cold_done["v"] = True
        if instrument:
            cur = {k: round(v, 4) for k, v in _cur.items()}
            cur["total"] = round(total, 4)
            phase_times.append(cur)
            _cur.clear()
        return total, metrics

    return one_fold, dict(hardware=hw, load_s=round(load_s, 2), n_msa=n_msa,
                          msa_dir=str(msa_dir), diffusion_samples=samples,
                          job_cfg=job_cfg, struct_dir=str(struct_dir),
                          timed_region=("model.fold only (featurization hoisted, "
                                        "CIF write suppressed)" if hoist else
                                        "predict_one (featurize + fold + CIF write)"),
                          phase_times=phase_times if instrument else None,
                          **_card_info()), state


def measure(model: str, repeat: int, msa_dir: Path, out_path: Path,
            target: Path, a3m: Path, label: str, fast: bool = False,
            keep_cif: Path | None = None, trace: bool = False) -> dict:
    one_fold, meta, state = build_fold(model, msa_dir, target, a3m, fast=fast, trace=trace)
    from tt_bio.tenstorrent import cleanup
    hw, load_s, n_msa = meta["hardware"], meta["load_s"], meta["n_msa"]

    # Cold fold: first-kernel compile. Never counted in the warm numbers. The MSA
    # cache was seeded above, so no fold here -- cold or warm -- runs a search.
    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"
    msa_hits = sorted(p.name for p in msa_dir.glob("*.a3m"))

    def _fold_record(t, m):
        """Per-fold provenance: the CIF hash the arm produced, its plDDT and the host load.

        An A/B whose arms differ in numerics must show what each arm actually built, and the
        load at the fold is what made an earlier 512 aa figure unusable. struct_dir is cleared
        at the START of the next fold, so this reads it while it is still the current one.
        """
        import hashlib
        cifs = {}
        for f in sorted(Path(meta["struct_dir"]).glob("*.cif")):
            cifs[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if keep_cif is not None:
            # struct_dir is cleared at the start of the next fold, so the copy happens here or
            # never. One directory per arm, named by the caller, so an RMSD control has a file.
            import shutil
            keep_cif.mkdir(parents=True, exist_ok=True)
            for f in sorted(Path(meta["struct_dir"]).glob("*.cif")):
                shutil.copy2(f, keep_cif / f.name)
        return {"s": round(t, 3), "plddt": m.get("plddt"), "cif_sha256": cifs,
                "loadavg": open("/proc/loadavg").read().split()[:3]}

    fold_recs = [dict(_fold_record(cold_s, cold_metrics), arm_fold="cold")]
    times = []
    for _ in range(repeat):
        t, _m = one_fold()
        times.append(t)
        fold_recs.append(_fold_record(t, _m))

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
        warm_folds=fold_recs,
        runtime_root=os.environ.get("TT_METAL_RUNTIME_ROOT", "<unset>"),
        kernel_cache=os.environ.get("TT_METAL_CACHE", "<unset>"),
        uptime=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
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
    ap.add_argument("--fast", action="store_true",
                    help="fold with the shipped --fast block-fp8 path (not the default arm)")
    ap.add_argument("--trace", action="store_true",
                    help="replay a captured ttnn trace of the diffusion step (reserves 1 GiB)")
    ap.add_argument("--keep-cif", type=Path, default=None,
                    help="copy each fold's CIF here, so an arm can be scored against another")
    args = ap.parse_args()
    measure(args.model, args.repeat, args.msa_dir, args.out,
            args.target, args.msa_a3m, args.label, fast=args.fast, keep_cif=args.keep_cif,
            trace=args.trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
