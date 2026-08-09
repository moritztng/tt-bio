#!/usr/bin/env python3
"""Fold-level A/B of the L1-resident trunk projections, interleaved in one process.

Arms alternate on the same loaded model with the same seed, so neither can drift:

  PROD  what main runs: only the qkv projection keeps its result in L1
  GO    this branch: qkv + gate + output projections

The monkeypatch discriminates by shape alone: the qkv weight is the only trunk
projection with n == 3*k, and the o projection is the only one called with
full_k=False. Whole-model bit-exactness is checked by capturing the coordinates
model.fold returns in each arm and comparing them with torch.equal. Timing is
the production predict_one boundary (featurize + fold + CIF write), the same
boundary the committed gpu_vs_tt baselines used. The cold fold runs the PROD
arm: it is warmup, and production-shaped warmup keeps compile-time cost out of
the A/B.

    TT_VISIBLE_DEVICES=0 python3 perf/qkv_rootcause/fold_ab.py \
        --model protenix-v2 --target examples/prot.yaml \
        --a3m scripts/gpu_vs_tt/fixtures/prot117.a3m --reps 3
"""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--a3m", default=None, help="seed the MSA cache with this alignment; omit for msa:empty targets")
ap.add_argument("--recycling", type=int, default=10)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--single-sequence", action="store_true")
ap.add_argument("--reps", type=int, default=3, help="warm folds per arm, interleaved")
ap.add_argument("--out", default=None)
args = ap.parse_args()


def main():
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name, cleanup
    import tt_bio.tenstorrent as T
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    noop = lambda *a, **k: None
    _E.set_progress(noop)
    get_device()

    target = Path(args.target)
    work = Path(tempfile.mkdtemp(prefix=f"foldab-{args.model}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = work / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        model=args.model, fast=False, output_format="cif",
        recycling_steps=args.recycling, sampling_steps=args.steps,
        diffusion_samples=args.samples, seed=0, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=args.single_sequence, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None, msa_server_password=None,
        api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    _ensure_local_artifacts(cfg)

    if args.a3m:
        from tt_bio.main import _read_bio_chains
        chains = _read_bio_chains(target)
        assert len(chains) == 1
        seq = chains[0][1]
        text = Path(args.a3m).read_text()
        assert text.split("\n")[1] == seq, "a3m query row does not match the target sequence"
        h = hashlib.sha256(seq.encode()).hexdigest()[:16]
        (msa_dir / f"{h}.a3m").write_text(text)

    state = _WorkerState("tenstorrent")
    t0 = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t0
    state.bind_run("foldab", cfg)
    state.pfn = noop
    job_cfg = dict(cfg)

    real_cfg = T._l1_resident_linear_config

    def cfg_prod(x, w, dtype, full_k=True):
        """PROD arm: only the qkv projection (n == 3*k) may use the L1 config."""
        if not full_k or int(w.shape[-1]) != 3 * int(x.shape[-1]):
            return None
        return real_cfg(x, w, dtype, full_k)

    # Capture the coordinates of every fold for the whole-model bit-exactness check.
    # protenix/opendde expose model.fold -> (coords, conf); boltz2 exposes
    # model.predict_step -> dict with "coords".
    fold_coords = []
    current_arm = {"name": "PROD"}  # the cold fold is warmup under the PROD arm

    def _coords_of(out):
        if isinstance(out, tuple):
            return out[0]
        if isinstance(out, dict) and "coords" in out:
            return out["coords"]
        return out

    if hasattr(state.model, "fold"):
        orig_call = state.model.fold
    else:
        orig_call = state.model.predict_step

    def capture_fold(*a, **k):
        out = orig_call(*a, **k)
        fold_coords.append((current_arm["name"], _coords_of(out)))
        return out

    if hasattr(state.model, "fold"):
        state.model.fold = capture_fold
    else:
        state.model.predict_step = capture_fold

    def one_fold():
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        t = time.perf_counter()
        metrics, _best, _feats = state.predict_one(target, job_cfg)
        return time.perf_counter() - t, metrics

    print(f"model={args.model} target={target} load={load_s:.1f}s hw={arch_name()}", flush=True)
    T._l1_resident_linear_config = cfg_prod
    t_cold, m_cold = one_fold()
    print(f"cold fold (PROD arm) {t_cold:.2f}s metrics={m_cold}", flush=True)

    times = {"PROD": [], "GO": []}
    for rep in range(args.reps):
        for arm, fn in (("GO", real_cfg), ("PROD", cfg_prod)):
            T._l1_resident_linear_config = fn
            current_arm["name"] = arm
            t, m = one_fold()
            times[arm].append(round(t, 3))
            print(f"  {arm} rep{rep}: {t:.2f}s plddt={m.get('plddt')}", flush=True)
    T._l1_resident_linear_config = real_cfg
    current_arm["name"] = "GO"

    def _as_tensor(c):
        if isinstance(c, torch.Tensor):
            return c
        try:
            import numpy as np
            if isinstance(c, np.ndarray):
                return torch.from_numpy(c)
        except Exception:
            pass
        return None

    # Whole-model bit-exactness: every fold ran at seed 0, so PROD and GO coordinates
    # must be identical tensors.
    prods = [c for a, c in fold_coords if a == "PROD"]
    gos = [c for a, c in fold_coords if a == "GO"]
    bit_exact = None
    if prods and gos:
        a, b = _as_tensor(prods[0]), _as_tensor(gos[0])
        if a is not None and b is not None:
            bit_exact = bool(a.shape == b.shape and torch.equal(a, b))

    med = lambda v: sorted(v)[len(v) // 2]
    summary = {
        "model": args.model, "target": str(target), "load_s": round(load_s, 2),
        "cold_s": round(t_cold, 3), "times": times,
        "median": {a: med(v) for a, v in times.items()},
        "speedup": round(med(times["PROD"]) / med(times["GO"]), 4),
        "saved_s": round(med(times["PROD"]) - med(times["GO"]), 3),
        "whole_model_bit_exact": bit_exact,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
        print("wrote", args.out, flush=True)
    cleanup()


if __name__ == "__main__":
    main()
