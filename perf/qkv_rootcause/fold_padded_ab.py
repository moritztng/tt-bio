#!/usr/bin/env python3
"""Fold-level A/B of the padded-tile L1-residency fix, interleaved in one process.

  BASE  what main effectively runs at a real target size: the guard's logical-shape
        tile test rejects every trunk projection (verified: 768 calls, 0 admitted at
        298 aa), so BASE forces both entry points to their fallback.
  GO    this branch: tiles derived from the padded shape, ragged row blocks allowed.

Whole-model bit-exactness is the coordinates the model returns, compared with
torch.equal at the same seed. The cold fold runs BASE, so compile time stays out of
the A/B.

    TT_VISIBLE_DEVICES=0 python3 perf/qkv_rootcause/fold_padded_ab.py \\
        --model protenix-v2 --target examples/prot300.yaml \\
        --a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --reps 3
"""
import argparse, hashlib, json, statistics, sys, tempfile, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--a3m", default=None)
ap.add_argument("--recycling", type=int, default=10)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--single-sequence", action="store_true")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default=None)
args = ap.parse_args()


def main():
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
    import tt_bio.tenstorrent as T
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    noop = lambda *a, **k: None
    _E.set_progress(noop)
    get_device()

    target = Path(args.target)
    work = Path(tempfile.mkdtemp(prefix="foldpad-"))
    struct_dir, msa_dir = work / "out", work / "msa"
    struct_dir.mkdir(parents=True, exist_ok=True)
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
        for ch in _read_bio_chains(target):
            seq, text = ch[1], Path(args.a3m).read_text()
            if text.split("\n")[1] == seq:
                (msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(text)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("foldpad", cfg)
    state.pfn = noop

    real_cfg, real_rb = T._l1_resident_linear_config, T._tri_att_row_block
    ARMS = {"BASE": (lambda *a, **k: None, lambda *a, **k: 0), "GO": (real_cfg, real_rb)}

    coords = []
    arm = {"name": "BASE"}
    orig = state.model.fold if hasattr(state.model, "fold") else state.model.predict_step

    def capture(*a, **k):
        out = orig(*a, **k)
        c = out[0] if isinstance(out, tuple) else (out["coords"] if isinstance(out, dict) else out)
        coords.append((arm["name"], c))
        return out

    if hasattr(state.model, "fold"):
        state.model.fold = capture
    else:
        state.model.predict_step = capture

    def one_fold():
        for p in struct_dir.glob("*"):
            p.unlink()
        t = time.perf_counter()
        m, _b, _f = state.predict_one(target, dict(cfg))
        return time.perf_counter() - t, m

    def set_arm(name):
        arm["name"] = name
        T._l1_resident_linear_config, T._tri_att_row_block = ARMS[name]

    print(f"model={args.model} target={target.name} hw={arch_name()}", flush=True)
    set_arm("BASE")
    t_cold, m_cold = one_fold()
    print(f"cold fold (BASE) {t_cold:.2f}s plddt={m_cold.get('plddt')}", flush=True)

    times = {"BASE": [], "GO": []}
    for rep in range(args.reps):
        for name in ("GO", "BASE"):
            set_arm(name)
            t, m = one_fold()
            times[name].append(round(t, 3))
            print(f"  {name} rep{rep}: {t:.2f}s plddt={m.get('plddt')}", flush=True)
    set_arm("GO")

    def tens(c):
        if isinstance(c, torch.Tensor):
            return c
        try:
            import numpy as np
            if isinstance(c, np.ndarray):
                return torch.from_numpy(c)
        except Exception:
            pass
        return None

    b = [tens(c) for a, c in coords if a == "BASE"]
    g = [tens(c) for a, c in coords if a == "GO"]
    bit_exact = None
    if b and g and b[0] is not None and g[0] is not None:
        bit_exact = bool(torch.equal(b[0], g[0]))
        if not bit_exact:
            d = (b[0].float() - g[0].float()).abs()
            print(f"  NOT bit-exact: max|d|={d.max():.6g} rmsd={d.pow(2).mean().sqrt():.6g}",
                  flush=True)

    med = {k: statistics.median(v) for k, v in times.items()}
    ratio = round(med["BASE"] / med["GO"], 4)
    print(f"\nBASE {med['BASE']:.3f}s  GO {med['GO']:.3f}s  ratio {ratio:.4f}x  "
          f"whole_model_bit_exact={bit_exact}", flush=True)
    out = {"model": args.model, "target": target.name, "arch": arch_name(),
           "times": times, "median": med, "ratio": ratio,
           "whole_model_bit_exact": bit_exact}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
