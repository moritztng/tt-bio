#!/usr/bin/env python3
"""GPU leg of the GPU-vs-Tenstorrent Protenix-v2 / OpenDDE head-to-head.

Runs the UPSTREAM reference implementations (bytedance/Protenix, aurekaresearch
OpenDDE) on an NVIDIA GPU with the model loaded once in-process: one cold fold
(CUDA init / autotune / first-kernel compile, reported separately), then N warm
timed folds (min/median/max). Same target, same precomputed MSA, same config as
the TT leg (scripts/gpu_vs_tt/tt_baseline.py): prot.yaml's 117-aa sequence,
10 cycles / 200 diffusion steps / 1 sample / seed 0.

The optimization ladder is vendor-sanctioned only: dtype (fp32 -> bf16), the
upstream triangle-kernel selectors (triattention / cuequivariance), Protenix's
built-in fusion/cache flags, TF32. Each rung is correctness-checked against the
fp32-eager rung's coordinates (CA Kabsch RMSD) -- a fast but wrong rung is
reported as INVALID, not as a speedup.

Usage (on the GPU box, after gpu_setup.sh):

    python3 gpu_bench.py --model protenix-v2 --repeat 3 --out protenix_gpu.json
    python3 gpu_bench.py --model opendde --repeat 3 --out opendde_gpu.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISA"
       "IVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")  # examples/prot.yaml, 117 aa

CYCLES = 10
STEPS = 200
SAMPLES = 1
SEED = 0

# Optimization ladder. Each rung overrides CLI-style config keys of the
# upstream runner. "fp32+torch kernels" is the eager reference point; the later
# rungs are the vendor's own published performance knobs.
LADDER = [
    dict(name="L0-eager-fp32", dtype="fp32", triangle_attention="torch",
         triangle_multiplicative="torch"),
    dict(name="L1-bf16-vendor-kernels", dtype="bf16",
         triangle_attention="triattention", triangle_multiplicative="cuequivariance"),
    dict(name="L2-bf16-fusion-cache", dtype="bf16",
         triangle_attention="triattention", triangle_multiplicative="cuequivariance",
         enable_efficient_fusion=True, enable_diffusion_shared_vars_cache=True,
         enable_tf32=True),
]


def _import_any(names: list[str]):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    raise ImportError(f"none of {names} importable")


def _build_configs(model_name: str, rung: dict, input_json: str, dump_dir: str,
                   checkpoint: str | None):
    """Replicate runner/inference.py::run()'s 3-pass config construction."""
    cb = _import_any(["configs.configs_base", "protenix.configs.configs_base"])
    cd = _import_any(["configs.configs_data", "protenix.configs.configs_data"])
    ci = _import_any(["configs.configs_inference", "protenix.configs.configs_inference"])
    cm = _import_any(["configs.configs_model_type", "protenix.configs.configs_model_type"])
    cc = _import_any(["protenix.config.config", "opendde.config.config"])
    from typing import Mapping

    arg_str = [
        "--model_name", model_name,
        "--input_json_path", input_json,
        "--dump_dir", dump_dir,
        "--seeds", str(SEED),
        "--use_default_params", "false",
        "--model.N_cycle", str(CYCLES),
        "--sample_diffusion.N_step", str(STEPS),
        "--sample_diffusion.N_sample", str(SAMPLES),
        "--use_msa", "true", "--use_template", "false", "--use_rna_msa", "false",
        "--dtype", rung["dtype"],
        "--triangle_attention", rung["triangle_attention"],
        "--triangle_multiplicative", rung["triangle_multiplicative"],
    ]
    if checkpoint:
        arg_str += ["--load_checkpoint_path", checkpoint]
    for k in ("enable_efficient_fusion", "enable_diffusion_shared_vars_cache",
              "enable_tf32"):
        if k in rung:
            arg_str += [f"--{k}", str(rung[k]).lower()]

    configs = {**cb.configs, **{"data": cd.data_configs}, **ci.inference_configs}
    configs = cc.parse_configs(configs=configs, arg_str=arg_str,
                               fill_required_with_null=True)
    base = {**cb.configs, **{"data": cd.data_configs}, **ci.inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d, v)
            else:
                d[k] = v
        return d

    deep_update(base, cm.model_configs[configs.model_name])
    configs = cc.parse_configs(configs=base, arg_str=arg_str,
                               fill_required_with_null=True)
    return configs


def _ca_kabsch_rmsd(c1, c2, ca_idx1=None, ca_idx2=None) -> float:
    import numpy as np
    a = np.asarray(c1, dtype=np.float64)
    b = np.asarray(c2, dtype=np.float64)
    if ca_idx1 is not None:
        a = a[ca_idx1]
        b = b[ca_idx2]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - a.mean(0)
    b = b - b.mean(0)
    H = a.T @ b
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    rot = Vt.T @ D @ U.T
    return float(np.sqrt(((a @ rot.T - b) ** 2).sum(-1).mean()))


def run_model(model: str, model_name: str, repeat: int, input_json: str,
              dump_root: Path, checkpoint: str | None, out_path: Path,
              rungs: list[dict]) -> dict:
    torch = importlib.import_module("torch")
    inf = _import_any(["runner.inference", "protenix.runner.inference",
                       "opendde.runner.inference"])
    dl = _import_any(["protenix.data.inference.infer_dataloader",
                      "opendde.data.inference.infer_dataloader"])
    sd = _import_any(["protenix.utils.seed", "opendde.utils.seed"])

    results = []
    ref_coords = None
    for rung in rungs:
        rung_dir = dump_root / rung["name"]
        rung_dir.mkdir(parents=True, exist_ok=True)
        configs = _build_configs(model_name, rung, input_json, str(rung_dir),
                                 checkpoint)
        if hasattr(inf, "update_gpu_compatible_configs"):
            configs = inf.update_gpu_compatible_configs(configs)
        inf.download_inference_cache(configs)

        t_load = time.perf_counter()
        runner = inf.InferenceRunner(configs)
        load_s = time.perf_counter() - t_load

        dataloader = dl.get_inference_dataloader(configs=configs)
        data, atom_array, err = next(iter(dataloader))[0]
        assert not err, f"featurization failed: {err}"
        new_configs = inf.update_inference_configs(configs, data["N_token"].item())
        runner.update_model_configs(new_configs)
        n_msa = int(data["N_msa"].item())

        def one_fold():
            sd.seed_everything(seed=SEED, deterministic=False)
            t0 = time.perf_counter()
            pred = runner.predict(data)
            torch.cuda.synchronize()
            return time.perf_counter() - t0, pred

        cold_s, pred = one_fold()
        times = []
        for _ in range(repeat):
            t, pred = one_fold()
            times.append(t)

        coords = pred["coordinate"]
        if hasattr(coords, "cpu"):
            coords = coords.cpu().numpy()
        rmsd = None
        if ref_coords is None:
            ref_coords = coords
        else:
            try:
                rmsd = _ca_kabsch_rmsd(ref_coords, coords)
            except Exception as e:
                rmsd = f"rmsd-error: {e}"

        ts = sorted(times)
        results.append(dict(
            rung=rung["name"], rung_config=rung, load_s=round(load_s, 2),
            cold_s=round(cold_s, 3), warm_times_s=[round(t, 3) for t in times],
            warm_min_s=round(ts[0], 3), warm_median_s=round(ts[len(ts) // 2], 3),
            warm_max_s=round(ts[-1], 3), n_msa=n_msa,
            ca_kabsch_rmsd_vs_L0=rmsd,
        ))
        print(f"[{model} {rung['name']}] warm median {ts[len(ts)//2]:.2f}s "
              f"(min {ts[0]:.2f}/max {ts[-1]:.2f}) cold {cold_s:.1f}s "
              f"rmsd_vs_L0={rmsd}", file=sys.stderr, flush=True)
        del runner
        torch.cuda.empty_cache()

    summary = dict(
        model=model, side="gpu", model_name=model_name,
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        input="prot.yaml sequence (117 aa), precomputed MSA (same a3m as TT side)",
        recycling_steps=CYCLES, sampling_steps=STEPS, diffusion_samples=SAMPLES,
        seed=SEED, rungs=results, date=time.strftime("%Y-%m-%d"),
    )
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_input_json(path: Path, msa_a3m: Path) -> None:
    job = {
        "name": "prot117",
        "modelSeeds": [SEED],
        "sequences": [{
            "proteinChain": {
                "sequence": SEQ,
                "count": 1,
                "msa": {
                    "precomputed_msa_dir": str(msa_a3m.parent),
                    "pairing_db": "uniref100",
                },
                "unpairedMsaPath": str(msa_a3m),
                "pairedMsaPath": "",
                "templatesPath": "",
            }
        }],
    }
    path.write_text(json.dumps([job], indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--msa-a3m", type=Path, default=HERE / "fixtures" / "prot117.a3m")
    ap.add_argument("--checkpoint", default=None,
                    help="local checkpoint path (skips the gated official download)")
    ap.add_argument("--rungs", default=",".join(r["name"] for r in LADDER),
                    help="comma-separated subset of rung names")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model_name = {"protenix-v2": "protenix-v2", "opendde": "opendde_v1"}[args.model]
    work = Path(tempfile.mkdtemp(prefix=f"gpubench-{args.model}-")) \
        if (tmp := os.environ.get("TMPDIR")) else Path(f"/tmp/gpubench-{args.model}")
    work.mkdir(parents=True, exist_ok=True)
    input_json = work / "input.json"
    write_input_json(input_json, args.msa_a3m.resolve())

    wanted = {r["name"] for r in LADDER if r["name"] in set(args.rungs.split(","))}
    rungs = [r for r in LADDER if r["name"] in wanted]
    if not rungs:
        ap.error("no rungs selected")
    run_model(args.model, model_name, args.repeat, str(input_json), work / "dump",
              args.checkpoint, args.out, rungs)
    return 0


if __name__ == "__main__":
    import tempfile
    sys.exit(main())
