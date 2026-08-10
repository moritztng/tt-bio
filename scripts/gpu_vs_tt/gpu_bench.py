#!/usr/bin/env python3
"""GPU leg of the GPU-vs-Tenstorrent Protenix-v2 / OpenDDE head-to-head.

Runs the UPSTREAM reference implementations (bytedance/Protenix, aurekaresearch
OpenDDE) on an NVIDIA GPU with the model loaded once in-process: one cold fold
(CUDA init / autotune / first-kernel compile, reported separately), then N warm
timed folds (min/median/max). Same target, same precomputed MSA, same config as
the TT leg (scripts/gpu_vs_tt/tt_baseline.py): 10 cycles / 200 diffusion steps /
1 sample / seed 0, over whichever of the two committed targets is selected --
prot117 (117 aa) or prot300 (CDK2, 298 aa), both with a 35-row alignment so that
token count is the only thing that changes between them.

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
FIXTURES = HERE / "fixtures"

CYCLES = 10
STEPS = 200
SAMPLES = 1
SEED = 0

# Optimization ladder, per model. Each rung sets EVERY performance knob
# explicitly: both vendors default fusion/cache/tf32 to ON for inference, so an
# unset flag is not a clean eager reference. "fp32+torch kernels, no fusion" is
# the eager reference point; the later rungs are the vendor's own published
# performance knobs. Kernel selector names differ between the two codebases
# (protenix: triattention/cuequivariance; opendde: cuequivariance/auto).
LADDERS = {
    "protenix-v2": [
        dict(name="L0-eager-fp32", dtype="fp32", triangle_attention="torch",
             triangle_multiplicative="torch", enable_efficient_fusion=False,
             enable_diffusion_shared_vars_cache=False, enable_tf32=False),
        dict(name="L1-bf16-vendor-kernels", dtype="bf16",
             triangle_attention="triattention", triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=False, enable_diffusion_shared_vars_cache=False,
             enable_tf32=False),
        dict(name="L2-bf16-fusion-cache", dtype="bf16",
             triangle_attention="triattention", triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=True, enable_diffusion_shared_vars_cache=True,
             enable_tf32=True),
        # Accuracy-isolation rungs (gpu-vs-tt-precision-fairness). At 298 aa the
        # L2 rung fails the CA Kabsch gate vs L0 (3.205 A). skip_amp keeps the
        # diffusion score model outside autocast at this size, so L2's diffusion
        # is fp32 already -- but with allow_tf32=True its matmuls run TF32. The
        # remaining L0->L2 deltas are TF32, the bf16 trunk, and the fused
        # kernels. These two rungs separate them:
        dict(name="L2-noTF32", dtype="bf16",
             triangle_attention="triattention", triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=True, enable_diffusion_shared_vars_cache=True,
             enable_tf32=False),
        dict(name="fp32-fused", dtype="fp32",
             triangle_attention="triattention", triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=True, enable_diffusion_shared_vars_cache=True,
             enable_tf32=True),
    ],
    "opendde": [
        dict(name="L0-eager-fp32", dtype="fp32", triatt_kernel="torch",
             trimul_kernel="torch", enable_fusion=False, enable_cache=False,
             enable_tf32=False),
        dict(name="L1-bf16-vendor-kernels", dtype="bf16",
             triatt_kernel="cuequivariance", trimul_kernel="cuequivariance",
             enable_fusion=False, enable_cache=False, enable_tf32=False),
        dict(name="L2-bf16-fusion-cache", dtype="bf16", triatt_kernel="auto",
             trimul_kernel="auto", enable_fusion=True, enable_cache=True,
             enable_tf32=True),
    ],
}


def _import_any(names: list[str]):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    raise ImportError(f"none of {names} importable")


def _build_configs(model_name: str, rung: dict, input_json: str, dump_dir: str,
                   checkpoint: str | None, samples: int = SAMPLES):
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
        "--model.N_cycle", str(CYCLES),
        "--sample_diffusion.N_step", str(STEPS),
        "--sample_diffusion.N_sample", str(samples),
        "--use_msa", "true", "--use_template", "false", "--use_rna_msa", "false",
        # The v2 checkpoint ships template_embedder weights; with templates off
        # the module is not built, so the sanctioned load is non-strict.
        "--load_strict", "false",
        "--dtype", rung["dtype"],
        "--triangle_attention", rung["triangle_attention"],
        "--triangle_multiplicative", rung["triangle_multiplicative"],
    ]
    if checkpoint:
        arg_str += ["--load_checkpoint_path", checkpoint]
        # protenix 2.0.0's runner loads load_checkpoint_dir/<model_name>.pt and
        # download_inference_cache 403s on the gated official URL if it is
        # absent there; point the dir at the local copy when the name matches.
        ck = Path(checkpoint)
        if ck.name == f"{model_name}.pt":
            arg_str += ["--load_checkpoint_dir", str(ck.parent)]
    for k in ("enable_efficient_fusion", "enable_diffusion_shared_vars_cache",
              "enable_tf32"):
        if k in rung:
            arg_str += [f"--{k}", str(rung[k]).lower()]

    # protenix 2.0.0 parse_configs does arg_str.split() -- it wants one string.
    arg_str = " ".join(arg_str)
    configs = {**cb.configs, **{"data": cd.data_configs}, **ci.inference_configs}
    configs = cc.parse_configs(configs=configs, arg_str=arg_str,
                               fill_required_with_null=True)
    base = {**cb.configs, **{"data": cd.data_configs}, **ci.inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
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
    # Vendors disagree on leading dims ([N,3] vs [N_sample,N,3]); flatten to
    # (N,3) and compare the first sample's atoms.
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
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


def _build_opendde_runner(rung: dict, input_json: str, dump_dir: str,
                          checkpoint: str | None, samples: int = SAMPLES):
    """OpenDDE's public runner factory (runner.batch_inference.get_default_runner)
    with the vendor's own knob names. Returns (runner, configs, inf_module,
    dataloader_module, seed_module)."""
    bi = importlib.import_module("runner.batch_inference")
    inf = importlib.import_module("runner.inference")
    dl = importlib.import_module("opendde.data.inference.infer_dataloader")
    sd = importlib.import_module("opendde.utils.seed")
    runner = bi.get_default_runner(
        seeds=[SEED], dump_dir=dump_dir, n_cycle=CYCLES, n_step=STEPS,
        n_sample=samples, dtype=rung["dtype"], model_name="opendde_v1",
        load_checkpoint_path=checkpoint or "", use_msa=True,
        trimul_kernel=rung["trimul_kernel"], triatt_kernel=rung["triatt_kernel"],
        enable_cache=rung["enable_cache"], enable_fusion=rung["enable_fusion"],
        enable_tf32=rung["enable_tf32"], deterministic=False, use_template=False,
        use_rna_msa=False, need_atom_confidence=True,
    )
    configs = runner.configs
    configs.input_json_path = input_json
    return runner, configs, inf, dl, sd


def build_fold(model: str, model_name: str, rung: dict, input_json: str,
               dump_dir: Path, checkpoint: str | None, n_msa_expected: int,
               samples: int = SAMPLES):
    """Load one ladder rung and return ``(one_fold, meta, runner)``.

    Split out of ``run_model`` so the concurrency launcher (``gpu_concurrency.py``)
    folds through exactly the same code path as the latency benchmark: an aggregate
    throughput number is only comparable to the committed per-fold latency if the fold
    itself is identical.
    """
    torch = importlib.import_module("torch")
    dump_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    if model == "opendde":
        runner, configs, inf, dl, sd = _build_opendde_runner(
            rung, input_json, str(dump_dir), checkpoint, samples=samples)
    else:
        inf = _import_any(["runner.inference", "protenix.runner.inference"])
        dl = _import_any(["protenix.data.inference.infer_dataloader"])
        sd = _import_any(["protenix.utils.seed"])
        configs = _build_configs(model_name, rung, input_json, str(dump_dir),
                                 checkpoint, samples=samples)
        if hasattr(inf, "update_gpu_compatible_configs"):
            configs = inf.update_gpu_compatible_configs(configs)
        inf.download_inference_cache(configs)
        runner = inf.InferenceRunner(configs)
    load_s = time.perf_counter() - t_load

    dataloader = dl.get_inference_dataloader(configs=configs)
    data, atom_array, err = next(iter(dataloader))[0]
    assert not err, f"featurization failed: {err}"
    new_configs = inf.update_inference_configs(configs, data["N_token"].item())
    runner.update_model_configs(new_configs)
    n_msa = int(data["N_msa"].item())
    n_token = int(data["N_token"].item())
    # Fairness is only real if both sides consume the SAME alignment rows. The
    # TT side reads all 35; if this side cropped or padded to something else the
    # head-to-head is invalid, so fail loudly rather than publish the number.
    assert n_msa == n_msa_expected, \
        f"GPU consumed {n_msa} MSA rows, TT side uses {n_msa_expected}"

    def one_fold():
        sd.seed_everything(seed=SEED, deterministic=False)
        t0 = time.perf_counter()
        # protenix's forward deletes the MSA keys from input_feature_dict
        # in place (protenix.py:524); hand each fold a fresh shallow copy.
        pred = runner.predict(
            {**data, "input_feature_dict": dict(data["input_feature_dict"])})
        torch.cuda.synchronize()
        return time.perf_counter() - t0, pred

    return one_fold, dict(load_s=round(load_s, 2), n_msa=n_msa, n_token=n_token,
                          diffusion_samples=samples), runner


def run_model(model: str, model_name: str, repeat: int, input_json: str,
              dump_root: Path, checkpoint: str | None, out_path: Path,
              rungs: list[dict], label: str, n_msa_expected: int) -> dict:
    torch = importlib.import_module("torch")

    def _write(path: Path, results: list) -> None:
        cpu = "unknown"
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
        summary = dict(
            model=model, side="gpu", model_name=model_name,
            gpu=torch.cuda.get_device_name(0), host_cpu=cpu,
            cpu_count=os.cpu_count(),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            input=f"{label}, precomputed MSA (same a3m as TT side)",
            recycling_steps=CYCLES, sampling_steps=STEPS, diffusion_samples=SAMPLES,
            seed=SEED, rungs=results, date=time.strftime("%Y-%m-%d"),
        )
        path.write_text(json.dumps(summary, indent=2) + "\n")

    results = []
    ref_coords = None
    for rung in rungs:
        one_fold, meta, runner = build_fold(
            model, model_name, rung, input_json, dump_root / rung["name"],
            checkpoint, n_msa_expected)
        load_s, n_msa, n_token = meta["load_s"], meta["n_msa"], meta["n_token"]

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
            rung=rung["name"], rung_config=rung, load_s=load_s,
            cold_s=round(cold_s, 3), warm_times_s=[round(t, 3) for t in times],
            warm_min_s=round(ts[0], 3), warm_median_s=round(ts[len(ts) // 2], 3),
            warm_max_s=round(ts[-1], 3), n_msa=n_msa, n_token=n_token,
            ca_kabsch_rmsd_vs_L0=rmsd,
        ))
        print(f"[{model} {rung['name']}] warm median {ts[len(ts)//2]:.2f}s "
              f"(min {ts[0]:.2f}/max {ts[-1]:.2f}) cold {cold_s:.1f}s "
              f"rmsd_vs_L0={rmsd}", file=sys.stderr, flush=True)
        # Persist after EVERY rung: on a metered rental a crash in a later rung
        # must not take the earlier rungs' results with it.
        _write(out_path, results)
        del runner
        torch.cuda.empty_cache()

    # Host CPU matters: at these token counts a good share of the wall clock is
    # kernel-launch dispatch, so a scaling ratio taken across two different rented
    # hosts is not clean. Recorded (inside _write) so a ratio can be shown to be
    # a within-host measurement (the 117-aa run of 2026-08-06 did not record it).
    _write(out_path, results)
    return json.loads(out_path.read_text())


def write_input_json(path: Path, msa_a3m: Path, seq: str, name: str) -> None:
    job = {
        "name": name,
        "modelSeeds": [SEED],
        "sequences": [{
            "proteinChain": {
                "sequence": seq,
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
    ap.add_argument("--msa-a3m", type=Path, default=FIXTURES / "prot117.a3m")
    ap.add_argument("--seq-file", type=Path, default=FIXTURES / "prot117.seq",
                    help="one-line target sequence; must match the a3m's query row")
    ap.add_argument("--label", default="prot.yaml sequence (117 aa)")
    ap.add_argument("--name", default="prot117", help="job name in the input JSON")
    ap.add_argument("--checkpoint", default=None,
                    help="local checkpoint path (skips the gated official download)")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated subset of rung names (default: all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model_name = {"protenix-v2": "protenix-v2", "opendde": "opendde_v1"}[args.model]
    work = Path(tempfile.mkdtemp(prefix=f"gpubench-{args.model}-")) \
        if (tmp := os.environ.get("TMPDIR")) else Path(f"/tmp/gpubench-{args.model}")
    work.mkdir(parents=True, exist_ok=True)
    input_json = work / "input.json"
    seq = args.seq_file.read_text().strip()
    a3m_rows = args.msa_a3m.read_text().split("\n")
    # Same identical-bytes check the TT harness makes, from the other side.
    assert a3m_rows[1] == seq, f"{args.msa_a3m} query row does not match {args.seq_file}"
    n_msa_expected = args.msa_a3m.read_text().count(">")
    print(f"target {args.label}: {len(seq)} aa, {n_msa_expected} MSA rows", file=sys.stderr)
    write_input_json(input_json, args.msa_a3m.resolve(), seq, args.name)

    ladder = LADDERS[args.model]
    selected = set(args.rungs.split(",")) if args.rungs else None
    rungs = [r for r in ladder if selected is None or r["name"] in selected]
    if not rungs:
        ap.error("no rungs selected")
    run_model(args.model, model_name, args.repeat, str(input_json), work / "dump",
              args.checkpoint, args.out, rungs, args.label, n_msa_expected)
    return 0


if __name__ == "__main__":
    import tempfile
    sys.exit(main())
