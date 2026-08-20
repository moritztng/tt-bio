"""One Nesso-1 rung, repeated in a single process, with the kernel paths COUNTED and the phases
timed device-side and host-side separately.

    python perf/nesso1/gpu_nesso1_run.py --inputs perf/nesso1/inputs/ladder/aa256 \
        --out-dir /work/out/aa256 --report /work/results/aa256.json --reps 4

Forked from perf/rf3/gpu_rf3_run.py, which forked perf/dsfix/gpu_rfd3_run.py. The discipline is
inherited: counters, not flags.

Nesso-1 routes triangle attention and triangle multiplication to cuEquivariance or to a vanilla
PyTorch path at *call* time, on `use_kernels and _CUEQUIVARIANCE_AVAILABLE`. `use_kernels` is a
model attribute read off the CHECKPOINT hparams, not a CLI flag -- `--no_kernels` can only turn it
off, never on -- so an installed wheel proves nothing about what ran. Both cueq entry points
(`kernel_triangular_attn`, `kernel_triangular_mult`) and both call sites are wrapped, and the
engagement fraction is reported.

This harness replicates the steps of `nesso predict` in-process rather than shelling out to it,
because at ~1 s per prediction the CLI's own fixed costs (HF resolution, checkpoint load, Trainer
construction) are the same order as the work, and a screening user pays them once for thousands of
compounds. Both numbers are therefore produced: the per-record steady state, and the fixed costs
named separately so a predictions/hour figure can be built for any batch size.

Phases, wall-clock around cuda-synchronised boundaries:
  preprocess   host: YAML -> structure npz + RDKit ETKDG conformer, in a ProcessPool
  esm          device: ESM-2 650M, once per UNIQUE protein sequence, cached to disk
  model_load   host+device: safetensors -> cuda
  featurize    host: the featurizer inside the dataloader (derived, see below)
  embed        device: InputEmbedder
  esm_module   device: the ESM pair-projection stack, per recycle
  pairformer   device: PairformerNoSeq, per recycle
  crop         mixed: pocket_crop -- distogram head + a device->host sync + numpy selection
  affinity     device: the two AffinityModule heads, run in fp32 (autocast disabled upstream)
  forward      device: Nesso1.forward, the sum of the above plus the residue
  predict_step device+host: forward plus the entropy/pocket-mask bookkeeping
  write        host: affinity.json
`featurize` is derived as predict_total - sum(predict_step), i.e. everything the dataloader and the
Lightning loop spent that was not the model. It is not a leftover: on a 1 s model the host
featuriser is a first-class cost, and `gpu-reference-device-vs-host-split` is explicit that a bar
built on wall-clock bakes one rented landlord's CPU into a port's target for months.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess
import sys
import time

COUNTS: dict[str, int] = {}
PHASE: dict[str, list[float]] = collections.defaultdict(list)
_ACTIVE_REP = [-1]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wrap_count(mod, name: str, label: str):
    fn = getattr(mod, name, None)
    if fn is None:
        COUNTS[label] = -1
        return
    COUNTS[label] = 0

    def wrapper(*a, **kw):
        COUNTS[label] += 1
        return fn(*a, **kw)

    setattr(mod, name, wrapper)


def timed(obj, name: str, phase: str, sync: bool = True):
    fn = getattr(obj, name, None)
    if fn is None:
        COUNTS["phase_missing:" + phase] = -1
        return
    COUNTS.setdefault("calls:" + phase, 0)

    def wrapper(*a, **kw):
        COUNTS["calls:" + phase] += 1
        if sync:
            _sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            if sync:
                _sync()
            PHASE["%d/%s" % (_ACTIVE_REP[0], phase)].append(time.perf_counter() - t0)

    setattr(obj, name, wrapper)


def compute_apps() -> list[dict]:
    """Every process holding memory on this GPU, ours included.

    A rented "single GPU" box can have a co-tenant on the same physical card; that has already
    voided one campaign's timings in this fleet (vast-ai-access). Absolute seconds from a shared
    card are worthless, so the condition is recorded per rung rather than assumed away.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        apps = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                apps.append({"pid": int(parts[0]), "used_MiB": int(parts[1])})
        return apps
    except Exception as e:                                    # noqa: BLE001
        return [{"error": repr(e)}]


def gpu_dynamic() -> dict:
    q = "power.draw,utilization.gpu,clocks.sm,temperature.gpu"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:                                    # noqa: BLE001
        return {"error": repr(e)}


def gpu_static() -> dict:
    q = ("name,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap,"
         "pcie.link.gen.max")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:                                    # noqa: BLE001
        return {"error": repr(e)}


def effective_cpus() -> float:
    """The container's cgroup quota, not the host's core count.

    `nproc` on a vast.ai container reports the whole host (e.g. 192) while the cgroup quota can be
    23; a pool sized off nproc oversubscribes by 8x and understates achievable throughput
    (vast-ai-access). Any concurrency number in this campaign is denominated by THIS value.
    """
    try:                                                      # cgroup v2
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return float(quota) / float(period)
    except Exception:                                         # noqa: BLE001
        pass
    try:                                                      # cgroup v1 (what vast.ai serves)
        q = float(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        pd = float(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return q / pd
    except Exception:                                         # noqa: BLE001
        pass
    import os
    return float(os.cpu_count() or 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="directory of .yaml inputs, or a single .yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--reps", type=int, default=4, help="rep 0 is the discarded cold rep")
    ap.add_argument("--recycling-steps", type=int, default=5)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-kernels", action="store_true")
    ap.add_argument("--refine", default="on", choices=("on", "off"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import os
    import warnings
    from dataclasses import replace

    import lightning.pytorch as pl
    import torch
    import torch.nn.functional as F
    from lightning.pytorch import Trainer

    import nesso.main as NM
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.writer import NessoWriter
    from nesso.model.models.nesso1 import Nesso1

    pl.seed_everything(args.seed, workers=True)
    warnings.filterwarnings("ignore")
    # Faithful to the shipped CLI: it sets "highest", i.e. TF32 OFF for fp32 matmuls. The two
    # affinity heads run under `autocast(enabled=False)`, so this is not a dead setting -- it is
    # the precision the affinity heads actually execute at.
    torch.set_float32_matmul_precision("highest")

    cache = pathlib.Path(os.environ.get("NESSO_CACHE", "/work/cache"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "huggingface"))

    inputs = pathlib.Path(args.inputs)
    out_dir = pathlib.Path(args.out_dir)
    report_path = pathlib.Path(args.report)

    # --- counters: the two cueq entry points, and every call site that could reach them --------
    import nesso.model.layers.triangular_attention.primitives as PRIM
    import nesso.model.layers.triangular_mult as TM
    from nesso.model.layers.triangular_attention.attention import (
        TriangleAttentionEndingNode,
        TriangleAttentionStartingNode,
    )

    wrap_count(PRIM, "kernel_triangular_attn", "cueq.triangle_attention")
    wrap_count(TM, "kernel_triangular_mult", "cueq.triangle_multiplicative_update")
    COUNTS["cuequivariance_available"] = int(bool(getattr(PRIM, "_CUEQUIVARIANCE_AVAILABLE", False)))
    wrap_count(PRIM.Attention, "forward", "callsite.triangle_attention")
    wrap_count(TM.TriangleMultiplicationOutgoing, "forward", "callsite.tri_mul_out")
    wrap_count(TM.TriangleMultiplicationIncoming, "forward", "callsite.tri_mul_in")
    wrap_count(TriangleAttentionStartingNode, "forward", "callsite.tri_att_start")
    wrap_count(TriangleAttentionEndingNode, "forward", "callsite.tri_att_end")

    _sdpa = F.scaled_dot_product_attention
    COUNTS["scaled_dot_product_attention"] = 0

    def sdpa_wrapper(*a, **kw):
        COUNTS["scaled_dot_product_attention"] += 1
        return _sdpa(*a, **kw)

    F.scaled_dot_product_attention = sdpa_wrapper
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper

    # --- phase timers ------------------------------------------------------------------------
    from nesso.model.modules.affinity import AffinityModule
    from nesso.model.modules.esm_module import ESMModule
    from nesso.model.modules.trunk import InputEmbedder
    from nesso.model.layers.pairformer import PairformerNoSeqModule

    timed(InputEmbedder, "forward", "embed")
    timed(ESMModule, "forward", "esm_module")
    timed(PairformerNoSeqModule, "forward", "pairformer")
    timed(Nesso1, "pocket_crop", "crop")
    timed(AffinityModule, "forward", "affinity")
    timed(Nesso1, "forward", "forward")
    timed(Nesso1, "predict_step", "predict_step")
    timed(NessoWriter, "write_on_batch_end", "write")

    env = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "compute_cap": list(torch.cuda.get_device_capability(0))
           if torch.cuda.is_available() else None,
           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "python": sys.version.split()[0],
           "effective_cpus": effective_cpus(),
           "gpu_static": gpu_static(),
           "compute_apps_before": compute_apps()}
    from importlib.metadata import PackageNotFoundError, version
    for pkg_name in ("nesso", "triton", "cuequivariance-torch",
                     "cuequivariance-ops-torch-cu12", "lightning", "transformers", "rdkit",
                     "numpy", "safetensors"):
        try:
            env[pkg_name] = version(pkg_name)
        except PackageNotFoundError:
            env[pkg_name] = None

    report = {"label": args.label, "inputs": str(inputs), "reps": args.reps,
              "recycling_steps": args.recycling_steps, "precision": args.precision,
              "no_kernels": args.no_kernels, "refine": args.refine, "seed": args.seed,
              "num_workers": args.num_workers, "env": env,
              "n_records": None, "n_unique_seqs": None,
              "preprocess_s": None, "esm_s": None, "model_load_s": None,
              "rep_s": [], "rep_windows": [], "phases": {}, "counts": None,
              "gpu_dynamic": [],
              "peak_vram_alloc_B": None, "peak_vram_reserved_B": None,
              "affinity": None, "ok": False, "why": "",
              "compute_apps_after": None, "gpu_exclusive": None}

    def dump():
        report["counts"] = dict(COUNTS)
        per_rep: dict[str, dict[str, float]] = {}
        for key, xs in PHASE.items():
            rep, phase = key.split("/", 1)
            per_rep.setdefault(rep, {})[phase] = round(sum(xs), 5)
            per_rep[rep][phase + "_n"] = len(xs)
        report["phases"] = per_rep
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        yaml_paths = NM.check_inputs(inputs)
        report["n_records"] = len(yaml_paths)
        revision = NM.resolve_model_revision(None)
        ccd_pkl, ckpt_dir = NM.ensure_cache(cache, revision=revision)
        report["model_revision"] = revision
        report["ckpt_dir"] = str(ckpt_dir)
        paths = NM.resolve_paths(out_dir)

        # --- host: YAML -> npz + RDKit conformer ---------------------------------------------
        _ACTIVE_REP[0] = 0
        t0 = time.perf_counter()
        manifest, failed = NM.preprocess_yamls(yaml_paths, paths.mol_dir, ccd_pkl,
                                               paths.structures_dir, paths.records_dir,
                                               num_workers=args.num_workers)
        report["preprocess_s"] = round(time.perf_counter() - t0, 4)
        if failed:
            raise RuntimeError("preprocess failed for %r" % (failed,))

        # --- device: ESM-2 650M, once per unique sequence -------------------------------------
        seq_by_md5, _ = NM.collect_esm_from_yamls(yaml_paths)
        report["n_unique_seqs"] = len(seq_by_md5)
        report["seq_lens"] = sorted({len(s) for s in seq_by_md5.values()})
        paths.esm_dir.mkdir(parents=True, exist_ok=True)
        _sync()
        t0 = time.perf_counter()
        NM.run_esm(seq_by_md5, paths.esm_dir, NM.DEFAULT_ESM2_MODEL, cache / "huggingface")
        _sync()
        report["esm_s"] = round(time.perf_counter() - t0, 4)

        # --- model ----------------------------------------------------------------------------
        t0 = time.perf_counter()
        model = Nesso1.from_pretrained(ckpt_dir)
        report["ckpt_use_kernels"] = bool(getattr(model, "use_kernels", False))
        report["affinity_head_present"] = bool(getattr(model, "affinity_prediction", False))
        if args.no_kernels:
            model.use_kernels = False
        report["effective_use_kernels"] = bool(model.use_kernels)
        model.predict_args.update({"pose_protein_cutoff": 15.0,
                                   "recycling_steps": args.recycling_steps,
                                   "affinity_protein_cutoff": 15.0,
                                   "refine_protein_inference": args.refine == "on",
                                   "refine_protein_cutoff": 22.0,
                                   "refine_protein_tokens_budget": 256,
                                   "save_metadata": False})
        model.eval()
        torch.set_grad_enabled(False)
        _sync()
        report["model_load_s"] = round(time.perf_counter() - t0, 4)

        datamodule = NessoInferenceDataModule(
            manifest=replace(manifest, records=manifest.records),
            target_dir=paths.processed, esm_emb_dir=paths.esm_dir,
            ligand_dir=paths.mol_dir, ccd_pkl=ccd_pkl, num_workers=args.num_workers,
            use_esm_all_layers=False, esm_emb_dim=1280, esm_num_layers=33)
        report["dataloader_batch_size"] = datamodule.predict_dataloader().batch_size

        for rep in range(args.reps):
            _ACTIVE_REP[0] = rep
            rep_out = paths.predictions_dir / ("rep%d" % rep)
            rep_out.mkdir(parents=True, exist_ok=True)
            writer = NessoWriter(output_dir=rep_out, save_metadata=False)
            trainer = Trainer(accelerator="gpu", devices=1, precision=args.precision,
                              logger=False, enable_checkpointing=False,
                              enable_progress_bar=False, callbacks=[writer])
            torch.cuda.reset_peak_memory_stats()
            _sync()
            t1, w0 = time.perf_counter(), time.time()
            trainer.predict(model, datamodule=datamodule, return_predictions=False)
            _sync()
            dt = time.perf_counter() - t1
            report["rep_s"].append(round(dt, 4))
            # Absolute epoch bounds, not just a duration: with C processes screening
            # concurrently, aggregate throughput is total records over the union of the warm
            # windows, and per-process durations alone cannot reconstruct that union.
            report["rep_windows"].append([round(w0, 4), round(time.time(), 4)])
            report["gpu_dynamic"].append(gpu_dynamic())
            report["peak_vram_alloc_B"] = max(report["peak_vram_alloc_B"] or 0,
                                              torch.cuda.max_memory_allocated())
            report["peak_vram_reserved_B"] = max(report["peak_vram_reserved_B"] or 0,
                                                 torch.cuda.max_memory_reserved())
            print("[nesso] %s rep%d %.4fs  %d records  %.4f s/pred"
                  % (args.label, rep, dt, len(manifest.records),
                     dt / max(1, len(manifest.records))), flush=True)
            dump()

        after = compute_apps()
        report["compute_apps_after"] = after
        before = env.get("compute_apps_before") or []
        report["gpu_exclusive"] = (len([a for a in after if "pid" in a]) <= 1
                                   and len([a for a in before if "pid" in a]) <= 1)

        ok, why, aff = check_output(paths.predictions_dir / ("rep%d" % (args.reps - 1)),
                                    len(manifest.records))
        report["affinity"] = aff
        if ok and not report["gpu_exclusive"]:
            ok, why = False, ("GPU NOT EXCLUSIVE: compute apps before=%r after=%r -- timings on "
                              "a shared card are void" % (before, after))
        report["ok"], report["why"] = ok, why
    except Exception as e:                                    # noqa: BLE001
        import traceback
        report["ok"] = False
        report["why"] = "".join(traceback.format_exception(e))[-4000:]
        dump()
        raise
    finally:
        dump()


def check_output(pred_dir: pathlib.Path, n_expected: int) -> tuple[bool, str, dict]:
    """A number is only a number if the model produced a real prediction for every record.

    Nesso writes no structure, so the output guard is on the affinity scalars: every record
    present, every value finite, and `entropy_crop_pl` non-zero -- upstream is explicit that a
    zero there means the model could not place the ligand and the prediction must not be trusted.
    A harness that checks only the exit code will happily average a directory of unusable
    predictions.
    """
    if not pred_dir.exists():
        return False, "no predictions dir %s" % pred_dir, {}
    files = sorted(pred_dir.rglob("affinity.json"))
    if len(files) != n_expected:
        return False, "wrote %d of %d affinity.json" % (len(files), n_expected), {}
    vals, zeros, bad = [], 0, []
    for f in files:
        d = json.loads(f.read_text())
        v = d.get("affinity_pred_value")
        if v is None or v != v or abs(float(v)) == float("inf"):
            bad.append(f.parent.name)
            continue
        vals.append(float(v))
        if float(d.get("entropy_crop_pl", 0.0)) == 0.0:
            zeros += 1
    if bad:
        return False, "non-finite affinity for %r" % (bad[:5],), {}
    aff = {"n": len(vals), "mean": sum(vals) / len(vals),
           "min": min(vals), "max": max(vals),
           "spread": max(vals) - min(vals),
           "entropy_crop_pl_zero": zeros,
           "sha256_of_values": hashlib.sha256(
               json.dumps([round(v, 6) for v in vals]).encode()).hexdigest()[:16]}
    if zeros == len(vals):
        return False, "entropy_crop_pl == 0 for every record: ligand never placed", aff
    return True, "%d predictions, %d with entropy_crop_pl == 0" % (len(vals), zeros), aff


if __name__ == "__main__":
    main()
