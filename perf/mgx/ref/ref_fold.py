#!/usr/bin/env python3
"""Fold one MGX reference cell with a model's own upstream code on a GPU.

    <venv>/bin/python perf/mgx/ref/ref_fold.py --model boltz2 --fixture 3abq_1536 --seed 0 \
        --out /root/refs

Runs on the rented box, one process per cell, from the repo root. It reads the shared fixture
yaml (perf/mgx/ref/fixtures), writes the input format the upstream wants with the SAME pinned
a3m per chain, runs the upstream CLI in-process with the model's shipped recycles and sampling
steps (the ones tt-bio runs, handed over in plan.json because tt_bio.main cannot be imported
here), one diffusion sample, and copies the structure to <out>/<model>/<fixture>/s<seed>.cif
beside a record.json: upstream package and version, checkpoint, GPU, dtype, the predict step's
own device time, peak memory, and the error if the upstream refused or ran out of memory. A
refusal is a result, so it is recorded, not retried.

Upstream map (package, venv, how it is driven):
  boltz2          boltz 2.2.1               venv-boltz      `boltz predict`, yaml as-is
  protenix-v2     protenix 2.0.0            venv-protenix   `protenix pred`, json with unpairedMsaPath
  protenix-v1     protenix 2.0.0            venv-protenix   same, v0.5.0 base checkpoint
  opendde(-abag)  opendde 1.0.3             venv-opendde    `opendde pred`, same json
  openfold3       openfold3 0.4.4           venv-of3        `run_openfold predict`, query json
  openbind        openfold3 0.5.0           venv-ob         same, of3-ob checkpoint
  esmfold2(-fast) esm + transformers        venv-esm312     ESMFold2Model, python API
  rf3             rc-foundry                v_rf3           `rf3 fold`, components with msa_path
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CKPT = Path(os.environ.get("MGX_CKPT", "/root/ckpt"))


# --------------------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------------------
def load_fixture(name: str) -> list[dict]:
    """[{ids: [...], sequence, msa: absolute a3m path}] in yaml order. Tiny parser on purpose:
    the fixture yaml is ours and flat, and not every upstream venv carries pyyaml."""
    chains, cur = [], None
    for line in (HERE / "fixtures" / f"{name}.yaml").read_text().split("\n"):
        s = line.strip()
        if s.startswith("- protein:"):
            cur = {}
            chains.append(cur)
        elif cur is not None and ":" in s and not s.startswith("#"):
            k, v = (x.strip() for x in s.split(":", 1))
            if k == "id":
                cur["ids"] = [x.strip() for x in v.strip("[]").split(",")]
            elif k == "sequence":
                cur["sequence"] = v
            elif k == "msa":
                cur["msa"] = str((ROOT / v).resolve())
    for c in chains:
        q = Path(c["msa"]).read_text().split("\n")[1].strip()
        assert q.replace("-", "") == c["sequence"], f"{c['msa']}: query row is not the chain"
    return chains


# --------------------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------------------
class StepTimer:
    """Wall time of the model's own predict call, cuda-synchronised both sides."""

    def __init__(self):
        self.times: list[float] = []

    def patch(self, obj, attr: str):
        import torch
        orig = getattr(obj, attr)
        times = self.times

        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            return r
        setattr(obj, attr, wrapper)


def pkg(*names: str) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            pass
    return out


def run_click(module: str, argv: list[str], attr: str | None = None):
    """Invoke a click CLI in-process so the step timer and memory counter see the fold."""
    mod = importlib.import_module(module)
    grp = getattr(mod, attr) if attr else next(
        getattr(mod, n) for n in dir(mod)
        if hasattr(getattr(mod, n), "main") and hasattr(getattr(mod, n), "commands"))
    try:
        grp.main(args=argv, standalone_mode=False)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"CLI exited {e.code}")


def newest(root: Path, pattern: str) -> Path:
    found = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise RuntimeError(f"upstream wrote no {pattern} under {root}: read the log above, "
                           "an exit code of 0 is not evidence of a fold")
    return found[-1]


# --------------------------------------------------------------------------------------
# per-model runners: each returns (structure path, facts dict)
# --------------------------------------------------------------------------------------
def run_boltz2(chains, seed, cfg, work):
    y = ["version: 1", "sequences:"]
    for c in chains:
        y += ["  - protein:", f"      id: [{', '.join(c['ids'])}]",
              f"      sequence: {c['sequence']}", f"      msa: {c['msa']}"]
    inp = work / "in"
    inp.mkdir()
    (inp / "t.yaml").write_text("\n".join(y) + "\n")
    timer = StepTimer()
    timer.patch(importlib.import_module("boltz.model.models.boltz2").Boltz2, "predict_step")
    argv = ["predict", str(inp), "--out_dir", str(work / "out"),
            "--cache", os.environ.get("BOLTZ_CACHE", "/root/.boltz"),
            "--devices", "1", "--accelerator", "gpu", "--seed", str(seed),
            "--recycling_steps", str(cfg["recycles"]), "--sampling_steps", str(cfg["steps"]),
            "--diffusion_samples", "1", "--output_format", "mmcif", "--num_workers", "2",
            "--override"]
    run_click("boltz.main", argv, "cli")
    return newest(work / "out", "*_model_0.cif"), dict(
        argv=argv, device_s=timer.times, pkgs=pkg("boltz", "torch", "cuequivariance-torch"),
        dtype="upstream default (boltz2 predict: bf16-mixed; no CLI switch to fp32)")


def protenix_json(chains, name: str) -> list:
    return [{"name": name, "modelSeeds": [], "sequences": [
        {"proteinChain": {"sequence": c["sequence"], "count": len(c["ids"]),
                          "unpairedMsaPath": c["msa"]}} for c in chains]}]


def run_protenix(chains, seed, cfg, work, *, model_name: str, ckpt: str, module: str):
    """Protenix 2.0.0 and OpenDDE 1.0.3 share the protenix runner and its json. OpenDDE takes an
    explicit --load_checkpoint_path; protenix 2.0.0 has no such flag and loads
    $PROTENIX_ROOT_DIR/checkpoint/<model_name>.pt, so the checkpoint is linked there."""
    js = work / "t.json"
    js.write_text(json.dumps(protenix_json(chains, "t"), indent=1))
    argv = ["pred", "-i", str(js), "-o", str(work / "out"), "-s", str(seed),
            "-c", str(cfg["recycles"]), "-p", str(cfg["steps"]), "-e", "1", "-d", cfg["dtype"],
            "-n", model_name, "--use_msa", "true"]
    if module == "runner.cli":
        argv += ["--load_checkpoint_path", ckpt]
    else:
        root = Path(os.environ.setdefault("PROTENIX_ROOT_DIR", "/root/protenix_root"))
        link = root / "checkpoint" / f"{model_name}.pt"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(ckpt)
    run_click(module, argv)
    return newest(work / "out", "*.cif"), dict(argv=argv, checkpoint=ckpt,
                                               dtype=f"{cfg['dtype']} (upstream --dtype, TF32 at its default)",
                                               pkgs=pkg("protenix", "opendde", "torch"))


def run_protenix_v2(chains, seed, cfg, work):
    return run_protenix(chains, seed, cfg, work, model_name="protenix-v2",
                        ckpt=str(CKPT / "protenix-v2.pt"), module="runner.batch_inference")


def run_protenix_v1(chains, seed, cfg, work):
    # tt-bio's protenix-v1 is upstream's v0.5.0 base checkpoint run through the one protenix
    # implementation; protenix 2.0.0 still ships that model's config under this name.
    return run_protenix(chains, seed, cfg, work, model_name="protenix_base_default_v0.5.0",
                        ckpt=str(CKPT / "model_v0.5.0.pt"), module="runner.batch_inference")


def run_opendde(chains, seed, cfg, work):
    return run_protenix(chains, seed, cfg, work, model_name="opendde_v1",
                        ckpt=str(CKPT / "opendde.pt"), module="runner.cli")


def run_opendde_abag(chains, seed, cfg, work):
    return run_protenix(chains, seed, cfg, work, model_name="opendde_v1",
                        ckpt=str(CKPT / "opendde_abag.pt"), module="runner.cli")


def run_of3(chains, seed, cfg, work, *, ckpt: str):
    # OF3 keys MSA files by basename against its database names; a precomputed unpaired main
    # MSA goes in the uniref90_hits slot (see scripts/gpu_vs_tt/gpu5_bench.py::run_of3).
    qchains = []
    for i, c in enumerate(chains):
        d = work / f"msa{i}"
        d.mkdir()
        shutil.copyfile(c["msa"], d / "uniref90_hits.a3m")
        qchains.append(dict(molecule_type="protein", chain_ids=c["ids"], sequence=c["sequence"],
                            main_msa_file_paths=[str(d / "uniref90_hits.a3m")]))
    q = work / "q.json"
    q.write_text(json.dumps(dict(seeds=[seed], queries={"t": dict(
        chains=qchains, use_msas=True, use_paired_msas=False, use_main_msas=True)}), indent=1))
    # The dataset reads experiment_settings.seeds, not the query set's (defaults to [42]).
    ry = work / "runner.yaml"
    ry.write_text(f"experiment_settings:\n  seeds: [{seed}]\n"
                  "model_update:\n  presets: [\"predict\"]\n  custom:\n    settings:\n"
                  "      memory:\n        eval:\n          use_cueq_triangle_kernels: true\n")
    timer = StepTimer()
    timer.patch(importlib.import_module("openfold3.projects.of3_all_atom.runner").OpenFold3AllAtom,
                "predict_step")
    argv = ["predict", "--query-json", str(q), "--output-dir", str(work / "out"),
            "--inference-ckpt-path", ckpt, "--runner-yaml", str(ry),
            "--num-diffusion-samples", "1", "--use-msa-server", "false", "--use-templates", "false"]
    run_click("openfold3.run_openfold", argv)
    cif = newest(work / "out", "*.cif")
    m = re.search(r"seed_(\d+)", str(cif))
    assert m and int(m.group(1)) == seed, f"openfold3 folded at {cif}, not seed {seed}"
    return cif, dict(argv=argv, device_s=timer.times, pkgs=pkg("openfold3", "torch"),
                     dtype="upstream predict preset",
                     settings_note="recycles and steps not passed: OF3's CLI has no switch for them, "
                                   "and tt-bio's 3 / 200 are documented as OF3's own defaults "
                                   "(tt_bio/main.py RECYCLING_STEPS)")


def run_openfold3(chains, seed, cfg, work):
    return run_of3(chains, seed, cfg, work, ckpt=str(CKPT / "of3-p2-155k.pt"))


def run_openbind(chains, seed, cfg, work):
    return run_of3(chains, seed, cfg, work, ckpt=str(CKPT / "of3-ob-2025-06-30-174k.pt"))


ESM_REPOS = {"esmfold2": ("biohub/ESMFold2", "8fc3ff471022fdce52c77030685eb775de0c00a3"),
             "esmfold2-fast": ("biohub/ESMFold2-Fast", "c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6")}


def run_esmfold2(chains, seed, cfg, work, *, model="esmfold2"):
    """Biohub esm (pip), EsmFold2Model in fp32 with the ESMC LM in fp32, driven through the same
    ESMFold2InputBuilder().fold() call tt-bio makes (tt_bio/esmfold2_runtime.py::fold_complex):
    one ProteinInput per chain copy, MSA.from_a3m(max_sequences=16384) per chain.

    tt-bio vendors an older esm whose fold() had no stochastic knobs. The current esm adds
    lm_dropout (default 0.3) and msa_max_depth / msa_column_mask_rate (1024 / 0.1). Those are
    passed as None here, which disables the dropout and hands depth and column masking back to
    the checkpoint config, the behaviour of the version the TT port was built against."""
    import torch
    from esm.models.esmfold2 import MSA, EsmFold2Model, ESMFold2InputBuilder, ProteinInput
    from esm.models.esmfold2.types import StructurePredictionInput

    # torch SDPA's cuDNN backend has no execution plan for ESMC's attention shapes on this stack
    # and raises before the first fold (scripts/gpu_vs_tt/gpu5_bench.py::run_esmfold2). Turning
    # off a backend that cannot run lets SDPA pick flash / mem-efficient / math.
    torch.backends.cuda.enable_cudnn_sdp(False)
    repo, rev = ESM_REPOS[model]
    m = EsmFold2Model.from_pretrained(repo, revision=rev, esmc_precision="fp32",
                                      device="cuda", dtype=torch.float32).eval()
    spi = StructurePredictionInput(sequences=[
        ProteinInput(id=cid, sequence=c["sequence"],
                     msa=MSA.from_a3m(c["msa"], max_sequences=16384))
        for c in chains for cid in c["ids"]])
    timer = StepTimer()
    timer.patch(EsmFold2Model, "forward")
    kw = dict(num_loops=cfg["recycles"], num_sampling_steps=cfg["steps"],
              num_diffusion_samples=1, seed=seed, lm_dropout=None,
              msa_max_depth=None, msa_column_mask_rate=None)
    with torch.no_grad():
        res = ESMFold2InputBuilder().fold(m, spi, **kw)
    cif = work / "pred.cif"
    cif.write_text(res.complex.to_mmcif())
    return cif, dict(fold_kwargs=kw, repo=repo, revision=rev, device_s=timer.times,
                     dtype="fp32 trunk, fp32 ESMC", pkgs=pkg("esm", "transformers", "torch"))


def run_esmfold2_fast(chains, seed, cfg, work):
    return run_esmfold2(chains, seed, cfg, work, model="esmfold2-fast")


def run_rf3(chains, seed, cfg, work):
    comps = []
    for c in chains:
        for cid in c["ids"]:
            comps.append({"seq": c["sequence"], "chain_id": cid, "msa_path": c["msa"]})
    js = work / "t.json"
    js.write_text(json.dumps({"name": "t", "components": comps}, indent=1))
    rf3 = Path(sys.executable).parent / "rf3"
    ckpt = CKPT / "rf3_foundry_01_24_latest_remapped.ckpt"   # tt_bio/weights.py "rf3"
    argv = [str(rf3), "fold", f"inputs={js}", f"out_dir={work / 'out'}", f"seed={seed}",
            f"n_recycles={cfg['recycles']}", f"num_steps={cfg['steps']}",
            "diffusion_batch_size=1", "skip_existing=False", f"ckpt_path={ckpt}",
            # upstream ships 0.5, which abandons a target after one recycle when pLDDT is low;
            # tt-bio runs without it unless --early_stop_plddt is passed.
            "early_stopping_plddt_threshold=0"]
    subprocess.run(argv, check=True)
    return newest(work / "out", "*.cif*"), dict(argv=argv, checkpoint=str(ckpt),
                                               pkgs=pkg("rc-foundry", "atomworks", "torch"),
                                               dtype="upstream default")


RUNNERS = {"boltz2": run_boltz2, "protenix-v2": run_protenix_v2, "protenix-v1": run_protenix_v1,
           "opendde": run_opendde, "opendde-abag": run_opendde_abag,
           "openfold3": run_openfold3, "openbind": run_openbind,
           "esmfold2": run_esmfold2, "esmfold2-fast": run_esmfold2_fast, "rf3": run_rf3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("/root/refs"))
    ap.add_argument("--plan", type=Path, default=HERE / "plan.json")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"),
                    help="protenix family only: bf16 is the fallback for a cell whose fp32 fold "
                         "does not fit the card, and the record says which one ran")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert set(RUNNERS) == set(plan["models"]), (
        f"runner table {sorted(RUNNERS)} is not PREDICT_MODELS {sorted(plan['models'])}")
    cfg = dict(plan["models"][args.model], dtype=args.dtype)
    chains = load_fixture(args.fixture)
    dest = args.out / args.model / args.fixture
    dest.mkdir(parents=True, exist_ok=True)
    work = Path("/root/work") / args.model / args.fixture / f"s{args.seed}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    import torch
    rec = dict(model=args.model, fixture=args.fixture, seed=args.seed,
               tokens=sum(len(c["sequence"]) * len(c["ids"]) for c in chains),
               recycles=cfg["recycles"], sampling_steps=cfg["steps"], samples=1,
               gpu=torch.cuda.get_device_name(0), started=time.strftime("%FT%TZ", time.gmtime()))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        cif, facts = RUNNERS[args.model](chains, args.seed, cfg, work)
        shutil.copyfile(cif, dest / f"s{args.seed}.cif")
        rec.update(facts, status="ok", upstream_output=str(cif))
    except BaseException as e:   # noqa: BLE001 -- an OOM or a refusal is the result
        tb = traceback.format_exc()
        oom = "out of memory" in tb.lower() or isinstance(e, torch.cuda.OutOfMemoryError)
        rec.update(status="oom" if oom else "error", error=f"{type(e).__name__}: {e}"[:2000],
                   traceback_tail=tb[-4000:])
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    rec["peak_mem_GiB_torch"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (dest / f"s{args.seed}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("REF", json.dumps({k: rec.get(k) for k in ("model", "fixture", "seed", "status",
                                                     "wall_s", "peak_mem_GiB_torch", "error")}))
    return 0 if rec["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
