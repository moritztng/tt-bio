"""Generate an OpenFold3 / OpenBind reference parity fixture: run the upstream
`openfold3` reference (CPU) for seeds 0..N on a fixed query JSON, then harvest
each seed's confidence-selected structure into the committed fixture layout:

    <fixture-dir>/seed<N>/results.json
    <fixture-dir>/seed<N>/structures/<target_id>.cif
    <fixture-dir>/msa.a3m            (the exact MSA bytes both sides consumed)
    <fixture-dir>/meta.json          (reference provenance + invalidation rule)

The query JSON must pin main_msa_file_paths to the SAME a3m the device leg will
consume (parity is only meaningful on identical inputs; the MSA content otherwise
differs run to run).

Two reference flavours, selected by --ref-run:
  * pip-installed openfold3 (preview2 legs):
        python3 -m venv /tmp/of3-venv && /tmp/of3-venv/bin/pip install openfold3
        --ref-run /tmp/of3-venv/bin/run_openfold
  * a source clone (the OpenBind v0.5.0 legs; the release is not on PyPI):
        --ref-run "<refenv>/bin/python <repo>/scripts/ob0_run_openfold.py"
        --ref-env PYTHONPATH=/home/ttuser/ob0_upstream:/home/ttuser/ob0_refdeps

Example:
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_ref_fixture.py \
        --query-json /tmp/of3_ubq_query.json --target-id ubq \
        --fixture-dir docs/implementation-parity-data/ref-fixtures/openfold3/ubq/<tag> \
        --seeds 0 1 2 3 4 --msa-a3m <the exact a3m the device leg stages>
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNNER_TEMPLATE = """\
experiment_settings:
  seeds: [{seed}]
  use_msa_server: false
  use_templates: {use_templates}
pl_trainer_args:
  accelerator: cpu
model_update:
  presets:
    - predict
  custom:
    settings:
      memory:
        eval:
          use_triton_triangle_kernels: false
          use_deepspeed_evo_attention: false
          use_cueq_triangle_kernels: false
{template_settings}"""

TEMPLATE_SETTINGS = """template_preprocessor_settings:
  mode: predict
  cache_directory: /home/ttuser/of3-bench/benchmark_templates
  structure_directory: /home/ttuser/of3-bench/template_structures
"""


def _harvest(out_dir: Path, query_name: str, target_id: str, seed: int,
             fixture_dir: Path, n_samples: int) -> dict:
    # Recursive: the reference's per-seed directory layout is not part of its API and
    # has moved between releases. The per-sample file names have not.
    aggs = sorted(out_dir.rglob("*_confidences_aggregated.json"))
    best_idx, best_score, best_agg, best_path = None, None, None, None
    for i, ap in enumerate(aggs):
        agg = json.loads(ap.read_text())
        score = agg["sample_ranking_score"]
        if best_score is None or score > best_score:
            best_idx, best_score, best_agg, best_path = i, score, agg, ap
    if best_idx is None:
        raise RuntimeError(f"no confidence summaries under {out_dir}")

    seed_dir = fixture_dir / f"seed{seed}"
    (seed_dir / "structures").mkdir(parents=True, exist_ok=True)
    src_cif = Path(str(best_path).replace("_confidences_aggregated.json", "_model.cif"))
    if not src_cif.exists():
        raise RuntimeError(f"selected sample has no structure: {src_cif}")
    shutil.copyfile(src_cif, seed_dir / "structures" / f"{target_id}.cif")
    record = {
        "id": target_id,
        "status": "ok",
        "n_residues": None,
        "n_chains": len(best_agg.get("chain_ptm", {}) or {target_id: 0}),
        "msa": True,
        "samples": len(aggs),
        "ptm": best_agg["ptm"],
        "iptm": best_agg.get("iptm", 0.0),
        "plddt": best_agg["avg_plddt"],
        "ranking_score": best_agg["sample_ranking_score"],
        "selected_sample_idx": best_idx,
    }
    (seed_dir / "results.json").write_text(json.dumps([record], indent=2))
    (seed_dir / "meta.json").write_text(json.dumps({
        "seed": seed,
        "target_id": target_id,
        "harvested_from": str(out_dir),
        "selected_structure": src_cif.name,
        "selected_record": record,
        "note": "real reference output copied verbatim; not regenerated or edited",
    }, indent=2))
    return record


def _seed_cmd(args, seed: int, work: Path) -> tuple[list[str], Path, Path]:
    runner = work / f"runner_seed{seed}.yml"
    runner.write_text(RUNNER_TEMPLATE.format(
        seed=seed, use_templates=str(args.use_templates).lower(),
        template_settings=(TEMPLATE_SETTINGS if args.use_templates else "")))
    out_dir = work / f"{args.target_id}_seed{seed}"
    cmd = shlex.split(args.ref_run) + [
        "predict", "--query-json", args.query_json,
        "--inference-ckpt-path", args.ckpt,
        "--num-diffusion-samples", str(args.num_diffusion_samples),
        "--use-msa-server", "False",
        "--use-templates", str(args.use_templates),
        "--output-dir", str(out_dir), "--runner-yaml", str(runner)]
    return cmd, out_dir, work / f"seed{seed}.log"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-json", required=True,
                    help="OF3 inference query JSON (MSA paths pinned).")
    ap.add_argument("--target-id", required=True,
                    help="Fixture-side id; must equal the device leg's yaml stem.")
    ap.add_argument("--fixture-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--num-diffusion-samples", type=int, default=5)
    ap.add_argument("--use-templates", action="store_true")
    ap.add_argument("--msa-a3m", default=None,
                    help="Copy these exact MSA bytes to <fixture-dir>/msa.a3m.")
    ap.add_argument("--ref-run", default="/tmp/of3-venv/bin/run_openfold",
                    help="Reference launcher, shell-split (a pip console script, or "
                         "'<python> <wrapper.py>' for a source clone).")
    ap.add_argument("--ref-env", action="append", default=[], metavar="NAME=VALUE",
                    help="Extra env for the reference process (repeatable).")
    ap.add_argument("--model-name", default="openfold3",
                    help="Written to meta.json 'model'; the gate keys fixtures on it.")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))
    ap.add_argument("--work-root", default="/tmp/of3_ref_fixture_runs")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Seeds to run concurrently. CPU reference folds are "
                         "thread-limited with --threads so N of them fit one box.")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch/OMP threads per seed process (0 = leave alone).")
    ap.add_argument("--meta", default=None,
                    help="JSON string merged into the fixture meta.json.")
    args = ap.parse_args()

    query = json.loads(Path(args.query_json).read_text())
    query_name = next(iter(query["queries"]))
    fixture_dir: Path = args.fixture_dir
    fixture_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_root)
    work.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    for kv in args.ref_env:
        k, _, v = kv.partition("=")
        env[k] = v
    if args.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            env[k] = str(args.threads)

    records = {}
    pending = list(args.seeds)
    while pending:
        batch, pending = pending[:max(1, args.jobs)], pending[max(1, args.jobs):]
        running = []
        for seed in batch:
            cmd, out_dir, log = _seed_cmd(args, seed, work)
            print(f"[seed {seed}] {' '.join(cmd)}", flush=True)
            fh = open(log, "w")
            running.append((seed, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                   env=env), out_dir, log, fh))
        for seed, proc, out_dir, log, fh in running:
            rc = proc.wait()
            fh.close()
            if rc != 0:
                print(Path(log).read_text()[-4000:], file=sys.stderr)
                raise SystemExit(f"reference run failed for seed {seed} (see {log})")
            records[seed] = _harvest(out_dir, query_name, args.target_id, seed,
                                     fixture_dir, args.num_diffusion_samples)
            print(f"[seed {seed}] selected sample {records[seed]['selected_sample_idx']} "
                  f"ptm={records[seed]['ptm']:.4f} plddt={records[seed]['plddt']:.2f}",
                  flush=True)

    if args.msa_a3m:
        shutil.copyfile(args.msa_a3m, fixture_dir / "msa.a3m")

    meta = {
        "model": args.model_name,
        "target": args.target_id,
        "reference_impl": "official aqlaboratory openfold3 (torch, CPU)",
        "reference_version": "openfold3 0.4.4, checkpoint of3-p2-155k.pt "
                             "(p2 preview, 155k steps)",
        "reference_commit": "c615a7f8 (aqlaboratory/openfold-3 main, 2026-08-05)",
        "command": "scripts/of3_ref_fixture.py (see docs/openfold3-port.md for the "
                   "CPU venv recipe)",
        "date": __import__("datetime").date.today().isoformat(),
        "seeds": list(args.seeds),
        "settings": {
            "diffusion_samples": args.num_diffusion_samples,
            "diffusion_steps": 200,
            "recycling_cycles": 4,
            "dtype": "fp32 (CPU)",
            "msa_source": "committed msa.a3m (identical bytes for device and reference)",
            "use_templates": bool(args.use_templates),
            "selection": "confidence-selected best-of-5 by sample_ranking_score",
        },
        "settings_tag": fixture_dir.name,
        "invalidation_rule": "Regenerate ONLY when the pinned reference version, "
                             "checkpoint, or settings change; the device side re-runs "
                             "every release.",
    }
    if args.meta:
        meta.update(json.loads(args.meta))
    (fixture_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote fixture {fixture_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
