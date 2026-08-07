"""Generate an OpenFold3 reference parity fixture: run the official upstream
`openfold3` pip package (CPU) for seeds 0..N on a fixed query JSON, then harvest
each seed's confidence-selected structure into the committed fixture layout:

    <fixture-dir>/seed<N>/results.json
    <fixture-dir>/seed<N>/structures/<target_id>.cif
    <fixture-dir>/msa.a3m            (the exact MSA bytes both sides consumed)
    <fixture-dir>/meta.json          (reference provenance + invalidation rule)

The query JSON must pin main_msa_file_paths to the SAME a3m the device leg will
consume (parity is only meaningful on identical inputs; the MSA content otherwise
differs run to run). Requires the CPU reference venv (see docs/openfold3-port.md):
    python3.12 -m venv /tmp/of3-venv && /tmp/of3-venv/bin/pip install openfold3

Example:
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_ref_fixture.py \
        --query-json /tmp/of3_ubq_query.json --target-id ubq \
        --fixture-dir docs/implementation-parity-data/ref-fixtures/openfold3/ubq/<tag> \
        --seeds 0 1 2 3 4 --msa-a3m <the exact a3m the device leg stages>
"""
import argparse
import json
import os
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
    seed_out = next((out_dir / query_name).glob(f"seed_{seed}"))
    best_idx, best_score, best_agg = None, None, None
    aggs = sorted(seed_out.glob(f"*_sample_*_confidences_aggregated.json"))
    for i, ap in enumerate(aggs):
        agg = json.loads(ap.read_text())
        score = agg["sample_ranking_score"]
        if best_score is None or score > best_score:
            best_idx, best_score, best_agg = i, score, agg
    if best_idx is None:
        raise RuntimeError(f"no confidence summaries under {seed_out}")

    seed_dir = fixture_dir / f"seed{seed}"
    (seed_dir / "structures").mkdir(parents=True, exist_ok=True)
    src_cif = seed_out / f"{query_name}_seed_{seed}_sample_{best_idx + 1}_model.cif"
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
        "selected_record": record,
        "note": "real reference output copied verbatim; not regenerated or edited",
    }, indent=2))
    return record


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
    ap.add_argument("--ref-run", default="/tmp/of3-venv/bin/run_openfold")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))
    ap.add_argument("--work-root", default="/tmp/of3_ref_fixture_runs")
    ap.add_argument("--meta", default=None,
                    help="JSON string merged into the fixture meta.json.")
    args = ap.parse_args()

    query = json.loads(Path(args.query_json).read_text())
    query_name = next(iter(query["queries"]))
    fixture_dir: Path = args.fixture_dir
    fixture_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_root)
    work.mkdir(parents=True, exist_ok=True)

    records = {}
    for seed in args.seeds:
        runner = work / f"runner_seed{seed}.yml"
        runner.write_text(RUNNER_TEMPLATE.format(
            seed=seed, use_templates=str(args.use_templates).lower(),
            template_settings=(TEMPLATE_SETTINGS if args.use_templates else "")))
        out_dir = work / f"{args.target_id}_seed{seed}"
        cmd = [args.ref_run, "predict", "--query-json", args.query_json,
               "--inference-ckpt-path", args.ckpt,
               "--num-diffusion-samples", str(args.num_diffusion_samples),
               "--use-msa-server", "False",
               "--use-templates", str(args.use_templates),
               "--output-dir", str(out_dir), "--runner-yaml", str(runner)]
        print(f"[seed {seed}] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"reference run failed for seed {seed}")
        records[seed] = _harvest(out_dir, query_name, args.target_id, seed,
                                 fixture_dir, args.num_diffusion_samples)
        print(f"[seed {seed}] selected sample {records[seed]['selected_sample_idx']} "
              f"ptm={records[seed]['ptm']:.4f} plddt={records[seed]['plddt']:.2f}",
              flush=True)

    if args.msa_a3m:
        shutil.copyfile(args.msa_a3m, fixture_dir / "msa.a3m")

    meta = {
        "model": "openfold3",
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
