"""The TT side of the internal comparison: Boltz-2 affinity on one p150a.

    python perf/nesso1/tt_boltz2_affinity.py --out perf/nesso1/results/tt_boltz2_affinity.json

Pass 1 left this owed. tt-bio ships binding affinity via Boltz-2 today, so the product question
"is Nesso-1 the better affinity path on our own hardware" is answerable before the port exists:
Nesso-1's device floor at 256 aa is known (4.158 s/prediction, perf/nesso1/results/
floor_timing_contended.json) and Boltz-2's cost on the same card can be measured now.

The protocol is copied from the GPU arm (perf/nesso1/gpu_boltz2_affinity.py and the
`boltz2_dir.json` directory arm) so the TT ratio and the GPU ratio are the same measurement on two
platforms:

  single      one 256 aa CDK2 target x the README ligand (22 heavy atoms), no MSA, one invocation.
  directory   the same target x 4 distinct peptide ligands in one invocation.
  marginal    (wall_dir - wall_single) / (n_dir - 1)

Marginal, not wall, is the number that matters. Affinity prediction is virtual screening: the
fixed per-invocation cost (weights load, kernel cache, ESM, preprocessing) is paid once against
millions of compounds, so a screen sees the marginal cost. The GPU arm measured Boltz-2 at
13.949 s marginal against Nesso-1's 0.722 s, i.e. 19.3x.

Settings are tt-bio's own shipped affinity fixture
(docs/implementation-parity-data/ref-fixtures/boltz2/affinity_fkg/nomsa_200step_5affsample_3recycle_bf16_mwcorr):
recycling_steps 3, sampling_steps 200, diffusion_samples 1, sampling_steps_affinity 200,
diffusion_samples_affinity 5, affinity_mw_correction, single-sequence.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_inputs import LADDER_LIGAND, cdk2, heavy_atoms, peptide, yaml_for  # noqa: E402

AFFINITY_ARGS = ["--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
                 "--sampling_steps_affinity", "200", "--diffusion_samples_affinity", "5",
                 "--affinity_mw_correction", "--output_format", "cif", "--single_sequence",
                 "--override", "--accelerator", "tenstorrent", "--model", "boltz2"]

# The GPU directory arm's four compounds, verbatim from results/boltz2_dir.json.
DIR_CODES = ("GGGGGG", "GGGGGA", "GGGGAG", "GGGGAA")


def run(cmd, timeout):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return round(time.time() - t0, 3), p


def affinity_values(out_dir: pathlib.Path):
    """Every affinity_*.json Boltz-2 wrote under out_dir, keyed by target stem."""
    vals = {}
    for f in sorted(out_dir.rglob("affinity_*.json")):
        try:
            vals[f.stem.replace("affinity_", "")] = json.loads(f.read_text())
        except Exception as e:
            vals[f.stem] = {"unreadable": str(e)}
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3, help="rep 0 is reported and discarded as cold")
    ap.add_argument("--dir-reps", type=int, default=2)
    ap.add_argument("--work", default=None, help="scratch dir (default: <repo>/perf/nesso1/tt_work)")
    ap.add_argument("--out", default=str(HERE / "results" / "tt_boltz2_affinity.json"))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tt-bio", default=None, help="tt-bio entry point (default: <python> -m tt_bio.main)")
    args = ap.parse_args()

    work = pathlib.Path(args.work or (HERE / "tt_work"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    base = [args.tt_bio] if args.tt_bio else [sys.executable, "-m", "tt_bio.main"]
    seq = cdk2(args.aa)

    report = {
        "arm": "boltz-2 affinity on one Blackhole p150a, tt-bio's shipped affinity settings",
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES", "(unset)"),
        "loadavg_at_start": os.getloadavg(),
        "protein_aa": args.aa,
        "cli_args": AFFINITY_ARGS,
        "single": {"ligand_heavy": 22, "reps_s": [], "affinity": None, "rc": None},
        "directory": {"ligand_codes": list(DIR_CODES),
                      "ligand_heavy": [heavy_atoms(c) for c in DIR_CODES],
                      "reps": []},
    }

    def save():
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n")

    # ---- single invocation ------------------------------------------------------------------
    sd = work / "single"
    sd.mkdir(parents=True)
    y = sd / ("cdk2_%d.yaml" % args.aa)
    y.write_text(yaml_for(seq, LADDER_LIGAND))
    for rep in range(args.reps):
        out = sd / ("out_rep%d" % rep)
        dt, p = run(base + ["predict", str(y), "--out_dir", str(out)] + AFFINITY_ARGS, args.timeout)
        report["single"]["reps_s"].append(dt)
        report["single"]["rc"] = p.returncode
        print("[tt boltz2] single aa%d rep%d %.3fs rc=%d" % (args.aa, rep, dt, p.returncode),
              flush=True)
        if p.returncode != 0:
            report["single"]["stderr_tail"] = p.stderr[-3000:]
            report["single"]["stdout_tail"] = p.stdout[-2000:]
            save()
            print("single arm failed, stopping", flush=True)
            return
        report["single"]["affinity"] = affinity_values(out)
        save()

    # ---- directory (4 compounds, one invocation) --------------------------------------------
    dd = work / "dir"
    din = dd / "in"
    din.mkdir(parents=True)
    for code in DIR_CODES:
        (din / ("cdk2_%d_%s.yaml" % (args.aa, code))).write_text(yaml_for(seq, peptide(code)))
    for rep in range(args.dir_reps):
        out = dd / ("out_rep%d" % rep)
        dt, p = run(base + ["predict", str(din), "--out_dir", str(out)] + AFFINITY_ARGS,
                    args.timeout)
        aff = affinity_values(out)
        report["directory"]["reps"].append({
            "wall_s": dt, "rc": p.returncode, "n_written": len(aff),
            "s_per_pred": round(dt / len(DIR_CODES), 4),
            "stderr_tail": p.stderr[-2000:] if p.returncode != 0 else "",
        })
        report["directory"]["affinity"] = aff
        print("[tt boltz2] dir aa%d rep%d %.3fs rc=%d n=%d"
              % (args.aa, rep, dt, p.returncode, len(aff)), flush=True)
        save()
        if p.returncode != 0:
            print("directory arm failed, stopping", flush=True)
            return

    # ---- marginal ---------------------------------------------------------------------------
    singles = report["single"]["reps_s"]
    warm_single = min(singles[1:]) if len(singles) > 1 else singles[0]
    dir_ok = [r["wall_s"] for r in report["directory"]["reps"] if r["rc"] == 0]
    if dir_ok:
        warm_dir = min(dir_ok)
        n = len(DIR_CODES)
        marginal = (warm_dir - warm_single) / (n - 1)
        report["matched_marginal"] = {
            "method": ("two measured points, one single-record invocation and a 4-record "
                       "directory, same card, same 256 aa protein, no MSA. marginal = "
                       "(wall_dir - wall_single) / (n_dir - 1). Warm arm = fastest non-cold rep, "
                       "which is the same reduction the GPU arm used."),
            "warm_single_s": warm_single,
            "warm_dir_s": warm_dir,
            "dir_n": n,
            "marginal_s_per_compound": round(marginal, 5),
            "fixed_s_per_invocation": round(warm_single - marginal, 3),
            "marginal_pred_per_hour": round(3600.0 / marginal, 1) if marginal > 0 else None,
        }
        report["loadavg_at_end"] = os.getloadavg()
        save()
        print(json.dumps(report["matched_marginal"], indent=1))


if __name__ == "__main__":
    main()
