"""The internal comparison arm: Boltz-2 affinity on the SAME GPU and the SAME fixtures.

    python perf/nesso1/gpu_boltz2_affinity.py --aa 128,256 --reps 2

tt-bio already ships binding affinity via Boltz-2, so Nesso-1 is a candidate upgrade to a
capability we have, not a new field. Two things follow. First, the number Nesso-1 has to beat is
Boltz-2's, measured here rather than inherited from the technical report's 10-20x claim. Second,
the comparison has to be matched: same card, same protein, same ligand, same single-sequence (no
MSA) setting, one prediction at a time. `perf-page-matched-batch-protocol-recurrence` has cost this
org three separate wrong published ratios; the inputs and the batch are pinned on both arms or the
ratio is void.

The Boltz-2 arm runs the settings tt-bio itself ships for affinity
(`docs/implementation-parity-data/ref-fixtures/boltz2/affinity_fkg/nomsa_200step_5affsample_3recycle_bf16_mwcorr`):
recycling_steps 3, sampling_steps 200, diffusion_samples 1, sampling_steps_affinity 200,
diffusion_samples_affinity 5, affinity_mw_correction, no MSA. Boltz-2 has to fold a structure
before it can score affinity; Nesso-1 does not. That is the architectural reason for the gap, and
it is exactly the thing being priced.
"""

import argparse
import json
import pathlib
import re
import subprocess
import time

HERE = pathlib.Path(__file__).parent
import sys                                                    # noqa: E402
sys.path.insert(0, str(HERE))
from make_inputs import LADDER_LIGAND, cdk2                   # noqa: E402

BOLTZ_ARGS = ["--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
              "--sampling_steps_affinity", "200", "--diffusion_samples_affinity", "5",
              "--affinity_mw_correction", "--output_format", "mmcif", "--num_workers", "2",
              "--accelerator", "gpu"]


def yaml_for(seq: str, smiles: str) -> str:
    return ("version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            "      sequence: %s\n"
            "      msa: empty\n"
            "  - ligand:\n"
            "      id: B\n"
            "      smiles: '%s'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n" % (seq, smiles))


def gpu_static() -> dict:
    q = "name,driver_version,power.limit,compute_cap"
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return dict(zip(q.split(","), [x.strip() for x in out.strip().splitlines()[0].split(",")]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", default="128,256")
    ap.add_argument("--reps", type=int, default=2, help="rep 0 pays the weight download / compile")
    ap.add_argument("--boltz", default="/work/v_boltz/bin/boltz")
    ap.add_argument("--work", default="/work/boltz2")
    ap.add_argument("--report", default="/work/results/boltz2_affinity.json")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ver = subprocess.run([args.boltz, "--version"], capture_output=True, text=True).stdout.strip()
    report = {"arm": "boltz-2 upstream, single sequence, tt-bio's shipped affinity settings",
              "boltz_version": ver, "boltz_args": BOLTZ_ARGS, "gpu": gpu_static(), "cells": []}

    for aa in [int(x) for x in args.aa.split(",")]:
        d = work / ("aa%d" % aa)
        d.mkdir(parents=True, exist_ok=True)
        y = d / ("cdk2_%d.yaml" % aa)
        y.write_text(yaml_for(cdk2(aa), LADDER_LIGAND))
        cell = {"aa": aa, "ligand_heavy": 22, "rep_s": [], "affinity": None, "ok": False,
                "why": ""}
        for rep in range(args.reps):
            out = d / ("out_rep%d" % rep)
            cmd = [args.boltz, "predict", str(y), "--out_dir", str(out)] + BOLTZ_ARGS
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            dt = time.time() - t0
            cell["rep_s"].append(round(dt, 3))
            print("[boltz2] aa%d rep%d %.3fs rc=%d" % (aa, rep, dt, p.returncode), flush=True)
            if p.returncode != 0:
                cell["why"] = p.stderr[-1500:]
                break
        # Boltz writes predictions/<stem>/affinity_<stem>.json
        affs = sorted(d.rglob("affinity_*.json"))
        if affs:
            cell["affinity"] = json.loads(affs[-1].read_text())
            cell["ok"] = "affinity_pred_value" in cell["affinity"]
            if not cell["ok"]:
                cell["why"] = "affinity json without affinity_pred_value: %r" % cell["affinity"]
        else:
            cell["why"] = cell["why"] or "no affinity_*.json written under %s" % d
        report["cells"].append(cell)
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps([{k: c[k] for k in ("aa", "rep_s", "ok")} for c in report["cells"]], indent=1))


if __name__ == "__main__":
    main()
