#!/usr/bin/env bash
# Can TT_BIO_FUSE_BIAS_STACKS reach a BoltzGen design at all?
#
# The flag was held default-off because "BoltzGen has the identical bias stacks". The stacks are
# identical; the code path is not. `_bias_stack` is defined and called only in tt_bio/boltz2.py,
# and BoltzGen builds its conditioning from tt_bio/boltzgen/model/modules/diffusion_conditioning.py,
# which inlines the per-layer loop. A design A/B cannot settle this on its own -- BoltzGen has no
# --seed and its output is not reproducible run to run even with torch, numpy and random all seeded
# (see bg_fusebias_ab.sh) -- so this counts invocations instead.
#
# Two legs, because a counter that never fires proves nothing about the counter:
#   boltz2    fold a short monomer with the flag on. Expect calls > 0.   <- the control
#   boltzgen  run a design with the flag on. Expect calls == 0.
#
# The counter lives in seedsite/sitecustomize.py, not in a wrapper here, because both CLIs fan
# their work into child processes: a monkeypatch installed in the launcher is never the one that
# runs, and the first attempt at this read 0 on a Boltz-2 fold that certainly makes the calls.
#
#   bash perf/b2z_levers/reach.sh <card>
set -u
card="${1:-0}"
WT=/home/ttuser/.coworker/wt/b2z-levers-default-on
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=$WT/perf/b2z_levers
W=$D/work
rm -rf "$W/cnt_b2" "$W/cnt_bg" "$W/reach_b2" "$W/reach_bg"

common=(TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_CARDS="$card"
        TT_BIO_LEASE_HOLDER=worker:b2z-levers-default-on
        PYTHONPATH="$D/seedsite:$WT" TT_BIO_FUSE_BIAS_STACKS=1)

env "${common[@]}" BG_COUNT_DIR="$W/cnt_b2" BG_LEG=boltz2 \
  "$PY" -m tt_bio.main predict "$W/ctl.fasta" --model boltz2 --out_dir "$W/reach_b2" \
        --sampling_steps 5 --recycling_steps 0 --diffusion_samples 1 > "$W/log_reach_b2.txt" 2>&1
env "${common[@]}" BG_COUNT_DIR="$W/cnt_bg" BG_LEG=boltzgen \
  "$PY" -m tt_bio.main design "$W/bg256.yaml" --model boltzgen --out_dir "$W/reach_bg" \
        --num_designs 1 --steps design --devices "$card" --debug > "$W/log_reach_bg.txt" 2>&1

"$PY" - "$W" "$D/reach_qb2c$card.json" "$card" <<'PY'
import json, pathlib, subprocess, sys
w, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def leg(name, artifact):
    recs = [json.loads(p.read_text()) for p in sorted((w / f"cnt_{name}").glob("*.json"))]
    return {"processes": len(recs), "bias_stack_calls": sum(r["n"] for r in recs),
            "fused_calls": sum(r["fused"] for r in recs),
            "artifact_written": (w / artifact).exists()}
rec = {"host": "tt-quietbox2", "card": sys.argv[3], "flag": "TT_BIO_FUSE_BIAS_STACKS=1",
       "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip(),
       "boltz2": leg("b2", "reach_b2/boltz2_results_ctl/results.json"),
       "boltzgen": leg("bg", "reach_bg/intermediate_designs/bg256.cif")}
rec["verdict"] = ("REACHES BOLTZ-2 ONLY"
                  if rec["boltz2"]["fused_calls"] > 0 and rec["boltzgen"]["bias_stack_calls"] == 0
                  else "INCONCLUSIVE")
out.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
PY
