#!/bin/bash
# Does the size ladder's own 6-step config produce a structure at all?
#
# The recorder's new structural read calls boltz2/256 exploded: 256 CA atoms, one chain,
# residues in order, and consecutive CA-CA distances with a MINIMUM of 17.6 A. The parse is
# not the suspect, so either the port is broken at 256 aa or six diffusion steps is not a
# fold. Same fixture, same seed, same card, only the step count moves.
set -u
WT=/home/ttuser/.coworker/wt/cov-ladder-p150a-p3
PY=/home/ttuser/kisoji_p2_fresh/env/bin/python3
cd "$WT" || exit 1
OUT=$WT/perf/sizegate/campaign/steps_ctl
mkdir -p "$OUT"
for S in 6 25 100 200; do
  rm -rf "$OUT/s$S"
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
  TT_BIO_LEASE_HOLDER=worker:cov-ladder-p150a-p3 PYTHONPATH="$WT" \
    "$PY" -u -m tt_bio.main predict "$WT/perf/size512/fixtures/cdk2x2_256.yaml" \
      --model boltz2 --single_sequence --sampling_steps "$S" --diffusion_samples 1 \
      --seed 0 --out_dir "$OUT/s$S" > "$OUT/s$S.log" 2>&1
  echo "steps=$S rc=$? $(date -u +%FT%TZ)"
  "$PY" - "$OUT/s$S" "$S" <<'PYEOF'
import json, pathlib, sys, importlib.util
spec = importlib.util.spec_from_file_location(
    "sg", "/home/ttuser/.coworker/wt/cov-ladder-p150a-p3/scripts/gpu_vs_tt/gpu5_accuracy_gate.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pathlib.Path(sys.argv[1])
cifs = sorted(d.rglob("*.cif"))
if not cifs:
    print("  steps=%s NO CIF" % sys.argv[2]); raise SystemExit
r = m.gate(cifs[0], None, None)
rt = None
for f in sorted(d.rglob("results.json")):
    rows = json.loads(f.read_text())
    ts = [x["runtime_s"] for x in rows if x.get("status") == "ok" and x.get("runtime_s")]
    if ts: rt = max(ts)
print("  steps=%s runtime=%s n_ca=%d ca_ca_median=%s break_frac=%s clash=%s rg=%s plddt=%s ok=%s"
      % (sys.argv[2], rt, r["n_ca"], r.get("ca_ca_median_A"), r.get("chain_break_frac"),
         r.get("clash_frac"), r.get("radius_of_gyration_A"), r.get("plddt_mean"), r["pass"]))
PYEOF
done
echo "=== steps control done $(date -u +%FT%TZ)"
