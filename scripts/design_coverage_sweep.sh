#!/usr/bin/env bash
# Regenerate the design/embedding feature-coverage evidence log.
#
# Reruns the three host-side probes and recomputes every checksum from the artifacts
# the device runs left in scratch/. Run it from the worktree root, on the card you were
# granted. Device runs themselves are not repeated here; the commands that produced the
# artifacts are echoed above each checksum block so they can be reproduced.
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WT"
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
export PYTHONPATH="$WT"
export FOLDSEEK_BIN=${FOLDSEEK_BIN:-/home/ttuser/scratch/foldseek/foldseek/bin/foldseek}

quiet() { grep -vE "nanobind|leaked|DEBUG|Config\{|refleaks|skipped remain|^ - "; }
coords() { grep -E "^ATOM" "$1" | awk '{print $11, $12, $13}' | md5sum | cut -c1-16; }

echo "=============================================================="
echo "design/embedding feature coverage — $(date -u +%FT%TZ) — $(hostname)"
echo "git $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "=============================================================="

echo
echo "### RFD3: which InputSpecification fields reach the model (host side)"
$PY scripts/rfd3_spec_field_coverage.py 2>&1 | quiet

echo
echo "### RFD3: hotspots on vs off, on device"
echo "# tt-bio design scratch/hs_{off,on}.json --model rfd3 --from_pdb --num_timesteps 4 --seed 7"
echo "# negative control: the same spec at --seed 99"
for f in scratch/out_hs_off scratch/out_hs_on scratch/out_neg_seed; do
    [ -f "$f/gate-binder.cif" ] && printf "  %-24s %s\n" "$f" "$(md5sum "$f/gate-binder.cif" | cut -c1-32)"
done

echo
echo "### RFD3: num_designs, seed, batch_size"
echo "# --num_designs 3 --seed 7 twice, then the same at --batch_size 1"
md5sum scratch/rf_n3/*.cif scratch/rf_n3b/*.cif scratch/rf_b1/*.cif 2>/dev/null | sed 's/^/  /'

echo
echo "### PXDesign: which YAML keys reach the model (host side)"
$PY scripts/pxdesign_input_coverage.py 2>&1 | quiet

echo
echo "### PXDesign: hotspots on device (coordinate checksums, filename excluded)"
echo "# tt-bio design scratch/px/{nohot,hot,badhot}.yaml --model pxdesign --n_step 20 --seed 5"
for f in scratch/px/o_nohot/nohot.cif scratch/px/o_hot/hot.cif scratch/px/o_badhot/badhot.cif; do
    [ -f "$f" ] && printf "  %-34s %s\n" "$f" "$(coords "$f")"
done
echo "# --num_designs 1 vs design 0 of --num_designs 3, same --seed 5"
for f in scratch/px/o_hot/hot.cif scratch/px/o_n3/hot_0.cif scratch/px/o_n3b/hot_0.cif; do
    [ -f "$f" ] && printf "  %-34s %s\n" "$f" "$(coords "$f")"
done

echo
echo "### ESMC + SaProt: assertions on the written artifacts"
$PY scripts/embed_output_checks.py scratch 2>&1 | quiet

echo
echo "### ESMC: is the batch drift padded-token leakage or reduction order?"
echo "# L37 alone, L37 batched with a 200-mer, L37 batched with a 600-mer"
$PY - <<'PY' 2>&1 | quiet
import numpy as np
ref = np.load("scratch/pad_out_alone/L37.npz")["per_residue"].astype(np.float64)
b1 = np.load("scratch/emb_b1/L37.npz")["per_residue"].astype(np.float64)
print("  L37 alone vs --batch_size 1        maxabs %.3e" % np.abs(ref - b1).max())
for tag in ("200", "600"):
    x = np.load("scratch/pad_out_%s/L37.npz" % tag)["per_residue"].astype(np.float64)
    print("  L37 padded to a %s-mer partner   maxabs %.3e  PCC %.8f"
          % (tag, np.abs(x - ref).max(), np.corrcoef(x.ravel(), ref.ravel())[0, 1]))
PY

echo
echo "### SaProt: 3Di alignment against a structure with unresolved residues"
$PY scripts/saprot_structure_alignment.py 2>&1 | quiet

echo
echo "### SaProt: manifest written without --logits"
$PY -c "import json;m=json.load(open('scratch/sap_seq/manifest.json'));print('  logits flag:',m['logits']);print('  shapes.logits:',m['shapes']['logits'])"

echo
echo "### BoltzGen: what the spec front door refuses"
B() { $PY -m tt_bio.main gen "$@" 2>&1 | quiet; }
echo "# an unknown YAML key"
B check scratch/bg_unknown.yaml | grep -iE "invalid keys" | sed 's/^/  /'
echo "# a Boltz-2 pocket constraint"
B check scratch/bg_pocket.yaml | grep -iE "does not support" | sed 's/^/  /'
echo "# binding_types past the end of the chain"
B check scratch/bg_binding_bad.yaml | grep -iE "higher than the length" | sed 's/^/  /'
echo "# binding_types in range: does it move the design spec?"
for f in bgf_ok.cif bgf_binding.cif; do
    [ -f "$f" ] && printf "  %-20s %s\n" "$f" "$(md5sum "$f" | cut -c1-32)"
done

echo
echo "### BoltzGen: --config key checking"
echo "# a real key lands"
grep -n "^sampling_steps" scratch/bg_cfg/config/design.yaml 2>/dev/null | sed 's/^/  bg_cfg  /'
echo "# a bogus key lands too, and the run succeeded"
grep -n "not_a_real_key" scratch/bg_bogus/config/design.yaml 2>/dev/null | sed 's/^/  bg_bogus /'
echo "# a bogus step name does not"
$PY -m tt_bio.main design scratch/bgf_binding.yaml --model boltzgen --steps design \
    --num_designs 1 --devices 2 --config nosuchstep k=1 --out_dir scratch/bg_bogus2 --debug 2>&1 \
    | grep -iE "Invalid step name" | sed 's/^/  /'

echo
echo "### docs/boltzgen-design.md's own example for a step subset"
$PY -m tt_bio.main design examples/binder.yaml --model boltzgen --steps analysis filtering 2>&1 \
    | grep -iE "^Error" | sed 's/^/  /'
echo "=============================================================="
