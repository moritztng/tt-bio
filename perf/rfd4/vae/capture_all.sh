#!/bin/bash
# Capture every real encoder batch the parity set uses: one upstream generate run each, on CPU, stopped at the
# encoder pre-hook. Run on pc in the UPSTREAM env /home/moritz/rfd4-ref-env/venv -- rfd4-cpu-venv has no
# `rfd4_proteina`. Each output .pt records the cif it came from.
#
#   bash perf/rfd4/vae/capture_all.sh <out_dir> [log_dir]
#
# The last four cases are the point: sm_sam and mcsa_13pk carry atomized ligand tokens, dna_7m5w and rna_pk
# nucleic ones, so they exercise the element/charge embedding and the per-token-type logit masking a
# protein-only batch leaves untested. The rna inputs are .cif.gz; every other benchmark file is .cif.zst.
set -u
OUT=${1:?out dir}; LOG=${2:-/tmp/rfd4vae_run}
B=/home/moritz/rfd4-upstream/RFD4-Proteina/data/benchmarks
mkdir -p "$OUT" "$LOG"
cd "$(dirname "$0")/../../.." || exit 1
. /home/moritz/rfd4-ref-env/venv/bin/activate
cap() {  # <name> <cif>
  timeout 1800 python -I perf/rfd4/vae/capture_real.py --cif "$2" --out "$OUT/$1.pt" > "$LOG/cap_$1.log" 2>&1
  echo "$1 rc=$?"
}
cap monomer_short  "$B/monomer/monomer-short.cif.zst"
cap monomer_medium "$B/monomer/monomer-medium.cif.zst"
cap monomer_long   "$B/monomer/monomer-long.cif.zst"
cap ppi_pd1        "$B/ppi/ppi_all_atom/01_PD1.cif.zst"
cap motif_2kl8     "$B/motif_scaffolding/motif_data_aa/2kl8_aa.cif.zst"
cap sm_sam         "$B/small_molecule_binder/SAM_hbonds.cif.zst"
cap dna_7m5w       "$B/dna_binder/dna_binder_denovo_val/7m5w_structure_context.cif.zst"
cap mcsa_13pk      "$B/mcsa/mcsa_37_fixed_ligand/M0040_13pk.cif.zst"
cap rna_pk         "$B/rna/pseudoknot/Q20.cif.gz"
