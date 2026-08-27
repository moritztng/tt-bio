#!/bin/bash
# Parallel-range fetch for PXDesign release data.
#
# The upstream origin (pxdesign.tos-cn-beijing.volces.com) serves a single stream at
# ~20 KB/s from outside CN, which is 8 hours for the 556 MB pxdesign checkpoint. It does
# honour Range requests, so N concurrent ranges get the full pipe. aria2c is not installed
# on the QuietBoxes, hence curl. Same problem the GPU-reference task hit and solved with
# aria2c -x16.
#
# Usage: [PYTHON=/path/to/python] fetch_release_data.sh <dest_dir> [name ...]
#        (default: all checkpoints; PYTHON must import torch for the .pt integrity check)
set -u
DEST="${1:?usage: fetch_release_data.sh <dest_dir> [name ...]}"; shift || true
BASE=https://pxdesign.tos-cn-beijing.volces.com
N=16

# The names are the same ones pxdesign/utils/infer.py URL maps, so a name here is a name
# upstream recognises. The CCD pair is needed to run the upstream featurizer at its pinned
# protenix: a substituted components.cif silently drops residues that fail CCD lookup (a
# 2024-06 CCD against a later one lost 61 of the PD-L1 target's 116 residues, and the only
# symptom was a smaller token count).
declare -A URLS=(
  [pxdesign_v0.1.0]="$BASE/release_model/pxdesign_v0.1.0.pt"
  [protenix_base_default_v0.5.0]="$BASE/release_model/protenix_base_default_v0.5.0.pt"
  [protenix_mini_default_v0.5.0]="$BASE/release_model/protenix_mini_default_v0.5.0.pt"
  [protenix_mini_tmpl_v0.5.0]="$BASE/release_model/protenix_mini_tmpl_v0.5.0.pt"
  [ccd_components_file]="$BASE/release_data/components.v20240608.cif"
  [ccd_components_rdkit_mol_file]="$BASE/release_data/components.v20240608.cif.rdkit_mol.pkl"
)
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=("${!URLS[@]}")

mkdir -p "$DEST"
for name in "${NAMES[@]}"; do
  url="${URLS[$name]:?unknown name $name}"
  out="$DEST/$(basename "$url")"
  total=$(curl -sI "$url" | tr -d "\r" | awk "/^[Cc]ontent-[Ll]ength:/{print \$2}")
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$total" ]; then
    echo "[skip] $name already complete ($total bytes)"; continue
  fi
  echo "[fetch] $name  $total bytes  in $N ranges"
  chunk=$(( (total + N - 1) / N ))
  pids=()
  for ((i=0;i<N;i++)); do
    s=$(( i * chunk )); e=$(( s + chunk - 1 )); [ $e -ge $total ] && e=$(( total - 1 ))
    [ $s -gt $e ] && continue
    part=$(printf "%s.part%03d" "$out" "$i")
    want=$(( e - s + 1 ))
    # Resume granularity is one whole part, deliberately. `curl -C -` OVERRIDES `-r`: given
    # both, curl drops the range and refetches from the local size to EOF, so every one of the
    # 16 "resumed" ranges pulls the entire object. Measured 2026-08-27 on this file: 16 parts of
    # ~1.47 GB each, 29 GB written for a 1.47 GB download, and the reassembly is garbage. So no
    # -C, and a part is either already exactly its range or fetched again from scratch.
    [ -f "$part" ] && [ "$(stat -c%s "$part")" = "$want" ] && continue
    curl -s --retry 20 --retry-delay 5 --retry-all-errors -r "$s-$e" "$url" -o "$part" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  # Reassemble in NUMERIC order. Zero-padded part names are load-bearing: a bare
  # "part$i" glob sorts lexicographically, so part10..part15 concatenate before part2 and
  # the result is a byte-complete but scrambled file -- the size check below still passes.
  cat $(ls "$out".part* | sort) > "$out" && rm -f "$out".part*
  got=$(stat -c%s "$out")
  if [ "$got" != "$total" ]; then echo "[FAIL] $name got $got want $total" >&2; exit 1; fi
  # Size alone does not prove a correct reassembly, so parse the file. One check per
  # format, each of which fails on out-of-order parts rather than merely on truncation.
  case "$out" in
    *.pt)
      check='import sys,torch; torch.load(sys.argv[1], map_location="cpu", weights_only=False)'
      what="torch.load" ;;
    *.pkl)
      # Walk the whole pickle stream without needing the classes it names (rdkit).
      check='import sys,pickle
class Any:
    def __new__(cls, *a, **k): return object.__new__(cls)
    def __init__(self, *a, **k): pass
    def __setstate__(self, st): pass
class U(pickle.Unpickler):
    def find_class(self, m, n): return type(n, (Any,), {})
U(open(sys.argv[1], "rb")).load()'
      what="pickle stream" ;;
    *.cif)
      # CCD blocks are emitted in ascending id order, so a scrambled concat shows up as a
      # descending step -- which a size check cannot see.
      check='import sys
ids = [l.split("_", 1)[1].strip() for l in open(sys.argv[1]) if l.startswith("data_")]
assert len(ids) > 30000, f"only {len(ids)} CCD blocks"
assert ids == sorted(ids), "CCD block ids are out of order: parts reassembled wrongly"'
      what="CCD block order" ;;
    *) check='import sys'; what="nothing (unknown format)" ;;
  esac
  # $PYTHON, not a bare python3: the *.pt branch needs torch, and the box's system python3
  # does not have it. A missing interpreter dep made a byte-perfect 1474265486-byte download
  # report "[FAIL] ... corrupt reassembly" (2026-08-27), which is the most expensive possible
  # way to be wrong -- it invites a re-download of a file that was already correct.
  if ! "${PYTHON:-python3}" -c "$check" "$out"; then
    echo "[FAIL] $name is $got bytes but $what failed -- corrupt reassembly" >&2; exit 1
  fi
  echo "[done] $name $got bytes, $what ok"
done
echo "[all done]"
