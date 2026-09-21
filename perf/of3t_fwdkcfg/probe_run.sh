#!/usr/bin/env bash
# Which softmax construction sites a shipped fold reaches, in what dtype, with what config.
# One fold per model on the granted card, probe on PYTHONPATH ahead of the tree.
#
# Two things this had to get right before its zeros meant anything:
#   * the out_dir is UNIQUE PER RUN. `tt_bio.main predict` reuses an existing result directory
#     and returns in ~6 s without touching the device, and a probe reporting zero calls on a
#     fold that never ran reads exactly like a site nothing reaches.
#   * the fold runs in a CHILD process. The parent's dump is a real dump of a process that did
#     no model work, so every probe file is collected, not the last line of the log.
set -u
WT=/home/ttuser/.coworker/wt/of3t-fwdkcfg
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$WT/perf/of3t_fwdkcfg"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/of3t-fwdkcfg-probe-XXXXXX")
mkdir -p "$WORK"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-fwdkcfg
export PYTHONPATH="$OUT/sitecustomize_probe:$WT"
export TT_BIO_SOFTMAX_PROBE_OUT="$WORK"
for m in "$@"; do
  echo "=== $m start $(date -u +%FT%TZ) ==="
  mkdir -p "$WORK/$m"
  TT_BIO_SOFTMAX_PROBE_OUT="$WORK/$m" "$PY" -m tt_bio.main predict \
      "$WT/perf/size512/fixtures/cdk2x2_128.yaml" \
      --model "$m" --single_sequence --sampling_steps 6 --diffusion_samples 1 \
      --seed 0 --out_dir "$WORK/$m/out" > "$WORK/$m/fold.log" 2>&1
  echo "=== $m exit=$? $(date -u +%FT%TZ) ==="
  find "$WORK/$m/out" -name '*.cif' | sed 's/^/  cif /'
done
"$PY" "$OUT/collect_census.py" "$WORK" "$OUT/SITE_DTYPE_CENSUS.json"
rm -rf "$WORK"
