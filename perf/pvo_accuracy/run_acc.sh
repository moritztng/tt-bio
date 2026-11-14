#!/usr/bin/env bash
# pvo-accuracy's GPU arms, run on pvo-gpu's box after the timing arms (state/pvo/GPU.md, modes state/pvo/MODES.md).
# OpenDDE and Mol-1, exact and fast, over the 11-complex accuracy set at each model's own defaults, same seeds in
# both modes: fast(s) vs exact(s) isolates the mode, exact(s) vs exact(s') is the seed floor.
# Interleaved (101-103, then 104-105) so a session cut short still leaves paired data, never one complete mode.
# Resumable: an arm whose rc is 0 is skipped on a re-run.
#   bash run_acc.sh [deadline_epoch]     inputs /root/kc/acc (make_acc.py), outputs /root/kc/pvo/acc/<model>/<arm>
# Clocks come from box_pvo.sh's box-wide smi.csv, cut to each arm's t_start/t_end; no second logger.
set -u
A=${ACC_DIR:-/root/kc/acc}; B=${PVO_DIR:-/root/kc/pvo}/acc; DEADLINE=${1:-$(( $(date +%s) + 86400 ))}
mkdir -p "$B"
say(){ echo "$(date -u +%FT%TZ) run_acc: $*"; }

# arm <model> <mode> <seeds> <env> <flags...>; ARM_IN overrides the input JSON, opendde-abag runs OpenDDE's venv
arm(){ local m=$1 mode=$2 seeds=$3 e=$4; shift 4
  local d=$B/$m/${mode}_${seeds//,/-}
  [ "$(cat "$d/rc" 2>/dev/null)" = 0 ] && return 0
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then say "deadline: skip $m $mode $seeds"; return 1; fi
  rm -rf "$d"; mkdir -p "$d"; say "arm $m $mode $seeds"
  date -u +%FT%T.%NZ > "$d/t_start"
  local bin=/opt/${m%-abag}-venv/bin cmd=opendde; [ "$m" = mol1 ] && cmd=mol
  (cd /tmp && env $e PATH=$bin:$PATH timeout "${ACC_TIMEOUT:-18000}" "$bin/$cmd" pred \
     -i "${ARM_IN:-$A/acc.json}" -o "$d/pred" --seeds "$seeds" --use_msa true "$@") > "$d/stdout.log" 2> "$d/stderr.log"
  local rc=$?
  date -u +%FT%T.%NZ > "$d/t_end"; echo $rc > "$d/rc"
  say "ARM-DONE $m $mode $seeds rc=$rc $(find "$d/pred" -name '*.cif' 2>/dev/null | wc -l) cifs"
  return $rc; }

# OpenDDE: 10 cycles, 200 steps, 5 samples, bf16 (MODES.md; = the CLI defaults, passed explicitly so the log says so).
# Checkpoint: the general opendde.pt, the box's default. opendde_abag.pt is a second checkpoint, not a mode.
ODDE="--cycle 10 --step 200 --sample 5 --dtype bf16"
ODDE_FAST="--enable_tf32 true --trimul_kernel cuequivariance --triatt_kernel cuequivariance"
# opendde_abag.pt, the antibody-antigen checkpoint (README.md:148), on the 5 antibody/VHH complexes only: same modes,
# same seeds 101-103, only --load_checkpoint_path differs. Staged and sha-checked here so the arm needs nothing else.
ABAG_CK=${OPENDDE_ROOT_DIR:-/root/oddedata}/checkpoint/opendde_abag.pt
ABAG_SHA=5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d
abag(){ local mode=$1; shift
  [ -f "$ABAG_CK" ] || wget -q https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt -O "$ABAG_CK"
  echo "$ABAG_SHA  $ABAG_CK" | sha256sum -c --quiet || { say "abag checkpoint sha mismatch"; rm -f "$ABAG_CK"; return 1; }
  /opt/opendde-venv/bin/python -c 'import json,sys; j=json.load(open(sys.argv[1])); json.dump([x for x in j if x["name"].split("_")[0].upper() in sys.argv[3:]],open(sys.argv[2],"w"),indent=1)' \
    "$A/acc.json" "$A/acc_abag.json" 9TH6 9W3L 9TY2 9W89 9W8A
  ARM_IN=$A/acc_abag.json arm opendde-abag "$mode" 101,102,103 "$@" --load_checkpoint_path "$ABAG_CK"; }

# Mol-1: 20 cycles, 200 steps, 5 samples, bf16; exact keeps chunk 1, TF32 off, auto kernels.
MOL="--cycle 20 --step 200 --sample 5 --dtype bf16"
# Fast = the kernel pair box_pvo.sh ranked fastest on THIS card (the B200 ranking does not carry; MODES.md caveat).
case "$(cat "${PVO_DIR:-/root/kc/pvo}/mol1/FAST-ARM" 2>/dev/null)" in
  cueq-cueq-c5-tf32)  MOL_K="--trimul_kernel cuequivariance --triatt_kernel cuequivariance";;
  fpf-triatt-c5-tf32) MOL_K="--trimul_kernel flashpairformer --triatt_kernel triattention";;
  *)                  MOL_K="--trimul_kernel cuequivariance --triatt_kernel triattention";;
esac
MOL_FAST="$MOL_K --enable_tf32 true --infer_setting.sample_diffusion_chunk_size 5"

for seeds in 101,102,103 104,105; do
  arm opendde exact "$seeds" LAYERNORM_TYPE=torch           $ODDE
  arm opendde fast  "$seeds" LAYERNORM_TYPE=fast_layernorm  $ODDE $ODDE_FAST
  if [ "$seeds" = 101,102,103 ]; then
    abag exact LAYERNORM_TYPE=torch          $ODDE
    abag fast  LAYERNORM_TYPE=fast_layernorm $ODDE $ODDE_FAST
  fi
  arm mol1    exact "$seeds" X=1 $MOL
  arm mol1    fast  "$seeds" X=1 $MOL $MOL_FAST
done
say "ACC-DONE $(find "$B" -name '*.cif' | wc -l) cifs under $B"
