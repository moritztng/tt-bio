#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 5 (workstream abag-xm-deepn-saturation-fullpanel).
#
# N=256 RUNG for protenix-v2 + esmfold2 -- the two models whose deep ladder was gated until
# the 2026-08-04 same-seed pairing root-cause licensed them on galaxy (a673a4d8). 4 chunks
# x 64 samples on the seed-nested ladder (chunk j seed = base+1000*j), so chunk 0 IS the
# N=64 fold:
#   protenix-v2 163 targets x 4 chunks, seeds 30000-33000, mps 5->1 narrowing
#   esmfold2    163 targets x 4 chunks, seeds 50000-53000, single-seq auto chunking
#     (never pass msa flags or mps -- the campaign measures the no-MSA regime)
#   both exclude 9j4c (documented WH DRAM capacity exclusion).
#
# SKIP-AND-LINK (same binding rule as p28): chunk-0 slots whose N=64 source provably
# matches -- same seed block by construction, engine tree unchanged since p28 launch
# (mtime gate below), p28 panel record rc=0 with 64/64 distinct CIFs, on-disk re-verify --
# are HARDLINKED from $PREV into this rung's dirs instead of re-folded, with schema-exact
# rung-256 records carrying the SOURCE fold's real seconds/mps/umd (never double-counted)
# plus a reused_chunks.jsonl provenance line and a pre-created claim. The 15+15 pilot
# targets have NO p28 panel record (they were PX_SKIP/ESM_SKIP there), so the link phase
# simply never links them and their chunk-0 folds fresh here -- no cross-window
# archaeology, self-healing for any panel target whose p28 fold failed. If the engine
# tree changed, the link phase disables itself and every chunk folds fresh.
#
# Task order hedges a truncated window: chunk-outer, model-interleaved, so a window that
# dies mid-rung leaves both models with the deepest UNIFORM partial pool (c0+c1 = N=128
# measured) rather than complete pools for a few targets and none for the rest.
#
# DEPLOY DISCIPLINE: single-file scp into $SRC/scripts/abag_xm/ -- never a full re-extract
# of $SRC (refreshes every mtime and trips the link gate).
#
# Launch procedure (pass-188 pattern -- NO maintenance-mode deploy): acquire
# galaxy_device_lock, stop ONLY the prod fold-worker tree (the spawn_main worker under
# tt-bio serve; web tier + SSH stay up -- NEVER touch japanfold.service or cloudflared),
# run this script detached, then arm p29_watchdog.sh by absolute path and READ ITS LOG
# for the respawn line before ending the pass (p27 lesson). Watchdog respawns the prod
# worker at P29_DONE; no service restart, landing/SSH never blip.
#
# Every attempt appends one JSON record to results.jsonl. DONE_CHECK convention: no
# literal percent strings in logs.
#
# RUNNER HARDENING (pass-185 spec; same guarded_fold as p28_fleet.sh): setsid process-
# group launch so kills reach spawn grandchildren, no-progress kill (<60s group CPU AND
# zero log growth for STALL_MIN consecutive minutes -> INT, GRACE_S, KILL; caps a hang at
# ~47min, not 6h), post-hang tt-smi -r quarantine before the slot's next fold, kill-class
# records normalized to rc=124 with real cifs (0 = hang, partial = slow) + GUARD log lines.
# Thresholds env-overridable for fixture testing (STALL_MIN/GRACE_S/CAP_S/MIN_CPU_S).
# PLUS the pass-260 binding third leg (p28 9d73 double-hang blind spot): a fold whose log
# is still ZERO BYTES after ZERO_MIN minutes is stuck in pre-banner engine init -- the
# spin class burns >=60 CPU-s/min forever with a frozen 0-byte log, so the two-leg rule
# never fires. Zero-log kills are false-positive-safe: every healthy fold on record
# writes its banner within ~2 min of launch.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
PREV=$H/p28
B=$H/p29; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
# CHIPS: space-separated chip ids to run (default 0..NCHIP-1). Escape hatch only --
# p27's wedge chips were refuted as host-kmd poison, cured by kmd reload (p28 header).
CHIPS=${CHIPS:-$(seq -s' ' 0 $((NCHIP-1)))}
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache
# 9j4c WH DRAM exclusion; the 15-target pilot sets fold chunk 0 fresh (no p28 panel
# record by design) but chunks 1-3 ride the same tasks.
EXCL="9j4c"

TASKS=$B/tasks.txt
{
  for j in 0 1 2 3; do
    for y in $SRC/examples/abag_xm/*.yaml; do
      t=$(basename $y .yaml)
      skip=0
      for e in $EXCL; do [ "$t" = "$e" ] && skip=1; done
      [ $skip = 1 ] && continue
      echo "esmfold2 $t 256 $((50000+1000*j)) $j 4"
      echo "protenix-v2 $t 256 $((30000+1000*j)) $j 4"
    done
  done
} > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $(wc -w <<<"$CHIPS") [$CHIPS]"

# ---- link phase: hardlink provably-identical chunk-0 slots from $PREV (N=64 panel) ----
$PY_SYS - "$PREV" "$B" "$SRC" <<'PY'
import hashlib, json, pathlib, subprocess, sys, time
prev, B, src = (pathlib.Path(x) for x in sys.argv[1:4])
MD = {"protenix-v2": "protenix", "esmfold2": "esmfold2"}
manifest = {"commit_equal": False, "linked": 0, "refold": [], "reason": None}

marker = prev / "tasks.txt"
rj = prev / "results.jsonl"
if not marker.exists() or not rj.exists():
    manifest["reason"] = "no previous window"; print("LINK 0: no previous window")
    (B / "link_manifest.json").write_text(json.dumps(manifest) + "\n"); sys.exit(0)
mt = marker.stat().st_mtime
newer = [str(p.relative_to(src)) for p in (src / "tt_bio").rglob("*.py")
         if p.stat().st_mtime > mt]
if newer:
    manifest["reason"] = f"engine tree changed: {newer[:5]}"
    print(f"LINK 0: engine tree changed since previous window: {newer[:5]}")
    (B / "link_manifest.json").write_text(json.dumps(manifest) + "\n"); sys.exit(0)

fp = {str(p.relative_to(src)): hashlib.md5(p.read_bytes()).hexdigest()
      for p in sorted((src / "tt_bio").rglob("*.py"))}
manifest["engine_fp"] = fp
manifest["commit_equal"] = True

# last-attempt-wins ok records at rung 64 (the N=64 panel folds), unchunked
ok = {}
for line in rj.read_text().splitlines():
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("rung") != 64 or r.get("model") not in MD:
        continue
    key = (r["model"], r["target"])
    if r.get("rc") == 0 and r.get("cifs", 0) == 64 and r.get("distinct", 0) == 64:
        ok[key] = r
    else:
        ok.pop(key, None)

idx = {}
for i, line in enumerate((B / "tasks.txt").read_text().splitlines(), 1):
    m, t, rung, seed, c, k = line.split()
    idx[(m, t, int(c))] = i

done = set()
bf = B / "results.jsonl"
if bf.exists():
    for line in bf.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("rung") == 256 and r.get("rc") == 0:
            done.add((r["model"], r["target"], r.get("chunk")))

def verify(d, t):
    """results.json ok + 64 CIFs + md5-distinct count (the binding reuse verify)."""
    rs = list(d.glob(f"*results_{t}"))
    if len(rs) != 1:
        return 0
    try:
        rec = json.loads((rs[0] / "results.json").read_text())[0]
        if rec.get("status") != "ok" or len(rec.get("all_runs") or []) != 64:
            return 0
    except Exception:
        return 0
    cifs = list((rs[0] / "structures").glob("*.cif"))
    if len(cifs) != 64:
        return 0
    return len({hashlib.md5(p.read_bytes()).hexdigest() for p in cifs})

now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
linked, refold = 0, []
with open(B / "results.jsonl", "a") as rf, open(B / "reused_chunks.jsonl", "a") as pf:
    for (m, t), r in sorted(ok.items()):
        if (m, t, 0) in done:
            linked += 1
            continue
        mdir = MD[m]
        srcdir = prev / mdir / t
        n_distinct = verify(srcdir, t)
        if n_distinct != 64:
            refold.append(f"{m}/{t}_c0")
            continue
        dst = B / mdir / f"{t}_c0"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-al", str(srcdir), str(dst)], check=True)
        rf.write(json.dumps({"model": m, "target": t, "rung": 256, "seed": r["seed"],
                             "chunk": 0, "chunks": 4, "mps": str(r["mps"]),
                             "umd": r["umd"], "rc": 0, "seconds": r["seconds"],
                             "cifs": 64, "distinct": n_distinct, "oom": 0},
                            separators=(",", ":")) + "\n")
        pf.write(json.dumps({"model": m, "target": t, "rung": 256, "chunk": 0,
                             "seed": r["seed"], "source_dir": str(srcdir),
                             "source_window": prev.name,
                             "source_seconds": r["seconds"],
                             "source_mps": str(r["mps"]),
                             "tree_gate": f"tt_bio mtimes < {prev.name}/tasks.txt",
                             "reused_at": now,
                             "claim_idx": idx.get((m, t, 0))}) + "\n")
        i = idx.get((m, t, 0))
        if i:
            (B / "claims" / str(i)).mkdir(exist_ok=True)
        linked += 1
manifest["linked"] = linked
manifest["refold"] = sorted(refold)
(B / "link_manifest.json").write_text(json.dumps(manifest) + "\n")
print(f"LINK {linked} chunk-0 slots hardlinked from {prev.name} N=64 panel; "
      f"{len(refold)} chunk-0 slots fold fresh")
PY

group_cpu() { # <pgid> -> total CPU seconds of every process in the group
  ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'
}

guarded_fold() { # <logfile> <chip> <cmd...> -- setsid launch + stall/cap group kills + quarantine
  local log=$1 u=$2; shift 2
  local stall_min=${STALL_MIN:-45} cap_s=${CAP_S:-21600} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-60}
  local zero_min=${ZERO_MIN:-30}  # pass-260 third leg: 0-byte log for this long = pre-banner spin hang
  local poll_s=${POLL_S:-60}   # one stall unit per poll; 60s default => stall_min ~ minutes
  setsid env TT_VISIBLE_DEVICES=$u "$@" > "$log" 2>&1 &
  local pid=$! t0=$(date +%s) last_cpu=-1 last_size=-1 stall=0 zero=0 killrc=0 g=0
  while kill -0 $pid 2>/dev/null; do
    sleep $poll_s
    kill -0 $pid 2>/dev/null || break
    local cpu=$(group_cpu $pid) size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" = "0" ]; then zero=$((zero+1)); else zero=0; fi
    if [ "$last_cpu" -ge 0 ]; then
      if [ $((cpu - last_cpu)) -ge $min_cpu ] || [ "$size" != "$last_size" ]; then
        stall=0
      else
        stall=$((stall+1))
      fi
    fi
    last_cpu=$cpu; last_size=$size
    if [ $zero -ge $zero_min ]; then
      echo "$(date -u +%FT%TZ) GUARD: zero-log kill (${zero}m at 0 bytes, pre-banner spin class) -> INT pgid $pid" >> "$log"
      kill -INT -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt $grace_s ]; do sleep 2; g=$((g+2)); done
      if kill -0 $pid 2>/dev/null; then
        echo "$(date -u +%FT%TZ) GUARD: INT grace expired -> KILL pgid $pid" >> "$log"
        kill -KILL -- -$pid 2>/dev/null
      fi
      killrc=124; break
    fi
    if [ $stall -ge $stall_min ]; then
      echo "$(date -u +%FT%TZ) GUARD: no-progress kill (${stall}m without cpu/log growth) -> INT pgid $pid" >> "$log"
      kill -INT -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt $grace_s ]; do sleep 2; g=$((g+2)); done
      if kill -0 $pid 2>/dev/null; then
        echo "$(date -u +%FT%TZ) GUARD: INT grace expired -> KILL pgid $pid" >> "$log"
        kill -KILL -- -$pid 2>/dev/null
      fi
      killrc=124; break
    fi
    if [ $(( $(date +%s) - t0 )) -ge $cap_s ]; then
      echo "$(date -u +%FT%TZ) GUARD: ${cap_s}s cap -> TERM pgid $pid" >> "$log"
      kill -TERM -- -$pid 2>/dev/null
      g=0; while kill -0 $pid 2>/dev/null && [ $g -lt 60 ]; do sleep 2; g=$((g+2)); done
      kill -0 $pid 2>/dev/null && kill -KILL -- -$pid 2>/dev/null
      killrc=124; break
    fi
  done
  wait $pid 2>/dev/null; local rc=$?
  if [ $killrc -ne 0 ]; then
    rc=$killrc
    # -r takes a bare UMD logical ID, the same namespace as TT_VISIBLE_DEVICES above.
    # /dev/tenstorrent/$u would reset a different chip: kernel node order is not BDF
    # order (node 0 is c1:00.0, UMD 0 is 01:00.0 on the galaxy).
    sudo -n tt-smi -r $u >> "$log" 2>&1 \
      || echo "$(date -u +%FT%TZ) GUARD: tt-smi reset failed on dev $u" >> "$log"
    sleep 10
  fi
  return $rc
}

record() {  # record <model> <target> <rung> <seed> <chunk> <chunks> <mps> <chip> <rc> <secs> <cifs> <distinct> <oom>
  printf '{"model":"%s","target":"%s","rung":%s,"seed":%s,"chunk":%s,"chunks":%s,"mps":"%s","umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" >> $B/results.jsonl
}

count_structs() { # <dir> -> echoes "n distinct"
  local d=$1 n distinct
  n=$(ls $d/structures/*.cif 2>/dev/null | wc -l)
  distinct=$(md5sum $d/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  echo "${n:-0} ${distinct:-0}"
}

outbase() { # <model> <target> <chunk> <chunks> -> per-task out dir under $B
  local m=$1 t=$2 c=$3 k=$4
  if [ "$k" -gt 1 ]; then echo "$B/$m/${t}_c$c"; else echo "$B/$m/$t"; fi
}

fold_px() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps 5, narrow 5->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d nd ob
  ob=$(outbase protenix $t $c $k)
  for mps in 5 1; do
    s=$(date +%s)
    guarded_fold $B/protenix_${t}_c${c}_mps$mps.log $u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model protenix-v2 --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/protenix_${t}_c${c}_mps$mps.log 2>/dev/null)
    record protenix-v2 $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_esm() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- single-seq, auto chunking
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d nd ob
  ob=$(outbase esmfold2 $t $c $k)
  s=$(date +%s)
  guarded_fold $B/esmfold2_${t}_c$c.log $u $PY_VENV -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model esmfold2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --recycling_steps 10 --sampling_steps 100 --seed $seed \
    --host_threads 2
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/esmfold2_${t}_c$c.log 2>/dev/null)
  record esmfold2 $t $rung $seed $c $k auto $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold() { # <model> <target> <rung> <seed> <chunk> <chunks> <chip>
  case "$1" in
    protenix-v2)  fold_px  "$2" "$3" "$4" "$5" "$6" "$7";;
    esmfold2)     fold_esm "$2" "$3" "$4" "$5" "$6" "$7";;
    *)            echo "SKIP: $1 not in this window" >> $B/slots.log;;
  esac
}

slot() {
  local chip=$1 idx n model t rung seed c k
  n=$(wc -l < $TASKS)
  for ((idx=1; idx<=n; idx++)); do
    mkdir $B/claims/$idx 2>/dev/null || continue
    read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
    ( cd $SRC && fold "$model" "$t" "$rung" "$seed" "$c" "$k" "$chip" )
  done
  echo "slot $chip done" >> $B/slots.log
}

for c in $CHIPS; do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo P29_DONE >> $B/results.jsonl
