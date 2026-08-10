#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 6 (workstream abag-xm-deepn-saturation-fullpanel).
#
# CONTINGENT WINDOW -- launch ONLY if the bz stop rule does NOT fire at N=512 (i.e. the
# 256->512 oracle gain still clears the seed-noise floor after the p28 analysis). If bz
# knees at 512 this script never runs; "no knee by N=1024" is a complete terminal answer.
#
# N=1024 RUNG for boltz2 only (opendde is TERMINAL at N*=64 per the checkpoint-1 analysis;
# px/esm deep rungs are p29/p31 decisions): 164 targets x 16 chunks x 64 samples, seeds
# 40000+1000*j for j=0..15. The ladder is seed-nested: chunk j seed = base+1000*j, so
# chunks 0-7 ARE the N=512 chunks and the N=1024 pool nests the N=512 pool exactly.
# Chunking is RAM-forced (boltz2 ~0.22 GB/sample host RAM; 64-sample chunks cap a fold at
# ~15 GB so 32 concurrent folds fit the galaxy's 566 GB).
#
# SKIP-AND-LINK (same binding rule as p28): chunks 0-7 whose N=512 source chunk provably
# matches -- same seed block by construction, engine tree unchanged since p28 launch
# (mtime gate below), p28 record rc=0 with 64/64 distinct CIFs, on-disk re-verify -- are
# HARDLINKED from $PREV into this rung's dirs instead of re-folded. The gate chains
# transitively (p28's manifest attested tree==p27's; this window's attests tree==p28's).
# If the engine tree changed, the link phase disables itself and every chunk folds fresh.
# COST-REFIT RULE (unchanged): dedupe seconds by (model,target,rung,chunk) across windows;
# linked records carry seconds paid in earlier windows, never double-counted.
# N=512 POOL REPAIR: a chunk j<8 folded FRESH here (its p28 source failed) also satisfies
# the n512 pool slot (identical seed/config); harvest applies that cross-pool link only
# when link_manifest.json says commit_equal=true (harvest-side step, manifest-gated).
#
# DEPLOY DISCIPLINE: ship this script to the galaxy by single-file scp into $SRC/scripts/
# abag_xm/ -- never a full `git archive` re-extract of $SRC, which refreshes every mtime
# and trips the link gate (safe, but re-folds ~400 card-h).
#
# Launch procedure (pass-188 pattern -- NO maintenance-mode deploy): acquire
# galaxy_device_lock, stop ONLY the prod fold-worker tree (the spawn_main worker under
# tt-bio serve; web tier + SSH stay up -- NEVER touch japanfold.service or cloudflared),
# run this script detached, then arm p30_watchdog.sh by absolute path and READ ITS LOG
# for the respawn line before ending the pass. Watchdog respawns the prod worker at
# P30_DONE; no service restart, landing/SSH never blip.
# LAUNCH GATE: verify the first ~5 bz folds complete rc=0 under guarded_fold (~20-30min)
# with one GUARD-free log each before trusting the window bulk.
#
# Every attempt appends one JSON record to results.jsonl. DONE_CHECK convention: no
# literal percent strings in logs.
#
# RUNNER HARDENING (pass-185 spec, inherited unchanged from p28/p29): setsid process-group
# launch, no-progress kill (STALL_MIN without CPU/log growth -> INT, grace, KILL; caps a
# hang at ~47min), post-hang tt-smi -r quarantine, rc=124 kill-class records with real
# cifs. Wedge forensics (pass-212 methodology): health = spawn-grandchild CPU accrual;
# main-pid freeze and cold-compile 0-byte logs never discriminate. Recovery ladder for a
# host-wide device-open epidemic: free handles -> rmmod+modprobe tenstorrent -> glx_reset
# -> grandchild-CPU canary; host reboot LAST (pass-212 cure).
# Thresholds are env-overridable for fixture testing (STALL_MIN/GRACE_S/CAP_S/MIN_CPU_S).
#
# CHIPS: space-separated chip ids to run (default 0..NCHIP-1). Escape hatch only --
# p27's chips-4/16/21/22 wedge class was refuted as host-kmd poison and cured by kmd
# reload (state doc pass 212/214); all 32 chips probe healthy. Exclude via CHIPS only
# if a fresh wedge class appears (detect by spawn-grandchild CPU stall, never 0B logs).
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
PREV=$H/p28
B=$H/p30; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
CHIPS=${CHIPS:-$(seq -s' ' 0 $((NCHIP-1)))}
PY_SYS=/usr/bin/python3.10
MSA=$H/abag_xm/msa_cache

TASKS=$B/tasks.txt
{
  for y in $SRC/examples/abag_xm/*.yaml; do
    t=$(basename $y .yaml)
    for j in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      echo "boltz2 $t 1024 $((40000+1000*j)) $j 16"
    done
  done
} > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $(wc -w <<<"$CHIPS") [$CHIPS]"

# ---- link phase: hardlink provably-identical chunks 0-7 from $PREV (rung 512) ----
$PY_SYS - "$PREV" "$B" "$SRC" <<'PY'
import hashlib, json, pathlib, subprocess, sys, time
prev, B, src = (pathlib.Path(x) for x in sys.argv[1:4])
MD = {"boltz2": "boltz2"}
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

# engine fingerprints for FUTURE windows' content-compare (p31+)
fp = {str(p.relative_to(src)): hashlib.md5(p.read_bytes()).hexdigest()
      for p in sorted((src / "tt_bio").rglob("*.py"))}
manifest["engine_fp"] = fp
manifest["commit_equal"] = True

# last-attempt-wins ok records at rung 512, chunks 0-7
ok = {}
for line in rj.read_text().splitlines():
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("rung") != 512 or r.get("model") not in MD:
        continue
    key = (r["model"], r["target"], r.get("chunk"))
    if r.get("rc") == 0 and r.get("cifs", 0) == 64 and r.get("distinct", 0) == 64:
        ok[key] = r
    else:
        ok.pop(key, None)

# task index map for claim pre-creation
idx = {}
for i, line in enumerate((B / "tasks.txt").read_text().splitlines(), 1):
    m, t, rung, seed, c, k = line.split()
    idx[(m, t, int(c))] = i

# idempotency: on relaunch, skip chunks already recorded at rung 1024
done = set()
bf = B / "results.jsonl"
if bf.exists():
    for line in bf.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("rung") == 1024 and r.get("rc") == 0:
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
    for (m, t, c), r in sorted(ok.items()):
        if (m, t, c) in done:
            linked += 1
            continue
        mdir = MD[m]
        srcdir = prev / mdir / f"{t}_c{c}"
        n_distinct = verify(srcdir, t)
        if n_distinct != 64:
            refold.append(f"{m}/{t}_c{c}")
            continue
        dst = B / mdir / f"{t}_c{c}"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-al", str(srcdir), str(dst)], check=True)
        rf.write(json.dumps({"model": m, "target": t, "rung": 1024, "seed": r["seed"],
                             "chunk": c, "chunks": 16, "mps": str(r["mps"]),
                             "umd": r["umd"], "rc": 0, "seconds": r["seconds"],
                             "cifs": 64, "distinct": n_distinct, "oom": 0},
                            separators=(",", ":")) + "\n")
        pf.write(json.dumps({"model": m, "target": t, "rung": 1024, "chunk": c,
                             "seed": r["seed"], "source_dir": str(srcdir),
                             "source_window": prev.name,
                             "source_seconds": r["seconds"],
                             "source_mps": str(r["mps"]),
                             "tree_gate": f"tt_bio mtimes < {prev.name}/tasks.txt",
                             "reused_at": now,
                             "claim_idx": idx.get((m, t, c))}) + "\n")
        i = idx.get((m, t, c))
        if i:
            (B / "claims" / str(i)).mkdir(exist_ok=True)
        linked += 1
manifest["linked"] = linked
manifest["refold"] = sorted(refold)
(B / "link_manifest.json").write_text(json.dumps(manifest) + "\n")
print(f"LINK {linked} chunks hardlinked from {prev.name}; {len(refold)} chunk-0..7 slots fold fresh")
PY

group_cpu() { # <pgid> -> total CPU seconds of every process in the group
  ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'
}

guarded_fold() { # <logfile> <chip> <cmd...> -- setsid launch + stall/cap group kills + quarantine
  local log=$1 u=$2; shift 2
  local stall_min=${STALL_MIN:-45} cap_s=${CAP_S:-21600} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-60}
  local poll_s=${POLL_S:-60}   # one stall unit per poll; 60s default => stall_min ~ minutes
  setsid env TT_VISIBLE_DEVICES=$u "$@" > "$log" 2>&1 &
  local pid=$! t0=$(date +%s) last_cpu=-1 last_size=-1 stall=0 killrc=0 g=0
  while kill -0 $pid 2>/dev/null; do
    sleep $poll_s
    kill -0 $pid 2>/dev/null || break
    local cpu=$(group_cpu $pid) size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$last_cpu" -ge 0 ]; then
      if [ $((cpu - last_cpu)) -ge $min_cpu ] || [ "$size" != "$last_size" ]; then
        stall=0
      else
        stall=$((stall+1))
      fi
    fi
    last_cpu=$cpu; last_size=$size
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

fold_bz() { # <target> <rung> <seed> <chunk> <chunks> <chip>
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d nd ob
  ob=$(outbase boltz2 $t $c $k)
  s=$(date +%s)
  guarded_fold $B/boltz2_${t}_c$c.log $u $PY_SYS -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model boltz2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --max_parallel_samples 1 --seed $seed --host_threads 2 \
    --msa_dir $MSA --msa_cache_only
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/boltz2_${t}_c$c.log 2>/dev/null)
  record boltz2 $t $rung $seed $c $k 1 $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

slot() {
  local chip=$1 idx n model t rung seed c k
  n=$(wc -l < $TASKS)
  for ((idx=1; idx<=n; idx++)); do
    mkdir $B/claims/$idx 2>/dev/null || continue
    read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
    ( cd $SRC && fold_bz "$t" "$rung" "$seed" "$c" "$k" "$chip" )
  done
  echo "slot $chip done" >> $B/slots.log
}

for c in $CHIPS; do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo P30_DONE >> $B/results.jsonl
