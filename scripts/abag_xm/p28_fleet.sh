#!/bin/bash
# AbAg-XM deep-N saturation, GALAXY window 4 (workstream abag-xm-deepn-saturation-fullpanel).
#
# N=512 RUNG for the PHASE-0-licensed models, 8 chunks x 64 samples. The ladder is
# seed-nested: chunk j seed = base+1000*j, so chunks 0-3 ARE the N=256 chunks and the
# N=512 pool nests the N=256 pool exactly.
#   boltz2 164 targets x 8 chunks, seeds 40000-47000, mps 1
#   opendde 160 targets x 8 chunks, seeds 20000-27000, mps 5->2->1 narrowing
#     (excluded: 9i3p 9j4c 9ivj 9q7y -- documented WH DRAM exclusions at mps=1)
# Chunking is RAM-forced (boltz2 ~0.22 GB/sample host RAM; 64-sample chunks cap a fold at
# ~15 GB so 32 concurrent folds fit the galaxy's 566 GB). px/esm legs are NOT in this
# window: their panel rungs stay gated on the qb1 N=64 cross-hardware gate verdict.
#
# SKIP-AND-LINK (binding, Moritz 2026-08-03 "as efficient as possible, but still correct"):
# chunks 0-3 whose N=256 source chunk provably matches -- same seed block by construction,
# engine tree unchanged since p27 launch (mtime gate below), p27 record rc=0 with 64/64
# distinct CIFs, on-disk re-verify -- are HARDLINKED from $PREV into this rung's dirs
# instead of re-folded. Each linked chunk gets a schema-exact rung-512 record carrying the
# SOURCE fold's real seconds/mps/umd plus a reused_chunks.jsonl provenance line, and its
# claim is pre-created so slots skip it. If the engine tree changed, the link phase
# disables itself and every chunk folds fresh -- correct first, efficient when provable.
# COST-REFIT RULE (unchanged): dedupe seconds by (model,target,rung,chunk) across windows;
# linked records carry seconds paid in earlier windows, never double-counted.
# N=256 POOL REPAIR: a chunk j<4 folded FRESH here (its p27 source failed) also satisfies
# the n256 pool slot (identical seed/config); harvest applies that cross-pool link only
# when link_manifest.json says commit_equal=true (harvest-side step, manifest-gated).
#
# DEPLOY DISCIPLINE: ship this script to the galaxy by single-file scp into $SRC/scripts/
# abag_xm/ -- never a full `git archive` re-extract of $SRC, which refreshes every mtime
# and trips the link gate (safe, but re-folds ~600 card-h).
#
# Launch procedure (maintenance window): acquire galaxy_device_lock, maint-deploy, run
# this script detached, then arm p28_watchdog.sh by absolute path and READ ITS LOG for
# the "armed" line before ending the pass (p27 lesson).
#
# Every attempt appends one JSON record to results.jsonl. DONE_CHECK convention: no
# literal percent strings in logs.
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
PREV=$H/p27
B=$H/p28; mkdir -p $B $B/claims
NCHIP=${1:-32}
STAGGER=${2:-8}
PY_SYS=/usr/bin/python3.10
MSA=$H/abag_xm/msa_cache
OD_EXCL="9i3p 9j4c 9ivj 9q7y"

TASKS=$B/tasks.txt
{
  for y in $SRC/examples/abag_xm/*.yaml; do
    t=$(basename $y .yaml)
    for j in 0 1 2 3 4 5 6 7; do
      echo "boltz2 $t 512 $((40000+1000*j)) $j 8"
    done
    skip=0
    for e in $OD_EXCL; do [ "$t" = "$e" ] && skip=1; done
    [ $skip = 1 ] && continue
    for j in 0 1 2 3 4 5 6 7; do
      echo "opendde-abag $t 512 $((20000+1000*j)) $j 8"
    done
  done
} > $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $NCHIP"

# ---- link phase: hardlink provably-identical chunks 0-3 from $PREV (rung 256) ----
$PY_SYS - "$PREV" "$B" "$SRC" <<'PY'
import hashlib, json, pathlib, subprocess, sys, time
prev, B, src = (pathlib.Path(x) for x in sys.argv[1:4])
MD = {"boltz2": "boltz2", "opendde-abag": "opendde"}
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

# engine fingerprints for FUTURE windows' content-compare (p29+)
fp = {str(p.relative_to(src)): hashlib.md5(p.read_bytes()).hexdigest()
      for p in sorted((src / "tt_bio").rglob("*.py"))}
manifest["engine_fp"] = fp
manifest["commit_equal"] = True

# last-attempt-wins ok records at rung 256, chunks 0-3
ok = {}
for line in rj.read_text().splitlines():
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("rung") != 256 or r.get("model") not in MD:
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

# idempotency: on relaunch, skip chunks already recorded at rung 512
done = set()
bf = B / "results.jsonl"
if bf.exists():
    for line in bf.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("rung") == 512 and r.get("rc") == 0:
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
        rf.write(json.dumps({"model": m, "target": t, "rung": 512, "seed": r["seed"],
                             "chunk": c, "chunks": 8, "mps": str(r["mps"]),
                             "umd": r["umd"], "rc": 0, "seconds": r["seconds"],
                             "cifs": 64, "distinct": n_distinct, "oom": 0},
                            separators=(",", ":")) + "\n")
        pf.write(json.dumps({"model": m, "target": t, "rung": 512, "chunk": c,
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
print(f"LINK {linked} chunks hardlinked from {prev.name}; {len(refold)} chunk-0..3 slots fold fresh")
PY

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
  timeout 21600 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
    examples/abag_xm/$t.yaml --model boltz2 --out_dir $ob --override \
    --diffusion_samples $((rung/k)) --max_parallel_samples 1 --seed $seed --host_threads 2 \
    --msa_dir $MSA --msa_cache_only > $B/boltz2_${t}_c$c.log 2>&1
  rc=$?; secs=$(( $(date +%s) - s ))
  d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
  read -r _n _di <<<$(count_structs "$d")
  oom=$(grep -c 'Out of Memory' $B/boltz2_${t}_c$c.log 2>/dev/null)
  record boltz2 $t $rung $seed $c $k 1 $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
}

fold_od() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d nd ob
  ob=$(outbase opendde $t $c $k)
  for mps in 5 2 1; do
    s=$(date +%s)
    timeout 21600 env TT_VISIBLE_DEVICES=$u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model opendde-abag --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/opendde_${t}_c${c}_mps$mps.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/opendde_${t}_c${c}_mps$mps.log 2>/dev/null)
    record opendde-abag $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold() { # <model> <target> <rung> <seed> <chunk> <chunks> <chip>
  case "$1" in
    boltz2)       fold_bz  "$2" "$3" "$4" "$5" "$6" "$7";;
    opendde-abag) fold_od  "$2" "$3" "$4" "$5" "$6" "$7";;
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

for (( c=0; c<NCHIP; c++ )); do
  slot "$c" &
  sleep "$STAGGER"
done
wait
echo P28_DONE >> $B/results.jsonl
