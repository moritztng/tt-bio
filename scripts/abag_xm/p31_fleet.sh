#!/bin/bash
# AbAg-XM deep-N, GALAXY window 7 (workstream abag-xm-deepn-n512).
#
# N=512 RUNG for all four models, 8 chunks x 64 samples, on the seed-nested ladder
# (chunk j seed = base+1000*j), so chunks 0-3 ARE the N=256 chunks and the N=512 pool
# nests the N=256 pool exactly:
#   boltz2       164 targets, seeds 40000-47000, mps 1
#   opendde-abag 160 targets, seeds 20000-27000, mps 5->2->1 narrowing
#                (9i3p 9j4c 9ivj 9q7y not in THIS window: they need the 2026-08-08
#                large-target OOM fix, which cannot deploy into this tree without tripping
#                the link gate -- their full 8-chunk cells fold in window p32 on the fixed
#                engine, deepn_src_oomfix)
#   protenix-v2  163 targets, seeds 30000-37000, mps 5->1 narrowing (9j4c likewise in p32)
#   esmfold2     163 targets, seeds 50000-57000, single-sequence auto chunking, no MSA
#                flags and no mps ever (the campaign measures the no-MSA regime; 9j4c in p32)
# Chunking is RAM-forced (boltz2 ~0.22 GB/sample host RAM; 64-sample chunks cap a fold at
# ~15 GB so 32 concurrent folds fit the galaxy's 566 GB).
#
# TASKS ARE CHUNKS 4-7 ONLY. Chunks 0-3 already exist and are LINKED, never re-folded and
# never repaired here. That is deliberate: a (target,model) cell short at N=256 (od 9rye
# 2/4, 9gvn 3/4, 9xqn 3/4; px 9d73 c0) stays short at N=512 and gets dropped from BOTH
# rung rows by the analysis completeness gate, so the paired 256-vs-512 comparison runs on
# one common target set and the already-published N=256 numbers cannot move. Repairing a
# short chunk here would silently restate a live headline.
#
# LINK PHASE (binding reuse rule, inherited from p28/p29/p30, with one change): chunks whose
# source provably matches -- same seed block by construction, engine tree unchanged since
# the p27 anchor (mtime gate below), and an on-disk verify of results.json status=ok +
# all_runs==64 + 64 md5-distinct CIFs -- are HARDLINKED into this window instead of folded.
# Sources:
#   boltz2, opendde-abag  <- p28, rung 512, chunks 0-7  (p28 is the cancelled N=512 window:
#                            its chunks 0-3 were themselves linked from p27, and 39 bz + 29
#                            od chunk-4..7 folds landed before the cap decision stopped it)
#   protenix-v2, esmfold2 <- p29, rung 256, chunks 0-3
# CHANGE vs p28/p30: the link phase is DISK-DRIVEN, not results.jsonl-driven. verify() was
# always the binding gate; filtering candidates by an ok record first is a strictly weaker
# redundant pre-filter that drops complete-on-disk chunks whose record was overwritten by a
# later killed attempt (measured on p28: 39 complete bz c4-7 dirs vs 24 ok records, 29 od
# dirs vs 20 records -- ~24 chunk-folds, ~11 card-h, thrown away for nothing).
# Linked chunks 4-7 get their claim pre-created so slots skip them. Linked chunks 0-3 get a
# record and a reused_chunks.jsonl line but no claim (they are not tasks).
# COST-REFIT RULE (unchanged): dedupe seconds by (model,target,rung,chunk) across windows;
# linked records carry seconds paid in earlier windows, never double-counted.
#
# DEPLOY DISCIPLINE: ship this script to the galaxy by single-file scp into
# $SRC/scripts/abag_xm/ -- never a full `git archive` re-extract of $SRC, which refreshes
# every mtime and trips the link gate (safe, but re-folds ~1000 card-h).
#
# Launch procedure (pass-188 pattern -- NO maintenance-mode deploy): acquire
# galaxy_device_lock, kill ONLY the prod fold-worker tree (the spawn_main worker pool under
# `tt-bio serve`; japanfold.service, the web tier, cloudflared and SSH all stay up), then
# VERIFY THE 503: the platform's zero-device capacity guard (tt_bio/platform/jobs.py
# compute_offline) turns the fold API honest the moment online_workers hits 0, which the
# controller reports within its 20 s staleness window. POST one trivial predict to /v1 and
# confirm 503, not a job parked at progress 0.0. If it does not 503, stop japanfold.service
# instead and say the outage plainly -- a silently-swallowing API over a multi-day window is
# worse than an honest outage. Then run this script detached, arm p31_watchdog.sh by
# absolute path, and READ ITS LOG for the respawn line before ending the pass (p27 lesson).
#
# Every attempt appends one JSON record to results.jsonl. DONE_CHECK convention: no literal
# percent strings in logs.
#
# RUNNER HARDENING (pass-185 spec, inherited verbatim from p29): setsid process-group launch
# so kills reach spawn grandchildren, no-progress kill (<MIN_CPU_S group CPU AND zero log
# growth for STALL_MIN consecutive minutes -> INT, GRACE_S, KILL; caps a hang at ~47 min),
# zero-log kill after ZERO_MIN minutes (the pre-banner spin class, which burns CPU forever
# behind a 0-byte log so the two-leg rule never fires), post-hang tt-smi -r quarantine
# before the slot's next fold, kill-class records normalized to rc=124 with real cifs.
# ZERO_MIN CALIBRATION LESSON (2026-08-08, p31 first launch): Rich writes to a non-tty log
# only on final repaint, so a healthy fold's log sits at 0 bytes until it COMPLETES (verified
# via scheduler sqlite events: folds killed at the 30 min wire were mid-diffusion, step
# 199/200, events flowing). p29's legit silent folds ran to 89 min. ZERO_MIN=30 decapitated
# ~40 pct of attempts and made every fold > 30 min uncompletable. 99 sits above the p29 max.
# Thresholds env-overridable for fixture testing (STALL_MIN/GRACE_S/CAP_S/MIN_CPU_S/ZERO_MIN).
set -u
H=$HOME/mthuening
SRC=$H/deepn_src
B=$H/p31; mkdir -p $B $B/claims $B/tries
ANCHOR=$H/p27/tasks.txt          # oldest source window: the mtime gate must clear THIS
NCHIP=${1:-32}
STAGGER=${2:-8}
CHIPS=${CHIPS:-$(seq -s' ' 0 $((NCHIP-1)))}
PY_SYS=/usr/bin/python3.10
PY_VENV=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache

# SECOND-INSTANCE MODE (2026-08-10). A window's slots walk the task index ONCE, upward, so a claim
# released below a slot's cursor is unreachable and its cell strands. Measured on p31: 58 of 59
# stranded cells sat in the chunk-7 block while every live slot's cursor was already near the top
# of it. Rather than take the window down to sweep them, deploy this same script under a second
# name and run it against the SAME $B with a fresh cursor on the chips whose slots have exited:
#   SKIP_LINK=1 KEEP_TASKS=1 DONE_MARK=P31B_DONE DONE_FILE=$B/p31b.done YIELD_ON=P31_DONE \
#   PGID_LOG=$B/p31b_pgids.log CHIPS="<idle chips>" bash p31b_fleet.sh 13 8
# Two copies of the script means two inodes, so this cannot corrupt the running driver (bash reads
# a script by byte offset as it executes). YIELD_ON makes the instance stop claiming and hand its
# chips back the moment the primary window finishes, before the watchdog chains the next one.
SKIP_LINK=${SKIP_LINK:-0}       # 1 = do not re-verify/hardlink sources (already done by instance 1)
KEEP_TASKS=${KEEP_TASKS:-0}     # 1 = reuse the existing tasks.txt instead of regenerating it
DONE_MARK=${DONE_MARK:-P31_DONE}
DONE_FILE=${DONE_FILE:-$B/results.jsonl}
YIELD_ON=${YIELD_ON:-}          # marker string; once it appears in $B/results.jsonl, stop claiming
YIELD_FLAG=$B/.yield.$DONE_MARK
PGID_LOG=${PGID_LOG:-$B/fold_pgids.log}

# Per-window target lists. The four large targets were WH DRAM exclusions until the
# 2026-08-08 OOM fix; this window runs the frozen p27-era engine tree, so they fold in
# window p32 on the fixed tree instead. Never a scoring or biology call.
# boltz2 has none: it folded all 164 targets incl. 9j4c on WH through p27/p28 (rung-512
# chunk-0 record count is 164). Do not "tidy" 9j4c into BZ_EXCL for symmetry.
BZ_EXCL=""
OD_EXCL="9i3p 9j4c 9ivj 9q7y"
PX_EXCL="9j4c"
ESM_EXCL="9j4c"
# The 16 p27 pilot targets lead chunk 4 so the launch gate lands inside the first hours.
PILOT="21tw 9d3j 9i3p 9j4c 9ly5 9m0j 9ma0 9obn 9ppw 9q6y 9rye 9ua5 9udq 9v0x 9wpm 9zen"

excluded() { # <target> <excl-list>
  local t=$1; shift
  for e in $*; do [ "$t" = "$e" ] && return 0; done
  return 1
}

emit() { # <model> <target> <seedbase> <chunk>
  echo "$1 $2 512 $(( $3 + 1000 * $4 )) $4 8"
}

# Task order: chunk-outer, model-interleaved, pilot-first inside each chunk. A window that
# dies mid-rung then leaves all four models at the same uniform measured depth (N=320, 384,
# 448) rather than complete pools for a few targets and none for the rest.
TASKS=$B/tasks.txt
if [ "$KEEP_TASKS" = 1 ] && [ -s "$TASKS" ]; then
  echo "tasks: reusing $(wc -l < $TASKS) existing lines  chips: $(wc -w <<<"$CHIPS") [$CHIPS]"
else
# Written via a temp file and mv: the rename is atomic, so a driver already running against this
# window keeps reading a complete file. A bare `> $TASKS` truncates it, and a concurrent
# `sed -n Np` would then return an empty line and fold garbage.
{
  for j in 4 5 6 7; do
    for pass in pilot rest; do
      for y in $SRC/examples/abag_xm/*.yaml; do
        t=$(basename $y .yaml)
        if excluded "$t" $PILOT; then [ $pass = pilot ] || continue
        else [ $pass = rest ] || continue; fi
        excluded "$t" $BZ_EXCL  || emit boltz2       "$t" 40000 $j
        excluded "$t" $OD_EXCL  || emit opendde-abag "$t" 20000 $j
        excluded "$t" $PX_EXCL  || emit protenix-v2  "$t" 30000 $j
        excluded "$t" $ESM_EXCL || emit esmfold2     "$t" 50000 $j
      done
    done
  done
} > $TASKS.new && mv $TASKS.new $TASKS
echo "tasks: $(wc -l < $TASKS)  chips: $(wc -w <<<"$CHIPS") [$CHIPS]"
fi

# ---- link phase: hardlink provably-identical chunks from p28 (bz/od) and p29 (px/esm) ----
if [ "$SKIP_LINK" = 1 ]; then
echo "link phase skipped (SKIP_LINK=1): instance 1 already linked and claimed the reused chunks"
else
$PY_SYS - "$H" "$B" "$SRC" "$ANCHOR" <<'PY'
import hashlib, json, pathlib, subprocess, sys, time
H, B, src, anchor = (pathlib.Path(x) for x in sys.argv[1:5])
# model -> (source window, source rung, source chunks, output dir name)
SRCS = {
    "boltz2":       ("p28", 512, range(0, 8), "boltz2"),
    "opendde-abag": ("p28", 512, range(0, 8), "opendde"),
    "protenix-v2":  ("p29", 256, range(0, 4), "protenix"),
    "esmfold2":     ("p29", 256, range(0, 4), "esmfold2"),
}
manifest = {"commit_equal": False, "linked": 0, "by_model": {}, "reason": None}

def bail(reason):
    manifest["reason"] = reason
    print(f"LINK 0: {reason}")
    (B / "link_manifest.json").write_text(json.dumps(manifest) + "\n")
    sys.exit(0)

if not anchor.exists():
    bail(f"no anchor {anchor}")
mt = anchor.stat().st_mtime
newer = [str(p.relative_to(src)) for p in (src / "tt_bio").rglob("*.py")
         if p.stat().st_mtime > mt]
if newer:
    bail(f"engine tree changed since {anchor.name}: {newer[:5]}")

manifest["engine_fp"] = {str(p.relative_to(src)): hashlib.md5(p.read_bytes()).hexdigest()
                         for p in sorted((src / "tt_bio").rglob("*.py"))}
manifest["commit_equal"] = True

def verify(d, t):
    """results.json ok + 64 CIFs + md5-distinct count. The binding reuse gate."""
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

# task index, so a linked chunk 4-7 can pre-create its claim
idx = {}
for i, line in enumerate((B / "tasks.txt").read_text().splitlines(), 1):
    m, t, rung, seed, c, k = line.split()
    idx[(m, t, int(c))] = i

# source seconds/mps/umd, keyed by (model,target,chunk) at the source rung
def src_records(win, rung):
    out = {}
    f = H / win / "results.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("rung") == rung:
            out[(r["model"], r["target"], r.get("chunk"))] = r
    return out

# already linked/folded in this window (idempotent re-run)
done = set()
bf = B / "results.jsonl"
if bf.exists():
    for line in bf.read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("rung") == 512 and r.get("rc") == 0:
                done.add((r["model"], r["target"], r.get("chunk")))

now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
linked_total = 0
with open(B / "results.jsonl", "a") as rf, open(B / "reused_chunks.jsonl", "a") as pf:
    for model, (win, rung, chunks, mdir) in SRCS.items():
        recs = src_records(win, rung)
        sdir = H / win / mdir
        linked = 0
        for c in chunks:
            for srcdir in sorted(sdir.glob(f"*_c{c}")):
                t = srcdir.name[: -len(f"_c{c}")]
                if (model, t, c) in done:
                    linked += 1
                    continue
                n_distinct = verify(srcdir, t)
                if n_distinct != 64:
                    continue
                dst = B / mdir / f"{t}_c{c}"
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["cp", "-al", str(srcdir), str(dst)], check=True)
                r = recs.get((model, t, c), {})
                rf.write(json.dumps({"model": model, "target": t, "rung": 512,
                                     "seed": r.get("seed"), "chunk": c, "chunks": 8,
                                     "mps": str(r.get("mps", "?")), "umd": r.get("umd", -1),
                                     "rc": 0, "seconds": r.get("seconds", 0),
                                     "cifs": 64, "distinct": n_distinct, "oom": 0},
                                    separators=(",", ":")) + "\n")
                pf.write(json.dumps({"model": model, "target": t, "rung": 512, "chunk": c,
                                     "seed": r.get("seed"), "source_dir": str(srcdir),
                                     "source_window": win, "source_rung": rung,
                                     "source_seconds": r.get("seconds"),
                                     "source_mps": str(r.get("mps", "?")),
                                     "record_backed": bool(r),
                                     "tree_gate": f"tt_bio mtimes < {anchor.parent.name}/tasks.txt",
                                     "reused_at": now,
                                     "claim_idx": idx.get((model, t, c))}) + "\n")
                i = idx.get((model, t, c))
                if i:
                    (B / "claims" / str(i)).mkdir(exist_ok=True)
                linked += 1
        manifest["by_model"][model] = linked
        linked_total += linked
manifest["linked"] = linked_total
(B / "link_manifest.json").write_text(json.dumps(manifest) + "\n")
print(f"LINK {linked_total} chunks hardlinked: {manifest['by_model']}")
PY
fi

group_cpu() { # <pgid> -> total CPU seconds of every process in the group
  ps -eo pgid=,times= | awk -v g="$1" '$1==g {s+=$2} END {print s+0}'
}

guarded_fold() { # <logfile> <chip> <cmd...> -- setsid launch + stall/cap group kills + quarantine
  local log=$1 u=$2; shift 2
  local stall_min=${STALL_MIN:-45} cap_s=${CAP_S:-21600} grace_s=${GRACE_S:-120} min_cpu=${MIN_CPU_S:-60}
  local zero_min=${ZERO_MIN:-99}
  local poll_s=${POLL_S:-60}
  setsid env TT_VISIBLE_DEVICES=$u "$@" > "$log" 2>&1 &
  local pid=$! t0=$(date +%s) last_cpu=-1 last_size=-1 stall=0 zero=0 killrc=0 g=0
  # setsid makes the child a group leader, so $! is the pgid. Logging it turns a window takedown
  # into `kill -- -<pgid>` over this file instead of reconstructing the groups by hand from ps.
  echo "$(date -u +%FT%TZ) pgid=$pid dev=$u log=$log" >> "$PGID_LOG"
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
  # Mark the claim satisfied, so slot() can tell a completed cell from a killed one. Success is
  # the criterion the link phase and the analysis already use: rc 0 plus a full set of
  # md5-distinct CIFs (rung/chunks of them). CLAIM is exported by slot() into the fold subshell.
  if [ -n "${CLAIM:-}" ] && [ "$9" = 0 ] && [ "${11}" -eq $(( $3 / $6 )) ] && [ "${12}" -eq "${11}" ]; then
    touch "$CLAIM/ok"
  fi
}

count_structs() { # <dir> -> echoes "n distinct"
  local d=$1 n distinct
  n=$(ls $d/structures/*.cif 2>/dev/null | wc -l)
  distinct=$(md5sum $d/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  echo "${n:-0} ${distinct:-0}"
}

outbase() { # <mdir> <target> <chunk> -> per-task out dir under $B
  echo "$B/$1/${2}_c$3"
}

fold_bz() { # <target> <rung> <seed> <chunk> <chunks> <chip>
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d ob
  ob=$(outbase boltz2 $t $c)
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

fold_od() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps narrowing 5->2->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d ob
  ob=$(outbase opendde $t $c)
  for mps in 5 2 1; do
    s=$(date +%s)
    guarded_fold $B/opendde_${t}_c${c}_mps$mps.log $u $PY_SYS -u -m tt_bio.main predict \
      examples/abag_xm/$t.yaml --model opendde-abag --out_dir $ob --override \
      --diffusion_samples $((rung/k)) --max_parallel_samples $mps --seed $seed --host_threads 2 \
      --msa_dir $MSA --msa_cache_only
    rc=$?; secs=$(( $(date +%s) - s ))
    d=$(ls -d $ob/*results_$t 2>/dev/null | head -1)
    read -r _n _di <<<$(count_structs "$d")
    oom=$(grep -c 'Out of Memory' $B/opendde_${t}_c${c}_mps$mps.log 2>/dev/null)
    record opendde-abag $t $rung $seed $c $k $mps $u $rc $secs ${_n:-0} ${_di:-0} ${oom:-0}
    [ "${oom:-0}" -gt 0 ] && [ "${_n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
}

fold_px() { # <target> <rung> <seed> <chunk> <chunks> <chip>  -- mps 5, narrow 5->1 on OOM
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 mps s rc secs oom d ob
  ob=$(outbase protenix $t $c)
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
  local t=$1 rung=$2 seed=$3 c=$4 k=$5 u=$6 s rc secs oom d ob
  ob=$(outbase esmfold2 $t $c)
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
    boltz2)       fold_bz  "$2" "$3" "$4" "$5" "$6" "$7";;
    opendde-abag) fold_od  "$2" "$3" "$4" "$5" "$6" "$7";;
    protenix-v2)  fold_px  "$2" "$3" "$4" "$5" "$6" "$7";;
    esmfold2)     fold_esm "$2" "$3" "$4" "$5" "$6" "$7";;
    *)            echo "SKIP: $1 not in this window" >> $B/slots.log;;
  esac
}

# A claim used to be taken with mkdir and never released, so a cell whose fold the guard killed
# was skipped by every later slot for the rest of the driver's life. That is how the 08-10 tail
# ended up 67 cells short with all 2600 claims held: the driver was claim-exhausted, not
# work-exhausted, and P31_DONE fires on `wait`, not on completeness. One missing chunk drops the
# whole (target, rung) from the analysis, which is the rung this campaign exists to measure.
# Fix: release a failed claim so another slot retries it, bounded by ATTEMPT_MAX so an unfoldable
# cell cannot burn the window, and re-walk the list until a full pass claims nothing.
ATTEMPT_MAX=${ATTEMPT_MAX:-3}
slot() {
  local chip=$1 idx n model t rung seed c k tries claimed
  n=$(wc -l < $TASKS)
  while true; do
    claimed=0
    for ((idx=1; idx<=n; idx++)); do
      [ -e "$YIELD_FLAG" ] && break 2      # cheap stat, not a grep: this runs 2600x per pass
      mkdir $B/claims/$idx 2>/dev/null || continue
      claimed=1
      read -r model t rung seed c k <<<"$(sed -n "${idx}p" $TASKS)"
      ( cd $SRC && export CLAIM=$B/claims/$idx && fold "$model" "$t" "$rung" "$seed" "$c" "$k" "$chip" )
      [ -e $B/claims/$idx/ok ] && continue
      tries=$(( $(cat $B/tries/$idx 2>/dev/null || echo 0) + 1 ))
      echo $tries > $B/tries/$idx
      if [ $tries -lt $ATTEMPT_MAX ]; then
        rm -rf $B/claims/$idx
        echo "$(date -u +%FT%TZ) RELEASE idx=$idx $model $t c$c try=$tries" >> $B/slots.log
      else
        echo "$(date -u +%FT%TZ) EXHAUSTED idx=$idx $model $t c$c after $tries tries" >> $B/slots.log
      fi
    done
    [ "$claimed" = 0 ] && break
  done
  echo "slot $chip done" >> $B/slots.log
}

# Yield poller (second-instance mode only). One process watching results.jsonl, converting the
# primary window's marker into a flag file the hot loop can stat, then INTing this instance's own
# fold groups so the chips are genuinely free before the watchdog chains the next window ~2-12 min
# later. Only groups from THIS instance's $PGID_LOG are touched, never the primary's.
if [ -n "$YIELD_ON" ]; then
  rm -f "$YIELD_FLAG"
  ( while ! grep -q "$YIELD_ON" $B/results.jsonl 2>/dev/null; do sleep 30; done
    : > "$YIELD_FLAG"
    echo "$(date -u +%FT%TZ) YIELD: $YIELD_ON seen, no new claims; INT of in-flight groups in 60 s" \
      >> $B/slots.log
    sleep 60
    while read -r _ p _; do
      p=${p#pgid=}
      kill -0 -- -$p 2>/dev/null && kill -INT -- -$p 2>/dev/null
    done < "$PGID_LOG"
  ) &
  YIELD_PID=$!
fi

# Wait on the slot pids specifically, not a bare `wait`: the yield poller is also a background job
# and it never returns if its marker never appears, which would hang the driver past its last slot
# and leave the done marker unwritten.
PIDS=""
for c in $CHIPS; do
  slot "$c" &
  PIDS="$PIDS $!"
  sleep "$STAGGER"
done
for p in $PIDS; do wait $p; done
[ -n "${YIELD_PID:-}" ] && kill $YIELD_PID 2>/dev/null
echo $DONE_MARK >> $DONE_FILE
