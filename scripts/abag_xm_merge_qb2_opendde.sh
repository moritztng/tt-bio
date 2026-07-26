#!/usr/bin/env bash
# abag_xm_merge_qb2_opendde.sh — merge qb2's opendde-abag Tier-A results into qb1.
#
# Run ON qb1 AFTER qb2's opendde campaign (scripts/abag_xm_opendde_qb2.sh) has completed
# (qb2's ~/abag_xm/tier_a/progress.jsonl shows all 164 opendde targets status=ok, or as
# many as will complete). qb1 and qb2 have no shared mount, so this pulls qb2's opendde
# output + progress records into qb1's campaign dir, then the qb1 labels loop labels them.
#
# Safe to re-run: rsync is idempotent (same files overwrite identically); the progress
# append dedups on (target,model,host) so re-running won't double-count.
#
# What it does:
#   1. rsync qb2:~/abag_xm/tier_a/opendde_abag/ -> qb1:~/abag_xm/tier_a/opendde_abag/
#      (result_dir paths match after rsync: both use ~/abag_xm/tier_a/opendde_abag/)
#   2. Append qb2's opendde `ok` records to qb1's progress.jsonl, rewriting `host`
#      to tt-quietbox2 (truthful — D12 records host per-record; the campaign is
#      intentionally cross-host). Dedup on (target,model="opendde-abag",host).
#   3. Report counts (merged ok opendde, total ok, any qb2 failures to review).
#
# Usage:  bash scripts/abag_xm_merge_qb2_opendde.sh [--dry-run]
set -u
QB2=ttuser@tt-quietbox2
QB1_DIR=$HOME/abag_xm/tier_a
QB2_DIR='~/abag_xm/tier_a'
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

log(){ echo "[$(date +%H:%M:%S)] $*"; }
[ -d "$QB1_DIR" ] || { echo "ABORT: $QB1_DIR not found (run on qb1)"; exit 1; }

log "STEP 1: rsync qb2 opendde_abag/ -> qb1"
if [ "$DRY" -eq 1 ]; then
  rsync -an "$QB2:$QB2_DIR/opendde_abag/" "$QB1_DIR/opendde_abag/"
else
  mkdir -p "$QB1_DIR/opendde_abag"
  rsync -a "$QB2:$QB2_DIR/opendde_abag/" "$QB1_DIR/opendde_abag/"
fi

log "STEP 2: append qb2 opendde ok records to qb1 progress.jsonl (dedup)"
python3 - "$QB1_DIR/progress.jsonl" "$QB2" "$DRY" <<'PY'
import json, sys, subprocess
qb1_prog, qb2_host, dry = sys.argv[1], sys.argv[2], int(sys.argv[3])
# read existing qb1 (target,model,host) keys to dedup
seen=set()
with open(qb1_prog) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get("status")=="ok":
            seen.add((r.get("target"),r.get("model"),r.get("host","tt-quietbox")))
# fetch qb2 progress.jsonl over ssh
out=subprocess.run(["ssh","-o","ConnectTimeout=8",f"ttuser@{qb2_host}",
                   "cat ~/abag_xm/tier_a/progress.jsonl"],capture_output=True,text=True)
qb2_recs=[json.loads(l) for l in out.stdout.splitlines() if l.strip()]
ok_qb2=[r for r in qb2_recs if r.get("status")=="ok" and r.get("model")=="opendde-abag"]
fails_qb2=[r for r in qb2_recs if r.get("status")!="ok" and r.get("model")=="opendde-abag"]
new=[]
for r in ok_qb2:
    r=dict(r); r["host"]="tt-quietbox2"  # truthful per-record
    k=(r.get("target"),r.get("model"),r["host"])
    if k in seen: continue
    seen.add(k); new.append(r)
print(f"qb2 opendde: {len(ok_qb2)} ok, {len(fails_qb2)} non-ok (review), {len(new)} new to merge")
if dry:
    print(f"[dry-run] would append {len(new)} records to {qb1_prog}")
else:
    with open(qb1_prog,"a") as f:
        for r in new: f.write(json.dumps(r)+"\n")
    print(f"appended {len(new)} records to {qb1_prog}")
# report any qb2 failures so they can be retried via resume_opendde.sh on qb1
if fails_qb2:
    from collections import Counter
    c=Counter(r.get("status") for r in fails_qb2)
    print(f"qb2 non-ok opendde records (review/retry): {dict(c)}")
PY

log "DONE — run labels campaign next (scripts/abag_xm_labels_loop.sh) to label the merged opendde folds"
