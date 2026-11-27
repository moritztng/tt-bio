#!/bin/bash
# RESET-RESUME on a Galaxy: a DP run of the real RFD4 step survives a whole-box reset.
# A: 8 steps uninterrupted over CARDS, checkpoint at step 4. B: same run, every rank SIGKILLed (process group)
# after step 5, the /dev/shm rendezvous wiped (what a reboot does to tmpfs), `tt-smi -glx_reset` of all 32 chips,
# then a fresh launch resumes from B's step-4 checkpoint to step 8. C (broken arm): resume from a copy of that
# checkpoint with the Adam moments stripped. PASS = B's steps 5-8 master digests equal A's on every rank and C's
# all differ. The reset is whole-box, so this runs only inside a booked HOLDER.windows slot and aborts if any
# process outside this run holds a /dev/tenstorrent node at reset time.
# usage: gx_reset_resume.sh CARDS OUT   (anx-glx, tree ~/fleet/r4t/r4t-scale, wrapper ~/fleet/r4t/gx_trunk.sh)
set -u
C=${1:-0,1,2,3}; O=${2:-$HOME/fleet/r4t/rr}; R=$HOME/fleet/r4t; PY=$HOME/fleet/env/bin/python
mkdir -p "$O"; log() { echo "$(date -u +%T) $*" >> "$O/progress"; }
holders() {
  for f in /proc/[0-9]*/fd/*; do
    case "$(readlink "$f" 2>/dev/null)" in /dev/tenstorrent/*) echo "${f#/proc/}" | cut -d/ -f1 ;; esac
  done | sort -u | tr '\n' ' '
}
[ -e ~/fleet/HANDBACK ] && { log HANDBACK; exit 4; }
rm -rf "$O/A" "$O/B" "$O/Bres" "$O/C" "$O/Cck" "$O/verdict.json"
OMP=16 "$R/gx_trunk.sh" "$C" "$O/A" 8 --shard --save-at 4; log "A $(tail -1 "$O/A/run.log")"
setsid "$R/gx_trunk.sh" "$C" "$O/B" 8 --shard --save-at 4 & P=$!
until grep -q '"step": 5' "$O/B/rank0.log" 2>/dev/null || ! kill -0 $P 2>/dev/null; do sleep 0.5; done
kill -9 -- -$P; sleep 3; rm -rf /dev/shm/r4t-trunk-dp-*
H=$(holders); [ -n "$H" ] && { log "ABORT: /dev/tenstorrent still held by pids $H"; exit 1; }
log "B killed after step 5 (session $P), rendezvous wiped, no device holders"
timeout 900 tt-smi -glx_reset > "$O/reset.log" 2>&1; log "glx_reset rc=$?"
CK=$(grep -ho '"saved": "[^"]*"' "$O/B/rank0.log" | head -1 | cut -d'"' -f4); log "resume from $CK"
[ -f "$CK" ] || { log "ABORT: no step-4 checkpoint"; exit 1; }
mkdir -p "$O/Cck"
"$PY" - "$CK" "$O/Cck/stripped.safetensors" <<'PY'
import sys
from safetensors.numpy import load_file, save_file
from safetensors import safe_open
src, dst = sys.argv[1:]
with safe_open(src, "np") as f:
    md = f.metadata()
b = {k: v for k, v in load_file(src).items() if not k.startswith("exp_avg")}
save_file(b, dst, metadata=md)
PY
OMP=16 "$R/gx_trunk.sh" "$C" "$O/Bres" 8 --shard --resume "$CK"; log "Bres $(tail -1 "$O/Bres/run.log")"
OMP=16 "$R/gx_trunk.sh" "$C" "$O/C" 8 --shard --resume "$O/Cck/stripped.safetensors"; log "C $(tail -1 "$O/C/run.log")"
"$PY" - "$O" "$C" <<'PY' > "$O/verdict.json"
import glob, json, os, sys
O, cards = sys.argv[1:]
def dig(d):
    out = {}
    for f in sorted(glob.glob(f"{O}/{d}/rank*.log")):
        r = int(os.path.basename(f)[4:-4])
        for ln in open(f):
            if ln.startswith('{"step"'):
                j = json.loads(ln)
                out.setdefault(j["step"], {})[r] = (j["master_digest"], j["aiclk"], j["loss"])
    return out
A, B, Bres, Cc = dig("A"), dig("B"), dig("Bres"), dig("C")
st = range(5, 9)
agree = lambda X: bool(X) and all(len({v[0] for v in X[s].values()}) == 1 for s in X)
v = dict(cards=cards, A_steps=sorted(A), B_steps=sorted(B), Bres_steps=sorted(Bres), C_steps=sorted(Cc),
         ranks_agree=dict(A=agree(A), Bres=agree(Bres), C=agree(Cc)),
         prekill_match=bool(B) and all(B[s][0][0] == A[s][0][0] for s in B if 0 in B[s]),
         resume_match=all(s in Bres and s in A and Bres[s][0][0] == A[s][0][0] for s in st),
         broken_mismatch=all(s in Cc and s in A and Cc[s][0][0] != A[s][0][0] for s in st),
         digests={k: {s: X[s][0][0] for s in X} for k, X in (("A", A), ("B", B), ("Bres", Bres), ("C", Cc))},
         loss8={k: X[8][0][2] for k, X in (("A", A), ("Bres", Bres), ("C", Cc)) if 8 in X},
         aiclk={k: [min(v[1][0] for s in X for v in X[s].values()), max(v[1][1] for s in X for v in X[s].values())]
                for k, X in (("A", A), ("Bres", Bres), ("C", Cc)) if X})
v["PASS"] = v["prekill_match"] and v["resume_match"] and v["broken_mismatch"] and all(v["ranks_agree"].values())
print(json.dumps(v, indent=1))
PY
log "verdict $(grep -o '"PASS": [a-z]*' "$O/verdict.json")"
