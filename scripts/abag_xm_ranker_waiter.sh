#!/usr/bin/env bash
# Wait until this host's Tier-A slab is fully folded AND fully labelled, then run the
# learned-ranker scoring (endgame step 2's exact command) exactly once, and exit.
#
# Why this exists: DeepRank-Ab cannot run while the cards fold (it takes every core and ignores
# every cap -- see abag_xm_ranker_scores.py::_run_deeprank_batched), and the scoring itself only
# covers LABELLED folds. Both preconditions become true hours after whoever is watching has gone:
# generation drains, then the labels loop chews the last folds. The endgame script refuses to
# score under either condition, so somebody would otherwise have to be present at the drain
# minute to start a ~5 h leg -- the same "schedule improvement nobody is present to make" class
# the self-adapting labels loop fixed. This waiter is present instead.
#
# Self-healing: ranker_scores.py --all skips already-scored folds, so a rerun after a crash
# resumes rather than restarting. CPU-only scoring -- never touches a device.
#
#   Usage:  setsid nohup scripts/abag_xm_ranker_waiter.sh >> $HOME/abag_xm/logs/ranker_waiter.log 2>&1 &
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/.abag_xm_label_venv/bin/python3"
[ -x "$PY" ] || PY=/home/ttuser/tt-bio/env/bin/python3
TIER=$HOME/abag_xm/tier_a
LOGDIR=$HOME/abag_xm/logs
mkdir -p "$LOGDIR"
say(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%S+0000)] $*"; }

say "ranker waiter up on $(hostname) -- waiting for folds to drain and labels to cover"
while :; do
  if pgrep -f "tt_bio.mai[n] predict" >/dev/null 2>&1; then
    sleep 300
    continue
  fi
  ok=$("$PY" - <<'PY'
import json
from pathlib import Path
p = Path.home()/"abag_xm"/"tier_a"/"progress.jsonl"
print(len({(r["target"],r["model"]) for r in map(json.loads, open(p)) if r.get("status")=="ok"}))
PY
)
  lab=$(ls "$TIER"/labels/*.json 2>/dev/null | wc -l)
  say "cards idle; labels=$lab ok_folds=$ok"
  [ "$ok" -gt 0 ] && [ "$lab" -ge "$ok" ] && break
  sleep 300
done

say "preconditions met -- scoring learned rankers (this is the long leg)"
"$PY" "$WT/scripts/abag_xm_ranker_scores.py" --all --out "$TIER"/ranker_scores.csv \
    --with_deeprank --with_abagrank
rc=$?
say "scoring exited rc=$rc -- ranker_scores.csv rows: $(($(wc -l < "$TIER"/ranker_scores.csv) - 1))"
exit $rc
