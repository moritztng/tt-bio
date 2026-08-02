#!/bin/bash
# Symlinks runtime/sfpi into throwaway tt_hw_planner worktrees (isolation only links
# python_env/build*/generated/model_cache; runtime/ is lost -> JIT falls back to old /opt sfpi).
REPO=/home/ttuser/tt-metal-hwplanner
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
touch $WT/logs/WATCHER_ACTIVE
while [ -f $WT/logs/WATCHER_ACTIVE ]; do
  for d in /tmp/tt_hw_planner_*/; do
    [ -d "$d" ] || continue
    if [ ! -e "$d/runtime/sfpi" ]; then
      mkdir -p "$d/runtime" 2>/dev/null && ln -sfn "$REPO/runtime/sfpi" "$d/runtime/sfpi" 2>/dev/null && echo "$(date -u) linked sfpi into $d" >> $WT/logs/watcher.log
    fi
  done
  sleep 5
done
