#!/usr/bin/env bash
# Session s3: the powered session. Detached, cwd inside THIS worktree, so a torn-down sibling
# worktree cannot delete it mid-run and the launching turn ending cannot kill it.
#
# 48 reps because pass 33 measured the instrument rather than guessing at it: at 5 reps the paired
# A/A 95 % CI is +/-0.79 s against the 0.5258 s the two banked levers are worth, i.e. 1.5x the
# effect, so every session before this one was under-powered before contention was even considered.
# CI ~ 1/sqrt(n), so 48 reps takes the same instrument to about +/-0.25 s, half the effect.
# 48 is even, so the palindrome ordering gives every interior arm 24 early and 24 late slots.
#
# MAXLOAD is raised from 3.0 to 6.0 deliberately and the reason is recorded rather than hidden: the
# only co-tenant on this box is gatechain7's single size_ladder_record arm, pinned to card 1 on the
# OTHER board pair (serial ...4103 against this pair's ...410D), so it shares host CPU but not the
# power budget. A constant co-tenant load cancels in a paired interleaved A/B; the 3.0 bar would
# refuse every session for the next eight hours and buy nothing the design does not already handle.
cd /home/ttuser/.coworker/wt/c12-compose-fold || exit 1
MAXLOAD=6.0 setsid nohup ./perf/c12_compose/run.sh s3 \
  --reps 48 --palindrome --arms base,silu,hoist,both,base \
  > perf/c12_compose/out/s3_launch.log 2>&1 < /dev/null &
echo "s3 launched pid=$! at $(date -u +%FT%TZ)"
