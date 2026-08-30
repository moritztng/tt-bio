# opendde 512 aa residual bisect, arms

One JSON per measured commit: `deb9b307` 0.725246 -> `626be0b5` 0.725015 on
`perf/size512/fixtures/cdk2x2_512.{yaml,a3m}`, 10 recycles / 200 sampling steps / 1 sample /
seed 0, pc card 0 (p150a, 13x10), ttnn 0.68.0. `F_forcegrid_626be0b5.json` is main tip with
`PROTENIX_PAIRCOND_MM_FORCE_GRID=1`, which restores the pre-`86df9db9` digest exactly.

Each arm's own `tt_bio_git` is the commit it measures. Check it before reading a number: the
first run of this search recorded four arms against commits they had never been checked out at.

The driver is deliberately NOT here. `git checkout --detach <older commit>` deletes a tracked
driver out from under a running search, which is how those four arms happened. Keep it in an
untracked directory.
