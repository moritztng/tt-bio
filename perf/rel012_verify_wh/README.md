# rel012-verify-wh: the Wormhole ladder re-measured at the merge tree

One chip of a Wormhole Galaxy, at `wk/rel012-integrate` 8a73696b3, confirming or refusing the
512 -> 896 token claim `wk/bcw-slowmode` measured on 2026-10-01. State doc:
`~/.coworker/state/rel012-verify-wh.md`.

The tree is shipped to the box with `git archive` of the ref, so what runs there is the merge tree
and not a branch, with the sha in `REL012_SHA` beside it:

    git archive --format=tar origin/wk/rel012-integrate | \
      ssh ubuntu@172.16.102.107 'mkdir -p ~/bcw-slowmode/tt-bio-rel012 && tar -x -C ~/bcw-slowmode/tt-bio-rel012'

The sitting, on `.107` (UF-EV-A4-GWH01) chip 30, through `perf/bcw_slowmode/launch.sh` with the
lease holder renamed to this row:

    bash perf/bcw_slowmode/launch_rel012.sh rel012wh 30 \
      --params ~/bwx/af2_params --rounds 5 --footprint-rounds 3 --timeout 18000 \
      r:hEGFR:150:f:auto r:hEGFR:250:f:offload r:hEGFR:280:f:offload r:hIL2R:90:t:fast

The four legs are the brief's minimum: 800 with no mode flag (`auto`, the user's path, the EGFR
ectodomain plus a 150 aa binder), 896 (the claimed ceiling), 928 (which must refuse), and 512 in
`fast`. Each is the same target and binder last night's ledger used for that rung, so the rungs
are the same shapes rather than new ones.

`cluster.txt` and `idle-before.txt` are the customer-impact evidence: `/v1/cluster` before and
after the launch, and the box's own lease table, agent journal and load before the chip was taken.
