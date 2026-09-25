# Why these files are in git

`~/.coworker/state/` is gitignored, lives on one machine and has no backup. The published copies in
`../` exist so the branch carries the campaign's reasoning and not only its artifacts.

`scripts/rotate_state_docs.sh` archives the middle of a long doc every 30 minutes, which keeps the
working surface cheap to re-read — a state doc is re-read every pass, so its length is a recurring
cost, and one orchestrator doc reached 1,168,747 bytes at $12.83/pass before this was fixed. But
rotation moves text OUT of the published copy, and if the archive it moves text into is not also
committed, the only copy of that reasoning is on one disk.

So: **whenever a rotation shrinks a published record here, the archive it points at is committed
beside it in the same pass.** Together, `../LEDGER.md` (the live tail) and
`of3t-LEDGER.20260925-123559.md` (the middle) are the whole ledger; `R173`-`R180` are in the
archive and nowhere else in git.

`of3t-orchestrator.pre-bwd-sprint-20260925.md` is the state doc as it stood at the end of the
correctness campaign, 39,299 B, before pass 445 cut it to carry the backward sprint instead.
