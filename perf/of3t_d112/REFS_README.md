# of3t campaign upstream references — where they live and why here

Restored 2026-09-21 by `of3t-d112`. D112: a concluded row's worktree was pruned and took
`/home/ttuser/of3t_rebase/` with it, including the 0.4.3 reference. Every surviving copy was
inside a directory named after a **concluded** slug — `of3t_refprec`, `of3t_trunk043ref`,
`of3t_gradients`, `scratch/of3t-bondcov` — so the same thing was queued up to happen again.

## Where

| host | path | holds |
|---|---|---|
| pc | `~/.coworker/artifacts/of3t-refs/` | both sdists, both extracted package trees, `MANIFEST.json` |
| tt-quietbox2 | `/home/ttuser/of3t-campaign-refs/` | both package trees, `bundle_min_043/`, `cap/`, `MANIFEST.json` |

The tensors stay on qb2 only: 8.6 GB against 18 GB free on pc, and qb2 is the host that computes
with them. They are hard links to the surviving copies, so they cost no extra space and an
`rm -rf` of the other name leaves the inode intact.

## Why these two paths survive a prune

Three mechanisms destroy a reference in this fleet, and neither path is reachable by any of them:

1. `worker.sh:343` runs `git worktree remove --force $WT` when a row concludes. `$WT` is
   `~/.coworker/wt/<slug>` on pc and `/home/ttuser/.coworker/wt/<slug>` on a qb host. Neither
   path is under a worktree.
2. `disk_guard.sh` `sweep_pc` tier 3 removes `/tmp/*` entries 3+ days old whose basename
   contains a concluded slug. Neither path is under `/tmp`, and neither basename names a slug —
   that is deliberate, because slug-attribution is what the sweep matches on.
3. `disk_guard.sh`'s only other `rm` targets are the pip and npm caches and
   `$HOME/.cache/tt-metal-cache*`. `~/.coworker/artifacts/` appears in no tier.

`~/.coworker/artifacts/` is also in `~/.coworker/.gitignore`, so a stray `git add` cannot stage
237 MB into the knowledge-base repo.

## What makes a rebuild checkable

Location stops a prune. It does not help if every copy is lost, so the manifest is the durable
half: `make_manifest.py` records a whole-tree digest under a stated rule plus the sha256 of every
non-package file, and **exits non-zero** when a tree disagrees with `EXPECTED_DIGESTS.json`.
Verified against a deliberately corrupted expectation, which exits 1 and names the tree.

The package trees are reproducible from PyPI. `pip download openfold3==0.4.3 --no-deps
--no-binary :all:` on 2026-09-21 gave a tree digest of `1b27f575…`, the value
`of3t-trunk043ref` recorded, and 0.5.0 gave `092fb575…`, the value `of3t-gradients` recorded. So
the reference can be rebuilt from upstream and checked, not merely copied.

`bundle_min_043/grads_f64_043.pt` is not reproducible from PyPI — it is a 2.9 GB float64
gradient bundle from a 26-minute CPU replay. Its sha256 is
`1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4`, which is the value recorded
inside every 0.4.3 instrument-A arm on `wk/of3t-rebase`, so the restored file is the same bytes
those arms were scored against.

## Checking it

    python3 make_manifest.py . EXPECTED_DIGESTS.json     # exit 0, or it names the tree that moved
