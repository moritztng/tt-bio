"""Audit ~/.cache/huggingface/hub for superseded revisions.

Live revision = pointed to by at least one ref under <repo>/refs/.
Superseded    = a snapshot revision with no refs, in a repo that has at
                least one live revision (so the repo stays usable).
Repos with zero refs are left untouched entirely.

Reclaimable bytes come from delete_revisions().expected_freed_size, which
refcounts blobs correctly (revisions can share blobs, so summing
per-revision sizes double counts).
"""
import json
import sys

from huggingface_hub import scan_cache_dir

DRY = "--delete" not in sys.argv

info = scan_cache_dir()

rows = []
all_delete = []
total_freed = 0

for repo in sorted(info.repos, key=lambda r: -r.size_on_disk):
    live = [r for r in repo.revisions if r.refs]
    dead = [r for r in repo.revisions if not r.refs]
    reclaimable = [r for r in dead if live]
    freed = 0
    if reclaimable:
        freed = info.delete_revisions(
            *(r.commit_hash for r in reclaimable)
        ).expected_freed_size
    total_freed += freed
    rows.append(
        {
            "repo": repo.repo_id,
            "type": repo.repo_type,
            "size_gib": round(repo.size_on_disk / 2**30, 2),
            "nb_revisions": len(repo.revisions),
            "live": sorted(r.commit_hash for r in live),
            "superseded": sorted(r.commit_hash for r in reclaimable),
            "kept_unreferenced": sorted(
                r.commit_hash for r in dead if not live
            ),
            "freed_gib": round(freed / 2**30, 2),
        }
    )
    all_delete.extend(r.commit_hash for r in reclaimable)

print(f"{'repo':40s} {'total':>8s} {'reclaim':>8s}  refs  superseded hashes")
for r in rows:
    print(
        f"{r['repo']:40s} {r['size_gib']:7.2f}G {r['freed_gib']:7.2f}G  "
        f"{len(r['live'])}/{r['nb_revisions']}   "
        + (",".join(h[:8] for h in r["superseded"]) or "-")
    )
print(f"\nTotal cache {sum(r['size_gib'] for r in rows):.2f} GiB, "
      f"reclaimable {total_freed/2**30:.2f} GiB")

if DRY:
    print("\nDRY RUN — pass --delete to remove superseded revisions")
elif all_delete:
    strategy = info.delete_revisions(*all_delete)
    print(f"Expected to free {strategy.expected_freed_size/2**30:.2f} GiB")
    strategy.execute()
    print("Deleted.")
else:
    print("Nothing to delete.")

with open(
    "/home/moritz/.coworker/wt/pc-hf-cache-revision-audit/perf/pc-hf-cache-revision-audit/audit.json",
    "w",
) as f:
    json.dump(
        {
            "rows": rows,
            "total_reclaimable_gib": round(total_freed / 2**30, 2),
            "deleted": not DRY,
        },
        f,
        indent=2,
    )
