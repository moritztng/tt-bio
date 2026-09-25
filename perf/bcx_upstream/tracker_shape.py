"""What upstream BindCraft 2 merges, and what it closes.

Reads the public tracker rather than quoting it, so the claim in
state/bcx/UPSTREAM-SEAM.md can be re-run. No auth: the two endpoints used are
unauthenticated, which is also why code search is avoided (it requires a token).

The question this answers is NOT "were we ignored" -- merged PRs in this repo
carry 0 comments too, so silence discriminates nothing. It is "what shape
lands": size of diff against merged-vs-closed.
"""
import json
import urllib.request

REPO = "PacesaLab/BindCraft2"


def get(path):
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/{path}", timeout=60) as r:
        return json.load(r)


def main():
    head = get("commits/main")
    rows = []
    for issue in get("issues?state=all&per_page=50&sort=created"):
        if "pull_request" not in issue:
            continue
        pr = get(f"pulls/{issue['number']}")
        rows.append({"number": pr["number"], "merged": bool(pr["merged_at"]),
                     "comments": issue["comments"], "additions": pr["additions"],
                     "deletions": pr["deletions"], "files": pr["changed_files"],
                     "user": pr["user"]["login"], "title": pr["title"]})
    merged = [r for r in rows if r["merged"]]
    closed = [r for r in rows if not r["merged"] and not r["number"] in (16, 10)]
    print(json.dumps({
        "main": {"sha": head["sha"], "date": head["commit"]["author"]["date"]},
        "merged_with_zero_comments": sum(1 for r in merged if r["comments"] == 0),
        "merged_count": len(merged),
        "median_additions_merged": sorted(r["additions"] for r in merged)[len(merged) // 2] if merged else None,
        "prs": rows,
    }, indent=1))


if __name__ == "__main__":
    main()
