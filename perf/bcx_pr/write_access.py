#!/usr/bin/env python3
"""Which PRs BindCraft 2 merges, split by whether the head branch is inside the lab's own repo.

Two earlier readings of this tracker disagreed. The first counted by ACCOUNT and got "0 of 5
outside PRs merged"; the correction counted by account too and got "8 outside, 3 merged", naming
#1 (rclune) and #14/#15 (zyzllik) as outside. Neither is measurable: whether an account is "the
lab" is a guess.

`head.repo.full_name` is not a guess. You cannot push a branch into `PacesaLab/BindCraft2`
without write access to it, so the field partitions every PR into "opened from a branch the
author could push to the lab's repo" and "opened from a fork". That is the discriminator this
prints, and it is what the merge decision has actually tracked.

Run:  gh auth status && python3 write_access.py
"""
import json
import subprocess
import sys

REPO = 'PacesaLab/BindCraft2'


def api(path):
    out = subprocess.run(['gh', 'api', path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def outcome(pr):
    if pr['merged_at']:
        return 'MERGED'
    return 'CLOSED' if pr['closed_at'] else 'open'


def main():
    pulls = api(f'repos/{REPO}/pulls?state=all&per_page=100')
    rows = []
    for pr in sorted(pulls, key=lambda p: p['number']):
        head = (pr['head']['repo'] or {}).get('full_name', '(deleted fork)')
        rows.append((pr['number'], pr['user']['login'], outcome(pr), head,
                     pr['created_at'], pr['closed_at'] or '-'))

    print(f'{REPO}, read {len(rows)} PRs\n')
    print(f'{"#":>3}  {"author":<16} {"outcome":<7} {"head repo":<24} {"opened":<21} closed')
    for number, user, state, head, opened, closed in rows:
        inside = 'lab' if head == REPO else 'fork'
        print(f'{number:>3}  {user:<16} {state:<7} {head:<24} {opened:<21} {closed}   [{inside}]')

    for label, want in (('head branch in ' + REPO, True), ('head branch in a fork', False)):
        group = [r for r in rows if (r[3] == REPO) is want]
        merged = [r for r in group if r[2] == 'MERGED']
        closed = [r for r in group if r[2] == 'CLOSED']
        opened = [r for r in group if r[2] == 'open']
        resolved = len(merged) + len(closed)
        rate = f'{len(merged)}/{resolved}' if resolved else 'n/a'
        print(f'\n{label}: {len(group)} PRs -- {len(merged)} merged, {len(closed)} closed '
              f'unmerged, {len(opened)} open. Merged of resolved: {rate}')
        if opened:
            print('  still open: ' + ', '.join(f'#{r[0]} ({r[1]})' for r in opened))
    return 0


if __name__ == '__main__':
    sys.exit(main())
