#!/usr/bin/env python3
"""When does PacesaLab/BindCraft2 act, and how long after an outside contribution lands?

Reads the GitHub API live. Every PR and issue: who opened it, when, and the first action by
someone in the lab (a merge, a close, a comment, a review, a reopen, a review request). The
point is a resume time backed by the repo's own behaviour rather than a guess.
"""
import json, subprocess, datetime as dt, statistics

LAB = {'martinpacesa', 'ErikMaeots', 'LeonardoTredese', 'rbedi'}

def api(path):
    out = subprocess.run(['gh', 'api', '--paginate', path], capture_output=True, text=True, check=True).stdout
    # --paginate concatenates arrays; gh returns one array per page already merged for --paginate
    return json.loads(out) if out.strip().startswith('[') else json.loads(out)

def when(s):
    return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)

items = api('repos/PacesaLab/BindCraft2/issues?state=all&per_page=100')
rows, lab_actions = [], []
for item in items:
    number, opener, opened = item['number'], item['user']['login'], when(item['created_at'])
    kind = 'PR' if item.get('pull_request') else 'issue'
    events = api(f"repos/PacesaLab/BindCraft2/issues/{number}/events")
    comments = api(f"repos/PacesaLab/BindCraft2/issues/{number}/comments")
    acts = [(when(e['created_at']), e['event'], (e.get('actor') or {}).get('login')) for e in events]
    acts += [(when(c['created_at']), 'comment', c['user']['login']) for c in comments]
    if kind == 'PR':
        acts += [(when(r['submitted_at']), 'review', r['user']['login'])
                 for r in api(f"repos/PacesaLab/BindCraft2/pulls/{number}/reviews") if r.get('submitted_at')]
    by_lab = sorted(a for a in acts if a[2] in LAB)
    lab_actions += by_lab
    first = by_lab[0] if by_lab else None
    rows.append((number, kind, opener, opener in LAB, opened, first,
                 (first[0] - opened).total_seconds() / 3600 if first else None))

print(f"{'#':>3} {'kind':<5} {'opener':<14} {'outside':<7} {'opened (UTC)':<20} {'first lab action':<20} {'h':>7}")
outside_latency = []
for number, kind, opener, is_lab, opened, first, hours in sorted(rows):
    label = f"{first[1]} by {first[2]}" if first else '-'
    print(f"{number:>3} {kind:<5} {opener:<14} {'no' if is_lab else 'YES':<7} "
          f"{opened:%Y-%m-%d %H:%M}     {label:<20} {'-' if hours is None else format(round(hours,2)):>7}")
    if not is_lab and hours is not None:
        outside_latency.append(hours)

print(f"\noutside items with a lab action: n={len(outside_latency)}")
if outside_latency:
    s = sorted(outside_latency)
    print(f"  hours to first lab action: min {s[0]:.2f}  median {statistics.median(s):.2f}  max {s[-1]:.2f}")

hours_of_day = sorted(a[0].hour for a in lab_actions)
print(f"\nlab actions total: n={len(hours_of_day)}, UTC hour-of-day:")
print("  " + " ".join(f"{h:02d}" for h in hours_of_day))
in_window = [h for h in hours_of_day if 7 <= h < 18]
print(f"  inside 07:00-18:00Z: {len(in_window)}/{len(hours_of_day)} = {100*len(in_window)/len(hours_of_day):.0f}%")
print(f"  earliest {min(hours_of_day):02d}:xx   latest {max(hours_of_day):02d}:xx")
by_weekday = {}
for a in lab_actions:
    by_weekday.setdefault(a[0].strftime('%a'), 0)
    by_weekday[a[0].strftime('%a')] += 1
print(f"  by weekday: {by_weekday}")
