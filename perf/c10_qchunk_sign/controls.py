"""Known-answer controls for the parts of this instrument that c10-bare-baseline does not already
cover: the ABBA schedule, the pairing, the sign convention, the drift immunity of the rep contrast,
the census gate and the shipped chunk rule. CPU only, no device.

c10-bare-baseline's own 13 controls (timer units, clock rejection, holder rejection, geometry,
route counters, host CPU witness) are closed and are not repeated here; the modules that carry them
are imported unmodified and pinned by sha256 in imports.json."""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from capture import schedule
from reduce import pair_up, abba_reps, census_verdict, boot_median

PASS, FAIL = [], []
def check(name, got, want):
    (PASS if got == want else FAIL).append((name, got, want))
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: {got!r}" + ('' if got == want else f" != {want!r}"))

def rows(times, reps=8):
    """Build the timed sequence the schedule produces, with `times` keyed by arm."""
    out = []
    for label, arm in schedule(reps)[2:]:
        out.append({'label': label, 'arm': arm, 'accepted': True, 'elapsed_s': times[arm]})
    return out

# ---- 1-3. the schedule is ABBA and balanced
plan = schedule(8)
check('schedule warms one census fold per arm', [x[1] for x in plan[:2]], ['on', 'off'])
check('every rep is on off off on', {tuple(x[1] for x in plan[2 + 4 * i:6 + 4 * i]) for i in range(8)},
      {('on', 'off', 'off', 'on')})
check('arms are balanced over the timed sequence',
      [sum(1 for x in plan[2:] if x[1] == a) for a in ('on', 'off')], [16, 16])

# ---- 4-5. the pairing splits the sequence the way the design says
cross, same = pair_up(rows({'on': 10.0, 'off': 11.0}))
check('8 reps give 16 cross pairs', len(cross), 16)
check('8 reps give 15 same-arm A/A pairs', len(same), 15)

# ---- 6-7. the sign convention, in both directions
check('off slower by 1.000 s reads +1.000', round(boot_median([c['off_minus_on_s'] for c in cross])['median'], 6), 1.0)
cross_loss, _ = pair_up(rows({'on': 11.0, 'off': 10.0}))
check('off FASTER by 1.000 s reads -1.000 (a live regression is visible)',
      round(boot_median([c['off_minus_on_s'] for c in cross_loss])['median'], 6), -1.0)

# ---- 8. an identical-arm pair reads exactly zero and its CI contains zero
_, same_flat = pair_up(rows({'on': 10.0, 'off': 10.0}))
flat = boot_median([s['delta_s'] for s in same_flat])
check('a true A/A reads 0 and does not exclude zero', (flat['median'], flat['excludes_zero']), (0.0, False))

# ---- 9. the ABBA rep contrast is exactly immune to a linear drift, which adjacent pairing is not
drift = rows({'on': 10.0, 'off': 10.0})
for i, r in enumerate(drift): r['elapsed_s'] += 0.01 * i          # pure ramp, no lever at all
check('a pure linear ramp leaves the ABBA rep contrast at 0',
      round(boot_median([r['off_minus_on_s'] for r in abba_reps(drift, 8)])['median'], 9), 0.0)
blocked = [{'label': f'B{i}', 'arm': 'on' if i < 16 else 'off', 'accepted': True, 'elapsed_s': 10.0 + 0.01 * i}
           for i in range(32)]                                   # same ramp, arms run in blocks
naive = (statistics.median([r['elapsed_s'] for r in blocked if r['arm'] == 'off'])
         - statistics.median([r['elapsed_s'] for r in blocked if r['arm'] == 'on']))
check('the same ramp in a BLOCKED order fakes a 0.160 s lever out of pure drift', round(naive, 9), 0.16)

# ---- 10-13. the census gate, which is what proves the arm switch reached the device
good = {'on': {'calls': 6000, 'sites_moved': ['token_dit']}, 'off': {'calls': 6000, 'sites_moved': []}}
check('a census with one site moved ON and none OFF passes', census_verdict(good), True)
check('a census that moves nothing with the flag ON is refused',
      census_verdict({'on': {'calls': 6000, 'sites_moved': []}, 'off': {'calls': 6000, 'sites_moved': []}}), False)
check('a census that moves a site with the flag OFF is refused',
      census_verdict({'on': {'calls': 6000, 'sites_moved': ['token_dit']}, 'off': {'calls': 6000, 'sites_moved': ['token_dit']}}), False)
check('arms that did not make the same calls are refused',
      census_verdict({'on': {'calls': 6000, 'sites_moved': ['token_dit']}, 'off': {'calls': 5999, 'sites_moved': []}}), False)

# ---- 14. a rejected fold is never paired across the gap it leaves
holed = rows({'on': 10.0, 'off': 11.0}); holed[1]['accepted'] = False
check('a rejected fold drops its own two pairs (one cross, one A/A) and its whole rep, never bridging the gap',
      (len(pair_up(holed)[0]), len(pair_up(holed)[1]), len(abba_reps(holed, 8))), (15, 14, 7))

# ---- 15-16. the shipped rule itself, replayed on CPU from tt_bio
import predict_picks
picks = {(r['size'], r.get('site', 'token_dit')): r for r in predict_picks.rows()}
check('512 aa token_dit: the rule takes 256 -> 128, 32 -> 64 units on 110 cores',
      (picks[(512, 'token_dit')]['shipped_chunk'], picks[(512, 'token_dit')]['rule_chunk'],
       picks[(512, 'token_dit')]['shipped_units'], picks[(512, 'token_dit')]['rule_units']), (256, 128, 32, 64))
check('298 aa token_dit: q pads to 320, 256 does not divide it, the rule takes 64, 32 -> 80 units',
      (picks[(298, 'token_dit')]['padded'], picks[(298, 'token_dit')]['shipped_chunk'],
       picks[(298, 'token_dit')]['rule_chunk'], picks[(298, 'token_dit')]['rule_units']), (320, 256, 64, 80))
check('the atom site has one tile of queries, so the rule provably cannot move it at either size',
      [picks[(s, 'atom')]['moved'] for s in (512, 298)], [False, False])

print(f"\n{len(PASS)} pass, {len(FAIL)} fail")
sys.exit(1 if FAIL else 0)
