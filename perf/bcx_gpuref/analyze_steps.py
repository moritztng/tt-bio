"""Turn the step log into the denominator: a distribution, and a measured compile split.

    python analyze_steps.py steps.jsonl --report denominator.json

The report separates four populations, because averaging them is how the previous reference ended
up 40 % a restatement of itself:

  * `warm`     -- steps whose shape was already compiled. This is the denominator. It is reported
                  as a distribution (n, min, p50, p90, max, mean, stdev), never as one number.
  * `cold`     -- the first step at a shape, which carries that shape's compilation.
  * `precompile` -- `compile_only=True` calls, BindCraft 2's background pre-compile of the next
                  length bucket (`campaign.py:102-106`). They are compilation, and they overlap
                  the running trajectory rather than blocking it, so they are counted apart from
                  both of the above.
  * `background` -- anything off the main thread, kept visible so an overlap never lands in the
                  denominator unnoticed.

A step is `cold` when its own measured `compile_s` is a material share of its wall, so the split
comes from the clock rather than from an assumption about which call index compiles.

The same log carries the PHASE records, and they answer a different question. `8,069.9 chip-s` on
the Tenstorrent side is the 125-step gradient phase; BindCraft 2's published 90.5 s/trajectory on a
GH200 is the whole cycle, gradient design plus ProteinMPNN redesign plus validation. `trajectories`
reports both on the same card in the same run: `gradient_phase_s` against `design_s` against
`whole_cycle_s`, per trajectory, so the ratio the campaign quotes can be phase-matched instead of
comparing our part to their whole.
"""
import argparse
import json
import statistics
import sys

COLD_COMPILE_SHARE = 0.10


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split(records):
    """Step records and phase records share one file, written in one pass, in order."""
    steps = [r for r in records if r.get('record', 'step') == 'step']
    phases = [r for r in records if r.get('record') == 'phase']
    return steps, phases


def trajectories(phases):
    """One row per trajectory: the gradient phase, the design wall and the whole cycle.

    A phase record is written when its call RETURNS, so the four `gradient_stage` records and the
    `mutate` record arrive before the `design` record that encloses them, and `mpnn_validation`
    arrives after it. Children are claimed by step range rather than by position, so a background
    pre-compile landing in the middle cannot be mistaken for a stage of this trajectory.
    """
    rows, pending = [], []
    for record in phases:
        if record.get('background'):
            continue
        name = record['phase']
        if name in ('gradient_stage', 'mutate'):
            pending.append(record)
            continue
        if name == 'design':
            low, high = record['first_step'], record['last_step']
            mine = [p for p in pending if low <= p['first_step'] and p['last_step'] <= high]
            pending = [p for p in pending if p not in mine]
            gradient = [p for p in mine if p['phase'] == 'gradient_stage']
            mutate = [p for p in mine if p['phase'] == 'mutate']
            rows.append({'design_s': record['wall_s'],
                         'gradient_phase_s': sum(p['wall_s'] for p in gradient),
                         'gradient_steps': sum(p['gradient_steps'] for p in gradient),
                         'gradient_stages': [round(p['wall_s'], 3) for p in gradient],
                         'mutate_s': sum(p['wall_s'] for p in mutate),
                         'mpnn_validation_s': None,
                         'whole_cycle_s': None,
                         'first_step': low,
                         'last_step': high})
            continue
        if name == 'mpnn_validation' and rows and rows[-1]['mpnn_validation_s'] is None:
            rows[-1]['mpnn_validation_s'] = record['wall_s']
    for row in rows:
        mpnn = row['mpnn_validation_s']
        row['whole_cycle_s'] = row['design_s'] + (mpnn or 0.0)
        row['reached_mpnn'] = mpnn is not None
    return rows


def bucket_of(record):
    padded = record.get('padded_residues')
    if isinstance(padded, dict) and padded:
        return max(padded.values())
    return None


def classify(record):
    if record.get('compile_only'):
        return 'precompile'
    if record.get('background'):
        return 'background'
    if record.get('compile_s', 0.0) > COLD_COMPILE_SHARE * max(record.get('call_s', 0.0), 1e-9):
        return 'cold'
    return 'warm'


def distribution(values):
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    return {'n': len(ordered),
            'min': ordered[0],
            'p50': statistics.median(ordered),
            'p90': ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
            'max': ordered[-1],
            'mean': statistics.fmean(ordered),
            'stdev': statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
            'total': sum(ordered)}


def report(all_records):
    records, phases = split(all_records)
    populations = {}
    for record in records:
        populations.setdefault(classify(record), []).append(record)
    summary = {'steps_logged': len(records)}
    for name, group in sorted(populations.items()):
        summary[name] = {'call_s': distribution([r['call_s'] for r in group]),
                         'exec_s': distribution([r.get('exec_s', 0.0) for r in group]),
                         'compile_s': distribution([r.get('compile_s', 0.0) for r in group]),
                         'buckets': sorted({b for b in (bucket_of(r) for r in group) if b})}
    warm = populations.get('warm', [])
    by_bucket = {}
    for record in warm:
        by_bucket.setdefault(bucket_of(record), []).append(record['call_s'])
    summary['warm_by_padded_residues'] = {str(k): distribution(v) for k, v in sorted(by_bucket.items(), key=lambda kv: (kv[0] is None, kv[0]))}

    rows = trajectories(phases)
    summary['trajectories'] = rows
    complete = [r for r in rows if r['reached_mpnn']]
    summary['phase_split'] = {
        'trajectories_logged': len(rows),
        'trajectories_through_mpnn': len(complete),
        'gradient_phase_s': distribution([r['gradient_phase_s'] for r in rows]),
        'gradient_steps': distribution([r['gradient_steps'] for r in rows]),
        'mutate_s': distribution([r['mutate_s'] for r in rows]),
        'design_s': distribution([r['design_s'] for r in rows]),
        'mpnn_validation_s': distribution([r['mpnn_validation_s'] for r in complete]),
        'whole_cycle_s': distribution([r['whole_cycle_s'] for r in complete]),
        # The share is what decides whether 'our gradient phase vs their whole cycle' flatters us,
        # and by how much. Only trajectories that ran the whole cycle can answer it.
        'gradient_share_of_cycle': distribution([r['gradient_phase_s'] / r['whole_cycle_s']
                                                 for r in complete if r['whole_cycle_s'] > 0]),
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('jsonl')
    parser.add_argument('--report', help='write the summary here as JSON')
    arguments = parser.parse_args(argv)
    summary = report(load(arguments.jsonl))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if arguments.report:
        with open(arguments.report, 'w') as handle:
            handle.write(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
