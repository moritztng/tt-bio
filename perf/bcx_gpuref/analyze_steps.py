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
"""
import argparse
import json
import statistics
import sys

COLD_COMPILE_SHARE = 0.10


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def report(records):
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
