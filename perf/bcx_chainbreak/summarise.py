"""Turn the leg JSONs in out/ into the tables this row reports.

Every number quoted in TABLES.md and in the upstream issue comes from here, so a
transcription error has nowhere to hide.
"""
import collections, glob, json, os, statistics, sys

WINDOW = 32
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

def load(directory=None):
    directory = directory or OUT
    legs = {}
    paths = glob.glob(f'{directory}/*.json')
    if not paths:
        sys.exit(f'no leg JSONs under {directory} -- nothing to summarise')
    for path in paths:
        leg = json.load(open(path))
        legs[(leg['leg'], leg['binder'], leg['target'], leg['target_start'],
              leg['chains'], leg.get('seed', 0))] = leg
    return legs

def in_window(binder_length, target_length, target_start):
    return sum(1 for i in range(1, binder_length + 1)
               for j in range(target_start, target_start + target_length) if abs(j - i) <= WINDOW)

def paired(legs):
    pairs = collections.defaultdict(dict)
    for (leg, binder, target, start, chains, seed), value in legs.items():
        if leg in ('predict', 'predict_identity') and chains == 2:
            pairs[(binder, target, start, seed)][leg] = value
    return {key: value for key, value in pairs.items() if len(value) == 2}

def distributions(pairs):
    by_arm = collections.defaultdict(list)
    for (binder, target, start, seed), value in pairs.items():
        by_arm[(binder, target, start)].append(
            {metric: value['predict_identity'][metric] - value['predict'][metric]
             for metric in ('iptm', 'ptm', 'plddt', 'binder_plddt')})
    return by_arm

def main():
    legs = load(sys.argv[1] if len(sys.argv) > 1 else OUT)
    pairs = paired(legs)

    print('## Gradient-path numbering minus predict-path numbering, per binder draw\n')
    print('| binder | target | target numbered from | draws | in-window pairs | in-window share | i_pTM gap (each draw) | median |')
    print('|---|---|---|---|---|---|---|---|')
    for (binder, target, start), gaps in sorted(distributions(pairs).items()):
        values = sorted(gap['iptm'] for gap in gaps)
        inside = in_window(binder, target, start)
        print('| %d | %d | %d | %d | %d | %.1f%% | %s | %+.4f |' % (
            binder, target, start, len(values), inside, 100 * inside / (binder * target),
            ', '.join('%+.4f' % value for value in values), statistics.median(values)))

    print('\n## Other metrics, same pairs (median over draws)\n')
    print('| binder | target | i_pTM | pTM | pLDDT | binder pLDDT |')
    print('|---|---|---|---|---|---|')
    for (binder, target, start), gaps in sorted(distributions(pairs).items()):
        if start != 1:
            continue
        print('| %d | %d | %s |' % (binder, target, ' | '.join(
            '%+.4f' % statistics.median([gap[metric] for gap in gaps])
            for metric in ('iptm', 'ptm', 'plddt', 'binder_plddt'))))

    print('\n## Identities (bit-for-bit checks)\n')
    def show(label, left, right):
        if left not in legs or right not in legs:
            return
        same = [metric for metric in ('ptm', 'iptm', 'plddt', 'binder_plddt')
                if legs[left].get(metric) == legs[right].get(metric)]
        worst = max((abs(legs[left][metric] - legs[right][metric])
                     for metric in ('ptm', 'iptm', 'plddt', 'binder_plddt')
                     if metric in legs[left] and metric in legs[right]), default=0.0)
        print('- %s: %d of 4 metrics bit-identical, largest difference %.3g' % (label, len(same), worst))
    show('sequence_gradients vs predict with the chain break made the identity',
         ('gradient', 32, 96, 1, 2, 0), ('predict_identity', 32, 96, 1, 2, 0))
    show('sequence_gradients fed the chain-broken numbering vs predict',
         ('gradient', 32, 96, 82, 2, 0), ('predict', 32, 96, 1, 2, 0))
    show('single chain, sequence_gradients vs predict',
         ('gradient', 64, 64, 1, 1, 0), ('predict', 64, 64, 1, 1, 0))
    show('zero-window control, the two numberings against each other',
         ('predict', 32, 96, 4000, 2, 0), ('predict_identity', 32, 96, 4000, 2, 0))

if __name__ == '__main__':
    main()
