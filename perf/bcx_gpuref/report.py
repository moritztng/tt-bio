"""Turn the step log and the design set into the figures the state doc quotes.

Everything here is read off `steps_*.jsonl` and the validated design set. The one number that
comes from outside is the device-side gradient phase, passed in, so the phase-matched ratio is
computed in one place instead of being retyped.
"""
import argparse
import json
import statistics as st

# bcx-orchestrator.md:526 -- the 125-step GRADIENT PHASE on qb2, not a whole trajectory.
TT_GRADIENT_PHASE_S = 8069.9
TT_STEPS = 124.9


def dist(values):
    values = sorted(values)
    if not values:
        return None
    return {'n': len(values), 'min': values[0], 'p50': st.median(values), 'mean': st.mean(values),
            'p90': values[int(0.9 * (len(values) - 1))], 'max': values[-1],
            'stdev': st.stdev(values) if len(values) > 1 else 0.0}


def line(name, d, unit='s'):
    if not d:
        return f'{name}: no sample'
    return (f'{name}: n={d["n"]} min={d["min"]:.4g} p50={d["p50"]:.4g} mean={d["mean"]:.4g} '
            f'p90={d["p90"]:.4g} max={d["max"]:.4g} sd={d["stdev"]:.3g} {unit}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    ap.add_argument('--designs')
    ap.add_argument('--ranked')
    args = ap.parse_args()
    r = json.load(open(args.report))

    warm, cold, phase = r['warm'], r['cold'], r['phase_split']
    print('== DENOMINATOR (compile excluded) ==')
    print(line('  warm exec_s ', warm['exec_s']))
    print(line('  warm call_s ', warm['call_s']))
    print(f'  warm steps: {warm["call_s"]["n"]}   padded buckets: {warm["buckets"]}')
    print(f'  spread: sd/mean = {warm["exec_s"]["stdev"] / warm["exec_s"]["mean"]:.2%} on exec, '
          f'{warm["call_s"]["stdev"] / warm["call_s"]["mean"]:.2%} on call')
    print('== COMPILE, measured apart ==')
    print(line('  cold call_s ', cold['call_s']))
    print(line('  cold compile_s', cold['compile_s']))
    print(f'  compile is paid {cold["call_s"]["n"]}x, once per process, not once per shape per card')

    print('== PHASE SPLIT, per trajectory ==')
    for k in ('gradient_phase_s', 'mutate_s', 'design_s', 'mpnn_validation_s', 'whole_cycle_s'):
        print(line(f'  {k:19s}', phase[k]))
    print(line('  gradient_steps    ', phase['gradient_steps'], unit='steps'))
    print(line('  gradient_share    ', phase['gradient_share_of_cycle'], unit=''))
    print(f'  trajectories logged {phase["trajectories_logged"]}, through MPNN '
          f'{phase["trajectories_through_mpnn"]}')

    # Only full-length trajectories are comparable to the device arm, which ran all 125 steps.
    full = [t for t in r['trajectories'] if t['gradient_steps'] == 125]
    gp = dist([t['gradient_phase_s'] for t in full])
    print('== PHASE-MATCHED, 125-step trajectories only ==')
    print(line('  gradient_phase_s  ', gp))
    if gp:
        print(f'  per step from the phase wall: {gp["p50"] / 125:.4f} s')
        print(f'  RATIO phase-matched (TT {TT_GRADIENT_PHASE_S} chip-s / H200 p50 {gp["p50"]:.2f} s)'
              f' = {TT_GRADIENT_PHASE_S / gp["p50"]:.1f}x')
        print(f'  RATIO per step (TT {TT_GRADIENT_PHASE_S / TT_STEPS:.1f} s / H200 warm p50 '
              f'{warm["call_s"]["p50"]:.4f} s) = '
              f'{(TT_GRADIENT_PHASE_S / TT_STEPS) / warm["call_s"]["p50"]:.1f}x')
    wc = dist([t['whole_cycle_s'] for t in r['trajectories'] if t['reached_mpnn']])
    if wc:
        print(f'  H200 whole cycle p50 {wc["p50"]:.1f} s against BC2\'s published 90.5 s GH200 '
              f'whole cycle = {wc["p50"] / 90.5:.2f}x')

    if args.designs:
        d = json.load(open(args.designs))
        print('== QUALITY ==')
        print(f'  accepted designs checked {d["checked"]}, passing {d["passing"]}, '
              f'failing {len(d["failing"])}')
        for des in d['designs']:
            print(f'    {des["path"].split("/")[-1]} chains={des["chains"]} '
                  f'binder={des["binder_chain"]} longest_run={des["binder_longest_residue_run"]} '
                  f'ok={des["ok"]} {des["problems"] or ""}')
    if args.ranked:
        import csv
        rows = list(csv.DictReader(open(args.ranked)))
        print(f'  ranked rows {len(rows)}')
        for row in rows:
            print(f'    rank {row["rank"]} len {row["length"]} i_pTM {row["i_pTM"]} '
                  f'pLDDT {row["pLDDT"]} pTM {row["pTM"]} i_pAE {row["i_pAE"]} '
                  f'i_pDAE {row["i_pDAE"]}')
        for k in ('i_pTM', 'pLDDT', 'pTM'):
            vals = [float(row[k]) for row in rows]
            print(line(f'  {k:19s}', dist(vals), unit=''))


main()
