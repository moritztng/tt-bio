"""Print the reduced run as the table this row reports. Reads analysis.json only."""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('analysis', type=Path); a = ap.parse_args()
A = json.loads(a.analysis.read_text())
print('VERDICT', A['verdict'], '|', A['flag']['name'], 'shipped default', A['flag']['shipped_default'])
for size, t in A['targets'].items():
    s = t['statistics']; p = t['parity']
    print(f"\n===== {size} aa  verdict {t['verdict']}  accepted {t['accepted_folds']} folds "
          f"({s['arm_n']['on']} on / {s['arm_n']['off']} off)  clock {t['clock']['during_min_MHz']}-{t['clock']['during_max_MHz']} MHz "
          f"on {t['clock']['total_during_samples']} during-samples")
    for k in ['adjacent_cross_pairs', 'abba_rep_contrast', 'AA_same_arm_adjacent']:
        v = s[k]
        print(f"  {k:22s} n={v['n']:2d}  median {v['median']:+.4f} s  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
              f"  +{v['positive']}/-{v['negative']}  excludes 0: {v['excludes_zero']}"
              f"  = {v['Mcycles_at_1350']['median']:+.0f} Mcyc")
    r = s['adjacent_cross_ratio']
    print(f"  ratio off/on           median {1+r['median']:.5f}x  95% CI [{1+r['ci95'][0]:.5f}, {1+r['ci95'][1]:.5f}]")
    print(f"  arm medians            on {s['arm_medians_s']['on']:.4f} s   off {s['arm_medians_s']['off']:.4f} s")
    print(f"  A/A floor              this session |delta| median {s['AA_same_arm_adjacent_abs_median_s']:.4f} s"
          f"   c10-bare-baseline single-pair floor {s['baseline_single_pair_floor_s']:.3f} s")
    print(f"  parity                 bit-exact across arms: {p['bit_exact_across_arms']}"
          f"   max cross-arm domain RMSD {p['max_cross_arm_domain_A']:.2e} A over {p['cross_arm_pairs']} pairs"
          f"   bar {p['bar_A']} A   pLDDT on {p['plddt_by_arm']['on']} off {p['plddt_by_arm']['off']}")
    print(f"  above-cap route        {t['above_cap_route_counters']} in every fold")
    print(f"  host                   max foreign CPU {t['ambient']['max_foreign_cpu_pct']} %  max loadavg {t['ambient']['max_loadavg']:.2f}")
    for armname in ('on', 'off'):
        for row in t['census'][armname]:
            print(f"  census/{armname:3s} {row['site']:10s} q{row['q_len']:5d} k{row['k_len']:5d} work {row['work']:5d} d {row['d']:3d}"
                  f"  calls {row['calls']:5d}  {row['shipped_chunk']:4d} -> {row['chunk']:4d}  units {row['units']:4d}"
                  f"  {'MOVED' if row['moved'] else 'declined'}")
