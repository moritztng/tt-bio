"""Choose leg-(i) targets: widest dockq_wave spread, unambiguous single Ab-Ag interface.

Classification is by the H/L/scFv/Fab/nanobody tokens in the RCSB entity description, matched
hyphen-tolerantly and without word-boundary assumptions after "Fab" -- real depositions write
"Heavy-chain of scFv clone 2", "Fab1b heavy chain", "IgG heavy chain" and "4F11 Heavy Chain",
all of which a naive /\bfab\b|heavy chain/ screen silently drops into the antigen bucket.

Exclusions, each for a stated reason:
  - more than one distinct antibody (two heavy-chain entities): which Ab-Ag interface is "the"
    interface is undefined, so our per-interface label has no unique referent.
  - more than one antigen entity (e.g. a 5-subunit acetylcholine receptor): the Fab binds one
    subunit while dockq_wave averages every intra-antigen interface too, so the offset stops
    being interpretable as the H-L dilution effect we are testing for.
  - any of H / L / antigen present in more than one copy: dodges
    dockq-multicopy-chain-mapper-false-zero, where the auto mapper picks a non-contacting copy
    and reports 0.0.
Finally at most one target per antigen, so a single deposition series cannot supply the sample.
"""
import json
import re
import sys
import urllib.request

HEAVY = re.compile(r'heavy[- ]chain|chain.{0,4}heavy|\bvhh\b|nanobody|single[- ]domain', re.I)
LIGHT = re.compile(r'light[- ]chain|chain.{0,4}light|kappa chain|lambda chain', re.I)
IGISH = re.compile(r'heavy[- ]chain|light[- ]chain|chain.{0,4}(heavy|light)|kappa chain'
                   r'|lambda chain|\bscfv\b|fab|nanobody|\bvhh\b|single[- ]domain'
                   r'|\bfv\b|immunoglobulin|\bigg\b|antibody', re.I)

spread = json.load(open(sys.argv[1]))
top = [r for r in spread if r["iqr"] >= float(sys.argv[3] if len(sys.argv)>3 else 0.15)]
ids = [r["id"] for r in top]

Q = ('{entries(entry_ids:[%s]){rcsb_id polymer_entities{'
     'rcsb_polymer_entity{pdbx_description}'
     'rcsb_polymer_entity_container_identifiers{auth_asym_ids}'
     'entity_poly{rcsb_sample_sequence_length}}}}')
req = urllib.request.Request(
    'https://data.rcsb.org/graphql',
    data=json.dumps({'query': Q % ','.join('"%s"' % x for x in ids)}).encode(),
    headers={'Content-Type': 'application/json'})
data = json.load(urllib.request.urlopen(req, timeout=120))

iqr = {r["id"]: r for r in top}
resolved = {e['rcsb_id']: [{
    'desc': (pe['rcsb_polymer_entity'] or {}).get('pdbx_description') or '',
    'chains': (pe['rcsb_polymer_entity_container_identifiers'] or {}).get('auth_asym_ids') or [],
    'len': (pe['entity_poly'] or {}).get('rcsb_sample_sequence_length') or 0,
} for pe in e['polymer_entities']] for e in data['data']['entries']}

eligible, rejected = [], []
for pid in ids:
    ents = resolved.get(pid)
    if not ents:
        rejected.append((pid, 'not resolved by RCSB')); continue
    igs = [x for x in ents if IGISH.search(x['desc'])]
    ags = [x for x in ents if not IGISH.search(x['desc']) and x['len'] >= 30]
    Hs = [x for x in igs if HEAVY.search(x['desc'])]
    Ls = [x for x in igs if LIGHT.search(x['desc'])]
    if len(Hs) > 1:
        rejected.append((pid, '%d antibodies (%s) -- no unique Ab-Ag interface'
                         % (len(Hs), '; '.join(x['desc'][:22] for x in Hs)))); continue
    if not Hs:
        rejected.append((pid, 'no antibody heavy/VHH entity: %s'
                         % '; '.join(x['desc'][:26] for x in ents)[:70])); continue
    if len(ags) != 1:
        rejected.append((pid, '%d antigen entities (%s)'
                         % (len(ags), '; '.join(x['desc'][:20] for x in ags)[:56]))); continue
    H, L, ag = Hs[0], (Ls[0] if Ls else None), ags[0]
    multi = [x for x in ([H, ag] + ([L] if L else [])) if len(x['chains']) != 1]
    if multi:
        rejected.append((pid, 'multi-copy: %s' % '; '.join(
            '%s x%d' % (x['desc'][:20], len(x['chains'])) for x in multi))); continue
    eligible.append({'id': pid, 'iqr': iqr[pid]['iqr'], 'median': iqr[pid]['median'],
                     'H': H['chains'][0], 'L': L['chains'][0] if L else None,
                     'A': ag['chains'][0], 'antigen': ag['desc'], 'ag_len': ag['len'],
                     'vhh_only': L is None})

print('candidates iqr>=0.15: %d | eligible: %d' % (len(ids), len(eligible)))
print('\nREJECTED (%d):' % len(rejected))
for pid, why in rejected:
    print('  %-6s %s' % (pid, why))

seen, picked = set(), []
for c in sorted(eligible, key=lambda c: -c['iqr']):
    key = re.sub(r'[^a-z0-9]', '', c['antigen'].lower())[:24]
    if key in seen:
        continue
    seen.add(key); picked.append(c)
    if len(picked) == 10:
        break

print('\nLEG (i) TARGET LIST (%d of 10 requested):' % len(picked))
print('%-6s %6s %6s  %-3s %-3s %-3s %5s  %s' % ('id', 'iqr', 'med', 'H', 'L', 'A', 'aglen', 'antigen'))
for c in picked:
    print('%-6s %6.3f %6.3f  %-3s %-3s %-3s %5d  %s'
          % (c['id'], c['iqr'], c['median'], c['H'], c['L'] or '-', c['A'], c['ag_len'], c['antigen'][:44]))
json.dump(picked, open(sys.argv[2], 'w'), indent=1)
print('\nwrote', sys.argv[2])
