#!/bin/bash
# score_all.sh [tree]: score every finished combination fold under <tree>/perf/mgx_combos/out
# against its crystal; the top-ranked structure first, then the best and worst sample.
cd "${1:-$(dirname "$0")/../..}"
declare -A GT=([pmhc]=perf/mgx_combos/templates/3gso.cif [mdh]=perf/mgx/ref/fixtures/gt/2ad6.cif.gz [eal]=perf/mgx/ref/fixtures/gt/3abq.cif.gz)
for d in perf/mgx_combos/out/*/*_s*/; do
  m=$(basename "$(dirname "$d")"); s=$(basename "$d"); stem=${s%_s*}
  cifs=$(find "$d" -name "*.cif" | sort); [ -n "$cifs" ] || continue
  top=$(find "$d" -name "$stem.cif" | head -1); [ -n "$top" ] || top=$(echo "$cifs" | head -1)
  $HOME/env/bin/python perf/mgx_combos/score.py --json "${GT[${stem%%_*}]}" $cifs 2>/dev/null | \
    $HOME/env/bin/python -c "
import json,sys
r=json.load(sys.stdin); top=[x for x in r if x['pred']=='$top'] or r[:1]
v=[x.get('ca_rmsd') for x in r if x.get('ca_rmsd') is not None]
t=top[0]; lig=' '.join(f\"{k}:{d['centroid_to_crystal']}/{d['min_contact']}\" for k,d in t.get('ligand_fit',{}).items())
print(f\"$m $s n={len(r)} chains={list(t['chains'].values())} top={t.get('ca_rmsd')} best={min(v) if v else None} worst={max(v) if v else None} lig(centroid/contact)={lig or '-'}\")"
done
