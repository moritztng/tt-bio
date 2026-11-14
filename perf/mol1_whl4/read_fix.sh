#!/bin/bash
# read_fix.sh OUTROOT : per case, pLDDT / pTM of every sample, rc, and the fold's AICLK on its chip (.111)
ssh -o BatchMode=yes ubuntu@172.16.102.111 "cd $1 && for c in */; do c=\${c%/}; python3 - \$c <<'PY'
import json, glob, sys
c = sys.argv[1]
rc = open(f'{c}/rc').read().strip() if glob.glob(f'{c}/rc') else 'running'
r = glob.glob(f'{c}/fold/mol1_results_*/results.json')
if not r:
    print(c, rc, 'no results'); sys.exit()
d = json.load(open(r[0]))[0]
s = d.get('all_runs') or []
pl = [round(x['plddt'], 4) for x in s]
print(c, rc, 'plddt', d['plddt'], 'samples', pl, 'ptm', d['ptm'], 'iptm', d.get('iptm'), 'runtime_s', d.get('runtime_s'))
PY
n=\$(cat \$c/nodes.txt 2>/dev/null | head -1 | grep -o '[0-9]*\$'); [ -n \"\$n\" ] && awk -F'\t' -v n=\$n '\$2==n{print \$3}' \$c/aiclk.tsv | sort -n | uniq -c | sort -rn | head -2 | tr '\n' ' '; echo; done"
