#!/usr/bin/env python3
"""How the PairformerLayer chain writes its linears and norms: 29 raw, 0 through a helper."""
import re
from collections import Counter
s = open('/home/ttuser/.coworker/wt/train-a1-defork/tt_bio/tenstorrent.py').read().splitlines()
starts = [(i, l) for i, l in enumerate(s) if re.match(r'^class ', l)]
def body(name):
    for k, (i, l) in enumerate(starts):
        if l.startswith('class ' + name):
            return i, (starts[k+1][0] if k+1 < len(starts) else len(s))
for n in ['PairformerLayer', 'TriangleMultiplication', 'TriangleAttention', 'Transition',
          'AttentionPairBias']:
    i, j = body(n)
    seg = '\n'.join(s[i:j])
    c = Counter(re.findall(r'ttnn\.(?:experimental\.)?([a-z_0-9]+)\(', seg))
    print(n)
    print("   raw ttnn.linear=%d  ttnn.matmul=%d  ttnn.layer_norm=%d"
          % (c['linear'], c['matmul'], c['layer_norm']))
    print("   self._lin=%d  self._ln=%d  batched_matmul=%d  _narrow_proj_linear=%d  _l1_layer_norm=%d"
          % (len(re.findall(r'self\._lin\(', seg)), len(re.findall(r'self\._ln\(', seg)),
             len(re.findall(r'batched_matmul\(', seg)),
             len(re.findall(r'_narrow_proj_linear\(', seg)),
             len(re.findall(r'_l1_layer_norm\(', seg))))
