#!/usr/bin/env python3
"""P0: the same device gradients, scored against BOTH float64 references.

of3t-apbgrad reports 0.38330656678074404 for its c64 RENORM arm and of3t-padshape reports
1.7043040667627918 for its width-64 RENORM arm. Both are the flipped arm, both are
TT_BIO_SOFTMAX_BW_RENORM=1, both are 56 real tokens in a 64-wide boundary. This scores the
banked tensors of both arms against both references with one scorer, so the axis that carries
the 4.45x is named rather than inferred.

REF-LOCAL is ref_grad.py's boundary-local float64 backward: upstream 0.4.3, every parameter and
activation float64, run on the SAME captured s_in/z_in with the SAME captured block-47
cotangent. REF-MODEL is the pinned grads_f64_043.pt, upstream's own full-model float64 backward
on batch_step003. They answer different questions and the campaign quotes both as 'float64'.
"""
import hashlib, json, sys, os
import numpy as np, torch

sys.path.insert(0, os.path.join(os.getcwd(), 'perf/of3t_trunkg043'))
from score import score, by_leaf

PIN = '1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4'

def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def load_grads(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    for k in ('grads', 'grad', 'g'):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return d

ARMS = {
  'apbgrad_scope_RENORM_c64': '/home/ttuser/of3t_apbgrad/dev_scope_RENORM_c64.pt',
  'padshape_RENORM_w64':      '/tmp/of3t/of3t-padshape/dev_RENORM_w64.pt',
  'padshape_CTRL_w64_A':      '/tmp/of3t/of3t-padshape/dev_CTRL_w64_A.pt',
}
REFS = {
  'REF_LOCAL_f64':  '/home/ttuser/of3t_trunkg043/ref_f64_c64.pt',
  'REF_LOCAL_bf16': '/home/ttuser/of3t_trunkg043/ref_bf16auto_c64.pt',
  'REF_MODEL_f64':  '/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt',
  'REF_MODEL_bf16': '/home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt',
}

out = {'what': __doc__.strip().splitlines()[0],
       'host': os.uname().nodename, 'card': 0, 'board': 'p300c Blackhole',
       'digests': {}, 'refs': {}, 'arms': {}, 'cross': {}}

if len(sys.argv) > 1 and sys.argv[1] == '--digest-only':
    p = REFS['REF_MODEL_f64']
    print(json.dumps({'path': p, 'sha256': sha256_file(p), 'matches_pin': sha256_file(p) == PIN}))
    raise SystemExit(0)

d = sha256_file(REFS['REF_MODEL_f64'])
out['digests']['REF_MODEL_f64'] = {'path': REFS['REF_MODEL_f64'], 'sha256': d,
                                   'matches_pin': d == PIN}
if d != PIN:
    raise SystemExit('REF_MODEL_f64 digest %s does not match the pin %s' % (d, PIN))

R = {}
for nm, p in REFS.items():
    g = load_grads(p)
    R[nm] = {k: v for k, v in g.items() if k.startswith('pairformer_stack.blocks.')}
    n2 = sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
             for v in R[nm].values() if v is not None)
    out['refs'][nm] = {'path': p, 'trunk_tensors': len(R[nm]),
                       'trunk_squared_gradient_norm': n2,
                       'trunk_gradient_norm': float(np.sqrt(n2))}
    print(nm, len(R[nm]), n2, flush=True)

for anm, ap in ARMS.items():
    if not os.path.exists(ap):
        out['arms'][anm] = {'path': ap, 'missing': True}
        continue
    G = load_grads(ap)
    G = {k: v for k, v in G.items() if k.startswith('pairformer_stack.blocks.') and v is not None}
    out['arms'][anm] = {'path': ap, 'tensors': len(G)}
    for rnm in ('REF_LOCAL_f64', 'REF_MODEL_f64'):
        keys = sorted(set(G) & set(R[rnm]))
        s = score(G, R[rnm], keys)
        rows = s.pop('_rows')
        s['leaf_error_mass'] = dict(list(by_leaf(rows).items())[:8]) if by_leaf(rows) else {}
        out['cross']['%s__vs__%s' % (anm, rnm)] = {
            'compared': len(keys), 'mass_weighted_rel_l2': s['mass_weighted_rel_l2'],
            'median': s['median_rel_l2_over_tensors'],
            'mass_weighted_norm_ratio': s['mass_weighted_norm_ratio'],
            'mass_weighted_cos': s['mass_weighted_cos'],
            'reference_squared_norm': s['reference_squared_norm'],
            'worst_by_error_mass': s['worst_by_error_mass'],
            'top8_by_error_mass': s['top8_by_error_mass']}
        print(anm, rnm, s['mass_weighted_rel_l2'], s['reference_squared_norm'], flush=True)
    del G

# the floors, same scorer, same compared sets
for rnm, fnm in (('REF_LOCAL_f64', 'REF_LOCAL_bf16'), ('REF_MODEL_f64', 'REF_MODEL_bf16')):
    keys = sorted(set(R[fnm]) & set(R[rnm]))
    s = score(R[fnm], R[rnm], keys)
    s.pop('_rows')
    out['cross']['FLOOR_%s__vs__%s' % (fnm, rnm)] = {
        'compared': len(keys), 'mass_weighted_rel_l2': s['mass_weighted_rel_l2'],
        'mass_weighted_cos': s['mass_weighted_cos'],
        'reference_squared_norm': s['reference_squared_norm']}
    print('FLOOR', fnm, rnm, s['mass_weighted_rel_l2'], flush=True)

with open(sys.argv[1], 'w') as fh:
    json.dump(out, fh, indent=2)
print(json.dumps({k: v.get('mass_weighted_rel_l2') for k, v in out['cross'].items()}, indent=1))
