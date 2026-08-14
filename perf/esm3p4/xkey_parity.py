#!/usr/bin/env python3
"""Cross-key parity for the gated move: the re-keyed kernel against the one it replaces.

`rowblock_parity.py` compares blocked against whole-tensor WITHIN one key, so it cannot see a
re-key that is wrong in the same way everywhere. This dumps the gated move's output at several N
and diffs the two keys byte for byte.

  --dump  writes <out>.pt   (run under the OLD code)
  --check reads <out>.pt and torch.equal's the current code against it

N=298 is in the list on purpose: it is the only case where D1 is not a multiple of 32, so the last
row tile is partly padding. The re-key moved the writer's staging-zero from once per (it, jt) to
once per (it, jt, ct), and that is exactly the path where getting it wrong is invisible at 512.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path("/home/ttuser/.coworker/wt/esmfold2-to-3p4x")
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T, tt_bio.reblock_permute as RP

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=('dump', 'check'), required=True)
ap.add_argument('--ref', type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
    m = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
    if m:
        os.environ['TT_MESH_GRAPH_DESC_PATH'] = m
dev = T.get_device()

CASES = [(128, 256, 1024), (256, 256, 1024), (298, 256, 1024), (320, 256, 1024),
         (512, 256, 1024)]
got = {}
for N, C, CW in CASES:
    torch.manual_seed(N)          # same draw in both runs, keyed on the case
    xw = ttnn.from_torch(torch.randn(1, N, N, CW), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    got[f'N{N}'] = ttnn.to_torch(RP.reblock_permute_gated(xw, 2 * C, 0, C))
    ttnn.deallocate(xw)
    print(f'  N={N} done', flush=True)

if a.mode == 'dump':
    torch.save(got, a.ref)
    print('dumped', a.ref)
else:
    ref = torch.load(a.ref)
    res = {k: bool(torch.equal(ref[k], v)) for k, v in got.items()}
    for k, v in res.items():
        print(f'  {k:8s} torch.equal vs old key: {v}', flush=True)
    Path(str(a.ref) + '.json').write_text(json.dumps(res, indent=1))
    print('ALL PASS' if all(res.values()) else 'FAILURES ' + str(res))
    sys.exit(0 if all(res.values()) else 1)
