import os, sys, statistics as st, time, json
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/esmfold2-to-3p4x"); sys.path.insert(0, str(ROOT))
import torch, ttnn
import tt_bio.tenstorrent as T, tt_bio.reblock_permute as RP
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get('TT_MESH_GRAPH_DESC_PATH'):
    m = _find_ttnn_mesh_graph_descriptor('p150_mesh_graph_descriptor.textproto')
    if m: os.environ['TT_MESH_GRAPH_DESC_PATH'] = m
dev = T.get_device()
res = {}
torch.manual_seed(0)
for N in (128, 256):
    C, CW = 256, 1024
    xw = ttnn.from_torch(torch.randn(1, N, N, CW), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    ref = ttnn.to_torch(RP.reblock_permute_gated(xw, 2 * C, 0, C))
    for Rb in (32, 64):
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, N, N]), ttnn.bfloat16,
                                             ttnn.TILE_LAYOUT, dev, xw.memory_config())
        for r in range(0, N, Rb):
            blk = ttnn.slice(xw, [0, r, 0, 0], [1, r + Rb, N, CW])
            RP.reblock_permute_gated(blk, 2 * C, 0, C, out=out, row_off=r)
            ttnn.deallocate(blk)
        eq = bool(torch.equal(ref, ttnn.to_torch(out)))
        res[f'N{N}_R{Rb}'] = eq
        print(f'N={N} rowblock={Rb}: torch.equal={eq}', flush=True)
        ttnn.deallocate(out)
    # the whole-tensor move must be untouched
    res[f'N{N}_whole_selfconsistent'] = bool(torch.equal(
        ref, ttnn.to_torch(RP.reblock_permute_gated(xw, 2 * C, 0, C))))
    print(f'N={N} whole-tensor unchanged: {res[f"N{N}_whole_selfconsistent"]}', flush=True)
    ttnn.deallocate(xw)
# the ungated forward move still works (its writer gained a common arg)
x = ttnn.from_torch(torch.randn(1, 320, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
g = ttnn.to_torch(RP.reblock_permute(x))
res['ungated_forward_vs_ttnn_permute'] = bool(torch.equal(
    g, ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2)))))
print('ungated forward vs ttnn.permute:', res['ungated_forward_vs_ttnn_permute'], flush=True)
Path(ROOT / 'perf/esm3p4/rowblock_parity_c0.json').write_text(json.dumps(res, indent=1))
print('ALL PASS' if all(res.values()) else 'FAILURES: ' + str(res))
