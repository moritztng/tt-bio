"""Normalized graph of one trimul call, one node per line, for a base/fix diff."""
import json, sys
import torch, ttnn
sys.path.insert(0, sys.argv[2])
import optrace as O
T = O.T
dev = T.get_device()
L = int(sys.argv[3]) if len(sys.argv) > 3 else 384
tm = T.TriangleMultiplication(ending=False, state_dict=O.weights(), compute_kernel_config=O.ck(dev))
x = ttnn.from_torch(torch.randn(1, L, L, 128, generator=torch.Generator().manual_seed(L)),
                    layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
tm(x)
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
tm(x)
gr = ttnn.graph.end_graph_capture()
with open(sys.argv[1], "w") as fh:
    for n in json.loads(O.norm(gr)):
        fh.write(json.dumps(n, sort_keys=True) + "\n")
