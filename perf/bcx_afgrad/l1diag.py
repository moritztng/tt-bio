import sys, pathlib, collections, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.bcx_afgrad.afgrad import load_models, Dev, embed
dm, ref = load_models("/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz")
dev = Dev(dm); ag = dev.ag; ttnn = dev.ttnn
m0, z0 = embed(ref["bf16"], torch.randn(128, 20), torch.arange(128))
ml, zl = dev.leaf(m0.detach()), dev.leaf(z0.detach())
kind = sys.argv[1] if len(sys.argv) > 1 else "extra"
with dev.tt.tape():
    if kind == "extra":
        zo = dev.extra(0, zl); roots = [zo]
    else:
        mo, zo = dev.evo(0, ml, zl); roots = [mo, zo]
order = ag._reverse_topo(roots)
# every Tensor reachable, including leaves, check L1
for t in order:
    try:
        bt = t.value.memory_config().buffer_type
        alloc = t.value.is_allocated()
    except Exception as e:
        continue
    if bt == ttnn.BufferType.L1 and alloc:
        prod = t.node.fn.__qualname__.split(".<locals>")[0] if t.node else "leaf/no-node"
        # consumers
        cons = [c.node.fn.__qualname__.split(".<locals>")[0] for c in order if c.node and any(p is t for p in c.node.parents)]
        print("L1", tuple(t.value.shape), t.value.dtype, "prod", prod, "pinned", t.pinned, "evictable", t.evictable, "consumers", cons, "addr", t.value.buffer_address())
