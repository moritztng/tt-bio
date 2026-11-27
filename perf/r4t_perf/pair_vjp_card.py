"""P2 block grade: PairOnCard.taped (winner z_init on the card + its backward) vs float64 upstream autograd.

Real batch, the K-candidate constant on the card as trunk_step puts it, winner row `win`, random masked dz.
Per pair-builder parameter: rel_l2 and cos of the card gradient vs float64, beside the host float32 floor
(upstream autograd in float32). BREAK=1 rolls the step one-hot by a bin in the features dW reads: must FAIL.
Bar (pre-registered here, D28 component class): every tensor rel_l2 <= 1e-2 and cos >= 0.9999."""
import sys, os, json, copy, torch
sys.path.insert(0, os.path.join(os.environ["WT"], "perf", "r4t"))
import ttnn
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as T, autograd as ag
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
from tt_bio.rfd4.pair import PairSplit, PairOnCard, PairInputs
from tt_bio.train.bridge import Bridge

batch, win = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
glue = N.FlowGlue(state, config)
bt, meta, *_ = load_batch(batch)
K = int(bt["padding_mask"].shape[0])
def take(d, i):
    if isinstance(d, dict): return {k: take(v, i) for k, v in d.items()}
    return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d
bw = take(bt, win)
n = int(bw["padding_mask"].shape[-1])
m = (bw["padding_mask"][0][:, None] * bw["padding_mask"][0][None, :]).float()
G = torch.randn(1, n, n, 384, generator=torch.Generator().manual_seed(7)) * m[None, ..., None]
pb = glue.pair_repr_builder
names = [nm for nm, _ in pb.named_parameters()]

def host(dtype):
    b = copy.deepcopy(pb).to(dtype).requires_grad_(True)
    for mm in (b.init_repr_factory.linear_out, b.cond_factory.linear_out):
        mm.register_forward_pre_hook(lambda mod, inp: tuple(i.to(dtype) for i in inp))
    z = b(bw); (z * G.to(dtype)).sum().backward()
    return z.detach(), {nm: p.grad.detach().double() for nm, p in b.named_parameters()}
z64, ref = host(torch.float64)
_, floor = host(torch.float32)

dev = T.get_device()
split = PairSplit(pb); glue.pair_split = split
pc = PairOnCard(split, dev)
if os.environ.get("BREAK"):
    f0 = split.features
    def rolled(b):
        f = f0(b).clone(); s = [i for i, c in enumerate(pb.init_repr_factory.feat_creators)
                                if type(c).__name__ == "XtBBCAPairwiseDistancesPairFeat"][0]
        a = sum(c.get_dim() for c in pb.init_repr_factory.feat_creators[:s]); e = a + 30
        f[..., a:e] = f[..., a:e].roll(1, -1); return f
    split.features = rolled
with torch.no_grad():
    z0 = glue.initial_states(bt, pair_constant=split.constant(bt))[1]
    zk = pc(z0, 0); ttnn.deallocate(zk)   # the candidate pass: puts the dedup constant on the card
pb.requires_grad_(True)
gamma, beta = split.modulation(bw)
br = Bridge(dev, ttnn.float32)
with ag.tape():
    zt = pc.taped(PairInputs(z0.constant, z0.bins[win:win + 1], z0.scale[win:win + 1], gamma, beta), br, bw, ttnn.float32)
    zc = ttnn.to_torch(zt.value).float().reshape(1, n, n, 384)
    ag.backward([zt], [ttnn.from_torch(G, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)])
for h, leaf in br.ups:
    if leaf.grad is not None:
        torch.autograd.backward([h], [ttnn.to_torch(leaf.grad).to(h.dtype).reshape(h.shape)])
card = {nm: p.grad.detach().double() for nm, p in pb.named_parameters()}
zerr = ((zc.double() - z64) * m[None, ..., None]).norm() / (z64 * m[None, ..., None]).norm()
rows, ok = [], True
for nm in names:
    r, g = ref[nm], card[nm]
    rel = ((g - r).norm() / r.norm()).item(); cos = torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), 0).item()
    fl = ((floor[nm] - r).norm() / r.norm()).item()
    ok &= rel <= 1e-2 and cos >= 0.9999
    rows.append(dict(name=nm, rel=rel, cos=cos, floor=fl, ratio=rel / max(fl, 1e-30)))
    print(f"{nm:42s} rel {rel:.3e} cos {cos:.8f} floor {fl:.2e} x{rel / max(fl, 1e-30):.1f}")
print("Z_REL", f"{zerr.item():.3e}")
print("P2_GRADE", json.dumps(dict(batch=os.path.basename(batch), win=win, broken=bool(os.environ.get("BREAK")),
                                  verdict="PASS" if ok else "FAIL", worst=max(r["rel"] for r in rows), z_rel=zerr.item())))
