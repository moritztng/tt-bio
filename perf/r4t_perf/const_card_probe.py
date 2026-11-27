"""C1 probe (r4t-perf): can the card build the candidate pass's pair constant and its norm scale
without the host's float64 table? Pre-registered 2026-11-27 16:58Z, before any card number:
  PASS iff  proj   max|card - f64| / rms(f64) <= 1e-6
       and  scale  max rel err vs float64 at the K candidates' bins <= 1e-5   (27x under the
            2.7e-4 card ttnn.sum route that moved 512 designs 2.1 A, pair.py docstring)
BREAK=1 sums with ttnn.sum (that failed route) and must FAIL.
Card: proj = sum_j F_bf16 @ split_bf16(W_const^T)_j (bf16 x bf16 products exact, fp32 dest);
row sums of x^2 and (x w)^2 as [hi|mid|lo] bf16 parts @ ones, fp32 dest (no tf32 unpack)."""
import json, os, sys, time
sys.path.insert(0, "perf/r4t")
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from loss_parity_reference import load_batch
from tt_bio import tenstorrent as T, autograd as ag
from tt_bio.rfd4 import network as N, engine as E
from tt_bio.rfd4.pair import PairOnCard, split_bf16

BREAK = os.environ.get("BREAK") == "1"
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
g = N.FlowGlue(state, config); b = load_batch(sys.argv[1])[0]
sp = g.pair_split; dev = T.get_device(); cfg = ag.precise_config()
n = int(b["padding_mask"].shape[-1]); c = int(sp.w_const.shape[0])
with torch.no_grad():
    host = sp.constant(b); bins = sp.bins(b)
    feats = torch.cat([(lambda f: f.reshape(f.shape[0], 1, 1, -1).expand(-1, n, n, -1) if f.ndim == 2 else f)(cr(b)).float()
                       for cr in sp.const_creators], -1)[:1]
k = int(feats.shape[-1]); kp = -(-k // 32) * 32
tile = dict(layout=ttnn.TILE_LAYOUT, device=dev)
pc = PairOnCard(sp, dev)

def bits3(q):
    """[hi|mid|lo] bf16 parts of an fp32 card tensor along the last axis: q - hi - mid - lo ~ 2^-25 q."""
    h = ttnn.typecast(q, ttnn.bfloat16); r = ttnn.subtract(q, ttnn.typecast(h, ttnn.float32))
    m = ttnn.typecast(r, ttnn.bfloat16); l = ttnn.typecast(ttnn.subtract(r, ttnn.typecast(m, ttnn.float32)), ttnn.bfloat16)
    return ttnn.concat([h, m, l], dim=-1)

ones = ttnn.from_torch(torch.ones(3 * c, 32), dtype=ttnn.bfloat16, **tile)
def rowsum(q):
    if BREAK:
        return ttnn.sum(q, dim=-1, keepdim=True, compute_kernel_config=cfg)
    s = ttnn.matmul(bits3(q), ones, dtype=ttnn.float32, compute_kernel_config=cfg)
    return ttnn.slice(s, [0] * (len(s.shape) - 1) + [0], list(s.shape[:-1]) + [1])

t0 = time.time()
F = ttnn.from_torch(torch.nn.functional.pad(feats.reshape(n * n, k), [0, kp - k]).bfloat16(), dtype=ttnn.bfloat16, **tile)
Wp = [ttnn.from_torch(torch.nn.functional.pad(p.float(), [0, 0, 0, kp - k]).bfloat16(), dtype=ttnn.bfloat16, **tile)
      for p in split_bf16(sp.w_const.t().contiguous())]
proj = None
for w in Wp:
    y = ttnn.matmul(F, w, dtype=ttnn.float32, compute_kernel_config=cfg)
    proj = y if proj is None else ttnn.add(proj, y)
proj = ttnn.reshape(proj, (1, n, n, c))
x = pc._x(proj, bins)                                                   # [K, n, n, c] fp32, exact gather
wt = ttnn.from_torch(sp.ln_weight.reshape(1, 1, 1, c), dtype=ttnn.float32, **tile)
s1 = rowsum(ttnn.multiply(x, x)); xw = ttnn.multiply(x, wt); s2 = rowsum(ttnn.multiply(xw, xw))
r1sq = ttnn.reciprocal(ttnn.add(ttnn.multiply(s1, 1.0 / c), sp.eps))
r2sq = ttnn.reciprocal(ttnn.add(ttnn.multiply(ttnn.multiply(r1sq, s2), 1.0 / c), float(sp.builder.adaln.norm.eps)))
scale = ttnn.sqrt(ttnn.multiply(r1sq, r2sq))
ttnn.synchronize_device(dev); card_s = time.time() - t0
pj = ttnn.to_torch(proj).float().reshape(1, n, n, c); sc = ttnn.to_torch(scale).float().reshape(bins.shape)
with torch.no_grad():
    p64 = feats.double() @ sp.w_const.double().t()
    # float64 scale at the candidates' bins, straight from the definition
    xx = p64 + sp.w_step.double()[bins]
    r1 = 1 / (xx.pow(2).mean(-1) + sp.eps); w2 = sp.ln_weight.double() ** 2
    r2 = 1 / (r1 * (xx.pow(2) * w2).mean(-1) + float(sp.builder.adaln.norm.eps))
    s64 = (r1 * r2).sqrt()
    host_sc = torch.take_along_dim(host.scales, bins[..., None], -1).squeeze(-1).double()
pe = float((pj.double() - p64).abs().max() / p64.pow(2).mean().sqrt())
se = float(((sc.double() - s64) / s64).abs().max()); he = float(((host_sc - s64) / s64).abs().max())
ok = pe <= 1e-6 and se <= 1e-5
print("C1_GRADE " + json.dumps(dict(batch=os.path.basename(sys.argv[1]), brk=BREAK, proj_rel=pe, scale_rel_max=se,
      host_table_scale_rel_max=he, card_s=round(card_s, 3), K=int(bins.shape[0]), verdict="PASS" if ok else "FAIL")), flush=True)
