"""Screen the token-axis shard on a REAL DiffusionTransformerLayer at Boltz-2 token geometry.

The sampler step is three tracks; only the 24-layer token transformer touches the [1,512,768]
track. Inside a token layer every op is row-local in the token axis except the attention, so a
sequence-parallel split costs one collective per layer. This measures what that is worth and
proves it is bit-exact, before any fold runs.

  single      one chip, the whole 512-token layer, in its own process
  mesh_shard  1x2 mesh over the p300c pair, 256 token rows per chip,
              TT_BIO_TOKEN_DIT_SEQ_SHARD=1 so the attention all-gathers k and v
  mesh_repl   1x2 mesh, everything replicated, no shard -- the mesh tax at this geometry

Bit-exactness is scored across processes: `single` writes its output, `mesh_shard` reads it and
demands torch.equal, against a negative control (chip 0's rows duplicated over chip 1's, which a
correct shard must NOT reproduce) that the check provably rejects.
"""

import json, os, sys, time
import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "single"
REPS = int(os.environ.get("TLS_REPS", "15"))
REF = "/tmp/b2z2_tls_ref.pt"
OUT = os.environ.get("TLS_OUT", "/tmp/b2z2_tls_%s.json" % MODE)

from tt_bio.device_lease import CardSetLease  # noqa: E402
if MODE != "single":
    CardSetLease().acquire()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import boltz2 as b2  # noqa: E402


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


if MODE == "single":
    dev = tt.get_device()
else:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
    tt._device = dev
log("mode=%s device=%s" % (MODE, dev))

KC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

S, D, H = int(os.environ.get("TLS_ROWS", "512")), tt.TOKEN_DIM, tt.TOKEN_N_HEADS
KV = int(os.environ.get("TLS_KV", str(S)))  # key/value rows; != S isolates the row-work term
torch.manual_seed(0)
ref_layer = b2.DiffusionTransformerLayer(H, D, D, False)
weights = {k: v.float() for k, v in ref_layer.state_dict().items()}
layer = tt.DiffusionTransformerLayer(D, H, False, weights, KC)
layer.attn_pair_bias.token_dit = True
log("layer built from %d weight tensors" % len(weights))

torch.manual_seed(1)
a_pt = (torch.randn(1, S, D) * 0.5).to(torch.bfloat16).float()
s_pt = (torch.randn(1, S, D) * 0.5).to(torch.bfloat16).float()
z_pt = (torch.randn(1, H, S, KV) * 0.1).to(torch.bfloat16).float()

shard = (MODE == "mesh_shard")


def to_dev(x, shard_dim=None):
    kw = {}
    if shard_dim is not None and shard:
        kw["mesh_mapper"] = ttnn.shard_tensor_to_mesh_mapper(dev, shard_dim)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, **kw)


a_tt = to_dev(a_pt, 1)
s_tt = to_dev(s_pt, 1)
z_tt = to_dev(z_pt, 2)
log("a=%s s=%s z=%s shard=%s" % (a_tt.shape, s_tt.shape, z_tt.shape, shard))

out = layer(a_tt, s_tt, z_tt)
ttnn.synchronize_device(dev)
log("out=%s" % (out.shape,))

if MODE == "single":
    got = ttnn.to_torch(out).float()
    torch.save(got, REF)
    parity = {"role": "reference", "shape": list(got.shape)}
else:
    comp = ttnn.concat_mesh_to_tensor_composer(dev, 0)
    per_chip = ttnn.to_torch(out, mesh_composer=comp).float()
    if shard:
        got = torch.cat([per_chip[0:1], per_chip[1:2]], dim=1)
    else:
        got = per_chip[0:1]
    ref = torch.load(REF)
    exact = torch.equal(got, ref)
    maxdiff = float((got - ref).abs().max())
    ctrl = ref.clone()
    ctrl[:, S // 2:] = ref[:, : S // 2]
    ctrl_rejects = not torch.equal(got, ctrl)
    parity = {"role": "arm", "bitexact": bool(exact), "max_abs_diff": maxdiff,
              "negctrl_rejected": bool(ctrl_rejects), "shape": list(got.shape)}
    log("PARITY exact=%s maxdiff=%s negctrl_rejected=%s" % (exact, maxdiff, ctrl_rejects))

for _ in range(3):
    o = layer(a_tt, s_tt, z_tt)
    ttnn.deallocate(o)
ttnn.synchronize_device(dev)
per = []
for _ in range(REPS):
    t0 = time.perf_counter()
    o = layer(a_tt, s_tt, z_tt)
    ttnn.synchronize_device(dev)
    per.append(time.perf_counter() - t0)
    ttnn.deallocate(o)
per.sort()
med = per[len(per) // 2]

# The eager number prices a dispatch regime the sampler does not run in. Capture the whole
# 24-layer token transformer as one ttnn trace and time the replay: that is the regime the
# sharded sampler would actually execute, and it is the only one in which the 37.2 us traced
# collective (gather_curve.py) is the cost that bills.
NL = tt.TOKEN_N_LAYERS
trace = None
try:
    def chain(x):
        for _ in range(NL):
            x = layer(x, s_tt, z_tt)
        return x
    _ = chain(a_tt)
    _w = chain(a_tt)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    _out = chain(a_tt)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(2):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    tper = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        tper.append(time.perf_counter() - t0)
    tper.sort()
    tmed = tper[len(tper) // 2]
    trace = {"captured": True, "token_transformer_ms": tmed * 1e3,
             "per_layer_ms": tmed * 1e3 / NL, "min_ms": tper[0] * 1e3, "max_ms": tper[-1] * 1e3}
    log("TRACE %d layers %.4f ms  = %.4f ms/layer" % (NL, tmed * 1e3, tmed * 1e3 / NL))
    ttnn.release_trace(dev, tid)
except Exception as e:
    trace = {"captured": False, "error": "%s: %s" % (type(e).__name__, e)}
    log("TRACE FAILED %s: %s" % (type(e).__name__, e))
res = {"mode": MODE, "rows": S, "reps": REPS, "layer_ms": med * 1e3, "min_ms": per[0] * 1e3,
       "max_ms": per[-1] * 1e3, "token_transformer_ms": med * 1e3 * tt.TOKEN_N_LAYERS,
       "shard_stats": list(tt.TOKEN_DIT_SEQ_SHARD_STATS), "parity": parity,
       "trace": trace}
log("LAYER %.4f ms  x%d = %.3f ms  shard_stats=%s"
    % (med * 1e3, tt.TOKEN_N_LAYERS, med * 1e3 * tt.TOKEN_N_LAYERS, tt.TOKEN_DIT_SEQ_SHARD_STATS))
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
log("wrote %s" % OUT)
