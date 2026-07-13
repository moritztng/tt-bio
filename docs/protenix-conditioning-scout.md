# Protenix-v2 diffusion conditioning path scout (no-go)

## Result

No production change is justified by this pass. The runtime path is unchanged.

The one genuinely unexplored component after the shared-trunk scouts
(triangle multiplication, Transition, TriangleAttention, OuterProductMean,
PairWeightedAveraging — all closed in `docs/boltz2-protenix-kernel-scout.md`
and `docs/msa-pwa-scout.md`) and the denoiser attention half
(`docs/atomattention-kernel-scout.md`, `docs/boltz2-dit-attention-kernel-scout.md`)
is the **diffusion conditioning path**: the per-block AdaLN modulation and the
s-gate sigmoids in the 24-block token DiT. They were profiled here on a real
warm 200-step fold. A real direct TTNN fusion exists for the modulation
(`ttnn.addcmul` collapses `multiply_` + `add_` into one eltwise) and is
bit-identical, but it produces no measurable wall-clock win, because the
denoiser is device-compute-bound and these are tiny-tensor dispatches already
hidden behind compute.

## What was profiled

Hardware: qb2 physical card 2, one Blackhole P150 chip, ttnn 0.68, real
Protenix-v2 checkpoint, `examples/prot.yaml` (117 residues), bf16 diffusion,
200 sampling steps, seed 0. Every timed fold is warm off the program cache
and ends with a device synchronize. Methodology matches the prior shared-trunk
scouts (`scripts/atomattention_kernel_scout.py`): the per-call AdaLN timing
adds a synchronization around each of the 48 AdaLN calls per DiT call, so its
share is an **upper bound** (it inflates AdaLN relative to its true compute).

The token DiT's per-block conditioning, from `DiffusionModule._token_dit_device`
and `AdaLN` (`tt_bio/tenstorrent.py`), is, per 24-block DiT call:

| component | per DiT call | per block |
|---|---:|---:|
| AdaLN modulations (pre-attn + pre-transition) | 48 calls | 2 |
| s-gate sigmoids (`sigmoid(linear_a_last)`, `sigmoid(linear_s)`) | 96 | 4¹ |

¹ The DiT call issues other sigmoids too; the AdaLN modulation itself fuses its
sigmoid into `multiply_` (`input_tensor_b_activations=[SIGMOID]`), so it is not
a separate dispatch.

A one-call dispatch decomposition of `_token_dit_device` (24 blocks), captured
on the first DiT call of the second warm fold:

| ttnn op | count |
|---|---:|
| linear | 576 |
| layer_norm | 192 |
| multiply / multiply_ | 192 / 96 |
| add / add_ | 96 / 96 |
| sigmoid | 96 |
| to_memory_config | 96 |
| reshape | 48 |
| deallocate | 480 |
| **total** | **1968** |

## Share in a real warm fold

| metric | value |
|---|---:|
| full warm fold (200 steps) | 8.29 s |
| token DiT (`_token_dit_device`, 200 calls) | 2.12 s — 24.6% of fold |
| AdaLN (9600 calls, synced upper bound) | 0.637 s — 7.37% of fold |
| AdaLN share of token DiT (upper bound) | 30.0% |
| AdaLN-free DiT ceiling (upper bound) | 1.43× |
| AdaLN per call (synced) | 66 µs |

The AdaLN share is an upper bound for the reasons above; the isolated A/B
below shows its true dispatch headroom is far smaller than 30%.

## Candidate fusion: `ttnn.addcmul` for the AdaLN modulation

`AdaLN.__call__` computes the AF3 modulation `a = LN(a) * sigmoid(s_scale) + s_bias`
as two eltwise dispatches on the full token tensor:

```
a = ttnn.multiply_(a, s_scale, input_tensor_b_activations=[SIGMOID])   # a * sigmoid(s_scale)
a = ttnn.add_(a, s_bias)
```

`ttnn.addcmul(input_a, input_b, input_c)` computes `input_a + input_b * input_c`,
so the modulation collapses to one eltwise:

```
sig = ttnn.sigmoid(s_scale)            # tiny s-tensor
a = ttnn.addcmul(s_bias, a, sig)       # s_bias + a * sigmoid(s_scale)
```

This removes one eltwise dispatch on the full `[1, NT, 768]` token tensor per
AdaLN call (9600 calls/fold). It does not fuse the sigmoid into the matmul, so
it adds one tiny `sigmoid` on `s_scale` (`[1, NT, 768]`); the net is one fewer
big-tensor eltwise per call.

### Isolated A/B (one AdaLN call, real captured DiT inputs, 80 synced repeats)

| path | median | PCC vs baseline | max abs |
|---|---:|---:|---:|
| baseline (`multiply_` + `add_`) | 0.0637 ms | — | — |
| `addcmul` fused | 0.0634 ms | **1.000001** | 0.0156 (1 bf16 ULP) |
| **isolated speedup** | **1.006×** | | |

Bit-identical (one bf16 ULP), but the isolated speedup is within timing noise.
The AdaLN call is dominated by its two `layer_norm` calls and two `linear`
calls on `s`, not by the two big-tensor eltwise ops the fusion collapses.

### End-to-end fold A/B (warm 200-step fold, two timed passes each)

| path | timed (s) |
|---|---:|
| baseline | 8.287 / 8.292 |
| `addcmul` fused (`TT_ADALN_ADDCMUL=1`) | 8.292 / 8.302 |
| **e2e speedup** | **0.999×** (flat) |

No measurable e2e win. This matches the compute-bound denoiser characterization
(`docs/protenix-accel-ceiling`-class finding, memory `protenix-accel-ceiling`):
traced diffusion at L256 spends only ~9% of a step on host round-trip, so
collapsing eltwise dispatch on tiny tensors is hidden behind device compute.

## Other conditioning-path fusions (not built — same ceiling)

Two further dispatch collapses are buildable in the same path and were rejected
without an A/B because the profile plus the `addcmul` result already bound them:

* **Pack AdaLN's two `s` linears** (`s_scale`, `s_bias`) into one `[dim, 2*dim]`
  matmul + `chunk`, removing one matmul dispatch per AdaLN call. The linears are
  on `s` (`[1, NT, 384] → [1, NT, 768]`) — tiny — and the `addcmul` A/B showed
  the AdaLN dispatch headroom is sub-1%. Same family as the
  `difftransformer-swiglu-scout` packed-matmul proxy (~1.6% with an instability
  spike), only smaller.
* **Fuse the s-gate `sigmoid` into its `linear`** (`ttnn.sigmoid(linb(s))` →
  `linb(s, activation="sigmoid")`). Removes 48 sigmoid dispatches per DiT call,
  but each is on the tiny `s_t` tensor, so the e2e effect is below noise.

## Verdict

The diffusion conditioning path is a dispatch-collapse no-go, like the rest of
the Protenix-v2 denoiser. The one direct TTNN fusion available
(`ttnn.addcmul` for the AdaLN modulation) is bit-identical and ships no e2e win.
Combined with the already-closed shared-trunk and denoiser-attention scouts,
this exhausts the dispatch-collapse fusion surface for Protenix-v2 at small N:
every remaining component is either compute-bound (denoiser) or already at a
measured ceiling. Further Protenix-v2 acceleration needs a deeper compute kernel
or a different model stage (the confidence head is host-bound and ~9% of e2e
per `protenix-accel-ceiling`, a separate axis), not another dispatch collapse.

## Reproduce

```bash
WT=<worktree>; PY=~/tt-bio-dev/env/bin/python3
M=TT_VISIBLE_DEVICES=2
G=TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

# share of AdaLN + DiT in a warm 200-step fold
env $M $G PYTHONPATH=$WT $PY $WT/scripts/protenix_conditioning_scout.py profile --steps 200

# dispatch decomposition of one _token_dit_device call (24 blocks)
env $M $G PYTHONPATH=$WT $PY $WT/scripts/protenix_conditioning_scout.py decomp

# isolated addcmul A/B (PCC + synced timing, 80 repeats)
env $M $G PYTHONPATH=$WT $PY $WT/scripts/protenix_conditioning_scout.py ab --repeats 80

# end-to-end fold A/B (baseline vs fused); the fused path is env-gated in AdaLN
# and is reverted in the shipped runtime — re-apply the TT_ADALN_ADDCMUL branch
# in tt_bio/tenstorrent.py AdaLN.__call__ to reproduce.
env $M $G PYTHONPATH=$WT $PY - <<'PY'
import os,time,json,yaml; from pathlib import Path
import torch,ttnn
from tt_bio.protenix import Protenix
from tt_bio.protenix_data import build_protein_features
from tt_bio.tenstorrent import get_device
seq=yaml.safe_load(Path("examples/prot.yaml").read_text())["sequences"][0]["protein"]["sequence"]
feats=build_protein_features(seq); dev=get_device()
cfg=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
m=Protenix.load_from_checkpoint("/home/ttuser/.boltz/protenix-v2.pt",compute_kernel_config=cfg,device=dev)
def f():
    ttnn.synchronize_device(dev);t=time.perf_counter()
    m.fold(feats,n_step=200,n_sample=1,seed=0,return_confidence=False)
    ttnn.synchronize_device(dev);return time.perf_counter()-t
f();print(json.dumps({"fused":bool(os.environ.get("TT_ADALN_ADDCMUL")),"s":f(),"s2":f()}))
PY
```
