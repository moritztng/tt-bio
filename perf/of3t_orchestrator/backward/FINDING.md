# The backward differentiates a different function than the forward computes, at every attention site

Orchestrator, pass 121. CPU-only: this is a reading of `tt_bio/autograd.py`,
`tt_bio/taped_ttnn.py` and the installed `ttnn`, with one hypothesis refuted and one raised.

## Refuted first: the backward's kernel config is not the defect

`tt_bio/autograd.py`'s `precise_config()` — the config every backward op in the tape uses —
hardcodes `ttnn.WormholeComputeKernelConfig`, and it is the **only** `ComputeKernelConfig(`
construction in `tt_bio/*.py` that does not branch on architecture. `tenstorrent.py:11268` does
branch:

    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if self.tt_device.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)

The gradient measurements run on Blackhole, so this looked like the forward getting a Blackhole
config and the backward a Wormhole one — which would have been an exact fit for a
backward-specific gap. **It is not.** On the row's own environment, ttnn 0.68.0 at
`/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/types.py:57`:

    BlackholeComputeKernelConfig = WormholeComputeKernelConfig

They are the same class; the `.so` exports only `WormholeComputeKernelConfig`. The arch branch is
cosmetic in this version and the tape's hardcoded class is byte-identical to what it would pick.
**No numerical difference. Nobody should spend a run on this.** The backward genuinely runs at
HiFi4 with `fp32_dest_acc_en` and `packer_l1_acc`.

## Raised: the tape differentiates the precise recompute, the forward ships a fused kernel

`taped_ttnn.py` wraps the shipped fused SDPA: it calls `shipped(*ra, **rk)` for the forward value
and hands that value to `ag.triangle_attention(..., value=out_v)`. `autograd.triangle_attention`'s
own docstring is explicit about what it then does:

> "The backward below reads q, k, v and bias and **recomputes the scores**; it never reads the
> forward output, so supplying the output changes nothing about the gradient"

and the recompute is `ttnn.matmul` + `ttnn.multiply` + `ttnn.add` + `ttnn.softmax`, all at
`precise_config()` (`autograd.py:801-807`).

The same docstring records what the fused kernel costs, measured on a p300c:

> "the error sits flat at **3.3e-02** for every mask magnitude tested (0, 0.05, 0.5, 2.0), which
> is **the fused softmax's own known deficit**"

So at every attention site the forward computes `fused(q,k,v,bias)`, carrying a 3.3e-02 deficit
against the exact composite, and the tape returns the gradient of `precise(q,k,v,bias)`. **The
function differentiated is not the function computed.** The resulting gradient is neither the
exact gradient of our forward nor the exact gradient of upstream's: it is the gradient of the
precise attention, evaluated at activations produced by the deficient one.

### Why this is a candidate for D30's backward-specific factor

1. **It is backward-specific by construction.** The forward's deficit is inside the fused kernel;
   the backward's is a different function entirely. A forward-only comparison cannot see it.
2. **It sits on the right module.** The docstring says this entry is what "`AttentionPairBias` in
   the pairformer and the denoiser's attention sites" both reach. The campaign's worst tensor in
   the trunk is `attn_pair_bias.layer_norm_a.weight` and in the diffusion transformer it is
   `attention_pair_bias.layer_norm_a.layer_norm_s.weight` — **the same module family, both
   stacks**, which is exactly the population this one code path serves.
3. **3.3e-02 per site is the right order.** The diffusion forward measures 8.34e-03 overall and
   its gradient 1.6588e-01. A per-site term of 3.3e-02 entering only the backward, across 24 DiT
   blocks, is a plausible source of a 19.6x factor. Plausible, not demonstrated.

### What the existing checks do NOT cover

The docstring says gradients are verified "by differencing two chunkings rather than trusting
one" (`perf/hallgrad/gradcheck.py --cases triatt_chunked`). That establishes the **recompute** is
chunking-invariant. It does not compare the recompute against the **fused kernel**, and no check
on the record does.

### The test, and it needs no reference

One arm, internal consistency only: on identical `q, k, v, bias`, compare the shipped fused SDPA's
output against `autograd.triangle_attention`'s own recompute path (call it with `value=None`, which
makes it materialise `out_blocks` through exactly the code the backward differentiates). Report the
relative L2 per attention site.

- If it reads ~3.3e-02, the mismatch is confirmed at the magnitude the docstring predicts and D31
  becomes the leading candidate for D30.
- If it reads ~1e-03 or below, the fused kernel and the recompute agree at these shapes and D31 is
  refuted for this model — record it refuted.

### Two things this finding is NOT

It is **not** a proposal to close the fused softmax's row-sum deficit. That lever has been tried
three times in this lineage and moved folds measurably worse each time; the deficit is entangled
with a compensating accumulation order. The defect here is the **mismatch**, and the cheaper
remedy if it is confirmed is to make the backward differentiate what the forward actually
computed, not to change the forward.

It is also **not** a claim that the shipped inference is wrong. The forward is the forward
production ships and its accuracy is measured elsewhere. This is about the **gradient**, which
only the training-reproduction charter cares about.

## Addendum, pass 122: which kernel, and what else the tape turns off

### The object in D31 is the stock ttnn SDPA, not tt-bio's custom one

`triatt_sdpa.py:340`:

    from . import ops
    if ops.taping():
        return None   # generic_op has no backward; the stock fused SDPA verb is taped

tt-bio's own fused triangle-attention SDPA **declines while a tape is open**. The object the tape
wraps, and therefore the object carrying the 3.3e-02 deficit in the taped forward, is the **stock
`ttnn` fused SDPA**. That is the same object the docstring's figure was measured against, so §"Raised"
above is unchanged in substance — but "the shipped fused SDPA" was the wrong name for it, because
the custom kernel is precisely the one that is not there.

### Twenty-one sites, nine modules

`ops.taping()` is consulted at 21 branch points in 9 shipped modules, each declining a fused path:

| module | sites | what stops happening |
|---|---|---|
| `triatt_qkv.py` | 74, 179, 276, 390 | fused QKV projection |
| `tenstorrent.py` | 1126, 4129, 4595, 8531 | trimul DRAM route; pair-proj L1 output; **L1 -> DRAM memory config** at 8531 |
| `eltwise_fusion.py` | 80, 104, 115 | `FUSE_MASK_ADD`, `FUSE_NORM_RESIDUAL` |
| `reblock_permute.py` | 373, 597, 877 | reblock/permute fusions |
| `softmax_generic.py` | 369, 514 | fused softmax |
| `triatt_sdpa.py` | 340, 486 | custom fused SDPA; QKV-from-pre-projection |
| `trimul_tail.py` | 233 | fused trimul tail |
| `mm_dualnoc.py` | 87 | dual-NoC matmul |
| `swiglu_fused.py` | 94 | fused SwiGLU |

One uniform reason, stated in the code: *"generic_op has no backward."*

**A training step is therefore a materially different execution than an inference step.** Every
gradient figure in this campaign is taken on the unfused path — self-consistent for training, but
it does not transfer to the inference path production ships, and inference accuracy does not
transfer here. And **a training s/step cannot be projected from an inference s/step**: when the
charter's second half is finally measured, the device number has to be taken with the tape open.
Recorded as D32.

### Refuted here as well

The bias pre-scale at `tenstorrent.py:7647` — `ttnn.multiply(bias, self.scale / self._bias_scale)`,
which fires only when `scale_pair_bias` is False — looked like it might be untaped, which would
break the gradient path to `attn_pair_bias.linear_z.weight`, the same tensor D26 shows going absent
under the opposite setting. It is taped: `taped_ttnn._swap` rebinds the name `ttnn` in **every**
`tt_bio` module, excluding only itself and `tt_bio.autograd`. No missing gradient path.
