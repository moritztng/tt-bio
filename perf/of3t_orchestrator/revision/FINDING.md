# D23 — the reference bundle runs the preview2 checkpoint on upstream 0.5.0, which upstream declares unsupported

Pass 89, `of3t-orchestrator`, CPU only, no card. Every number below is reproducible on pc with
`dit_apb_identity.py` and `dit_apb_control.py` in this directory.

## What upstream says

`entry_points/parameters.py` in the 0.5.0 sdist:

```
"openbind-2025-06-30-174k": CheckpointEntry(file_name="of3-ob-2025-06-30-174k.pt",
                                            version_compatibility=">=0.5.0"),
"openfold3-p2-155k":        CheckpointEntry(file_name="of3-p2-155k.pt",
                                            version_compatibility=">=0.4,<0.4.4dev0"),
...
DEFAULT_CHECKPOINT_NAME = "openbind-2025-06-30-174k"
LEGACY_CHECKPOINTS = ["openfold3-p1", "openfold3-p2-145k", "openfold3-p2-155k"]
# These checkpoints are not supported for download and use in the current version,
# but are left in the registry for record-keeping and compatibility checks.
```

The same file in the 0.4.3 sdist gives `of3-p2-155k.pt` an unbounded `">=0.4"`, makes it
`DEFAULT_CHECKPOINT_NAME`, and lists only `openfold3-p1` as legacy.

So `of3-p2-155k.pt` — the checkpoint tt-bio ships OpenFold3 on, and the checkpoint the campaign's
reference bundle was built with — is declared by upstream to be incompatible with the 0.5.0 code
the bundle was built on. Release names: v0.4.0 is "OpenFold3 Preview2", v0.5.0 is the "OpenBind
Model Release".

## Two code changes land between those revisions, one per track

**Trunk.** `transpose_bias=True` on `PairFormerBlock.tri_att_end` appears in 0.5.0 and nowhere
earlier. Occurrences of `transpose_bias=True` in `core/model/latent/base_blocks.py`:

| 0.4.0 | 0.4.3 | 0.4.4 | 0.4.5 | 0.5.0 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | **1** |

**Diffusion transformer.** 0.5.0 splits `AttentionPairBias` into a plain one and a new
`DiffusionAttentionPairBias`, and the new class has **no `layer_norm_z`** — neither constructed
nor applied. 0.4.3's single `AttentionPairBias` constructs it at line 107 and applies it to the
pair bias at line 156, on both the trunk and the diffusion path. 0.5.0 keeps it on the trunk
class (lines 85/134) and drops it on the diffusion class.

The p2 checkpoint carries the weights the dropped norm needs: 24 tensors
`diffusion_module.diffusion_transformer.blocks.{0..23}.attention_pair_bias.layer_norm_z.weight`,
shape `(128,)`, out of 4,935 total.

## Measured, float64, same p2 block-0 weights, both of upstream's own constructions

Loading the p2 block-0 attention weights into each:

```
[0.4.3] AttentionPairBias(use_ada_layer_norm=True)  missing=[]  unexpected=[]
[0.5.0] DiffusionAttentionPairBias                  missing=[]  unexpected=['layer_norm_z.weight']
```

0.5.0 has nowhere to put the checkpoint's per-block pair LayerNorm and drops it silently.

Relative L2 between the two constructions, iid normal inputs, float64:

| N | separation |
|---|---|
| 64 | **3.995e-01** |
| 384 | **3.868e-01** |

Flat in N, which matches the campaign's own DiT size ladder being flat from n=32 to 384.

**Sufficiency control.** Pre-normalising `z` with the checkpoint's own `layer_norm_z` weight
(`create_offset=False`, eps 1e-5) and then calling 0.5.0's class reproduces 0.4.3's output at
**0.000000e+00** — bit-identical. `layer_norm_z` is the entire difference between the two, not
one contribution among several.

## What this corrects

**D22 said every DiT-path layer file is functionally inert between 0.4.3 and 0.5.0.** That is
refuted. D22 compared the two `forward` methods and they are indeed line-for-line identical; the
change is in `_prep_bias`, which `forward` calls and D22 did not read.

**`of3t-reopen` reported that our shipped OpenFold3 trunk computes the ending-node function
`PairFormerBlock` does not call.** True of 0.5.0's `PairFormerBlock`, and 0.5.0 does not support
these weights. tt-bio sets `tri_att_end_bias_follows_pair = not is_openbind(state_dict)`, which
gives the preview2 convention to the preview2 checkpoint and the 0.5.0 convention to the OpenBind
checkpoint — upstream's own checkpoint-to-revision binding, reproduced. **The shipped setting is
correct and no flag should be flipped.** Flipping it would have put a real regression into
OpenFold3 inference on the strength of a reference that cannot run these weights.

**The DiT forward gap is the reference, not our module.** `of3t-reopen`'s own localisation
corroborates it: inside one DiT block, `d_attention_pair_bias` misses the bar at 5.522e-02 while
the shared AdaLN conditioning is clean at 6.027e-03. `layer_norm_z` sits on the pair-bias path and
not on the AdaLN path, so the one op that differs between the revisions is the one op the
bisection indicted.

The 3.9e-01 figure is a function-identity separation on iid inputs and is **not** a prediction of
the 2.07e-02 seen on real tokens; a trained `z` is closer to the norm's own output than iid normal
is, so the effect shrinks in distribution. The claim here is that the two are different functions
and which one the checkpoint describes, not that this number reproduces that one.

## Why the composition laws still say two mechanisms

Pass 85/88 refuted a shared mechanism from the composition laws — the trunk gap composes
near-linearly at 0.74x of block count, the DiT gap sub-linearly at 0.32x. That stands. These are
two different code changes on two different tracks. They have one **cause**: a reference built one
revision family away from the checkpoint it loads.

## What to do

Rebuild the reference bundle at **0.4.3** — the revision the port targets and the last one inside
upstream's declared window for this checkpoint. 0.4.4's model tree is byte-identical to 0.4.3's;
0.4.5 changes 13 model files, so 0.4.5 and later are not substitutes. The tree already installed
on pc at `/home/moritz/.coworker/scratch/of3-upstream/repo` (0.4.6.dev12+g72fc3a953) still carries
the unified `AttentionPairBias` with `layer_norm_z` and no `transpose_bias`, so it behaves as
0.4.3 on both changes.

Pre-registered predictions, written before the rebuild exists:

1. Our trunk's 48-block forward against a 0.4.3 bundle falls from **2.792e-01** toward the bf16
   composition floor, and the `tb-off` arm — better against the 0.5.0 bundle at 4.965e-02 — gets
   worse. If both arms move the same way, this explanation is wrong.
2. `d_attention_pair_bias` falls from **5.522e-02** toward `adaln_a_out`'s **6.027e-03**, and
   `block_out` from **2.072e-02**. If `d_attention_pair_bias` stays over bar, the DiT has a second
   defect and this closes only part of it.
3. D19's near-linear 0.74x composition disappears, because one differing sub-module per block is
   what produces a per-block constant that composes linearly.
