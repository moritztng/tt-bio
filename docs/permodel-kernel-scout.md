# Per-model kernel scout

## Result

No production change is justified by this pass. The shared-trunk and
model-specific scout lists are exhausted for now.

| Candidate | Warm stage | Full warm call | Share | Tested optimization |
|---|---:|---:|---:|---:|
| ESMFold2 structure head | 0.25725 s | 2.20278 s | 11.68% | trace: 0.9139x stage, 0.9716x full fold |
| Protenix-v2 confidence | 0.07875 s | 5.45464 s | 1.44% | 1.0146x ceiling if made free |
| BoltzGen diffusion | 10.88699 s | 17.26328 s | 63.06% | trace: 0.9674x stage, 0.9802x full batch |

ESMFold2 in this repository does not have an Invariant Point Attention
structure module. It uses an all-atom diffusion head. That head has no Python
loop over attention heads. Its repeated sampling-step stream was the only
plausible dispatch-collapse target, and trace replay made both the stage and
the full fold slower.

Protenix-v2 confidence runs once per returned sample. Its four-block
Pairformer is already on device, and the remaining distance-bin work is
vectorized. The complete head is too small to meet the 5% end-to-end threshold
even if it were free.

BoltzGen diffusion is substantial, but its step-invariant tensors are already
cached and each sampling step enters one device module. Replaying that module
for all 500 production steps was slower than the existing path. No smaller
per-head, per-bin, or per-atom-type dispatch loop remains to collapse.

The trace prototypes were removed after the negative A/B results. Runtime code
is unchanged, so no accuracy release gate was required.

## Method

Measurements used `pc` physical card 0, one Blackhole P150a, real checkpoints,
real inputs, and synchronized stage boundaries. Each reported number is the
second same-shape call in one process.

ESMFold2 used `examples/prot.yaml` at 117 residues, three trunk loops, 20
sampling steps, and one sample. Protenix-v2 used the same sequence, its default
10 recycling cycles, 20 sampling steps, and one sample. BoltzGen used
`examples/binder.yaml`, one checkpoint, two serial designs, and 500 sampling
steps. Its second batch was the warm result.

The BoltzGen baseline warm batch split into 6.18964 s before diffusion,
10.88699 s in diffusion, and 0.18664 s after diffusion. The trace A/B used the
same production path and changed the warm batch from 17.26328 s to 17.61124 s.

## Reproduce

```bash
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/esmfold2_profile.py \
  --protein prot --loops 3 --steps 20
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/permodel_kernel_scout.py \
  protenix-v2 --steps 20 --samples 1
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 -m tt_bio.main gen run \
  examples/binder.yaml --output /tmp/boltzgen-scout --num_designs 2 \
  --budget 2 --devices 1 --steps design \
  --design_checkpoints huggingface:moritztng/boltzgen:boltzgen1_diverse.ckpt \
  --debug --log
```
