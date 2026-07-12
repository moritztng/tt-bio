# AtomAttention kernel scout

## Result

No production change is justified by this pass.

The AF3-style atom-attention encoder/decoder (32-query x 128-key local windowed
SDPA, per-window gather, per-head pair bias) is a genuine, meaningful share of a
production fold in both models, but it is dispatch-bound and its gather +
projection are already collapsed to near-minimal op count. The one localized
fusion the profile allows (pack Protenix's separate K/V projection + window the
packed KV once) is bit-exact but net-negative.

| Model | Config | Atom-attention share | Free ceiling |
|---|---|---:|---:|
| Protenix-v2 | 200 steps, warm full fold | **17.1%** | 1.207x |
| Boltz-2 | 200 steps, warm per-step | **~20%** of the diffusion sampling loop | n/a |

* Protenix-v2 atom attention breaks down as the input-embedder encoder (0.17%,
  once per fold) plus the two per-step diffusion AtomTransformers (atxE 8.4% +
  atxD 8.6%). Even if all of it were free the fold is only 1.207x faster.
* The Protenix K+V-pack fusion is PCC 1.0 / max-abs 0.0 but runs at **0.928x**
  (regression). Recovering K and V from the packed gather costs more dispatch
  than the single linear + single windowing it removes. Same failure mode as the
  TriangleAttention QKV-pack (0.58-0.61x, docs/boltz2-protenix-kernel-scout.md).
* Boltz-2's atom-level attention already ships the fused form: packed KV
  projection (`kv_weight`), a single `keys_indexing` matmul gather, and
  `nlp_create_qkv_heads` for the head split. It is the exact shape that regresses
  when grafted onto Protenix, so it has no analogous headroom.

The only lever left for a dispatch-bound tiny-op stream is full trace replay,
which already exists for the entire denoise (`fold(trace=True)`) and is out of
scope for a per-component scout. Runtime code is unchanged, so no accuracy gate
was required.

## Method

Measurements used qb1 physical card 0, one Blackhole P150a, real Protenix-v2 and
Boltz-2 checkpoint weights, and `examples/prot.yaml` (117 tokens). Every timed
region is warm and bracketed by a device synchronization.

Protenix shares are the second same-shape `model.fold` in one process (the first
fold is the warm-up). Each AtomTransformer call was timed by wrapping the resident
`input_aae` / `diffusion.atxE` / `diffusion.atxD` instances.

Boltz-2 predict fans jobs to mp-spawn worker processes, so the atom-level
`DiffusionTransformer` (encoder + decoder) was timed with a spawn-safe
`sitecustomize.py` patched at interpreter startup. Within one diffusion sample the
200 sampling steps are warm same-shape repeats, so the reported per-step number is
the median over the steps.

## Share

Protenix-v2, `prot.yaml`, 1 sample:

| Sampling steps | Full warm fold | Atom attention | Share |
|---:|---:|---:|---:|
| 20 | 5.787 s | 0.205 s | 3.55% |
| 200 (production) | 11.098 s | 1.903 s | **17.15%** |

At 20 steps the once-per-fold 10-recycling trunk dominates and buries the atom
attention. Production predict uses 200 sampling steps (the CLI default), where the
per-step diffusion dominates and the atom-attention share is the honest number.

Boltz-2, `prot.yaml`, single-sequence, 200 steps (warm per-step medians, one
representative run):

| Phase | Per step | Share of denoise |
|---|---:|---:|
| atom encoder + decoder | 0.00733 s | **26.6%** |
| token DiT | 0.01848 s | 67.1% |
| full denoise network | 0.02755 s | 100% |

Atom attention is 0.00733 s x 200 = 1.467 s of the 7.738 s diffusion sampling
loop (**19.0%**). The per-step medians jitter run to run (atom 27-31% of the
denoise network, 19-21% of the sampling loop across runs); the conclusion is
stable. In Boltz-2 the token-level DiT, not the atom attention, is the larger
diffusion cost.

## Dispatch decomposition

One warm Protenix atxE call (3 blocks) issues 372 ttnn op launches on tiny
tensors (4 windows x 32 queries x 128 keys). It is host-dispatch bound.

| category | launches | category | launches |
|---|---:|---|---:|
| linear | 102 | add | 24 |
| to_layout | 48 | pad | 18 |
| layer_norm | 42 | embedding | 12 |
| permute | 36 | matmul | 12 |
| reshape | 36 | slice | 6 |
| multiply | 30 | softmax | 6 |

The per-window gather is already a single `ttnn.embedding` per K and per V window
(the historical nb-slice + concat loop was already removed). The windowing
data-movement (pad + embedding + permute + reshape + to_layout) is comparable in
launch count to the projections, so no single adjacent-op pair dominates.

## K+V pack A/B

Protenix projects K and V with two separate linears on the same `kv_norm`, then
windows each separately. The fusion packs them into one `[Wk|Wv]` linear and
windows the packed KV once (one pad, one gather, one permute), splitting K and V
after the gather. Measured on the real captured diffusion input, 50 warm repeats:

| | median | parity |
|---|---:|---:|
| baseline (separate K, V) | 0.005331 s | reference |
| packed KV + shared window | 0.005746 s | PCC 1.0, max-abs 0.0 |
| speedup | **0.928x** | |

The refactor is a bit-exact identity but regresses. On these tiny tensors dispatch
count is everything, and the reshape-to-`(nb,nk,2,H,dh)` + permute + two slices
needed to recover K and V add more launches than the one linear and one windowing
the pack removes. Even in the best case the full-fold win is bounded by the 1.207x
free ceiling times the ~4% of atom-attention launches this touches, i.e. under 1%.

## Reproduce

```bash
# Protenix-v2 atom-attention share (production 200 steps and 20-step reference)
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/atomattention_kernel_scout.py \
  protenix --steps 200 --samples 1
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/atomattention_kernel_scout.py \
  protenix --steps 20 --samples 1

# Per-call dispatch decomposition
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/atomattention_kernel_scout.py \
  protenix-decomp

# K+V pack A/B (bit-exact check + warm timing)
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/atomattention_kernel_scout.py \
  protenix-ab --repeats 50

# Boltz-2 atom-attention share (spawn-safe instrumentation)
AA_BOLTZ_OUT=/tmp/aa_boltz_timing \
  PYTHONPATH=scripts/aa_site:. TT_VISIBLE_DEVICES=0 \
  python3 -m tt_bio.main predict examples/prot.yaml --out_dir /tmp/aa_boltz_out \
  --accelerator tenstorrent --model boltz2 --single_sequence \
  --sampling_steps 200 --num_devices 1 --seed 0
cat /tmp/aa_boltz_timing.*.json
```

`scripts/aa_site/sitecustomize.py` must be imported as `sitecustomize`, which is
why its directory (`scripts/aa_site`) is prepended to `PYTHONPATH` above; spawn
workers re-run it at startup so the timing patch reaches them too.
